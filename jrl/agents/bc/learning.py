# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""BC learner implementation."""

import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from collections import OrderedDict

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
from jrl.agents.bc import networks as bc_networks



class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  key: networks_lib.PRNGKey
  img_encoder_params: Optional[networks_lib.Params] = {}


class BCLearner(acme.Learner):
  """BC learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      iterator,
      policy_lr = 1e-4,
      loss_type = 'MLE', # or MSE
      regularize_entropy = False,
      entropy_regularization_weight = 1.0,
      use_img_encoder = False,
      img_encoder_params_ckpt_path = '',
      counter = None,
      logger = None,
      num_sgd_steps_per_step = 1):
    """Initialize the BC learner.

    Args:
      networks: BC networks
      rng: a key for random number generation.
      iterator: an iterator over training data.
      policy_lr: learning rate for the policy
      regularize_entropy: whether to regularize the entropy of the policy.
      entropy_regularization_weight: weight for entropy regularization.
      use_img_encoder: whether to preprocess the image part of the observation
        using a pretrained encoder.
      img_encoder_params_ckpt_path: path to checkpoint for image encoder params
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    assert loss_type in ['MLE', 'MSE'], 'Invalid BC loss type!'
    num_devices = len(jax.devices())
    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    self._use_img_encoder = use_img_encoder
    policy_optimizer = optax.adam(learning_rate=policy_lr)

    def actor_loss(
        policy_params,
        transitions,
        key,
        img_encoder_params):
      obs = transitions.observation
      acts = transitions.action

      if use_img_encoder:
        img = obs['state_image']
        dense = obs['state_dense']
        obs = dict(
            state_image=networks.img_encoder.apply(img_encoder_params, img),
            state_dense=dense,)

      dist = networks.policy_network.apply(policy_params, obs)
      if loss_type == 'MLE':
        log_probs = networks.log_prob(dist, acts)
        loss = -1. * jnp.mean(log_probs)
      else:
        acts_mode = dist.mode()
        mse = jnp.sum((acts_mode - acts)**2, axis=-1)
        loss = 0.5 * jnp.mean(mse)

      total_loss = loss
      entropy_term = 0.
      if regularize_entropy:
        sample_acts = networks.sample(dist, key)
        sample_log_probs = networks.log_prob(dist, sample_acts)
        entropy_term = jnp.mean(sample_log_probs)
        total_loss = total_loss + entropy_regularization_weight * entropy_term

      return total_loss, (loss, entropy_term)

    actor_loss_and_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def actor_update_step(
        policy_params,
        optim_state,
        transitions,
        key,
        img_encoder_params):
      (total_loss, (bc_loss_term, entropy_term)), actor_grad = actor_loss_and_grad(
          policy_params,
          transitions,
          key,
          img_encoder_params)
      actor_grad = jax.lax.pmean(actor_grad, 'across_devices')

      policy_update, optim_state = policy_optimizer.update(actor_grad, optim_state)
      policy_params = optax.apply_updates(policy_params, policy_update)

      return policy_params, optim_state, total_loss, bc_loss_term, entropy_term

    pmapped_actor_update_step = jax.pmap(
        actor_update_step,
        axis_name='across_devices',
        in_axes=0,
        out_axes=0)


    def _full_update_step(
        state,
        transitions,
    ):
      """The unjitted version of the full update step."""

      metrics = OrderedDict()

      key = state.key

      # actor update step
      def reshape_for_devices(t):
        rest_t_shape = list(t.shape[1:])
        new_shape = [num_devices, t.shape[0]//num_devices,] + rest_t_shape
        return jnp.reshape(t, new_shape)
      transitions = jax.tree_map(reshape_for_devices, transitions)
      sub_keys = jax.random.split(key, num_devices + 1)
      key = sub_keys[0]
      sub_keys = sub_keys[1:]

      new_policy_params, new_policy_optimizer_state, total_loss, bc_loss_term, entropy_term = pmapped_actor_update_step(
          state.policy_params,
          state.policy_optimizer_state,
          transitions,
          sub_keys,
          state.img_encoder_params)
      metrics['total_actor_loss'] = jnp.mean(total_loss)
      metrics['BC_loss'] = jnp.mean(bc_loss_term)
      metrics['entropy_loss'] = jnp.mean(entropy_term)


      # create new state
      new_state = TrainingState(
          policy_optimizer_state=new_policy_optimizer_state,
          policy_params=new_policy_params,
          key=key,
          img_encoder_params=state.img_encoder_params)

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    self._update_step = utils.process_multiple_batches(
        _full_update_step,
        num_sgd_steps_per_step)


    def make_initial_state(key):
      """"""
      # policy stuff
      key, sub_key = jax.random.split(key)
      policy_params = networks.policy_network.init(sub_key)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      devices = jax.local_devices()
      replicated_policy_params = jax.device_put_replicated(
          policy_params, devices)
      replicated_optim_state = jax.device_put_replicated(
          policy_optimizer_state, devices)

      if use_img_encoder:
        """
        Load pretrained img_encoder_params and do:
        replicated_img_encoder_params = jax.device_put_replicated(
            img_encoder_params, devices)
        """
        class EncoderTrainingState(NamedTuple):
          encoder_params: hk.Params
        img_encoder_params = {}
        replicated_img_encoder_params = img_encoder_params
        raise NotImplementedError('Need to load a checkpoint.')
      else:
        img_encoder_params = {}
        replicated_img_encoder_params = img_encoder_params

      state = TrainingState(
          policy_optimizer_state=replicated_optim_state,
          policy_params=replicated_policy_params,
          key=key,
          img_encoder_params=replicated_img_encoder_params)
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None


  def step(self):
    with jax.profiler.StepTraceAnnotation('sampling batch'):
      sample = next(self._iterator)
    transitions = types.Transition(*sample.data)

    with jax.profiler.StepTraceAnnotation('train step'):
      self._state, metrics = self._update_step(self._state, transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(
        steps=self._num_sgd_steps_per_step, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': jax.tree_map(lambda x: x[0], self._state.policy_params),
    }
    if self._use_img_encoder:
      img_encoder_params = jax.tree_map(
          lambda x: x[0], self._state.img_encoder_params)
      variables['img_encoder'] = img_encoder_params
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
