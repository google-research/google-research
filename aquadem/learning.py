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

"""Aquadem learner implementation."""

import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb


class TrainingState(NamedTuple):
  """Contains training state of the Aquadem learner."""
  discrete_rl_state: Any
  pretraining_state: Any


class PretrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  encoder_params: networks_lib.Params
  random_key: jnp.ndarray
  steps: int


class MultiBCLearner(acme.Learner):
  """Multicategorical BC learner."""

  _state: PretrainingState

  def __init__(self,
               network,
               random_key,
               temperature,
               num_actions,
               optimizer,
               demonstrations,
               num_sgd_steps_per_step,
               logger = None,
               counter = None):

    def aqualoss(params, transitions, key):
      predicted_actions = network.apply(
          params,
          transitions.observation,
          is_training=True,
          rngs={'dropout': key})
      predicted_actions = jnp.squeeze(predicted_actions)

      action_distances = jnp.sum(
          jnp.square(predicted_actions -
                     jnp.expand_dims(transitions.action, axis=-1)),
          axis=0)

      # softmin
      softmin_action_distances = temperature * (
          jax.nn.logsumexp(-action_distances / temperature)
          - jnp.log(num_actions)
      )
      loss = - softmin_action_distances
      return loss

    def batch_aqualoss(params, transitions, key):
      batched_aqualoss = jax.vmap(aqualoss, in_axes=(None, 0, None), out_axes=0)
      return jnp.mean(batched_aqualoss(params, transitions, key))

    def sgd_step(
        state,
        transitions,
    ):

      loss_and_grad = jax.value_and_grad(batch_aqualoss, argnums=0)

      # Compute losses and their gradients.
      loss_key, random_key = jax.random.split(state.random_key)
      loss_value, gradients = loss_and_grad(state.encoder_params, transitions,
                                            loss_key)

      update, optimizer_state = optimizer.update(
          gradients, state.optimizer_state, params=state.encoder_params)
      encoder_params = optax.apply_updates(state.encoder_params, update)

      new_state = PretrainingState(
          optimizer_state=optimizer_state,
          encoder_params=encoder_params,
          random_key=random_key,
          steps=state.steps + 1,
      )

      metrics = {
          'encoder_loss': loss_value,
      }

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter(prefix='encoder')
    self._logger = logger or loggers.make_default_logger(
        'encoder', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._demonstrations = demonstrations

    # Use the JIT compiler.
    self._sgd_step = utils.process_multiple_batches(
        sgd_step, num_sgd_steps_per_step)
    self._sgd_step = jax.jit(self._sgd_step)

    self._num_actions = num_actions

    encoder_params = network.init(random_key)
    optimizer_state = optimizer.init(encoder_params)

    # Create initial state.
    self._state = PretrainingState(
        optimizer_state=optimizer_state,
        encoder_params=encoder_params,
        random_key=random_key,
        steps=0,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self, transitions=None):
    # Get a batch of Transitions.
    if transitions is None:
      transitions = next(self._demonstrations)
    self._state, metrics = self._sgd_step(self._state, transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'aquadem_encoder': self._state.encoder_params,
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state


def _generate_aquadem_samples(
    demonstration_iterator,
    replay_iterator, encoder_apply,
    params, ratio,
    min_demo_reward):
  """Generator which creates the sample iterator for Aquadem.

  Args:
    demonstration_iterator: Iterator of demonstrations.
    replay_iterator: Replay buffer sample iterator.
    encoder_apply: encoder apply function
    params: parameters of the encoder
    ratio: probability with which we sample the expert demonstration
    min_demo_reward: minimum reward

  Yields:
    A batch of demonstrations or a batch of interaction from the replay buffer.
  """
  if min_demo_reward is None:
    min_demo_reward = -1e10

  for demonstrations, replay_sample in zip(demonstration_iterator,
                                           replay_iterator):

    if np.random.random() < ratio:
      continuous_actions_candidates = encoder_apply(
          params, demonstrations.observation)

      discrete_actions = np.argmin(
          np.linalg.norm(
              continuous_actions_candidates - demonstrations.action[:, :, None],
              axis=1),
          axis=-1)

      demonstrations = demonstrations._replace(
          action=discrete_actions,
          reward=np.maximum(min_demo_reward, demonstrations.reward))
      yield reverb.ReplaySample(info=replay_sample.info, data=demonstrations)

    else:
      yield reverb.ReplaySample(
          info=replay_sample.info, data=replay_sample.data)


class AquademLearner(acme.Learner):
  """Aquadem learner."""

  def __init__(self,
               random_key,
               discrete_rl_learner_factory,
               iterator,
               demonstrations_iterator,
               optimizer,
               make_demonstrations,
               networks,
               encoder_num_steps,
               encoder_batch_size,
               encoder_eval_every,
               temperature,
               demonstration_ratio,
               min_demo_reward,
               num_actions,
               counter = None,
               logger = None):

    random_key, key1 = jax.random.split(random_key, 2)
    demonstrations = make_demonstrations(
        encoder_batch_size * encoder_eval_every)
    self._pretraining_learner = MultiBCLearner(
        networks.encoder,
        random_key=key1,
        temperature=temperature,
        num_actions=num_actions,
        optimizer=optimizer,
        demonstrations=demonstrations,
        num_sgd_steps_per_step=encoder_eval_every)

    assert encoder_num_steps % encoder_eval_every == 0
    for _ in range(encoder_num_steps // encoder_eval_every):
      # The training logs are currently as coarse as the (not yet existing) eval
      self._pretraining_learner.step()

    encoder_apply = jax.jit(networks.encoder.apply)
    lfd_iterator = _generate_aquadem_samples(
        demonstrations_iterator,
        iterator,
        encoder_apply,
        self._pretraining_learner._state.encoder_params,
        ratio=demonstration_ratio,
        min_demo_reward=min_demo_reward)

    self._discrete_rl_learner = discrete_rl_learner_factory(
        networks.discrete_rl_networks, lfd_iterator)

  def step(self):
    self._discrete_rl_learner.step()

  def get_variables(self, names):
    variables = []
    for name in names:
      if name == 'aquadem_encoder':
        variables.append(self._pretraining_learner.get_variables([name])[0])
      else:
        variables.append(self._discrete_rl_learner.get_variables([name])[0])
    return variables

  def save(self):
    return TrainingState(pretraining_state=self._pretraining_learner.save(),
                         discrete_rl_state=self._discrete_rl_learner.save())

  def restore(self, state):
    self._pretraining_learner.restore(state.pretraining_state)
    self._discrete_rl_learner.restore(state.discrete_rl_state)
