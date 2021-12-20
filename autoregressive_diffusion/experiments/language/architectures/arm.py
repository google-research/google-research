# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Contains training and sampling functions an autoregressive model."""
import functools
from typing import Any, Callable

from absl import logging
from flax import linen as nn
from flax import struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from autoregressive_diffusion.model import distributions
from autoregressive_diffusion.model.autoregressive_diffusion import ardm_utils
from autoregressive_diffusion.utils import util_fns


def cross_entropy(logits, targets):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  onehot_targets = common_utils.onehot(targets, vocab_size)

  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)

  d = np.prod(targets.shape[1:])

  loss = util_fns.sum_except_batch(loss) / d / np.log(2)

  return loss


def compute_accuracy(logits, targets):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.shape[:-1] != targets.shape[:-1]:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  logits = logits[:, :, None, :]  # Insert empty channel axis.
  d = float(np.prod(logits.shape[1:-1]))
  acc = jnp.equal(jnp.argmax(logits, axis=-1), targets) / d
  acc = util_fns.sum_except_batch(acc)
  return acc


class ARM(struct.PyTreeNode):
  """Static model object that wraps important model functions."""
  config: ml_collections.config_dict.config_dict.ConfigDict
  apply_fn: Callable[Ellipsis, Any]
  logprob_fn: Callable[Ellipsis, Any]
  sample_fn: Callable[Ellipsis, Any]
  neural_net: Any
  num_steps: int
  policy_support: bool = False
  num_stages: int = 1
  absorbing_state: int = 0
  random_order: bool = False

  def log_px(self, rng, params, x, train, context=None):
    batch_size = x.shape[0]
    if self.random_order:
      logging.info('Log-likelihood for a random-order ARM XLNet style.')
      rng, rng_perm = jax.random.split(rng)
      permutations = ardm_utils.get_batch_permutations(rng_perm, batch_size,
                                                       self.num_steps)
    else:
      logging.info('Log-likelihood for a standard ARM.')
      permutations = None

    net_out = self.apply_fn(
        {'params': params}, x, t=None, mask=None, train=train, context=context,
        permutations=permutations,
        rngs={'dropout': rng} if train else None)

    d = float(np.prod(net_out.shape[1:-1]))
    log_px_elementwise = util_fns.sum_except_batch(self.logprob_fn(x, net_out))
    log_px = log_px_elementwise / d / np.log(2)
    neg_acc = -compute_accuracy(logits=net_out, targets=x)

    t_batch_dummy = jnp.zeros((batch_size,), dtype=jnp.int32)
    loss_components_dummy = jnp.zeros((batch_size,))

    return log_px, loss_components_dummy, neg_acc, t_batch_dummy

  def elbo(self, rng, params, x, train, context=None):
    return self.log_px(rng, params, x, train, context)

  def sample(self, rng, params, batch_size, context=None):
    chain_sharded = self.p_sample(rng, params, batch_size, context)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(jax.pmap, in_axes=(None, None, 0, None, 0),
                     out_axes=1,
                     static_broadcasted_argnums=(0, 3), axis_name='batch')
  def p_sample(self, rng, params, batch_size, context):
    """Samples from the model, calls sample_step for every timestep."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    assert batch_size % jax.local_device_count() == 0
    per_device_batch_size = batch_size // jax.local_device_count()
    logging.info('Sampling from model, hope you are patient...')

    if self.random_order:
      rng, rng_perm = jax.random.split(rng)
      orders = ardm_utils.get_batch_permutations(rng_perm,
                                                 per_device_batch_size,
                                                 self.num_steps)
    else:
      orders = jnp.arange(0, self.num_steps)[None, :]
      orders = jnp.repeat(orders, repeats=per_device_batch_size, axis=0)

    chain = []
    x = jnp.full((per_device_batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state,
                 dtype=jnp.int32)
    chain.append(x)

    def next_sample_step(x, t):
      x = self.sample_step(
          jax.random.fold_in(rng, t), x,
          t, orders, params, context)
      return x, x

    ts = jnp.arange(self.num_steps)
    _, chain = jax.lax.scan(next_sample_step, init=x, xs=ts)

    return chain

  def get_naive_policy(self, budget=250):
    assert budget <= self.num_steps
    # We use budget+1 because a linspace contains the last step.
    naive_policy = ardm_utils.integer_linspace(0, self.num_steps, budget+1)

    # Last index does not need to be in policy.
    naive_policy = naive_policy[:-1]
    return naive_policy

  def sample_with_naive_policy(self,
                               rng,
                               params,
                               batch_size,
                               budget=250):
    logging.info('Sampling with naive policy.')
    naive_policy = self.get_naive_policy(budget)
    return self.sample_with_policy(rng, params, batch_size, naive_policy)

  def sample_with_policy(self, rng, params, batch_size, policy):
    """Wrapper for p_sample_with_policy that takes care of unsharding."""
    logging.info('Sampling from model (quickly)...')
    chain_sharded = self.p_sample_with_policy(rng, params, batch_size, policy)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(jax.pmap, in_axes=(None, None, 0, None, None),
                     out_axes=1,
                     static_broadcasted_argnums=(0, 3), axis_name='batch')
  def p_sample_with_policy(self, rng, params, batch_size, policy):
    """Samples from the model, calls sample_step for every policy step."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    assert batch_size % jax.local_device_count() == 0
    per_device_batch_size = batch_size // jax.local_device_count()

    rng, rng_perm = jax.random.split(rng)
    sigmas = ardm_utils.get_batch_permutations(rng_perm, per_device_batch_size,
                                               self.num_steps)

    policy_extended = jnp.concatenate(
        [policy, jnp.array([self.num_steps], dtype=jnp.int32)], axis=0)

    x = jnp.full((per_device_batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state,
                 dtype=jnp.int32)

    def next_sample_step(x, idx):
      left_t = policy_extended[idx]
      right_t = policy_extended[idx + 1]
      x = self.sample_step_with_policy(
          jax.random.fold_in(rng, idx), x, left_t, right_t, sigmas, params)
      return x, x

    x, chain = jax.lax.scan(next_sample_step, x, jnp.arange(len(policy)))
    return chain

  def sample_step_with_policy(self, rng, x, left_t, right_t, sigmas, params):
    """Sampling code for a single step starting at left_t until right_t."""
    batch_size = x.shape[0]
    left_t = jnp.full(batch_size, fill_value=left_t)
    right_t = jnp.full(batch_size, fill_value=right_t)

    prev_selection, current_selection = ardm_utils.get_selections_for_sigma_and_range(
        sigmas, left_t, right_t, self.config.data_shape)

    params_px = self.apply_fn(
        {'params': params},
        x, left_t, prev_selection, train=False)

    new_x = self.sample_fn(rng, params_px)
    x = (1 - current_selection) * x + current_selection * new_x
    x = jnp.asarray(x, jnp.int32)
    return x

  def sample_step(self, rng, x, t, sigmas, params, context):
    """Sampling code for a single step t."""
    batch_size = x.shape[0]
    t_batch = jnp.full(batch_size, fill_value=t)

    prev_selection, current_selection = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t_batch, self.config.data_shape)

    if self.random_order:
      permutations = sigmas
    else:
      permutations = None

    params_px = self.apply_fn(
        {'params': params},
        x, t_batch, prev_selection, train=False, context=context,
        permutations=permutations)

    new_x = self.sample_fn(rng, params_px)
    x = (1 - current_selection) * x + current_selection * new_x
    x = jnp.asarray(x, jnp.int32)
    return x

  def init_architecture(self, init_rng, tmp_x, tmp_t, context=None):
    tmp_mask = None
    if context is None:
      return self.neural_net.init(init_rng, tmp_x, tmp_t, tmp_mask, train=False)
    else:
      return self.neural_net.init(init_rng, tmp_x, tmp_t, tmp_mask,
                                  train=False, context=context)

  @classmethod
  def create(cls, config, get_architecture, random_order):
    """Creates a new instance with `step=0` and initialized `opt_state`."""

    required_num_outputs = config.num_classes
    num_steps = int(np.prod(config.data_shape))

    # We set num_steps=0 since this disables time conditioning, which is not
    # necessary for ARMs.
    neural_net = get_architecture(
        config.num_classes, required_num_outputs, num_steps=0, is_causal=True)

    out_dist = distributions.SoftmaxCategorical(config.data_shape[-1],
                                                config.num_classes)

    return cls(
        config,
        apply_fn=neural_net.apply,
        logprob_fn=out_dist.log_prob,
        sample_fn=out_dist.sample,
        neural_net=neural_net,
        num_steps=num_steps,
        random_order=random_order
    )
