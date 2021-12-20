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

"""Contains training and sampling functions for vanilla autoregressive model."""
import functools
from typing import Any, Callable

from absl import logging
from flax import struct
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from autoregressive_diffusion.model import distributions
from autoregressive_diffusion.model.autoregressive_diffusion import ardm_utils
from autoregressive_diffusion.utils import util_fns


class ARM(struct.PyTreeNode):
  """Static model object that wraps important model functions."""
  config: ml_collections.config_dict.config_dict.ConfigDict
  apply_fn: Callable[Ellipsis, Any]
  logprob_fn: Callable[Ellipsis, Any]
  sample_fn: Callable[Ellipsis, Any]
  out_dist: Any
  neural_net: Any
  num_steps: int
  policy_support: bool = False
  num_stages: int = 1
  absorbing_state: int = 0

  def log_px(self, rng, params, x, train, context=None):
    net_out = self.apply_fn(
        {'params': params}, x, t=None, mask=None, train=train, context=context,
        rngs={'dropout': rng} if train else None)

    d = float(np.prod(net_out.shape[1:-1]))
    log_px_elementwise = util_fns.sum_except_batch(self.logprob_fn(x, net_out))
    log_px = log_px_elementwise / d / np.log(2)

    # Not implemented, but did not want to push through all the logic.
    # TODO(agritsenko): Implement this for DMoL.
    acc = jnp.ones_like(log_px)

    return log_px, acc, None, None

  def elbo(self, rng, params, x, train, context=None):
    return self.log_px(rng, params, x, train, context)

  def sample(
      self, rng, params, batch_size, context=None, chain_out_size = 50):
    assert chain_out_size >= 1
    if self.num_steps < chain_out_size:
      chain_out_size = self.num_steps

    chain_sharded = self.p_sample(
        rng, params, batch_size, context, chain_out_size)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(jax.pmap, in_axes=(None, None, 0, None, 0, None),
                     out_axes=1,
                     static_broadcasted_argnums=(0, 3, 5),
                     axis_name='batch')
  def p_sample(self, rng, params, batch_size, context, chain_out_size):
    """Samples from the model, calls sample_step for every timestep."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    assert batch_size % jax.local_device_count() == 0
    per_device_batch_size = batch_size // jax.local_device_count()
    logging.info('Sampling from model, hope you are patient...')

    orders = jnp.arange(0, self.num_steps)[None, :]
    orders = jnp.repeat(orders, repeats=per_device_batch_size, axis=0)

    chain = []
    x = jnp.full((per_device_batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state,
                 dtype=jnp.int32)
    chain.append(x)

    def next_sample_step(state, t):
      chain, x = state
      x = self.sample_step(
          jax.random.fold_in(rng, t), x,
          t, orders, params, context)

      # Compute the write index. Minimum is 0, maximum is chain_out_size - 1.
      write_index = (t * chain_out_size) // self.num_steps

      # May simply overwrite if write_index lands on the same index again, this
      # is desired behaviour and as a result the final index will also be the
      # complete sample.
      chain = jax.lax.dynamic_update_slice(
          chain, jnp.expand_dims(x, axis=0), (write_index,) + (0,) * x.ndim)
      return (chain, x), None

    # Every step of the generative process.
    ts = jnp.arange(self.num_steps)

    # `chain` is an output buffer that will contain intermediate states.
    chain = jnp.zeros(
        (chain_out_size, per_device_batch_size) + self.config.data_shape,
        dtype=x.dtype)

    state, _ = jax.lax.scan(
        next_sample_step, init=(chain, x), xs=ts)
    chain, _ = state

    return chain

  def sample_step(self, rng, x, t, sigmas, params, context):
    """Sampling code for a single step t."""
    batch_size = x.shape[0]
    t_batch = jnp.full(batch_size, fill_value=t)

    _, current_selection = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t_batch, self.config.data_shape)

    params_px = self.apply_fn(
        {'params': params},
        x, t=None, mask=None, train=False, context=context)

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
  def create(cls, config, get_architecture):
    """Creates a new instance with `step=0` and initialized `opt_state`."""

    if config.output_distribution == 'softmax':
      logging.info('Using softmax distribution')
      out_dist = distributions.SoftmaxCategorical(
          config.data_shape[-1], config.num_classes)
    elif config.output_distribution == 'discretized_logistic':
      logging.info('Using discretized logistic distribution')
      out_dist = distributions.DiscretizedMixLogistic(
          config.data_shape[-1], config.num_classes,
          n_mixtures=config.num_mixtures)
    else:
      raise ValueError

    required_num_outputs = out_dist.get_required_num_outputs()
    num_steps = int(np.prod(config.data_shape))

    # We set num_steps=0 since this disables time conditioning, which is not
    # necessary for ARMs.
    neural_net = get_architecture(
        config.num_classes, required_num_outputs, num_steps=0, is_causal=True)

    return cls(
        config,
        apply_fn=neural_net.apply,
        logprob_fn=out_dist.log_prob,
        sample_fn=out_dist.sample,
        out_dist=out_dist,
        neural_net=neural_net,
        num_steps=num_steps)
