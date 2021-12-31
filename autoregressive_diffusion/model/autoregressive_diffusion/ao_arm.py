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

"""Contains training and sampling functions for Autoregressive Diffusion."""
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
from autoregressive_diffusion.utils import dynamic_programming
from autoregressive_diffusion.utils import util_fns


Array = jnp.ndarray


class ArbitraryOrderARM(struct.PyTreeNode):
  """Static model object that wraps important model functions."""
  config: ml_collections.config_dict.config_dict.ConfigDict
  apply_fn: Callable[Ellipsis, Any]
  logprob_fn: Callable[Ellipsis, Any]
  sample_fn: Callable[Ellipsis, Any]
  out_dist: Any
  required_num_outputs: int
  neural_net: Any
  num_steps: int
  policy_support: bool = True
  num_stages: int = 1
  absorbing_state: int = 0

  def corrupt(self, x, intact_selection):
    assert intact_selection.dtype == jnp.int32

    # The corruption selection is the opposite of the intact selection.
    corrupt_selection = 1 - intact_selection

    # Keep the intact selection of x but let the
    corrupted = x * intact_selection + corrupt_selection * self.absorbing_state
    return corrupted

  def sample_qt(self, rng, batch_size):
    if self.config.elbo_mode == 'uniform':
      # Sample t ~ Uniform({1, ..., d-1}). Note randint is _exclusive_ maxval.
      t = jax.random.randint(rng, shape=(batch_size,), minval=0,
                             maxval=self.num_steps)
      prob_qt = 1. / self.num_steps
    elif self.config.elbo_mode == 'antithetic':
      t, prob_qt = ardm_utils.sample_antithetic(rng, batch_size, self.num_steps)

    else:
      raise ValueError

    return t, prob_qt

  def elbo(self, rng, params, x, train, context=None):
    d = np.prod(x.shape[1:])
    n_steps = np.prod(self.config.mask_shape)
    batch_size = x.shape[0]

    rng1, rng2, rng3 = jax.random.split(rng, 3)

    # Get random permutation sigma ~ Unif(S_n_steps)
    sigmas = ardm_utils.get_batch_permutations(rng1, x.shape[0], n_steps)

    # Sample t and prob_qt.
    t, prob_qt = self.sample_qt(rng2, batch_size)

    prev_selection, _ = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t, self.config.mask_shape)
    future_selection = (1. - prev_selection)

    corrupted = self.corrupt(x, prev_selection)

    net_out = self.apply_fn(
        {'params': params}, corrupted, t, prev_selection, train, context,
        rngs={'dropout': rng3} if train else None)

    log_px_sigma_geq_t = self.logprob_fn(x, net_out)

    log_px_sigma_geq_t = future_selection.reshape(
        log_px_sigma_geq_t.shape) * log_px_sigma_geq_t
    log_px_sigma_geq_t = util_fns.sum_except_batch(log_px_sigma_geq_t)

    # Negative cross-entropy.
    nce = log_px_sigma_geq_t / d / np.log(2)

    # Reweigh for summation over i.
    reweighting_factor_expectation_i = 1. / (n_steps - t)
    elbo_per_t = reweighting_factor_expectation_i * log_px_sigma_geq_t

    # Reweigh for expectation over t.
    reweighting_factor_expectation_t = 1. / prob_qt
    elbo = elbo_per_t * reweighting_factor_expectation_t

    elbo = elbo / d / np.log(2)
    elbo_per_t = elbo_per_t / d / np.log(2)

    return elbo, elbo_per_t, nce, t

  def sample(self, rng, params, batch_size,
             context=None, chain_out_size = 50):
    assert chain_out_size >= 1
    if self.num_steps < chain_out_size:
      chain_out_size = self.num_steps

    chain_sharded = self.p_sample(
        rng, params, batch_size, context, chain_out_size)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(jax.pmap,
                     in_axes=(None, None, 0, None, 0, None),
                     out_axes=1,
                     static_broadcasted_argnums=(0, 3, 5),
                     axis_name='batch')
  def p_sample(self, rng, params, batch_size, context, chain_out_size):
    """Samples from the model, calls sample_step for every timestep."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    assert batch_size % jax.local_device_count() == 0
    per_device_batch_size = batch_size // jax.local_device_count()
    logging.info('Sampling from model, hope you are patient...')

    rng, rng_perm = jax.random.split(rng)
    sigmas = ardm_utils.get_batch_permutations(rng_perm, per_device_batch_size,
                                               self.num_steps)

    x = jnp.full((per_device_batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state,
                 dtype=jnp.int32)

    def next_sample_step(state, t):
      chain, x = state
      x = self.sample_step(
          jax.random.fold_in(rng, t), x,
          t, sigmas, params, context)

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

  def sample_with_naive_policy(self,
                               rng,
                               params,
                               batch_size,
                               budget = 250,
                               context=None):
    logging.info('Sampling with naive policy.')
    naive_policy = self.get_naive_policy(budget)
    return self.sample_with_policy(rng, params, batch_size, naive_policy,
                                   context)

  def sample_with_policy(self, rng, params, batch_size, policy, context=None):
    """Wrapper for p_sample_with_policy that takes care of unsharding."""
    logging.info('Sampling from model (quickly)...')
    chain_sharded = self.p_sample_with_policy(rng, params, batch_size, policy,
                                              context)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(jax.pmap, in_axes=(None, None, 0, None, None, 0),
                     out_axes=1,
                     static_broadcasted_argnums=(0, 3), axis_name='batch')
  def p_sample_with_policy(self, rng, params, batch_size, policy, context):
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
          jax.random.fold_in(rng, idx), x, left_t, right_t, sigmas, params,
          context)
      return x, x

    x, chain = jax.lax.scan(next_sample_step, x, jnp.arange(len(policy)))
    return chain

  def sample_step_with_policy(self, rng, x, left_t, right_t, sigmas, params,
                              context):
    """Sampling code for a single step starting at left_t until right_t."""
    batch_size = x.shape[0]
    left_t = jnp.full(batch_size, fill_value=left_t)
    right_t = jnp.full(batch_size, fill_value=right_t)

    prev_selection, current_selection = ardm_utils.get_selections_for_sigma_and_range(
        sigmas, left_t, right_t, self.config.mask_shape)

    params_px = self.apply_fn(
        {'params': params},
        x, left_t, prev_selection, train=False, context=context)

    new_x = self.sample_fn(rng, params_px)
    x = (1 - current_selection) * x + current_selection * new_x
    x = jnp.asarray(x, jnp.int32)
    return x

  def sample_step(self, rng, x, t, sigmas, params, context):
    """Sampling code for a single step t."""
    batch_size = x.shape[0]
    t_batch = jnp.full(batch_size, fill_value=t)

    prev_selection, current_selection = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t_batch, self.config.mask_shape)
    params_px = self.apply_fn(
        {'params': params},
        x, t_batch, prev_selection, train=False, context=context)

    new_x = self.sample_fn(rng, params_px)
    x = (1 - current_selection) * x + current_selection * new_x
    x = jnp.asarray(x, jnp.int32)
    return x

  def sample_policy_t(self, rng, batch_size, policy):
    """Samples a step from a policy."""
    len_policy = len(policy)

    idx = jax.random.randint(
        rng, shape=(batch_size,), minval=0, maxval=len_policy)
    p_idx = 1. / float(len_policy)

    # Include final step in policy to compute ranges.
    policy_extended = jnp.concatenate(
        [policy, jnp.array([self.num_steps], dtype=jnp.int32)], axis=0)

    low_t = policy_extended[idx]
    high_t = policy_extended[idx + 1]

    num_tokens_in_parallel = high_t - low_t

    weight_policy = jnp.asarray(num_tokens_in_parallel, jnp.float32) / p_idx

    return low_t, high_t, weight_policy

  def elbo_with_policy(self, rng, params, x, policy, train, context=None):
    """Computes the ELBO for AO-ARMs using uniform distribution over policy.

    Args:
      rng: Random number key.
      params: Parameters for the apply_fn.
      x: Input image.
      policy: An array of integers describing the generative model,
        parallelizing sampling steps if integers are missing. For example, the
        list [0, 2, 4, 5] indicates that step 0 & 1 should be generated in
        parallel, then then 2 & 3 in parallel and then 4 (individually) and then
        5, ..., n_steps - 1 (in parallel).
      train: Is the model in train or eval mode?
      context: Anything the model might want to condition on.

    Returns:
      elbo: batch of stochastic elbo estimates.
      ce: batch of the direct cross-entropy loss
      t: batch timesteps that were sampled
    """
    d = np.prod(x.shape[1:])
    batch_size = x.shape[0]

    rng_perm, rng_t, rng_drop = jax.random.split(rng, 3)

    # Get random sigma ~ Unif(S_n_steps)
    sigmas = ardm_utils.get_batch_permutations(rng_perm, x.shape[0],
                                               self.num_steps)

    # Sample t from policy.
    t, _, weight_policy = self.sample_policy_t(rng_t, batch_size, policy)

    prev_selection, _ = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t, self.config.mask_shape)
    future_selection = (1. - prev_selection)

    corrupted = self.corrupt(x, prev_selection)

    net_out = self.apply_fn(
        {'params': params}, corrupted, t, prev_selection, train,
        rngs={'dropout': rng_drop} if train else None, context=context)

    log_px_sigma_geq_t = self.logprob_fn(x, net_out)

    log_px_sigma_geq_t = future_selection.reshape(
        log_px_sigma_geq_t.shape) * log_px_sigma_geq_t
    log_px_sigma_geq_t = util_fns.sum_except_batch(log_px_sigma_geq_t)

    ce = log_px_sigma_geq_t / d / np.log(2)

    # Reweigh for expectation over i.
    reweighting_factor_expectation_i = 1. / (self.num_steps - t)
    elbo_per_t = reweighting_factor_expectation_i * log_px_sigma_geq_t

    # Reweigh for expectation over policy.
    elbo = elbo_per_t * weight_policy

    elbo = elbo / d / np.log(2)
    elbo_per_t = elbo_per_t / d / np.log(2)

    return elbo, elbo_per_t, ce, t

  def log_prob_with_policy_and_sigma(self, rng, params, x, policy, sigmas,
                                     train, context=None):
    """Expected log prob with specific policy and generation order sigma.

    Computes the log probability for AO-ARMs using a specific policy _and_ a
    specific permutation sigma. The given permutation makes this exact (hence
    log prob), the policy ensures that the estimator has reasonable variance.

    Args:
      rng: Random number key.
      params: Parameters for the apply_fn.
      x: Input image.
      policy: An array of integers describing the generative model,
        parallelizing sampling steps if integers are missing. For example, the
        list [0, 2, 4, 5] indicates that step 0 & 1 should be generated in
        parallel, then then 2 & 3 in parallel and then 4 (individually) and then
        5, ..., n_steps - 1 (in parallel).
      sigmas: An array describing the generation order that is being enforced.
      train: Is the model in train or eval mode?
      context: Anything the model might want to condition on.

    Returns:
      log_prob: batch of stochastic log probability estimates.
    """
    d = np.prod(x.shape[1:])
    batch_size = x.shape[0]

    # Expand the dimensions of sigma if only a single order is given.
    if len(sigmas.shape) == 1:
      sigmas = jnp.repeat(sigmas[None], repeats=batch_size, axis=0)
    assert sigmas.shape == (batch_size, self.num_steps), (
        f'{sigmas.shape} does not match')

    rng_t, rng_drop = jax.random.split(rng, 2)

    # Sample t from policy.
    left_t, right_t, weight_policy = self.sample_policy_t(
        rng_t, batch_size, policy)
    num_tokens_in_parallel = right_t - left_t

    prev_selection, current_selection = ardm_utils.get_selections_for_sigma_and_range(
        sigmas, left_t, right_t, self.config.mask_shape)

    corrupted = self.corrupt(x, prev_selection)

    net_out = self.apply_fn(
        {'params': params}, corrupted, left_t, prev_selection, train,
        rngs={'dropout': rng_drop} if train else None, context=context)

    log_px_sigma_geq_t = self.logprob_fn(x, net_out)

    current_selection = current_selection.reshape(log_px_sigma_geq_t.shape)
    log_px_sigma_geq_t = current_selection * log_px_sigma_geq_t
    log_px_sigma_geq_t = util_fns.sum_except_batch(log_px_sigma_geq_t)

    # Reweigh for expectation over policy.
    log_prob = log_px_sigma_geq_t / num_tokens_in_parallel * weight_policy
    log_prob = log_prob / d / np.log(2)

    return log_prob

  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 5))
  def p_apply(self, params, x, t, selection, train, context=None):
    return self.apply_fn(
        {'params': params}, x, t, selection, train=train,
        rngs=None, context=context)

  def batch_and_apply(self, params, x, t, selection, train, context=None):
    """Batches and applies the network. Useful in non-jitted routines."""
    batch_size = x.shape[0]
    assert batch_size == selection.shape[0]
    assert batch_size == t.shape[0]
    x = util_fns.batch(x, jax.local_device_count())
    t = util_fns.batch(t, jax.local_device_count())
    selection = util_fns.batch(selection, jax.local_device_count())
    if context is not None:
      context = util_fns.batch(context, jax.local_device_count())
    net_out = self.p_apply(params, x, t, selection, train, context)
    return util_fns.unbatch(net_out)

  def get_random_order(self, rng):
    return jax.random.permutation(rng, jnp.arange(self.num_steps))

  def decode_with_policy_and_sigma(self, streams, params, policy, sigmas,
                                   batch_size,
                                   context=None):
    """Decode variables from streams with policy and generation order sigma.

    Losslessly decodes the batch of variables to a batch of bitstreams using
    entropy coding. Note that policy and sigma should be the _exact_ same as
    were used to do the encoding.

    Args:
      streams: A batch of bitstream objects.
      params: Parameters for the apply_fn.
      policy: An array of integers describing the generative model,
        parallelizing sampling steps if integers are missing. For example, the
        list [0, 2, 4, 5] indicates that step 0 & 1 should be generated in
        parallel, then then 2 & 3 in parallel and then 4 (individually) and then
        5, ..., n_steps - 1 (in parallel).
      sigmas: An array describing the generation order that is being enforced.
      batch_size: Number of items in a batch.
      context: Anything the model might want to condition on.

    Returns:
      A tuple containing the decoded variables, and the batch of bitstreams.
    """
    # Initialize starting x.
    x = jnp.full((batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state,
                 dtype=jnp.int32)

    # Expand the dimensions of sigma if only a single order is given.
    if len(sigmas.shape) == 1:
      sigmas = jnp.repeat(sigmas[None], repeats=batch_size, axis=0)
    assert sigmas.shape == (batch_size, self.num_steps), (
        f'{sigmas.shape} does not match ({batch_size}, {self.num_steps})')

    # Extend policy.
    policy_extended = jnp.concatenate(
        [policy, jnp.array([self.num_steps], dtype=jnp.int32)], axis=0)

    # Loop over policy indices.
    for idx in range(0, len(policy)):
      left_t = policy_extended[idx]
      right_t = policy_extended[idx + 1]

      left_t_batch = jnp.full(batch_size, fill_value=left_t)

      prev_selection, _ = ardm_utils.get_selection_for_sigma_and_t(
          sigmas, left_t_batch, self.config.mask_shape)

      net_out = self.batch_and_apply(
          params, x, left_t_batch, prev_selection,
          train=False, context=context)

      # Loop over timesteps in the current policy.
      for t in range(left_t, right_t):
        t_batch = jnp.full(batch_size, fill_value=t)
        _, current_selection = ardm_utils.get_selection_for_sigma_and_t(
            sigmas, t_batch, self.config.mask_shape)

        # Decode the current value at place t.
        decoded_value, streams = self.out_dist.decode(streams, net_out,
                                                      current_selection)

        # Write decode value to x.
        x = x * (1 - current_selection) + decoded_value * current_selection

      del net_out, prev_selection

    return x, streams

  def encode_with_policy_and_sigma(self,
                                   streams,
                                   params,
                                   x,
                                   policy,
                                   sigmas,
                                   context=None):
    """Encode x to streams with specific policy and generation order sigma.

    Losslessly encodes the batch of x to a batch of bitstreams using entropy
    coding.

    Args:
      streams: A batch of bitstream objects.
      params: Parameters for the apply_fn.
      x: Input image.
      policy: An array of integers describing the generative model,
        parallelizing sampling steps if integers are missing. For example, the
        list [0, 2, 4, 5] indicates that step 0 & 1 should be generated in
        parallel, then then 2 & 3 in parallel and then 4 (individually) and then
        5, ..., n_steps - 1 (in parallel).
      sigmas: An array describing the generation order that is being enforced.
      context: Anything the model might want to condition on.

    Returns:
      The batch of bitstream objects.
    """
    batch_size = x.shape[0]

    # Expand the dimensions of sigma if only a single order is given.
    if len(sigmas.shape) == 1:
      sigmas = jnp.repeat(sigmas[None], repeats=batch_size, axis=0)
    assert sigmas.shape == (batch_size, self.num_steps), (
        f'{sigmas.shape} does not match ({batch_size}, {self.num_steps})')

    # Extend policy.
    policy_extended = jnp.concatenate(
        [policy, jnp.array([self.num_steps], dtype=jnp.int32)], axis=0)

    # Reversed, recall our entropy coder behaves like a stack.
    for idx in reversed(range(len(policy))):
      left_t = policy_extended[idx]
      right_t = policy_extended[idx + 1]

      # Compute current masking.
      left_t_batch = jnp.full(batch_size, fill_value=left_t)
      prev_selection, _ = ardm_utils.get_selection_for_sigma_and_t(
          sigmas, left_t_batch, self.config.mask_shape)

      # Take next step in destruction process.
      corrupted = self.corrupt(x, prev_selection)

      # The network call is pmapped here.
      net_out = self.batch_and_apply(
          params, corrupted, left_t_batch, prev_selection, train=False,
          context=context)

      # Again reversed, entropy coder behaves like a stack. Here the same
      # net_out is used for all range(left_t, right_t).
      for t in reversed(range(left_t, right_t)):
        t_batch = jnp.full(batch_size, fill_value=t)
        _, current_selection = ardm_utils.get_selection_for_sigma_and_t(
            sigmas, t_batch, self.config.mask_shape)

        streams = self.out_dist.encode(streams, x, net_out, current_selection)

      del net_out, prev_selection

    return streams

  def compute_policies_and_costs(self, kl_per_t, budgets):
    assert kl_per_t.shape[0] == self.num_steps

    # Sorting reduces bias in the dynamic programming computation. Otherwise
    # it easily exploits the non-monotonicity due to stochastic estimates.
    kl_values_sorted = jnp.sort(kl_per_t)[::-1]
    policies, costs = dynamic_programming.compute_fixed_budget(
        kl_values_sorted, budgets)

    return policies, costs

  def get_naive_policy(self, budget = 250):
    assert budget <= self.num_steps
    # We use budget+1 because a linspace contains the last step.
    naive_policy = ardm_utils.integer_linspace(0, self.num_steps, budget+1)

    # Last index does not need to be in policy.
    naive_policy = naive_policy[:-1]
    return naive_policy

  def compute_cost_with_policy(self, kl_per_t, policy):
    return dynamic_programming.compute_cost_with_policy(kl_per_t, policy)

  def init_architecture(self, init_rng, tmp_x, tmp_t, context=None):
    tmp_mask = jnp.ones([1, *self.config.mask_shape])
    return self.neural_net.init(init_rng, tmp_x, tmp_t, tmp_mask, train=False,
                                context=context)

  @classmethod
  def create(cls, config, get_architecture, absorbing_state):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    assert absorbing_state >= 0

    if absorbing_state == config.num_classes:
      num_input_classes = config.num_classes + 1
    elif absorbing_state < config.num_classes:
      num_input_classes = config.num_classes
    else:
      # A waste of space, at most you want to append the absorbing state to the
      # first new class at the end.
      raise ValueError(f'Absorbing state {absorbing_state} invalid for '
                       f'num_classes {config.n_classes}')

    if config.output_distribution == 'softmax':
      logging.info('Using softmax distribution')
      out_dist = distributions.SoftmaxCategorical(
          config.data_shape[-1], config.num_classes)
      mask_shape = config.data_shape
    elif config.output_distribution == 'discretized_logistic':
      logging.info('Using discretized logistic distribution')
      out_dist = distributions.DiscretizedMixLogistic(
          config.data_shape[-1], config.num_classes,
          n_mixtures=config.num_mixtures)
      mask_shape = config.data_shape
    elif config.output_distribution == 'discretized_logistic_rgb':
      logging.info('Using discretized logistic distribution')
      out_dist = distributions.DiscretizedMixLogisticRGB(
          config.data_shape[-1], config.num_classes, n_mixtures=30)
      mask_shape = config.data_shape[:-1] + (1,)
    else:
      raise ValueError

    config.mask_shape = mask_shape
    required_num_outputs = out_dist.get_required_num_outputs()
    num_steps = int(np.prod(mask_shape))

    neural_net = get_architecture(
        num_input_classes, required_num_outputs, num_steps)

    return cls(
        config,
        apply_fn=neural_net.apply,
        logprob_fn=out_dist.log_prob,
        sample_fn=out_dist.sample,
        out_dist=out_dist,
        required_num_outputs=required_num_outputs,
        neural_net=neural_net,
        num_steps=num_steps,
        absorbing_state=absorbing_state,
    )
