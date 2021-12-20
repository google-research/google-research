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

from autoregressive_diffusion.model.autoregressive_diffusion import ardm_utils
from autoregressive_diffusion.model.autoregressive_diffusion import transitions

from autoregressive_diffusion.utils import distribution_utils
from autoregressive_diffusion.utils import dynamic_programming
from autoregressive_diffusion.utils import util_fns


Array = jnp.ndarray


class BitUpscaleAutoregressiveDiffusion(struct.PyTreeNode):
  """Static model object that wraps important model functions."""
  config: ml_collections.config_dict.config_dict.ConfigDict
  apply_fn: Callable[Ellipsis, Any]
  required_num_outputs: int
  neural_net: Any
  num_steps_per_stage: int
  num_steps: int
  num_stages: int
  num_input_classes: int
  num_output_classes: int
  transition_matrices: jnp.ndarray
  cum_matmul_transition_matrices: jnp.ndarray
  absorbing_state: int
  branch_factor: int
  direct_parametrization: bool
  policy_support: bool = True
  max_frames_per_stage: int = 50

  def transform_with_matrix(self, x_onehot, matrix):
    assert x_onehot.shape[0] == matrix.shape[0], (f'{x_onehot.shape} does not '
                                                  f'match {matrix.shape}')
    assert x_onehot.shape[-1] == matrix.shape[-2], (f'{x_onehot.shape} does not'
                                                    f' match {matrix.shape}')

    num_data_dims = len(x_onehot.shape) - 2  # minus batch, category axes.
    new_axes = tuple(range(1, num_data_dims + 1))
    matrix_inflated = jnp.expand_dims(matrix, axis=new_axes)

    out = jnp.matmul(matrix_inflated, x_onehot[Ellipsis, None])
    out = jnp.squeeze(out, axis=-1)

    assert out.shape == x_onehot.shape
    return out

  def log_prob_vector_future_given_past(self, logits, x_past, stage):
    probs = jax.nn.softmax(logits, axis=-1)
    if self.num_input_classes > self.num_output_classes:
      # We have to pad probs with empty probabilities for augmented classes.
      pad_size = self.num_input_classes - self.num_output_classes

      # Only padding at the end of the last axis.
      padding = ((0, 0),) * (len(logits.shape) - 1) + ((0, pad_size),)

      # Vector is zero-padded since it is in log-space.
      probs = jnp.pad(probs, pad_width=padding,
                      mode='constant', constant_values=0)

    stage_reversed = self.num_stages - 1 - stage
    transition_future = self.cum_matmul_transition_matrices[stage_reversed]

    transition_single_step = self.transition_matrices[stage_reversed]

    probs_future = self.transform_with_matrix(probs, transition_future)

    x_past_onehot = util_fns.onehot(x_past, self.num_input_classes)

    # We only select possible futures:
    num_data_axes = len(x_past.shape) - 1  # minus batch size.
    new_axes = tuple(range(1, num_data_axes + 1))
    possible_futures = jnp.sum(
        jnp.expand_dims(transition_single_step, axis=new_axes) *
        x_past_onehot[Ellipsis, None],
        axis=-2)

    # Collect logits given past, using only possible futures.
    infty = 1e20
    logits_given_past = jnp.log(probs_future * possible_futures +
                                1e-10) - (1 - possible_futures) * infty

    # Forces stability. Corresponds to -jnp.log(prob_past[..., None]) but
    # this is numerically more stable.
    log_probs = logits_given_past - jax.nn.logsumexp(
        logits_given_past, axis=-1, keepdims=True)

    return log_probs

  def sample_q_t(self, rng, batch_size):
    if self.config.elbo_mode == 'uniform':
      # Sample t ~ Uniform({1, ..., d-1}). Note randint is _exclusive_ maxval.
      t = jax.random.randint(rng, shape=(batch_size,), minval=0,
                             maxval=self.num_steps)
      stage = t // self.num_steps_per_stage
      t_in_stage = t % self.num_steps_per_stage
      prob_qt = 1. / self.num_steps
    elif self.config.elbo_mode == 'antithetic':
      rng_stage, rng_t_in_stage = jax.random.split(rng, 2)
      stage, _ = ardm_utils.sample_antithetic(rng_stage, batch_size,
                                              self.num_stages)
      t_in_stage, _ = ardm_utils.sample_antithetic(rng_t_in_stage, batch_size,
                                                   self.num_steps_per_stage)
      t = t_in_stage + stage * self.num_steps_per_stage
      prob_qt = 1. / self.num_steps
    else:
      raise ValueError

    return t, prob_qt, stage, t_in_stage

  def sample_policy_t(self, rng, policy):
    """Samples a timestep t from a policy distribution.

    This function takes in a random key and a policy. The first axis of policy
    is the batch dimension, the second is the policy axis itself. This is
    in this particular way because depending on the sampled stage, different
    datapoints may have a different policy. After sampling an index uniformly,
    the left and right bounds in the policy are retrieved, then the elbo is
    computed as if all tokens between these indices are generated.

    Args:
      rng: An RNG key.
      policy: A policy array with dimensions (batch_size, policy_length).

    Returns:
      low_t: The left bound of the generation step.
      high_t: The right bound of the generation step.
      weight_policy: The weight of the step when computing an ELBO.
    """
    batch_size = policy.shape[0]
    len_policy = policy.shape[1]

    idx = jax.random.randint(
        rng, shape=(batch_size,), minval=0, maxval=len_policy)
    p_idx = 1. / float(len_policy)

    # Include final step in policy to compute ranges.
    extension = jnp.repeat(
        jnp.array([[self.num_steps_per_stage]], dtype=jnp.int32),
        batch_size, axis=0)
    policy_extended = jnp.concatenate([policy, extension], axis=1)

    arange = jnp.arange(batch_size)
    low_t = policy_extended[arange, idx]
    high_t = policy_extended[arange, idx+1]

    assert low_t.shape == (batch_size,)
    assert high_t.shape == (batch_size,)

    num_tokens_in_parallel = high_t - low_t

    weight_policy = jnp.asarray(num_tokens_in_parallel, jnp.float32) / p_idx

    return low_t, high_t, weight_policy

  def corrupt(self, x, stage):
    """Corrupts x to a lower depth version for each stage."""
    # Note: we take the _generative process_ order. This is the reverse of the
    # diffusion process. However, the transition matrices are defined in the
    # order of the diffusion process. So here future is 's' and past 's+1'.
    stage_reverse = self.num_stages - 1 - stage
    if self.direct_parametrization:
      t = jnp.expand_dims(stage_reverse, np.arange(1, len(x.shape)))
      # The branch_factor determines at which speed the integers are reduced.
      # For example: suppose we have a 16-bit problem and we want to model it
      # in two stages, we could pick a branch factor of 256. Then, 16383 // 256
      #  = 255 for the first corruption and 255 // 256 = 0 for the second
      # corruption.
      x_past = (x // self.branch_factor**(t+1)) * self.branch_factor**(t+1)
      x_future = (x // self.branch_factor**t) * self.branch_factor**t

    else:
      transition_future = self.cum_matmul_transition_matrices[stage_reverse]
      transition_past = self.cum_matmul_transition_matrices[stage_reverse+1]

      # Here we compute the past and future x representations.
      x_onehot = util_fns.onehot(x, self.num_input_classes)
      x_past_onehot = self.transform_with_matrix(x_onehot, transition_past)
      x_past = jnp.argmax(x_past_onehot, axis=-1)
      x_future_onehot = self.transform_with_matrix(x_onehot,
                                                   transition_future)
      x_future = jnp.argmax(x_future_onehot, axis=-1)

    return x_future, x_past

  def log_prob_for_x_future_given_past(self, x_future, x_past, net_out, stage):
    """Computes the log probability of x_future given x_past and net_out.

    Args:
      x_future: The variable to compute the log probability for.
      x_past: The variable to condition on.
      net_out: Network output containing the logits.
      stage: The stage of the specific datapoint.

    Returns:
      The elementwise log probabilities of x_future.
    """
    batch_size = x_future.shape[0]
    if self.direct_parametrization:
      logits_per_stage = net_out.reshape(*x_future.shape, self.num_stages,
                                         self.branch_factor)

      # Retrieve the logits for this specific stage.
      logits = logits_per_stage[jnp.arange(batch_size), Ellipsis, stage, :]
      log_probs = jax.nn.log_softmax(logits, axis=-1)

      t = self.num_stages - 1 - stage
      t = jnp.expand_dims(t, np.arange(1, len(x_future.shape)))
      x_target = (x_future // self.branch_factor**t) % self.branch_factor

      x_target_onehot = util_fns.onehot(x_target, self.branch_factor)

      log_prob_future_given_past = jnp.sum(log_probs * x_target_onehot, axis=-1)

    else:
      logits = net_out.reshape(*x_future.shape, self.num_output_classes)
      log_prob_future_given_past = self.log_prob_vector_future_given_past(
          logits, x_past, stage)
      x_future_onehot = util_fns.onehot(x_future, self.num_input_classes)
      log_prob_future_given_past = jnp.sum(
          log_prob_future_given_past * x_future_onehot, axis=-1)

    return log_prob_future_given_past

  def sample_x_future_given_past(self, rng, x_past, net_out, stage):
    """Samples a future x given x_past with parameters given by net_out.

    Args:
      rng: Random number key.
      x_past: Previous x in the generative process.
      net_out: The network output containing the logits.
      stage: The stage that is currently being generated.

    Returns:
      The new sampled x.
    """
    batch_size = x_past.shape[0]
    if self.direct_parametrization:
      logits_per_stage = net_out.reshape(*x_past.shape, self.num_stages,
                                         self.branch_factor)

      # Retrieve the logits for this specific stage.
      logits = logits_per_stage[jnp.arange(batch_size), Ellipsis, stage, :]
      log_probs = jax.nn.log_softmax(logits, axis=-1)
      sample = distribution_utils.sample_categorical(rng, log_probs)

      t = self.num_stages - 1 - stage
      t = jnp.expand_dims(t, np.arange(1, len(x_past.shape)))
      x_sample = x_past + sample * self.branch_factor**t

    else:
      logits = net_out.reshape(*x_past.shape, self.num_output_classes)

      log_probs_future = self.log_prob_vector_future_given_past(
          logits, x_past=x_past, stage=stage)
      x_sample = distribution_utils.sample_categorical(rng, log_probs_future)

    return x_sample

  def elbo(self, rng, params, x, train, context=None):
    d = np.prod(x.shape[1:])
    batch_size = x.shape[0]

    rng_perm, rng_t, rng_drop = jax.random.split(rng, 3)

    t, prob_qt, stage, t_in_stage = self.sample_q_t(rng_t, batch_size)

    x_future, x_past = self.corrupt(x, stage)

    # Get random permutation sigma ~ Unif(S_n_steps) for a stage.
    sigma_in_stage = ardm_utils.get_batch_permutations(rng_perm, x.shape[0],
                                                       self.num_steps_per_stage)

    already_predicted, _ = ardm_utils.get_selection_for_sigma_and_t(
        sigma_in_stage, t_in_stage, self.config.mask_shape)
    to_predict = (1 - already_predicted)

    model_inp = already_predicted * x_future + to_predict * x_past

    net_out = self.apply_fn(
        {'params': params}, model_inp, t,
        self.prepare_additional_input(stage, already_predicted), train,
        context, rngs={'dropout': rng_drop} if train else None)

    log_prob_future_given_past = self.log_prob_for_x_future_given_past(
        x_future, x_past, net_out, stage)

    log_prob = log_prob_future_given_past * to_predict

    log_prob = util_fns.sum_except_batch(log_prob)

    ce = log_prob / d / np.log(2)

    # Reweigh for summation over i.
    reweighting_factor_expectation_i = 1. / (
        self.num_steps_per_stage - t_in_stage)
    elbo_per_t = reweighting_factor_expectation_i * log_prob

    # Reweigh for expectation over t.
    reweighting_factor_expectation_t = 1. / prob_qt
    elbo = elbo_per_t * reweighting_factor_expectation_t

    elbo = elbo / d / np.log(2)
    elbo_per_t = elbo_per_t / d / np.log(2)

    return elbo, elbo_per_t, ce, t

  def sample_step(self, rng, x, t_in_stage, stage, sigmas, params, context):
    """Sampling code for a single step t."""
    t_in_stage = t_in_stage * jnp.ones(x.shape[0], dtype=jnp.int32)
    t = t_in_stage + stage * self.num_steps_per_stage
    stage = stage * jnp.ones(x.shape[0], dtype=jnp.int32)

    already_predicted, current_pred = ardm_utils.get_selection_for_sigma_and_t(
        sigmas, t_in_stage, self.config.mask_shape)

    model_inp = jnp.asarray(x, jnp.int32)
    net_out = self.apply_fn(
        {'params': params},
        model_inp, t,
        self.prepare_additional_input(stage, already_predicted),
        train=False,
        context=context)
    new_x = self.sample_x_future_given_past(rng, x, net_out, stage)

    x = (1 - current_pred) * x + current_pred * new_x
    return x

  def sample(self, rng, params, batch_size, context=None, chain_out_size=64):
    chain_sharded = self.p_sample(rng, params, batch_size, context,
                                  chain_out_size)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(
      jax.pmap, in_axes=(None, None, 0, None, 0), out_axes=1,
      static_broadcasted_argnums=(0, 3, 5), axis_name='batch')
  def p_sample(self, rng, params, batch_size, context, chain_out_size):
    """Samples from the model, calls sample_step for every timestep."""
    assert batch_size % jax.local_device_count() == 0

    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    per_device_batch_size = batch_size // jax.local_device_count()
    logging.info('Sampling from model, hope you are patient...')
    n_steps = np.prod(self.config.mask_shape)

    x = jnp.full((per_device_batch_size, *self.config.data_shape),
                 fill_value=self.absorbing_state)

    assert chain_out_size % self.num_stages == 0
    chain_per_stage_size = chain_out_size // self.num_stages

    def stage_step(x, stage):
      """Describes the sampling of a stage."""
      rng_stage = jax.random.fold_in(rng, stage)
      rng_perm, rng_sample = jax.random.split(rng_stage)
      sigma_in_stage = ardm_utils.get_batch_permutations(
          jax.random.fold_in(rng_perm, stage), per_device_batch_size, n_steps)

      def next_sample_step(state, t_in_stage):
        """Describes the sampling of a single step within a stage."""
        x, chain_in_stage = state
        x = self.sample_step(
            jax.random.fold_in(rng_sample, t_in_stage), x, t_in_stage, stage,
            sigma_in_stage, params, context)

        # Compute the write index. Minimum is 0, maximum is chain_out_size - 1.
        write_index = (t_in_stage *
                       chain_per_stage_size) // self.num_steps_per_stage

        # Write to the chain in this stage.
        chain_in_stage = jax.lax.dynamic_update_slice(
            chain_in_stage, jnp.expand_dims(x, axis=0),
            (write_index,) + (0,) * x.ndim)

        state = (x, chain_in_stage)
        return state, None

      # `chain_in_stage` is an output array that contains intermediate states.
      chain_in_stage = jnp.zeros(
          (chain_per_stage_size, per_device_batch_size) +
          self.config.data_shape,
          dtype=x.dtype)

      ts_in_stage = jnp.arange(self.num_steps_per_stage)

      # For-loop over timesteps within the stage.
      (x, chain_in_stage), _ = jax.lax.scan(next_sample_step,
                                            (x, chain_in_stage), ts_in_stage)

      return x, chain_in_stage

    # For-loop over stages.
    x, chain = jax.lax.scan(stage_step, x, jnp.arange(self.num_stages))
    chain = chain.reshape(chain.shape[0] * chain.shape[1],
                          per_device_batch_size, *self.config.data_shape)

    return chain

  def compute_policies_and_costs(self, kl_per_t, budgets):
    assert kl_per_t.shape[0] == self.num_steps

    policies = [[] for _ in range(len(budgets))]
    costs = [0 for _ in range(len(budgets))]

    for stage in range(self.num_stages):
      kls_stage = kl_per_t[stage * self.num_steps_per_stage:
                           (stage + 1) * self.num_steps_per_stage]
      # Sorting reduces bias in the dynamic programming computation. Otherwise
      # it easily exploits the non-monotonicity due to stochastic estimates.
      kls_stage_sorted = jnp.sort(kls_stage)[::-1]

      policies_stage, costs_stage = dynamic_programming.compute_fixed_budget(
          kls_stage_sorted, budgets)

      for i in range(len(policies_stage)):
        policies[i].append(policies_stage[i])
        costs[i] += costs_stage[i]

    policies = [jnp.asarray(p, dtype=jnp.int32) for p in policies]

    return policies, costs

  def sample_step_with_policy(self, rng, x, left_t_in_stage, right_t_in_stage,
                              stage, sigmas, params):
    """Sampling code for a single step t."""
    left_t_in_stage = left_t_in_stage * jnp.ones(x.shape[0], dtype=jnp.int32)
    right_t_in_stage = right_t_in_stage * jnp.ones(x.shape[0], dtype=jnp.int32)
    stage = stage * jnp.ones(x.shape[0], dtype=jnp.int32)

    already_predicted, current_pred = ardm_utils.get_selections_for_sigma_and_range(
        sigmas, left_t_in_stage, right_t_in_stage, self.config.mask_shape)

    left_t = left_t_in_stage + stage * self.num_steps_per_stage

    model_inp = jnp.asarray(x, jnp.int32)
    net_out = self.apply_fn(
        {'params': params},
        model_inp,
        left_t,
        self.prepare_additional_input(stage, already_predicted),
        train=False)
    new_x = self.sample_x_future_given_past(rng, x, net_out, stage)

    x = (1 - current_pred) * x + current_pred * new_x
    return x

  def get_naive_policy(self, budget_per_stage = 500):
    assert budget_per_stage <= self.num_steps_per_stage
    # We use budget_per_stage+1 because a linspace contains the last step.
    naive_policy = ardm_utils.integer_linspace(
        0, self.num_steps_per_stage, budget_per_stage+1)

    # Last index does not need to be in policy
    naive_policy = naive_policy[:-1]

    naive_policy = jnp.repeat(naive_policy[None, :], self.num_stages, axis=0)
    return naive_policy

  def sample_with_naive_policy(self,
                               rng,
                               params,
                               batch_size,
                               budget_per_stage = 500,
                               context=None):
    logging.info('Sampling with naive policy.')
    naive_policy = self.get_naive_policy(budget_per_stage)
    return self.sample_with_policy(rng, params, batch_size, naive_policy,
                                   context)

  def sample_with_policy(self, rng, params, batch_size, policy, context=None):
    """Wrapper that unshards p_sample_with_policy."""
    logging.info('Sampling from model with policy...')
    chain_sharded = self.p_sample_with_policy(
        rng, params, batch_size, policy, context)
    chain = chain_sharded.reshape(
        chain_sharded.shape[0], batch_size, *chain_sharded.shape[3:])
    return chain

  @functools.partial(
      jax.pmap, in_axes=(None, None, 0, None, None, 0), out_axes=1,
      static_broadcasted_argnums=(0, 3), axis_name='batch')
  def p_sample_with_policy(self, rng, params, batch_size, policy, context):
    """Samples from the model, calls sample_step for every policy step."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    assert batch_size % jax.local_device_count() == 0
    per_device_batch_size = batch_size // jax.local_device_count()

    x = jnp.full((per_device_batch_size,) + self.config.data_shape,
                 fill_value=self.absorbing_state)

    # Include final step in policy to compute ranges.
    extension = jnp.repeat(
        jnp.array([[self.num_steps_per_stage]]), self.num_stages, axis=0)
    policy_extended = jnp.concatenate([policy, extension], axis=1)
    num_policy_steps = policy.shape[1]

    def stage_step(x, stage):
      rng_stage = jax.random.fold_in(rng, stage)
      rng_sample, rng_perm = jax.random.split(rng_stage)
      sigma_in_stage = ardm_utils.get_batch_permutations(
          rng_perm, per_device_batch_size,
          self.num_steps_per_stage)

      # Sample_step wrapper that only has a carry and iterator variable.
      def next_sample_step(x, idx):
        left_t_in_stage = policy_extended[stage, idx]
        right_t_in_stage = policy_extended[stage, idx+1]
        x = self.sample_step_with_policy(
            jax.random.fold_in(rng_sample, idx),
            x, left_t_in_stage, right_t_in_stage,
            stage, sigma_in_stage, params)

        return x, x

      # For-loop over timesteps within the stage.
      x, chain_in_stage = jax.lax.scan(next_sample_step, x,
                                       jnp.arange(num_policy_steps))

      # Keep limited number of frames, mainly for memory reasons.
      chain_in_stage = ardm_utils.prune_chain(
          chain_in_stage, self.max_frames_per_stage)

      return x, chain_in_stage

    # For-loop over stages.
    x, chain = jax.lax.scan(stage_step, x, jnp.arange(self.num_stages))
    chain = chain.reshape(chain.shape[0] * chain.shape[1],
                          per_device_batch_size, *self.config.data_shape)

    return chain

  def elbo_with_policy(self, rng, params, x, policy, train, context=None):
    """Computes the ELBO for AO-ARMs using uniform distribution over policy.

    Args:
      rng: random number key.
      params: parameters for the apply_fn.
      x: input image
      policy: An array of integers describing the generative model,
        parallelizing sampling steps if integers are missing. For example, the
        list [0, 2, 4, 5] indicates that step 0 & 1 should be generated in
        parallel, then then 2 & 3 in parallel and then 4 (individually) and then
        5, ..., n_steps - 1 (in parallel).
      train: Is the model in train or eval mode?
      context: Optional context to condition on.

    Returns:
      elbo: batch of stochastic elbo estimates.
      ce: batch of the direct cross-entropy loss
      t: batch timesteps that were sampled
    """
    d = np.prod(x.shape[1:])
    batch_size = x.shape[0]

    rng_stage, rng_perm, rng_t, rng_dropout = jax.random.split(rng, 4)

    # Get random stage s ~ Unif({0, 1, ..., num_stages-1})
    stage = jax.random.randint(
        rng_stage, shape=(batch_size,), minval=0, maxval=self.num_stages)

    x_future, x_past = self.corrupt(x, stage)

    # Get random permutation sigma ~ Unif(S_n_steps) for a stage.
    sigma_in_stage = ardm_utils.get_batch_permutations(rng_perm, x.shape[0],
                                                       self.num_steps_per_stage)

    # Sample t from policy.
    t_in_stage, _, weight_policy = self.sample_policy_t(
        rng_t, policy[stage])
    t = t_in_stage + stage * self.num_steps_per_stage

    already_predicted, _ = ardm_utils.get_selection_for_sigma_and_t(
        sigma_in_stage, t_in_stage, self.config.mask_shape)
    to_predict = (1 - already_predicted)

    model_inp = already_predicted * x_future + to_predict * x_past

    net_out = self.apply_fn(
        {'params': params},
        model_inp,
        t,
        self.prepare_additional_input(stage, already_predicted),
        train,
        context=context,
        rngs={'dropout': rng_dropout} if train else None)

    log_prob_future_given_past = self.log_prob_for_x_future_given_past(
        x_future, x_past, net_out, stage)

    log_prob = log_prob_future_given_past * to_predict

    log_prob = util_fns.sum_except_batch(log_prob)

    # Negative cross-entropy.
    nce = log_prob / d / np.log(2)

    # Reweigh for summation over i.
    reweighting_factor_expectation_i = 1. / (
        self.num_steps_per_stage - t_in_stage)
    elbo_per_t = reweighting_factor_expectation_i * log_prob

    # Reweigh for expectation over policy and stages.
    elbo = elbo_per_t * weight_policy * self.num_stages

    elbo = elbo / d / np.log(2)
    elbo_per_t = elbo_per_t / d / np.log(2)

    return elbo, elbo_per_t, nce, t

  def log_prob_with_policy_and_sigma(self, rng, params, x, policy, sigmas,
                                     train, context=None):
    """Expected log prob with specific policy and generation order sigma.

    Computes the log probability for AO-ARMs using a specific policy _and_ a
    specific permutation sigma. The given permutation makes this exact (hence
    log prob), the policy ensures that the estimator has reasonable variance.

    Args:
      rng: Random number key.
      params: Parameters for the apply_fn.
      x: Input.
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

    assert sigmas.shape == (self.num_stages, self.num_steps_per_stage)
    sigmas = jnp.repeat(sigmas[:, None], repeats=batch_size, axis=1)
    assert policy.shape[0] == self.num_stages

    rng_stage, rng_t, rng_dropout = jax.random.split(rng, 3)

    # Get random stage s ~ Unif({0, 1, ..., num_stages-1})
    stage = jax.random.randint(
        rng_stage, shape=(batch_size,), minval=0, maxval=self.num_stages)

    x_future, x_past = self.corrupt(x, stage)

    # Retrieve the relevant permutation.
    sigma_in_stage = sigmas[stage, jnp.arange(batch_size)]

    # Sample t from policy.
    t_in_stage, right_t_in_stage, weight_policy = self.sample_policy_t(
        rng_t, policy[stage])
    num_tokens_in_parallel = right_t_in_stage - t_in_stage
    t = t_in_stage + stage * self.num_steps_per_stage

    already_predicted, to_predict = ardm_utils.get_selections_for_sigma_and_range(
        sigma_in_stage, t_in_stage, right_t_in_stage, self.config.mask_shape)

    model_inp = already_predicted * x_future + (1 - already_predicted) * x_past

    net_out = self.apply_fn(
        {'params': params},
        model_inp,
        t,
        self.prepare_additional_input(stage, already_predicted),
        train,
        context=context,
        rngs={'dropout': rng_dropout} if train else None)

    log_prob_future_given_past = self.log_prob_for_x_future_given_past(
        x_future, x_past, net_out, stage)

    log_prob = log_prob_future_given_past * to_predict

    log_prob = util_fns.sum_except_batch(log_prob)

    # Reweigh for expectation over policy and stages.
    log_prob = log_prob / num_tokens_in_parallel * weight_policy * self.num_stages
    log_prob = log_prob / d / np.log(2)

    return log_prob

  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 6, 7))
  def p_compute_logits_and_x_future(self, params, x, t_in_stage, stage,
                                    already_predicted, train, corrupt,
                                    context=None):
    """Parallel mapped function to compute logits and the future x.

    Args:
      params: The network parameters.
      x: The input to be corrupted.
      t_in_stage: The current timestep.
      stage: The current stage.
      already_predicted: A selection mask denoting what has already been
        predicted in the current stage.
      train: Boolean denoting whether the model is in train / eval mode.
      corrupt: Boolean denoting whether the corrupted x should be computed.
        This is necessary for encoding, but not for decoding.
      context: Optional context to pass to the neural network.

    Returns:
      A tuple containing the logits for the current step, and the future x to
      be encoded.
    """

    # For encoding steps the input needs to be corrupted. When decoding the
    # procedure already ensures that the current x already contains the correct
    # corruptions at the relevant locations.
    if corrupt:
      x_future, x_past = self.corrupt(x, stage)
      to_predict = 1 - already_predicted
      model_input = already_predicted * x_future + to_predict * x_past
    else:
      # Decoding corruption and masking is entirely handled automatically by the
      # the generative process itself.
      model_input = x
      x_past = x
      x_future = x

    t = t_in_stage + stage * self.num_steps_per_stage

    # Call neural network.
    net_out = self.apply_fn(
        {'params': params}, model_input, t,
        self.prepare_additional_input(stage, already_predicted), train=train,
        rngs=None, context=context)

    # Retrieve the logits based on the parametrization choice.
    if self.direct_parametrization:
      batch_size = x.shape[0]
      logits_for_stages = net_out.reshape(*x.shape, self.num_stages,
                                          self.branch_factor)

      # Retrieve the logits for this specific stage.
      logits = logits_for_stages[jnp.arange(batch_size), Ellipsis, stage, :]
    else:
      logits = self.log_prob_vector_future_given_past(
          net_out.reshape(*x_past.shape, self.num_output_classes), x_past,
          stage)
    return logits, x_future

  def batch_compute_logits_and_x_future(
      self, params, x, t_in_stage, stage, already_predicted, train,
      corrupt, context=None):
    """Batches and applies the network. Useful in non-jitted routines."""
    batch_size = x.shape[0]
    assert batch_size == already_predicted.shape[0]
    assert batch_size == t_in_stage.shape[0]
    num_local_devices = jax.local_device_count()
    x = util_fns.batch(x, num_local_devices)
    t_in_stage = util_fns.batch(t_in_stage, num_local_devices)
    stage = util_fns.batch(stage, num_local_devices)
    already_predicted = util_fns.batch(already_predicted, num_local_devices)
    if context is not None:
      context = util_fns.batch(context, num_local_devices)
    logits, x_future = self.p_compute_logits_and_x_future(
        params, x, t_in_stage, stage, already_predicted, train,
        corrupt, context)

    return util_fns.unbatch(logits), util_fns.unbatch(x_future)

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

    assert sigmas.shape == (self.num_stages, self.num_steps_per_stage)
    sigmas = jnp.repeat(sigmas[:, None], repeats=batch_size, axis=1)
    target_shape = (self.num_stages, batch_size, self.num_steps_per_stage)
    assert sigmas.shape == target_shape, (
        f'{sigmas.shape} does not match {target_shape}')
    assert policy.shape[0] == self.num_stages

    for stage in range(self.num_stages):
      stage_batch = jnp.full(batch_size, fill_value=stage)
      # Extend policy.
      policy_in_stage = policy[stage]
      policy_in_stage_extended = jnp.concatenate(
          [policy_in_stage,
           jnp.array([self.num_steps_per_stage], dtype=jnp.int32)], axis=0)
      sigma_in_stage = sigmas[stage]

      # Loop over policy indices.
      for idx in range(0, len(policy_in_stage)):
        left_t_in_stage = policy_in_stage_extended[idx]
        right_t_in_stage = policy_in_stage_extended[idx + 1]

        left_t_in_stage_batch = jnp.full(batch_size, fill_value=left_t_in_stage)

        already_predicted, _ = ardm_utils.get_selection_for_sigma_and_t(
            sigma_in_stage, left_t_in_stage_batch, self.config.mask_shape)

        # The network call and logit computation is pmapped here. Corruption is
        # not needed as the decoding process itself ensures the correct
        # variables.
        logits, _ = self.batch_compute_logits_and_x_future(
            params, x,
            left_t_in_stage_batch, stage_batch, already_predicted,
            train=False, corrupt=False, context=context)

        # Loop over timesteps in the current policy.
        for t_in_stage in range(left_t_in_stage, right_t_in_stage):
          t_in_stage_batch = jnp.full(batch_size, fill_value=t_in_stage)
          _, current_selection = ardm_utils.get_selection_for_sigma_and_t(
              sigma_in_stage, t_in_stage_batch, self.config.mask_shape)

          # Retrieves coding probabilities from the logits.
          probs = ardm_utils.get_probs_coding(logits, current_selection)

          # Decode the current value at place t.
          decoded_value, streams = ardm_utils.decode(streams, probs)

          # Map code value to x. For the direct parametrization, the code needs
          # to be mapped to its corresponding value in x first. For the indirect
          # parametrization, the code and x value are equivalent.
          if self.direct_parametrization:
            # Convert the code to the value it attains in x.
            s_reverse = self.num_stages - 1 - stage_batch
            s_reverse = jnp.expand_dims(s_reverse,
                                        jnp.arange(1, len(decoded_value.shape)))
            decoded_value = decoded_value * self.branch_factor**s_reverse

            # Add to the already decoded values in x.
            x = x + current_selection * decoded_value

          else:
            # Overwrite decode value to x directly.
            x = x * (1 - current_selection) + decoded_value * current_selection

        del logits, already_predicted

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

    assert sigmas.shape == (self.num_stages, self.num_steps_per_stage)
    sigmas = jnp.repeat(sigmas[:, None], repeats=batch_size, axis=1)
    target_shape = (self.num_stages, batch_size, self.num_steps_per_stage)
    assert sigmas.shape == target_shape, (
        f'{sigmas.shape} does not match {target_shape}')
    assert policy.shape[0] == self.num_stages

    for stage in reversed(range(self.num_stages)):
      stage_batch = jnp.full(batch_size, fill_value=stage)
      # Extend policy.
      sigma_in_stage = sigmas[stage]
      policy_in_stage = policy[stage]
      policy_in_stage_extended = jnp.concatenate(
          [policy_in_stage,
           jnp.array([self.num_steps_per_stage], dtype=jnp.int32)], axis=0)

      # Reversed, recall our entropy coder behaves like a stack.
      for idx in reversed(range(len(policy_in_stage))):
        left_t_in_stage = policy_in_stage_extended[idx]
        right_t_in_stage = policy_in_stage_extended[idx + 1]

        # Compute current masking.
        left_t_batch = jnp.full(batch_size, fill_value=left_t_in_stage)
        already_predicted, _ = ardm_utils.get_selection_for_sigma_and_t(
            sigma_in_stage, left_t_batch, self.config.mask_shape)

        # The network call is pmapped here.
        logits, x_future = self.batch_compute_logits_and_x_future(
            params, x,
            left_t_batch, stage_batch, already_predicted,
            train=False, corrupt=True, context=context)

        # Again reversed, entropy coder behaves like a stack. Here the same
        # net_out is used for all range(left_t, right_t).
        for t_in_stage in reversed(range(left_t_in_stage, right_t_in_stage)):
          t_batch = jnp.full(batch_size, fill_value=t_in_stage)
          _, current_selection = ardm_utils.get_selection_for_sigma_and_t(
              sigma_in_stage, t_batch, self.config.mask_shape)

          probs, x_encode = ardm_utils.get_probs_coding(
              logits, current_selection, x=x_future)

          if self.direct_parametrization:
            s_rev = self.num_stages - 1 - stage_batch
            s_rev = jnp.expand_dims(s_rev, jnp.arange(1, len(x_encode.shape)))
            x_encode = (x_encode //
                        self.branch_factor**s_rev) % self.branch_factor
          streams = ardm_utils.encode(streams, x_encode, probs)
        del logits, already_predicted

    return streams

  def get_random_order(self, rng):
    """Retrieves a random permutation for each stage."""
    return ardm_utils.get_batch_permutations(rng, batch_size=self.num_stages,
                                             n_steps=self.num_steps_per_stage)

  def prepare_additional_input(self, stage, already_predicted):
    new_axes = tuple(range(1, len(already_predicted.shape)-1))
    stage = jnp.expand_dims(stage, axis=new_axes)
    stage_onehot = util_fns.onehot(stage, num_classes=self.num_stages)
    stage_onehot = jnp.broadcast_to(
        stage_onehot, shape=already_predicted.shape[:-1] + (self.num_stages,))

    add_info = jnp.concatenate([already_predicted, stage_onehot], axis=-1)

    return add_info

  def init_architecture(self, init_rng, tmp_x, tmp_t, context=None):
    tmp_already_predicted = jnp.ones([1, *self.config.mask_shape])
    tmp_stage = jnp.ones([1])
    tmp_mask = self.prepare_additional_input(tmp_stage, tmp_already_predicted)

    return self.neural_net.init(init_rng, tmp_x, tmp_t, tmp_mask,
                                train=False, context=context)

  @classmethod
  def create(cls, config, get_architecture):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    mask_shape = config.data_shape

    config.mask_shape = mask_shape
    num_steps_per_stage = int(np.prod(mask_shape))

    if config.upscale_direct_parametrization:
      branch_factor = config.upscale_branch_factor
      num_classes = config.num_classes
      assert branch_factor >= 1
      num_stages = int(np.ceil(np.log(num_classes) / np.log(branch_factor)))
      num_input_classes = config.num_classes
      absorb = 0
      transition_mats, cum_matmul_transition_mats = None, None
      required_num_outputs = config.data_shape[-1] * num_stages * branch_factor
    else:
      # Parametrization via matrices.
      if config.upscale_mode == 'zero_least_significant':
        transition_mats, cum_matmul_transition_mats, absorb = transitions.zero_least_significant_bit(
            config.num_classes, config.upscale_branch_factor)
      elif config.upscale_mode == 'augment_least_significant':
        assert not config.upscale_direct_parametrization
        transition_mats, cum_matmul_transition_mats, absorb = transitions.augment_least_significant_bit(
            config.num_classes, config.upscale_branch_factor)
      elif config.upscale_mode == 'augment_text8':
        assert not config.upscale_direct_parametrization
        transition_mats, cum_matmul_transition_mats, absorb = transitions.augment_text8(
            config.num_classes)
      else:
        raise ValueError

      required_num_outputs = config.data_shape[-1] * config.num_classes
      num_stages = transition_mats.shape[0]
      num_input_classes = transition_mats.shape[2]

    num_steps = num_steps_per_stage * num_stages

    neural_net = get_architecture(
        num_input_classes, required_num_outputs, num_steps)

    return cls(
        config,
        apply_fn=neural_net.apply,
        required_num_outputs=required_num_outputs,
        neural_net=neural_net,
        num_steps_per_stage=num_steps_per_stage,
        num_steps=num_steps,
        num_stages=num_stages,
        num_input_classes=num_input_classes,
        num_output_classes=config.num_classes,
        transition_matrices=transition_mats,
        cum_matmul_transition_matrices=cum_matmul_transition_mats,
        absorbing_state=absorb,
        branch_factor=config.upscale_branch_factor,
        direct_parametrization=config.upscale_direct_parametrization
    )
