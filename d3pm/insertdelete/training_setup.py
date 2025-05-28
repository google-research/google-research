# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Sentinel insert/delete task."""

import dataclasses
import functools
from typing import Any, Optional, Tuple, Protocol

import gin
import jax
import jax.experimental.host_callback
import jax.numpy as jnp
import numpy as np
import scipy

from d3pm.insertdelete import forward_process
from d3pm.insertdelete import math_util
from d3pm.insertdelete import schedules
from d3pm.insertdelete import transition_operator
from d3pm.insertdelete import util
from d3pm.text import types

NDArray = Any
PRNGKey = NDArray

PAD_TOKEN_ID = 0


# pylint: disable=invalid-name


class PredictionFunction(Protocol):
  """Type signature for model callables as used in loss functions."""

  def __call__(
      self,
      timestep,
      noisy_sequence_xtplus1,
      precomputed_Attplus1_xtplus1 = None,
  ):
    """Predicts a marginal distribution for the reverse process.

    Args:
      timestep: The current timestep.
      noisy_sequence_xtplus1: The current noisy sequence to predict from (at
        step `timestep + 1`.)
      precomputed_Attplus1_xtplus1: Optional precomputed transition matrix
        d_t_to_tplus1.A.observe(xtplus1.tokens), to save computation time.

    Returns:
      Tuple (marginals, length_distribution), where `marginals` is a
      prediction of the sequence at step `timestep`, and `length_distribution`
      is a vector of log probabilities giving a prediction of the length of the
      sequence at the final timestep. Usually `length_distribution` is
      independent of the inputs, and is produced here just to make computing the
      loss easier.
    """
    raise NotImplementedError


@gin.configurable
def preprocess_targets(sequence, how=gin.REQUIRED, pad_to=None):
  """Preprocesses a list of targets by padding and wrapping."""
  orig_len = sequence.shape[-1]
  if pad_to is None:
    padded_sequence = sequence
  else:
    padded_sequence = jnp.pad(sequence, [(0, pad_to - orig_len)], "constant")

  if how == "fixed":
    return forward_process.DynamicLengthSentinelSequence(
        tokens=padded_sequence, length=orig_len
    )
  elif how == "padding":
    return forward_process.DynamicLengthSentinelSequence(
        tokens=padded_sequence,
        length=jnp.count_nonzero(sequence != PAD_TOKEN_ID),
    )
  else:
    raise NotImplementedError()


@gin.configurable
def sample_based_elbo_term_loss(
    x0,
    schedule,
    denoise_prediction_fn,
    rng,
    num_resamples = 0,
    resample_in_dynamic_band = False,
    resample_band_size = 32,
    force_timestep = None,
):
  r"""Estimates one term in the ELBO from samples.

  Chooses a single timestep, draws a single sample of x_{t+1} from the forward
  process, runs the model on this one sample, then estimates one term in the
  ELBO based on that one sample.

  We specifically estimate the following two terms:

    log q(x_{t+1} | x_t) - log p_theta(x_t | x_{t+1})

  and

    log p_theta(x_T)

  with appropriate weights based on how often they are sampled.

  We assume that the schedule is such that x_T will always consist entirely of
  DEL sentinels, so that `log p_theta(x_T)` is equivalent to just predicting
  the length of the sequence `x_T`.

  Optionally, resamples multiple alignments of x_0 and x_{t+1} to reduce
  variance at the cost of increased computation time.

  Args:
    x0: Initial sequence.
    schedule: Noise schedule to use.
    denoise_prediction_fn: Function (t, x{t+1}) -> predicted x_0 distribution
      and length estimate.
    rng: Random number generator to use to draw the samples.
    num_resamples: How many alignments to resample before computing the loss.
      Reduces the variance of the loss at the expense of increased computation
      time (running many more hypothetical aligned forward process computations)
    resample_in_dynamic_band: If resampling, whether to only resample inside a
      "dynamic band" of alignments that consume and produce tokens at about the
      same rate. To ensure unbiasedness, if the original alignment is not in the
      band, we instead copy the original alignment.
    resample_band_size: If resample_in_dynamic_band is True, this specifies how
      far two sequences are allowed to be unaligned before we give up on
      resampling. Otherwise, this specifies how many terms in the realignment
      table we compute at once, but we still compute all entries in the table
      eventually.
    force_timestep: Optional timestep to force the loss to be computed at.
      Otherwise, samples one at random.

  Returns:
    Dictionary of metric totals, along with a denominator to divide by, and
    extras.

    Denominator will count the number of valid samples that were used to
    compute these metrics. A sample is invalid if either x_{t+1} or x_t are
    longer than the maximum sequence length (and thus some tokens are missing).

    Metrics dict will include keys:
      "neg_elbo_timestep_term": Estimate of the quantity
          log q(x_{t+1} | x_t) - log p_theta(x_t | x_{t+1})
      "nll_length": Estimate of -log p_theta(x_T).
      "neg_elbo_total_importance_sample": Unbiased high-variance estimate of the
          full (negated) ELBO, with an importance-sampling correction.
  """
  t_key, xtplus1_key, xt_resample_key = jax.random.split(rng, 3)

  # Sample a single timestep from the schedule.
  if force_timestep is not None:
    t = force_timestep
    t_weight = schedule.weights[t]
    d_0_to_t, d_t_to_tplus1 = schedule.distns_at_step(t)
  else:
    t, t_weight, d_0_to_t, d_t_to_tplus1 = schedule.sample_step_distns(t_key)
  d_0_to_tplus1, _ = schedule.distns_at_step(t + 1)

  precomputed_x0_A0tplus1 = jax.vmap(
      functools.partial(d_0_to_tplus1.A.apply, is_distn=False, log=True)
  )(x0.tokens)

  # Sample xtplus1 and an alignment.
  xtplus1, align_0_tplus1 = forward_process.sample_noise_step(
      x0, d_0_to_tplus1, xtplus1_key, precomputed_xA=precomputed_x0_A0tplus1
  )

  precomputed_x0_A0t = jax.vmap(
      functools.partial(d_0_to_t.A.apply, is_distn=False, log=True)
  )(x0.tokens)
  precomputed_Attplus1_xtplus1 = jax.vmap(
      functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True)
  )(xtplus1.tokens)

  # Possibly resample the alignment, then compute intermediate
  # alignment-conditioned marginals for the model target.
  if num_resamples:
    realignment_table = forward_process.compute_multi_step_alignment_table(
        x0,
        xtplus1,
        d_0_to_tplus1,
        pack_block_size=resample_band_size,
        use_dynamic_band=resample_in_dynamic_band,
        precomputed_xA=precomputed_x0_A0tplus1,
    )

    def draw_sample(resample_key):
      realignment_0_tplus1 = (
          forward_process.rao_blackwellize_dynamic_band_alignment(
              x0,
              xtplus1,
              d_0_to_tplus1,
              align_0_tplus1,
              realignment_table,
              resample_key,
          )
      )
      return realignment_0_tplus1

    realign_0_tplus1 = jax.vmap(draw_sample)(
        jax.random.split(xt_resample_key, num_resamples)
    )
    all_align_0_tplus1 = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b], axis=0),
        align_0_tplus1,
        realign_0_tplus1,
    )

    all_targets, all_expected_q_xtplus1_given_xt = util.vmap_with_kwargs(
        forward_process.intermediate_marginals, alignment_axis=0
    )(
        x0=x0,
        xtplus1=xtplus1,
        alignment=all_align_0_tplus1,
        d_0_to_t=d_0_to_t,
        d_t_to_tplus1=d_t_to_tplus1,
        precomputed_x0_A0t=precomputed_x0_A0t,
        precomputed_Attplus1_xtplus1=precomputed_Attplus1_xtplus1,
    )

    denoise_targets = all_targets.mean_across_batch()
    expected_q_xtplus1_given_xt = jnp.mean(all_expected_q_xtplus1_given_xt)

  else:
    denoise_targets, expected_q_xtplus1_given_xt = (
        forward_process.intermediate_marginals(
            x0=x0,
            xtplus1=xtplus1,
            alignment=align_0_tplus1,
            d_0_to_t=d_0_to_t,
            d_t_to_tplus1=d_t_to_tplus1,
            precomputed_x0_A0t=precomputed_x0_A0t,
            precomputed_Attplus1_xtplus1=precomputed_Attplus1_xtplus1,
        )
    )

  # Run the model.
  denoise_prediction, length_prediction = denoise_prediction_fn(
      t, xtplus1, precomputed_Attplus1_xtplus1
  )

  # Compute our timestep ELBO term.
  expected_p_xt_given_xtplus1, extra_cross_entropy_info = (
      forward_process.ReverseProcessMarginalDistribution.cross_entropy(
          samples_from=denoise_targets,
          model_output=denoise_prediction,
          return_extra=True,
      )
  )
  elbo_term_at_t = expected_p_xt_given_xtplus1 - expected_q_xtplus1_given_xt

  elbo_across_terms_estimate = elbo_term_at_t / t_weight

  # Compute "prior" term p_theta(x_T)
  true_xT_length_distn = forward_process.terminal_sentinel_distribution(
      x0, schedule.terminal_distn()
  )
  lp_of_xT_length = jnp.sum(
      math_util.safe_exp_weighted(true_xT_length_distn, length_prediction)
  )

  # Estimate the ELBO
  elbo_estimate = elbo_across_terms_estimate + lp_of_xT_length

  results = {
      "neg_elbo_timestep_term": -elbo_term_at_t,
      "nll_length": -lp_of_xT_length,
      "neg_elbo_total_importance_sample": -elbo_estimate,
      "expected_p_xt_given_xtplus1": expected_p_xt_given_xtplus1,
      "expected_q_xtplus1_given_xt": expected_q_xtplus1_given_xt,
      **extra_cross_entropy_info,
  }
  valid = xtplus1.is_valid()
  results = jax.tree.map(
      lambda v: jnp.where(valid, v, jnp.zeros_like(v)), results
  )
  denominator = (valid).astype(jnp.float32)

  return results, denominator, {"t": t}


def sample_forward_chain(
    x0,
    schedule,
    rng,
):
  """Utility function to sample a forward process trajectory."""
  T = schedule.num_steps

  def step(xt, step_inputs):
    t, rng = step_inputs
    _, d_t_to_tplus1 = schedule.distns_at_step(t)

    # Rejection sample until we get a sequence that fits in max_len.
    def generate_sample(key):
      sample, _ = forward_process.sample_noise_step(
          sequence=xt, distn=d_t_to_tplus1, rng=key
      )
      return sample.is_valid(), sample

    rng, key = jax.random.split(rng)
    xtplus1 = util.rejection_sample(generate_sample, key)
    xtplus1 = xtplus1.fill_padding(PAD_TOKEN_ID)

    return xtplus1, xtplus1

  _, sequences = jax.lax.scan(
      step, init=x0, xs=(jnp.arange(T), jax.random.split(rng, T))
  )
  return jax.tree.map(
      lambda a, b: jnp.concatenate([jnp.array(a)[None], b]), x0, sequences
  )


@gin.configurable
def sample_based_elbo(
    x0,
    schedule,
    denoise_prediction_fn,
    rng,
    num_samples,
    max_rejects_inner = 4,
    max_rejects_outer = 32,
):
  """Estimate the ELBO across all diffusion steps.

  We make a small modifications to the forward process `q` to ensure that we can
  compute an accurate likelihood lower bound. Specifically, we correct
  q(x_{t+1} | x_t) so that it never generates sequences that are
  too long. We do this by rejection sampling until it generates a short-enough
  sequence, and then adjusting the log probs by the probability of generating
  something short enough. Note that this means it's no longer quite correct
  to marginalize over intermediate steps, so we must sample steps in order.
  This is intended as a slight correction only; the probability of staying
  in bounds should nevertheless be made as high as is feasible.

  Args:
    x0: Initial sequence.
    schedule: Noise schedule to use.
    denoise_prediction_fn: Function (t, x{t+1}) -> predicted x_0 distribution,
      length.
    rng: Random number generator to use to draw the samples.
    num_samples: How many samples to use when computing the ELBO.
    max_rejects_inner: Maximum times to try to sample from forward process in
      the inner loop. Might introduce some bias.
    max_rejects_outer: Maximum times to try to sample from forward process in
      the outer loop. Might introduce some bias.

  Returns:
    Dictionary of metrics and extras.
  """
  max_len = x0.tokens.shape[0]
  T = schedule.num_steps

  # ELBO = E_q[
  #   log p_{final}(x_T)
  #   + \sum_t log p_theta(x_t | x_{t+1}) - log q'(x_{t+1} | x_t)
  # ]
  # where we use q' to denote the lightly-modified version of q described
  # above.

  def step(state, step_inputs):
    xt, was_good = state
    t, rng = step_inputs
    d_0_to_t, d_t_to_tplus1 = schedule.distns_at_step(t)
    del d_0_to_t
    # d_0_to_tplus1 = d_0_to_t.then(d_t_to_tplus1)

    in_bounds_log_prob = forward_process.compute_in_bounds_log_probs(
        xt, d_t_to_tplus1, max_len
    )

    # Rejection sample until we get a sequence that fits in max_len.
    def generate_sample(key):
      sample, _ = forward_process.sample_noise_step(
          sequence=xt, distn=d_t_to_tplus1, rng=key
      )
      return sample.is_valid(), sample

    rng, key = jax.random.split(rng)
    xtplus1 = util.rejection_sample(
        generate_sample, key, max_rejects=max_rejects_inner
    )
    is_good = was_good & xtplus1.is_valid()

    # Note: We wish to compute the value
    #   log p_theta(x_t | x_{t+1}) - log q'(x_{t+1} | x_t)
    # where (for samples that passed the rejection sampling check)
    #   log q'(x_{t+1} | x_t) = log q(x_{t+1} | x_t) - log in_bounds_prob

    # Generate a prediction.
    xt_prediction, length_prediction = denoise_prediction_fn(t, xtplus1)

    # Construct the term of our ELBO.
    true_xt_point_mass = (
        forward_process.ReverseProcessMarginalDistribution.point_mass(
            xt,
            xtplus1,
            vocab_size=xt_prediction.previous_token_log_probs.shape[-1],
        )
    )
    log_p_xtplus1_to_xt = (
        forward_process.ReverseProcessMarginalDistribution.cross_entropy(
            samples_from=true_xt_point_mass, model_output=xt_prediction
        )
    )
    log_q_xt_to_xtplus1 = forward_process.sentinel_one_step_log_prob(
        xt, xtplus1, d_t_to_tplus1
    )

    elbo_term = log_p_xtplus1_to_xt - (log_q_xt_to_xtplus1 - in_bounds_log_prob)
    aux_metrics = {
        "log_p_xtplus1_to_xt": log_p_xtplus1_to_xt,
        "log_q_xt_to_xtplus1": log_q_xt_to_xtplus1,
        "in_bounds_log_prob": in_bounds_log_prob,
    }
    extras = {
        "xt": xt,
    }
    return (
        (xtplus1, is_good),
        (elbo_term, length_prediction, aux_metrics, extras),
    )

  def go(rng):
    (xT, good), (elbo_terms, length_predictions, aux_metrics, extras) = (
        jax.lax.scan(
            step, init=(x0, True), xs=(jnp.arange(T), jax.random.split(rng, T))
        )
    )

    # xT must be all delete sentinels now.
    # Score the length prediction.
    reached_all_deletes = jnp.all(xT.delete_sentinel_mask() == xT.token_mask())
    length_prediction = length_predictions[-1, :]
    p_xT_term = length_prediction[xT.length]
    p_xT_term = jnp.where(reached_all_deletes, p_xT_term, -jnp.inf)

    results = elbo_terms, p_xT_term, aux_metrics, {**extras, "final_x": xT}
    results = jax.tree.map(
        lambda v: jnp.where(good, v, jnp.zeros_like(v)), results
    )

    return good, results

  def go2(rng):
    good, results = go(rng)
    return good, (good, results)

  def go3(rng):
    return util.rejection_sample(go2, rng, max_rejects=max_rejects_outer)

  good, (elbo_terms, p_xT_term, aux_metrics, extras) = jax.vmap(go3)(
      jax.random.split(rng, num_samples)
  )
  good_ct = jnp.sum(good)
  elbo_terms = jnp.sum(elbo_terms, axis=0) / good_ct
  p_xT_term_mean = jnp.sum(p_xT_term, axis=0) / good_ct
  aux_metrics_mean = jax.tree.map(
      lambda v: jnp.sum(v, axis=0) / good_ct, aux_metrics
  )

  neg_elbo = -(jnp.sum(elbo_terms) + p_xT_term_mean)
  bits_per_char = (neg_elbo / jnp.log(2)) / x0.length

  extras = {
      # **extras,
      # **{f"no_mean/{k}": v for k, v in aux_metrics.items()},
      # "terminal_term": p_xT_term
  }

  return {
      "neg_elbo": neg_elbo,
      "bits_per_char": bits_per_char,
      "elbo_terms": elbo_terms,
      "terminal_term": p_xT_term_mean,
      "elbo_valid_count": good_ct,
      **aux_metrics_mean,
  }, extras


def make_pseudogaussian_standard_normal(beta, dim):
  """Builds a transition matrix based on a discretized Gaussian."""
  bounds = np.linspace(-2, 2, dim + 1)
  xs = (bounds[1:] + bounds[:-1]) / 2
  mu = np.sqrt(1 - beta) * xs
  distn = scipy.stats.norm(mu[:, None], np.sqrt(beta))
  cdfvals = distn.cdf(bounds)
  res = cdfvals[:, 1:] - cdfvals[:, :-1]
  return res / np.sum(res, axis=-1)


def gaussian_schedule(t1, t2, dim, num_steps):
  """Helper function to build a schedule for Gaussian noise."""
  # ranges from -2 to 2
  # Manually adjusted to make this a bit noisier at the beginning, because
  # the remap below isn't exactly correct.
  min_standard_dev = 4 / dim
  min_beta = np.square(min_standard_dev)

  def remap(t):
    return -np.expm1(
        scipy.interpolate.interp1d(
            [1 / num_steps, 1],
            [np.log(min_beta), np.log(1)],
            fill_value="extrapolate",
        )(t)
    )

  v1 = remap(t1)
  v2 = remap(t2)
  return np.minimum(1 - v2 / v1, 0.999)


def state_init_fn(
    dataset_info,
    *,
    transition_type=gin.REQUIRED,
    insert_type=gin.REQUIRED,
    num_steps=gin.REQUIRED,
    final_relative_size=1.0,
    final_refresh_prob=0.0,
    initial_slope=1.0,
    acceleration=8,
    mask_token=None,
    max_len=None,
):
  """Initialization helper for training logic."""
  dim = len(dataset_info.vocab)  # pytype: disable=wrong-arg-types  # dataclasses-replace

  if transition_type == "text8_nn":
    assert dim == 27
    TRANSITION_RATE = 10.0
    vowels = "aeiouy"
    space_ix = 26
    vowel_ixs = np.array([ord(x) - ord("a") for x in vowels])
    consonant_ixs = np.array(
        [x for x in range(27) if x not in vowel_ixs and x != space_ix]
    )
    rate_matrix = np.full([27, 27], 0.5 * 1 / 27)
    rate_matrix[vowel_ixs[:, None], vowel_ixs[None, :]] += (
        0.5 * 1 / len(vowel_ixs)
    )
    rate_matrix[consonant_ixs[:, None], consonant_ixs[None, :]] += (
        0.5 * 1 / len(consonant_ixs)
    )
    rate_matrix = rate_matrix - np.diagflat(np.sum(rate_matrix, axis=0))

    def transition_distn_fn(t1, t2):
      return transition_operator.MatrixOperator(
          scipy.linalg.expm(
              TRANSITION_RATE * (t2**1.8 - t1**1.8) * rate_matrix
          )
      )

  elif transition_type == "mask":
    assert mask_token is not None

    def transition_distn_fn(t1, t2):
      diagonal_value = (1 - t2) / (1 - t1)
      return transition_operator.MaskDiffusionOperator(
          dim, mask_token, jnp.log(diagonal_value)
      )

  elif transition_type == "delayed_mask":
    assert mask_token is not None

    def transition_distn_fn(t1, t2):
      diagonal_value = (1 - jnp.maximum(0.0, 2 * t2 - 1)) / (
          1 - jnp.maximum(0.0, 2 * t1 - 1)
      )
      return transition_operator.MaskDiffusionOperator(
          dim, mask_token, jnp.log(diagonal_value)
      )

  elif transition_type == "uniform":

    def transition_distn_fn(t1, t2):
      diagonal_value = (1 - t2) / (1 - t1)
      return transition_operator.UniformDiffusionOperator(
          dim, jnp.log(diagonal_value)
      )

  elif transition_type == "pseudogaussian":

    def transition_distn_fn(t1, t2):
      beta = gaussian_schedule(t1, t2, dim - 2, num_steps)
      mat = make_pseudogaussian_standard_normal(beta, dim - 2)
      mat = np.pad(mat, [(2, 0), (2, 0)], "constant")
      mat[0, 0] = 1
      mat[1, 1] = 1
      return transition_operator.MatrixOperator(mat)

  else:
    raise NotImplementedError()

  def insertion_distn_fn(t):
    if insert_type == "uniform":
      # Modulate automatically by the forward transitions, so that if we were
      # going to mask, we still insert masks
      logits = jax.nn.log_softmax(jnp.zeros([dim]))
      return transition_distn_fn(0, t).apply(logits, is_distn=True, log=True)
    elif insert_type == "mask":
      assert mask_token is not None
      return jnp.log(jax.nn.one_hot(mask_token, dim))

  def relative_size_fn(t):
    return final_relative_size + (1.0 - final_relative_size) * (1.0 - t)

  def refresh_prob_fn(t):
    return final_refresh_prob * t

  def interpolator(u):
    return initial_slope * u + (1 - initial_slope) * u**acceleration

  schedule = schedules.schedule_from_interpolators(
      num_steps=num_steps,
      transition_distn_fn=transition_distn_fn,
      insertion_distn_fn=insertion_distn_fn,
      relative_size_fn=relative_size_fn,
      refresh_prob_fn=refresh_prob_fn,
      interpolator=interpolator,
  )

  # Precompute marginals for the number of deletes we would see.
  if max_len:
    precomputed_cumulative = jax.lax.map(
        lambda distn: distn.with_precomputed_delete_count_marginals(max_len),
        schedule.cumulative,
    )
    schedule = dataclasses.replace(schedule, cumulative=precomputed_cumulative)

  return types.State({"schedule": schedule}, {}, None, 1)
