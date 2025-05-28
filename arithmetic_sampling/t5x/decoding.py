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

"""T5X decoding routine for arithmetic sampling."""
import functools

from typing import Any, Callable, Mapping, Optional, Tuple, Union
import flax
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from t5x import decoding

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = np.array(-1.0e7)

# Temperatures lower than this are considered 0.0, which is handled specially
# with a conditional. This is to avoid numeric issues from exponentiating on
# 1.0/temperature when temperature is close to 0.0.
MIN_TEMPERATURE = np.array(1e-4)

#------------------------------------------------------------------------------
# Arithmetic Sampling
#------------------------------------------------------------------------------


@flax.struct.dataclass
class ArithmeticSamplingLoopState:
  """Holds sampling state data.

  Attributes:
    cur_index: [batch_size] array position of the sampling loop in the length
      dimension.
    sequences: [batch_size * num_decodes, max_decode_len] array of current
      sampled sequence prefixes.
    cache: any mapping of arrays, e.g. flax attention cache.
    cur_token: [batch_size, num_decodes] single timestep slice containing
      current tokens.
    ended: [batch_size, num_decodes] binary array marking completed sequences.
    rng: Jax PRNGKey
    log_prob: [batch_size, num_decodes] array of log probs for each sequence.
    codes: [batch_size, num_decodes] array containing the arithmetic codes for
      the remainder of the sequence at the current time step for each sample.
  """
  cur_index: jnp.ndarray
  sequences: jnp.ndarray
  cache: Mapping[str, jnp.ndarray]
  cur_token: jnp.ndarray
  ended: jnp.ndarray
  rng: jnp.ndarray
  log_prob: jnp.ndarray
  codes: jnp.ndarray


_dynamic_update_vector_slice_in_dim = jax.vmap(
    lax.dynamic_update_slice_in_dim, in_axes=(0, 0, 0, None))


def _is_tracer(value):
  return isinstance(value, jax.core.Tracer)


def _sequential_cumsum(arr, axis):
  """Sequential scan-based implementation of cumulative sum for Jax.

  The Jax implementation of cumulative sum does not guarantee that the output
  array is nondecreasing when applied to nonnegative outputs. This breaks
  the use of cumulative sum for bucketing. Using scan guarantees forces the
  sum to happen sequentially, which avoids the floating point nonsense that
  causes normal Jax cumsum to exhibit bad behavior.

  Args:
    arr: Jax array to sum.
    axis: axis to sum over.

  Returns:
    Jax array of partial cumulative sums.
  """

  # Swap axes so that the axis to be scanned over is the leading axis.
  xs = jnp.swapaxes(arr, 0, axis)
  init_carry = jnp.zeros(xs.shape[1:], xs.dtype)
  _, res = jax.lax.scan(lambda c, x: (c + x, c + x), init_carry, xs)
  return jnp.swapaxes(res, 0, axis)


def _arithmetic_categorical(
    rng, logits,
    codes):
  """Sample from a categorical using arithmetic sampling.

  Returns samples from an arithmetic codebook based on provided codes. This
  gives an unbiased sample for each code randomly picked from the unit interval.

  Args:
    rng: JAX PRNGKey.
    logits: array: [batch_size, vocab_size] float32 sequence of logits.
    codes: array: [batch_size] float32 codes for each batch element.

  Returns:
    A tuple (samples, new_codes) where `samples` are sampled indices with shape
    [batch_size], and `new_codes` are shape [batch_size] containing codes for
    the remaining suffix if doing ancestral sampling.
  """
  # We randomly permute the logits here at each timestep to avoid depending on
  # The default order of the vocabulary. This isn't strictly necessary.
  # We need to invert this permutation at the end cause it changes the
  # identities of the sampled indices.
  _, vocab_size = logits.shape
  perm = jax.random.permutation(rng, vocab_size)
  invperm = jnp.argsort(perm)

  logits = logits[:, perm]

  # Now we want to, for each element in the batch, get the normalized
  # probabilities, stack them in the unit interval into buckets, and figure
  # out what bucket the code falls into.
  probs = jax.nn.softmax(logits, axis=1)

  # Use the numpy cumsum with host callback to guarantee nondecreasing array
  # of partial sums.
  cumprobs = _sequential_cumsum(probs, axis=1)

  # Because of precision, make sure the max value (and everything with that
  # value, to not change bucket widths) is at least 1.0.
  max_probs = jnp.expand_dims(jnp.max(cumprobs, axis=1), 1)
  all_bucket_maxes = jnp.where((cumprobs == max_probs) & (cumprobs < 1.0), 1.0,
                               cumprobs)

  # Now the cumulative probabilities represent the max value of each of the
  # buckets. So let's make a mask of all the buckets whose maxes are less
  # than and greater than the given codes.
  expanded_codes = jnp.expand_dims(codes, axis=1)
  bucket_maxes_lte_codes = all_bucket_maxes <= expanded_codes
  bucket_maxes_gt_codes = all_bucket_maxes > expanded_codes

  # Pick the minimum value for the bucket for the code. Note this will be
  # 0.0 if the code falls into the zero'th bucket, as desired.
  code_bucket_mins = jnp.max(all_bucket_maxes * bucket_maxes_lte_codes, axis=1)

  # We have to do some masking here, and for probabilities, anything > 1.0
  # is as good as infinity.
  prob_infty = 1.1
  # Pick the maximum value for the bucket, the first bucket whose max is
  # greater than the code.
  code_bucket_maxes = jnp.min(
      all_bucket_maxes * bucket_maxes_gt_codes +
      bucket_maxes_lte_codes * prob_infty,
      axis=1)
  # We have to take the argmin before inverting the permutation,
  # otherwise it messes up the default tie breaking behavior for size zero
  # buckets (take lowest index).
  sampled_indices_permed = jnp.argmin(
      (all_bucket_maxes * bucket_maxes_gt_codes +
       bucket_maxes_lte_codes * prob_infty),
      axis=1)
  sampled_indices = jnp.argmax(
      jax.nn.one_hot(sampled_indices_permed, vocab_size)[:, invperm], axis=1)

  remainder_codes = (codes - code_bucket_mins) / (
      code_bucket_maxes - code_bucket_mins)

  samples = sampled_indices
  new_codes = remainder_codes

  return samples, new_codes


def arithmetic_sample(
    inputs,
    cache,
    tokens_to_logits,
    eos_id,
    decode_rng = None,
    num_decodes = 1,
    temperature = 1.0,
    topk = 1,
    topp = 0.0,
    cache_offset = 0,
    initial_index = None,
    max_decode_steps = None,
    max_decode_steps_hard_limit = None,
    rescale_log_probs = True,
    state_callback_fn = None,
    logit_callback_fn = None
):
  """Arithmetic sampling for language model generation.

  The sampling is performed `num_decodes` times in a vectorized
  manner by expanding the batch dimension. This is similar to how beam search
  expands the batch dimension to process each batch element with multiple beams.

  Args:
    inputs: array: [batch_size, max_decode_len] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    eos_id: int: end-of-sentence token for target vocabulary.
    decode_rng: JAX PRNGKey.
    num_decodes: number of decoded sequences to be returned.
    temperature: float: sampling temperature factor. As it approaches zero this
      becomes equivalent to greedy sampling.
    topk: integer: if nonzero only use the top-k logits to sample next token, if
      zero don't use any cutoff and sample from full logits over vocabulary.
    topp: float: if nonzero only use the smallest number of logits whose
      cumulative sum of probs adds up to (at least) topp. Will raise ValueError
      if it's nonzero when topk is nonzero.
    cache_offset: axis offset for cache, arising from scanned layers.
    initial_index: Optional[array]: [batch_size] int32 a vector of loop indexes
      to start decoding at.
    max_decode_steps: int: an optional maximum number of decoding steps. If
      None, it will decode until the full input shape `inputs.shape[1]` is
      filled. max_decode_steps begins counting after the prompt, so it will
      decode at most len(prompt) + max_decode_steps tokens.
    max_decode_steps_hard_limit: int: an optional fixed hard limit on
      max_decode_steps. If this is set (not None and > 0), and max_decode_steps
      is also set, then max_decode_steps will be clipped to this limit. The
      value max_decode_steps can be an ndarray, but max_decode_steps_hard_limit
      must be a Python integer or None.
    rescale_log_probs: bool: whether to apply temperature, topp, and topk
      rescaling to the log probs which are returned. If True, the log_probs will
      include these transformations (for example, with topk=1, all log_probs
      will be identically 0.0). If False, the log_probs will not be affected,
      and topk/topp/temperature will not affect sequence probabilities.
    state_callback_fn: Function that modifies the sampling loop state before
      each step. This can be used to manipulate any part of the state either on
      the accelerator or on the host using host callback. The function should
      take a SamplingLoopState as argument, and it returns the updated state.
      See `decoding_test.py` for an example usage.
    logit_callback_fn: Function that modifies the logits before each temperature
      sampling step. The function should take arguments (logits, state) and it
      should return the modified logits. See `decoding_test.py` for an example
      usage.

  Returns:
    A tuple (decodes, log_prob) where `decodes` is sampled sequences with shape
    [batch_size, num_decodes, max_decode_len] sorted by `log_prob`, which is log
    probability of each of the sampled sequences.
  """
  if decode_rng is None:
    decode_rng = jax.random.PRNGKey(0)

  if (max_decode_steps_hard_limit is not None and
      max_decode_steps_hard_limit > 0 and max_decode_steps is not None):
    max_decode_steps = jnp.minimum(max_decode_steps,
                                   max_decode_steps_hard_limit)

  initial_codes = _make_default_codes(inputs.shape[0], num_decodes, decode_rng)
  flattened_codes = decoding.flatten_beam_dim(initial_codes)

  # [batch, len] -> [batch * num_decodes, len]
  expanded_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)
  expanded_cache = decoding.cache_map(
      functools.partial(
          decoding.flat_batch_beam_expand,
          beam_size=num_decodes,
          offset=cache_offset),
      cache,
      # When we start with a prefilled cache, the cache index is no longer a
      # scalar that will broadcast across multiple decodes, it is a vector and
      # needs to be updated to handle the multiple decodes.
      apply_to_index=initial_index is not None)
  if initial_index is not None:
    initial_index = decoding.flat_batch_beam_expand(initial_index, num_decodes)

  # expanded_decodes: [batch * num_decodes, len]
  # expanded_log_prob: [batch * num_decodes]
  expanded_decodes, expanded_log_prob = _arithmetic_sample_single_trial(
      expanded_inputs,
      flattened_codes,
      expanded_cache,
      tokens_to_logits,
      eos_id,
      decode_rng,
      temperature,
      topk,
      topp,
      initial_index=initial_index,
      max_decode_steps=max_decode_steps,
      rescale_log_probs=rescale_log_probs,
      state_callback_fn=state_callback_fn,
      logit_callback_fn=logit_callback_fn)

  batch_size = inputs.shape[0]
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  decodes = decoding.unflatten_beam_dim(expanded_decodes, batch_size,
                                        num_decodes)
  # [batch * num_decodes] -> [batch, num_decodes]
  log_prob = decoding.unflatten_beam_dim(expanded_log_prob, batch_size,
                                         num_decodes)

  # Sort `decodes` and `log_prob` by increasing log probabilities of the sampled
  # sequence.
  # [batch, num_decodes, 1]
  idxs = jnp.expand_dims(jnp.argsort(log_prob, axis=-1), axis=-1)

  # returns [batch, num_decodes, len], [batch, num_decodes] in sorted order.
  return jnp.take_along_axis(
      decodes, idxs, axis=1), jnp.take_along_axis(
          log_prob, jnp.squeeze(idxs, axis=-1), axis=-1)


def _make_default_codes(batch_size, num_decodes,
                        rng):
  """Make default codebook for a batch of `num_decodes` samples.

  The codes are initialized evenly spaced in the unit interval, with a random
  offset applied. This lets them evenly cover the sample space while also
  providing an unbiased estimate of any sample average.

  Args:
    batch_size: size of input batch.
    num_decodes: number of samples per batch element.
    rng: random seed.

  Returns:
    [batch_size, num_decodes] array of codes.
  """
  offset = jax.random.uniform(rng, (batch_size, 1))
  codes = jnp.tile(
      jnp.expand_dims(
          jnp.arange(1, num_decodes + 1, dtype=jnp.float32) / (num_decodes + 1),
          axis=0), (batch_size, 1))
  return jnp.mod(codes + offset, 1.0)


def _arithmetic_sample_single_trial(
    inputs,
    initial_codes,
    cache,
    tokens_to_logits,
    eos_id,
    prng_key,
    temperature = 1.0,
    topk = 20,
    topp = 0.0,
    initial_index = None,
    max_decode_steps = None,
    rescale_log_probs = True,
    state_callback_fn = None,
    logit_callback_fn = None
):
  """A helper function for `arithmetic_sample`."""

  # We can check the values of topp and topk only if they are not dynamic.
  if not _is_tracer(topp) and topp and topk:
    raise ValueError('At most one of `topp` or `topk` may be non-zero.')

  batch_size, max_decode_len = inputs.shape

  if max_decode_steps is not None:
    # We can check the max_decode_steps bounds only if it is not dynamic.
    if not _is_tracer(max_decode_steps) and max_decode_steps > inputs.shape[1]:
      raise ValueError('Cannot decode more steps than the sequence length.')

    # The number of decode steps required to process the prefix is the number
    #   of non-zero tokens, since inputs[0] == 0 is the BOS token.
    # `max_decode_len[j]` is the number of non-padding tokens in the jth element
    #   of the returned sequences capped at `len(inputs)`, assuming that the
    #   early stop doesn't occur. This is true with or without
    #   `max_decode_steps`.
    # When the while loop index `i` for the `j`th element `i[j] =
    #   max_decode_len[j] - 1`, the generated token populate sequences[i[j]+1]].
    #   Since sequences[:, 0] is BOS token, the generated token is
    #   `max_decode_len[j]`th non-padding tokens and hence `j`th element is
    #   ended.
    max_decode_len = jnp.sum(inputs != 0, axis=1) + max_decode_steps
    max_decode_len = jnp.minimum(inputs.shape[1], max_decode_len)

  # In the case of starting generation from a non-zero index, it is possible for
  # one batch element to reach `max_decode_len` number of decoding steps before
  # another. In order to let the last element decoder all the way to
  # `max_decode_len` number of steps, we add a final garbage token to the end of
  # The sequences. Any element that has reached `max_decode_len` before the rest
  # of the elements will continually overwrite this token until all elements
  # finish.
  # [batch, length+1] -> [batch, length+2]
  extra_input_tokens = 2
  expanded_prompt_inputs = jnp.append(
      inputs,
      jnp.zeros((batch_size, extra_input_tokens), dtype=inputs.dtype),
      axis=1)
  end_marker = jnp.array(eos_id)

  temperature = jnp.asarray(temperature)

  # Initialize sampling loop state.
  # initial loop PRNGKey
  rng0 = prng_key

  # The per batch-item holding current token in loop.
  if initial_index is None:
    # The per batch-item loop position counter.
    i0 = jnp.zeros((batch_size), dtype=jnp.int32)
    # The per batch-item holding current token in loop.
    token0 = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  else:
    # The per batch-item loop position counter.
    i0 = initial_index
    # The per batch-item holding current token in loop.
    # Select the token that the initial index is pointing to.
    token0 = jnp.take_along_axis(
        expanded_prompt_inputs, jnp.expand_dims(i0, axis=1), axis=1)
  # per batch-item state bit indicating if sentence has finished.
  ended0 = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
  # (batch, length+2) array containing prefix prompt tokens for sampling loop
  # as well as the generated output of newly sampled tokens.
  sequences0 = expanded_prompt_inputs
  log_prob0 = jnp.zeros((batch_size,), dtype=jnp.float32)

  sampling_loop_init_state = ArithmeticSamplingLoopState(
      i0, sequences0, cache, token0, ended0, rng0, log_prob0, initial_codes)
  # Initial eos count to be used to determine whether eos is "generated". Many
  # inputs follow the format bos, inputs..., eos, targets..., eos. By counting
  # The number of eos tokens we can detect when a new one is added, instead of
  # just finding the one that probably ends the inputs.
  # [batch, 1]
  initial_eos_count = jnp.sum(sequences0 == end_marker, axis=-1, keepdims=True)

  def sampling_loop_cond_fn(state):
    """Sampling loop termination condition."""
    # Have all sampled sequences reached an end marker?
    # Different elements in the batch can be at different loop indices, if any
    # of our examples are not at the end, keep going.
    all_sequences_ended = jnp.all(state.ended)
    return ~all_sequences_ended  # pytype: disable=bad-return-type  # jnp-type

  def sampling_loop_body_fn(
      state):
    """Sampling loop state update."""

    if state_callback_fn is not None:
      state = state_callback_fn(state)

    # Split RNG for sampling.
    rng1, rng2 = random.split(state.rng)
    # Call fast-decoder model on current tokens to get next-position logits.
    decoding_state = decoding.DecodingState(
        cur_index=state.cur_index,
        sequences=state.sequences[:, :-extra_input_tokens],
        cur_token=state.cur_token,
        cache=state.cache)
    logits, new_cache = tokens_to_logits(decoding_state)
    # Sample next token from logits.

    if logit_callback_fn is not None:
      logits = logit_callback_fn(logits, state)

    def sample_logits_with_nonzero_temperature(logits):

      # Before setting up the arithmetic sampling, we preprocess the logits into
      # Their final form.
      scaled_logits = logits / jnp.maximum(temperature, MIN_TEMPERATURE)
      if topk:
        # Get top-k logits and their indices, sample within these top-k tokens.
        topk_logits, _ = lax.top_k(scaled_logits, topk)
        cutoff_logit = topk_logits[:, -1, None]
        scaled_logits = jnp.where(scaled_logits < cutoff_logit,
                                  jnp.full_like(scaled_logits, NEG_INF),
                                  scaled_logits)

      # When topp is dynamic, we always use it since we cannot check
      # non-zeroness (but it will have no effect if topp is 0.0).
      if _is_tracer(topp) or topp:
        logits_sorted = jnp.sort(
            scaled_logits, axis=-1)[:, ::-1]  # sort descending
        sorted_cum_probs = jnp.cumsum(
            jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
        cutoff_index = jnp.sum(sorted_cum_probs < topp, axis=-1, keepdims=True)
        cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
        scaled_logits = jnp.where(scaled_logits < cutoff_logit,
                                  jnp.full_like(scaled_logits, NEG_INF),
                                  scaled_logits)

      next_token, next_code = _arithmetic_categorical(rng1, scaled_logits,
                                                      state.codes)

      # log probability of the current token conditioned on the previously
      # sampled and prefix tokens.
      # [batch, vocab] -> [batch, vocab]
      if rescale_log_probs:
        log_probs = jax.nn.log_softmax(scaled_logits)
      else:
        log_probs = jax.nn.log_softmax(logits)
      # [batch, vocab] -> [batch]
      next_log_prob = jnp.squeeze(
          jnp.take_along_axis(
              log_probs, jnp.expand_dims(next_token, axis=1), axis=-1),
          axis=-1)

      return (next_token, next_log_prob, next_code)

    def sample_logits_with_zero_temperature(logits):
      # For zero temperature, we always want the greedy output, regardless
      # of the values of topk and topp.

      next_token = jnp.argmax(logits, -1).astype(jnp.int32)

      if rescale_log_probs:
        next_log_prob = jnp.zeros_like(next_token, dtype=jnp.float32)
      else:
        log_probs = jax.nn.log_softmax(logits)
        next_log_prob = jnp.squeeze(
            jnp.take_along_axis(
                log_probs, jnp.expand_dims(next_token, axis=1), axis=-1),
            axis=-1)

      return (next_token, next_log_prob, state.codes)

    # Perform sampling with temperature
    (next_token, next_log_prob,
     next_code) = lax.cond(temperature > MIN_TEMPERATURE,
                           sample_logits_with_nonzero_temperature,
                           sample_logits_with_zero_temperature, logits)

    # When different batch elements are at different points in the loop counter,
    # it is possible that an element that started at a higher index will reach
    # `max_decode_len` before other elements. When this happens we need to make
    # sure this element continuous overwrites our new garbage collection index.
    # Here we clamp `i` to `max_decode_len`. This will cause the a write to
    # `max_decode_len + 1` which is the final index in `sequences`. Subsequent
    # loop body executions will also get their value clamped causing continual
    # overwriting of the final garbage position until all examples are finished.
    i = jnp.minimum(state.cur_index, max_decode_len)

    # Only use sampled tokens if we're past provided prefix tokens.
    # Select the next token from sequences.
    # [batch]
    next_input_token = jnp.squeeze(
        jnp.take_along_axis(
            state.sequences, jnp.expand_dims(i + 1, axis=1), axis=1),
        axis=1)
    # Check if the next token is padding (a target) or non-padding (an input).
    # Mask will have `1` for targets and `0` for inputs.
    out_of_prompt = (next_input_token == 0)
    # Select the sampled next token for targets and the actual next token for
    # inputs (teacher forcing).
    # [batch]
    next_token = (
        next_token * out_of_prompt + next_input_token * ~out_of_prompt)

    # only add probability if outside prefix region
    # [batch] -> [batch]
    next_log_prob = state.log_prob + (
        next_log_prob * out_of_prompt) * jnp.squeeze(
            ~state.ended, axis=-1).astype(jnp.int32)

    # [batch] -> [batch, 1]
    next_token = jnp.expand_dims(next_token, axis=-1)

    # If end-marker reached for batch item, only emit padding tokens.
    # [batch, 1] * [batch, 1] -> [batch, 1]
    next_token_or_endpad = next_token * ~state.ended
    # Add current sampled tokens to recorded sequences.
    one_hot = jax.nn.one_hot(
        i + 1, state.sequences.shape[1], dtype=state.sequences.dtype)
    new_sequences = state.sequences * (1 -
                                       one_hot) + next_token_or_endpad * one_hot
    # new_sequences = dynamic_update_vector_slice_in_dim(sequences,
    #                                                    next_token_or_endpad,
    #                                                    i + 1,
    #                                                    0)
    # Count eos tokens in the sequences and compare to the initial count
    # [batch, 1]
    cur_eos_count = jnp.sum(new_sequences == end_marker, axis=-1, keepdims=True)
    # [batch, 1]

    # Have we reached max decoding length?
    # We generally index into sequences[:, i + 1], and sequences.shape[1] =
    # max_decode_len + 2, therefore i == max_decode_len - 1 will write to
    # sequences[-2] which is our last valid location. i == max_decode_len will
    # write to sequences[-1] which is our garbage collection token. Thus `i`
    # should be strictly less than max_decode_len.
    has_additional_eos = cur_eos_count > initial_eos_count
    ended = state.ended | has_additional_eos | jnp.expand_dims(
        i >= max_decode_len - 1, axis=1)

    return ArithmeticSamplingLoopState(i + 1, new_sequences, new_cache,
                                       next_token_or_endpad, ended, rng2,
                                       next_log_prob, next_code)

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                               sampling_loop_init_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state.sequences
  log_prob = final_state.log_prob
  # Drop the first position because they are dummy bos tokens. Drop the new
  # garbage collection token at the end too.
  return final_sequences[:, 1:-1], log_prob  # pytype: disable=bad-return-type  # jax-ndarray
