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

"""Fast decoding routines for duo-decoding sequences."""
import functools
from typing import Callable, Mapping, Optional, Tuple, Union
from absl import logging
import flax
import gin
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
from t5x import binary_search
from t5x.decoding import _is_tracer
from t5x.decoding import cache_map
from t5x.decoding import DecodingState
from t5x.decoding import flat_batch_beam_expand
from t5x.decoding import LogitCallbackFn
from t5x.decoding import MIN_TEMPERATURE
from t5x.decoding import NEG_INF
from t5x.decoding import SamplingLoopState
from t5x.decoding import StateCallbackFn
from t5x.decoding import unflatten_beam_dim


@flax.struct.dataclass
class DuoSamplingLoopState:
  with_context: SamplingLoopState
  wo_context: SamplingLoopState


def kl_between_two_logits(logits, logits_wo_context):
  """Compute the KL divergence from the uniform distribution.

  Args:
    logits: unnormalized logits with contexts in inputs.
    logits_wo_context: unnormalized logits without contexts in inputs.

  Returns:
    the kl divergence between the logits and uniform distribution.
  """
  # Clip lower bound for probs to avoid NaNs.
  probs = jnp.clip(jax.nn.softmax(logits), 1e-7, 1)
  probs_wo_context = jnp.clip(jax.nn.softmax(logits_wo_context), 1e-7, 1)
  q = probs * (jnp.log(probs) - jnp.log(probs_wo_context))
  kl = jnp.sum(q, axis=-1)
  return jax.nn.relu(kl)  # Guard against numerical issues giving negative KL.


@gin.configurable
def temperature_sample(
    inputs,
    inputs_wo,
    cache,
    cache_wo,
    tokens_to_logits,
    tokens_to_logits_wo,
    eos_id,
    decode_rng = None,
    num_decodes = 1,
    temperature = 1.0,
    kl_bandwidth = 1.0,
    topk = 1,
    topp = 0.0,
    cache_offset = 0,
    cache_offset_wo = 0,
    initial_index = None,
    initial_index_wo = None,
    max_decode_steps = None,
    max_decode_steps_hard_limit = None,
    rescale_log_probs = True,
    state_callback_fn = None,
    logit_callback_fn = None,
):
  """Temperature sampling for language model generation.

  Similar to t5x.decoding but with two decodings in parallel: one with contexts
  in inputs (or prefix) and one without. In each decoding step, we calculate the
  KL-divegence between the distributions over dictionary tokens of two decoding
  sequences. We map this KL-divergence to the temperature according to an
  exponential decay function (with hyperparameter kl_bandwidth). The actual
  temperature is in the range [0, temperature].

  Args:
    inputs: array: [batch_size, max_decode_len] int32 sequence of tokens.
    inputs_wo: similar to inputs but without contexts in inputs/prefix.
    cache: flax attention cache.
    cache_wo: similar to cache but without contexts in inputs/prefix.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    tokens_to_logits_wo: similar to tokens_to_logits but without contexts in
      inputs/prefix.
    eos_id: int: end-of-sentence token for target vocabulary.
    decode_rng: JAX PRNGKey.
    num_decodes: number of decoded sequences to be returned.
    temperature: float: sampling temperature factor. As it approaches zero this
      becomes equivalent to greedy sampling. You may also provide an array of
      floats of size batch_size to use different temperature values for each
      batch item.
    kl_bandwidth: the half-life hyperparamter for the exponential decay function
      that maps kl-divergence to the actual temperature.
    topk: integer: if nonzero only use the top-k logits to sample next token, if
      zero don't use any cutoff and sample from full logits over vocabulary.
    topp: float: if nonzero only use the smallest number of logits whose
      cumulative sum of probs adds up to (at least) topp. Will raise ValueError
      if it's nonzero when topk is nonzero.
    cache_offset: axis offset for cache, arising from scanned layers.
    cache_offset_wo: similar to cache_offset but without contexts in
      inputs/prefix.
    initial_index: Optional[array]: [batch_size] int32 a vector of loop indexes
      to start decoding at.
    initial_index_wo: similar to initial_index but without contexts in
      inputs/prefix.
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

  def maybe_expand_by_decodes(
      num_decodes, inputs, cache, cache_offset, initial_index):
    if num_decodes > 1:
      # [batch, len] -> [batch * num_decodes, len]
      expanded_inputs = flat_batch_beam_expand(inputs, num_decodes)
      expanded_cache = cache_map(
          functools.partial(
              flat_batch_beam_expand, beam_size=num_decodes,
              offset=cache_offset),
          cache,
          # When we start with a prefilled cache, the cache index is no longer a
          # scalar that will broadcast across multiple decodes, it is a vector
          # and needs to be updated to handle the multiple decodes.
          apply_to_index=initial_index is not None)
      if initial_index is not None:
        initial_index = flat_batch_beam_expand(initial_index, num_decodes)
    else:
      expanded_inputs = inputs
      expanded_cache = cache
    return expanded_inputs, expanded_cache, initial_index

  expanded_inputs, expanded_cache, initial_index = maybe_expand_by_decodes(
      num_decodes, inputs, cache, cache_offset, initial_index)
  expanded_inputs_wo, expanded_cache_wo, initial_index_wo = (
      maybe_expand_by_decodes(
          num_decodes, inputs_wo, cache_wo, cache_offset_wo, initial_index_wo))

  # expanded_decodes: [batch * num_decodes, len]
  # expanded_log_prob: [batch * num_decodes]
  expanded_decodes, expanded_log_prob = _temperature_sample_single_trial(
      expanded_inputs,
      expanded_inputs_wo,
      expanded_cache,
      expanded_cache_wo,
      tokens_to_logits,
      tokens_to_logits_wo,
      eos_id,
      decode_rng,
      num_decodes,
      temperature,
      kl_bandwidth,
      topk,
      topp,
      initial_index=initial_index,
      initial_index_wo=initial_index_wo,
      max_decode_steps=max_decode_steps,
      rescale_log_probs=rescale_log_probs,
      state_callback_fn=state_callback_fn,
      logit_callback_fn=logit_callback_fn)

  batch_size = inputs.shape[0]
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  decodes = unflatten_beam_dim(expanded_decodes, batch_size, num_decodes)
  # [batch * num_decodes] -> [batch, num_decodes]
  log_prob = unflatten_beam_dim(expanded_log_prob, batch_size, num_decodes)

  # Sort `decodes` and `log_prob` by increasing log probabilities of the sampled
  # sequence.
  # [batch, num_decodes, 1]
  idxs = jnp.expand_dims(jnp.argsort(log_prob, axis=-1), axis=-1)

  # returns [batch, num_decodes, len], [batch, num_decodes] in sorted order.
  return jnp.take_along_axis(
      decodes, idxs, axis=1), jnp.take_along_axis(
          log_prob, jnp.squeeze(idxs, axis=-1), axis=-1)


def _temperature_sample_single_trial(
    inputs,
    inputs_wo,
    cache,
    cache_wo,
    tokens_to_logits,
    tokens_to_logits_wo,
    eos_id,
    prng_key,
    num_decodes = 1,
    temperature = 1.0,
    kl_bandwidth = 0.0,
    topk = 20,
    topp = 0.0,
    initial_index = None,
    initial_index_wo = None,
    max_decode_steps = None,
    rescale_log_probs = True,
    state_callback_fn = None,
    logit_callback_fn = None,
):
  """A helper function for `temperature_sample`."""

  # We can check the values of topp and topk only if they are not dynamic.
  if not _is_tracer(topp) and topp and topk:
    raise ValueError('At most one of `topp` or `topk` may be non-zero.')

  # Assume inputs and inputs_wo has the same shape.
  batch_size, max_decode_len = inputs.shape
  max_decode_len_wo = max_decode_len

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
    max_decode_len_wo = jnp.sum(inputs_wo != 0, axis=1) + max_decode_steps
    max_decode_len_wo = jnp.minimum(inputs_wo.shape[1], max_decode_len_wo)

  # In the case of starting generation from a non-zero index, it is possible for
  # one batch element to reach `max_decode_len` number of decoding steps before
  # another. In order to let the last element decoder all the way to
  # `max_decode_len` number of steps, we add a final garbage token to the end of
  # the sequences. Any element that has reached `max_decode_len` before the rest
  # of the elements will continually overwrite this token until all elements
  # finish.
  # [batch, length+1] -> [batch, length+2]
  extra_input_tokens = 2
  expanded_prompt_inputs = jnp.append(
      inputs,
      jnp.zeros((batch_size, extra_input_tokens), dtype=inputs.dtype),
      axis=1)
  expanded_prompt_inputs_wo = jnp.append(
      inputs_wo,
      jnp.zeros((batch_size, extra_input_tokens), dtype=inputs.dtype),
      axis=1)
  end_marker = jnp.array(eos_id)

  temperature = jnp.asarray(temperature)
  kl_bandwidth = jnp.asarray(kl_bandwidth)
  topk = jnp.asarray(topk)
  topp = jnp.asarray(topp)

  # Initialize sampling loop state.
  def get_sampling_loop_init_state(
      initial_index, expanded_prompt_inputs, cache):
    # Outer variables: prng_key, end_marker, batch_size.
    step = jnp.zeros((), dtype=jnp.int32)
    # initial loop PRNGKey
    rng0 = prng_key
    # the per batch-item holding current token in loop.
    if initial_index is None:
      # the per batch-item loop position counter.
      i0 = jnp.zeros((batch_size), dtype=jnp.int32)
      # the per batch-item holding current token in loop.
      token0 = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    else:
      # the per batch-item loop position counter.
      i0 = initial_index
      # the per batch-item holding current token in loop.
      # Select the token that the initial index is pointing to.
      token0 = jnp.take_along_axis(
          expanded_prompt_inputs, jnp.expand_dims(i0, axis=1), axis=1)
    # per batch-item state bit indicating if sentence has finished.
    ended0 = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
    # (batch, length+2) array containing prefix prompt tokens for sampling loop
    # as well as the generated output of newly sampled tokens.
    sequences0 = expanded_prompt_inputs
    log_prob0 = jnp.zeros((batch_size,), dtype=jnp.float32)
    sampling_loop_init_state = SamplingLoopState(
        step, i0, sequences0, cache, token0, ended0, rng0, log_prob0)
    # Initial eos count to be used to determine whether eos is "generated". Many
    # inputs follow the format bos, inputs..., eos, targets..., eos. By counting
    # the number of eos tokens we can detect when a new one is added, instead of
    # just finding the one that probably ends the inputs.
    # [batch, 1]
    initial_eos_count = jnp.sum(
        sequences0 == end_marker, axis=-1, keepdims=True)
    return sampling_loop_init_state, initial_eos_count

  sampling_loop_init_state, initial_eos_count = get_sampling_loop_init_state(
      initial_index, expanded_prompt_inputs, cache)
  sampling_loop_init_state_wo, initial_eos_count_wo = (
      get_sampling_loop_init_state(
          initial_index_wo, expanded_prompt_inputs_wo, cache_wo))
  duo_sampling_loop_init_state = DuoSamplingLoopState(
      sampling_loop_init_state, sampling_loop_init_state_wo)

  def sampling_loop_cond_fn(state):
    """Sampling loop termination condition."""
    # Have all sampled sequences reached an end marker?
    # Different elements in the batch can be at different loop indices, if any
    # of our examples are not at the end, keep going.
    all_sequences_ended = jnp.all(state.ended)
    return ~all_sequences_ended  # pytype: disable=bad-return-type  # jnp-type

  def duo_sampling_loop_cond_fn(state):
    return sampling_loop_cond_fn(state.with_context)  # pytype: disable=bad-return-type  # jnp-type

  def duo_sampling_loop_body_fn(
      state):
    """Sampling loop state update."""

    def state_to_logits(state, tokens_to_logits):
      # Outer variables: extra_input_tokens, logit_callback_fn
      # Split RNG for sampling.
      rng1, rng2 = random.split(state.rng)
      # Call fast-decoder model on current tokens to get next-position logits.
      decoding_state = DecodingState(
          cur_index=state.cur_index,
          sequences=state.sequences[:, :-extra_input_tokens],
          cur_token=state.cur_token,
          cache=state.cache)
      logits, new_cache = tokens_to_logits(decoding_state)
      # Sample next token from logits.

      if logit_callback_fn is not None:
        logits = logit_callback_fn(logits, state)
      return logits, new_cache, rng1, rng2

    def sample_logits_with_nonzero_temperature(
        logits, temperature, rng, logits_wo):
      # Outer variables: topk, topp, rescale_log_probs, kl_bandwidth
      temperature_scaled = temperature
      if _is_tracer(kl_bandwidth) or kl_bandwidth:
        kl = kl_between_two_logits(logits, logits_wo)
        temperature_scaled *= jnp.where(
            kl_bandwidth == 0,
            jnp.ones_like(temperature),
            jnp.exp(-jnp.log(2) / kl_bandwidth * kl))
      scaled_logits = logits / jnp.maximum(
          jnp.expand_dims(temperature_scaled, axis=-1), MIN_TEMPERATURE)
      if _is_tracer(topk) or topk:
        scaled_logits = jnp.where(
            topk == 0,
            scaled_logits,
            binary_search.topk_mask(scaled_logits, topk, NEG_INF))  # pytype: disable=wrong-arg-types  # jax-ndarray

      # When topp is dynamic, we always use it since we cannot check
      # non-zeroness (but it will have no effect if topp is 0.0).
      if _is_tracer(topp) or topp:
        scaled_logits = jnp.where(
            topp == 0,
            scaled_logits,
            binary_search.topp_mask(scaled_logits, topp, NEG_INF))  # pytype: disable=wrong-arg-types  # jax-ndarray

      # [batch]
      next_token = random.categorical(rng, scaled_logits).astype(jnp.int32)

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

      return (next_token, next_log_prob)

    def sample_logits_with_zero_temperature(
        logits, temperature, rng, logits_wo):  # pylint: disable=unused-argument
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

      return (next_token, next_log_prob)

    def perform_sampling(logits, rng, logits_wo):
      # Outer variables: temperature, num_decodes, topk, topp, rescale_log_probs, kl_bandwidth.
      # Perform sampling with temperature
      if len(temperature.shape) == 1:
        # Each batch item can have different temperatures.
        def map_logits_with_different_temperatures(
            logits_batch_item, temperature_batch_item
        ):
          return lax.cond(
              temperature_batch_item > MIN_TEMPERATURE,
              sample_logits_with_nonzero_temperature,
              sample_logits_with_zero_temperature,
              jnp.expand_dims(logits_batch_item, axis=0),
              temperature_batch_item,
              rng,
              logits_wo,
          )

        (next_token, next_log_prob) = jax.vmap(
            map_logits_with_different_temperatures
        )(logits, jnp.repeat(temperature, num_decodes))
        next_token = jnp.squeeze(next_token, axis=-1)
        next_log_prob = jnp.squeeze(next_log_prob, axis=-1)
      else:
        # Single temperature value is applied to all batch items.
        (next_token, next_log_prob) = lax.cond(
            temperature > MIN_TEMPERATURE,
            sample_logits_with_nonzero_temperature,
            sample_logits_with_zero_temperature,
            logits,
            temperature,
            rng,
            logits_wo,
        )
      return next_token, next_log_prob

    def update_state(
        state,
        next_token,
        next_log_prob,
        max_decode_len,
        rng,
        new_cache,
        initial_eos_count,
    ):
      # Outer variables: end_marker.
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
      new_sequences = state.sequences * (
          1 - one_hot) + next_token_or_endpad * one_hot
      # new_sequences = dynamic_update_vector_slice_in_dim(sequences,
      #                                                    next_token_or_endpad,
      #                                                    i + 1,
      #                                                    0)
      # Count eos tokens in the sequences and compare to the initial count
      # [batch, 1]
      cur_eos_count = jnp.sum(
          new_sequences == end_marker, axis=-1, keepdims=True)
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

      return SamplingLoopState(state.step + 1, i + 1, new_sequences, new_cache,
                               next_token_or_endpad, ended, rng, next_log_prob)

    if state_callback_fn is not None:
      state = DuoSamplingLoopState(state_callback_fn(state.with_context),
                                   state_callback_fn(state.wo_context))
    logits, new_cache, rng1, rng2 = state_to_logits(state.with_context,
                                                    tokens_to_logits)
    logits_wo, new_cache_wo, _, _ = state_to_logits(state.wo_context,
                                                    tokens_to_logits_wo)

    # Merged perform_sampling
    # next_token, metrics = perform_sampling(logits, rng1)
    # next_token_wo, metrics_wo = perform_sampling(logits_wo, rng1)
    next_token, next_log_prob = perform_sampling(logits, rng1, logits_wo)

    state_with_context = update_state(
        state.with_context, next_token, next_log_prob, max_decode_len, rng2,
        new_cache, initial_eos_count)
    state_wo_context = update_state(
        state.wo_context, next_token, next_log_prob, max_decode_len_wo, rng2,
        new_cache_wo, initial_eos_count_wo)

    return DuoSamplingLoopState(state_with_context, state_wo_context)

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(duo_sampling_loop_cond_fn,
                               duo_sampling_loop_body_fn,
                               duo_sampling_loop_init_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state.with_context.sequences
  log_prob = final_state.with_context.log_prob
  # Drop the first position because they are dummy bos tokens. Drop the new
  # garbage collection token at the end too.
  return final_sequences[:, 1:-1], log_prob  # pytype: disable=bad-return-type  # jax-ndarray
