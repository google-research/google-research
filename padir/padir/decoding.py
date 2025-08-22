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

"""PaDIR routines for inference from a trained model."""

import functools
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import flax
from flax import traverse_util
import gin
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import binary_search

from padir.padir import config_options
from padir.padir.utils import padir_utils
from padir.padir.utils import vocab_utils

PyTree = Any
PyTreeDef = jax.tree_util.PyTreeDef

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = np.array(-1.0e7)

# Temperatures lower than this are considered 0.0, which is handled specially
# with a conditional. This is to avoid numeric issues from exponentiating on
# 1.0/temperature when temperature is close to 0.0.
MIN_TEMPERATURE = np.array(1e-4)


@flax.struct.dataclass
class DecodingState:
  """Holds decoding state data.

  Used to communicate the current decoding state to tokens_to_decoder_outputs
  methods.
  Note that we use a different class than `SamplingLoopState` or `Beamstate` to
  decouple the concerns of what data is useful for the loop vs. what the
  sampling method needs.
  Decodes for a given batch entry are flattened in a column-major way so that
  decodes from the same batch entry are grouped together.

  Attributes:
    cur_index: [batch_size * num_decodes, 1] array position of the sampling loop
      in the length dimension.
    sequences: [batch_size * num_decodes, max_decode_len] array of current
      sampled sequence prefixes.
    cur_token: [batch_size * num_decodes, block_size] single timestep slice
      containing current tokens.
    cache: any mapping of arrays, e.g. flax attention cache.
  """
  cur_index: Optional[jnp.ndarray]
  sequences: jnp.ndarray
  cur_token: jnp.ndarray
  cache: Optional[Mapping[str, jnp.ndarray]]


@flax.struct.dataclass
class RejecterState:
  """Holds rejecter state data.

  Used to communicate the current decoding state to tokens_to_rejecter_outputs
  methods.

  Attributes:
    step: Scalar decoding step count. Starts from zero.
    sequences: [batch_size * num_decodes, max_decode_len] array of current
      sampled sequences.
    decoder_outputs: mapping with decoder outputs.
    cache: any mapping of arrays, e.g. flax attention cache.
  """

  step: jnp.ndarray
  sequences: jnp.ndarray
  decoder_outputs: Mapping[str, jnp.ndarray]
  cache: Optional[Mapping[str, jnp.ndarray]]


# ------------------------------------------------------------------------------
# Parallel Temperature Sampling
# ------------------------------------------------------------------------------


@flax.struct.dataclass
class SamplingLoopState:
  """Holds sampling state data.

  Attributes:
    step: Scalar decoding step count. Starts from zero.
    cur_index: [batch_size * num_decodes] array position of the sampling loop in
      the length dimension.
    sequences: [batch_size * num_decodes, max_decode_len] array of current
      sampled sequence prefixes.
    cache: any mapping of arrays, e.g. flax attention cache.
    cur_token: [batch_size * num_decodes] single timestep slice containing
      current tokens.
    ended: [batch_size * num_decodes] binary array marking completed sequences.
    rng: Jax PRNGKey
    log_prob: [batch_size * num_decodes] array of log probs for each sequence.
    logits_sum: [batch_size * num_decodes, max_decode_len, vocab_size] sum of
      decoder logits across all decoder iterations so far.
    decoder_self_attentions: decoder self attentions.
    cross_attentions: encoder-decoder cross attentions.
  """
  step: jnp.ndarray
  cur_index: Optional[jnp.ndarray]
  sequences: jnp.ndarray
  cache: Optional[Mapping[str, jnp.ndarray]]
  cur_token: jnp.ndarray
  ended: jnp.ndarray
  rng: jnp.ndarray
  log_prob: jnp.ndarray
  logits_sum: jnp.ndarray
  decoder_self_attentions: jnp.ndarray
  cross_attentions: jnp.ndarray


_dynamic_update_vector_slice_in_dim = jax.vmap(
    lax.dynamic_update_slice_in_dim, in_axes=(0, 0, 0, None))


def _is_tracer(value):
  return isinstance(value, jax.core.Tracer)


StateCallbackFn = Callable[[SamplingLoopState], SamplingLoopState]
LogitCallbackFn = Callable[[jnp.ndarray, SamplingLoopState], jnp.ndarray]

DecoderFn = Callable[
    [DecodingState], Tuple[Mapping[str, jnp.ndarray], Mapping[str, jnp.ndarray]]
]
RejecterFn = Callable[[RejecterState], Mapping[str, jnp.ndarray]]


def temperature_sample(
    decoder_inputs,
    predicted_lengths,
    output_vocab,
    cache,
    tokens_to_decoder_outputs,
    tokens_to_rejecter_outputs,
    num_decode_iterations,
    decode_rng = None,
    num_decodes = 1,
    temperature = 0.0,
    topk = 1,
    topp = 0.0,
    cache_offset = 0,
    initial_index = None,
    max_decode_steps = None,
    max_decode_steps_hard_limit = None,
    rescale_log_probs = False,
    state_callback_fn = None,
    logit_callback_fn = None,
    decoder_input_scheme = config_options.DecoderInputScheme.RANDOM,
    beam_length = 1,
    parallel_decodes = 1,
    brevity_alpha = None,
    remask_stutter = False,
    remasking_scheme = config_options.RemaskingScheme.PREVIOUS_TOKENS,
    output_self_attentions = False,
    output_cross_attentions = False,
    num_decoder_layers = 1,
    num_decoder_heads = 1,
    encoder_input_len = 1,
):
  """Temperature sampling for language model generation.

  The temperature sampling is performed `num_decodes` times in a vectorized
  manner by expanding the batch dimension. This is similar to how beam search
  expands the batch dimension to process each batch element with multiple beams.

  This function dynamically updates the `inputs` array by sampling from the
  model logits, which is provided by `tokens_to_decoder_outputs` callable.

  There are a few important observations related to this function.

  1. The `inputs` is assumed to be a non-packed sequence.

  Args:
    decoder_inputs: array: [batch_size, max_decode_len] int32 sequence of
      tokens. Its content is ignored, replaced as needed based on
      decoder_input_scheme and predicted_lengths.
    predicted_lengths: [batch_size, top_k_lengths] integer array with top length
      predictions for each each input row (including EOS).
    output_vocab: the decoder vocabulary.
    cache: flax attention cache.
    tokens_to_decoder_outputs: decoder function taking decoder input tokens and
      cache and returning decoded logits and updated cache.
    tokens_to_rejecter_outputs: rejecter function taking decoder outputs and
      cache and returning the tokens to remask for the next decoding iteration.
    num_decode_iterations: int: The decoding iterative refinement process is run
      `num_decode_iterations` times.
    decode_rng: JAX PRNGKey.
    num_decodes: number of decoded sequences to be returned.
    temperature: float: sampling temperature factor. As it approaches zero this
      becomes equivalent to greedy sampling. You may also provide an array of
      floats of size batch_size to use different temperature values for each
      batch item.
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
    decoder_input_scheme: how to initialize the decoder input (random tokens,
      mask token, etc.).
    beam_length: decodes several hypotheses varying lengths to the left and
      right of the EOS position from the input.
    parallel_decodes: decodes several hypotheses maintaining the same length as
      the EOS position from the input (if beam_length==1) or for each possible
      beam length (if beam_length>1).
    brevity_alpha: Optional alpha parameter for the brevity penalty, penalizing
      shorter decoded outputs.
    remask_stutter: Whether to remask stutter, including after the last decoding
      iteration.
    remasking_scheme: how to select token replacements when remasking tokens
      across decoding iterations.
    output_self_attentions: whether to output attention scores.
    output_cross_attentions: whether to output cross attention scores.
    num_decoder_layers: number of layers in decoder.
    num_decoder_heads: number of heads in decoder.
    encoder_input_len: number of tokens in encoder input.

  Returns:
    A mapping containing (decodes, log_prob, decoder_self_attentions,
      cross_attentions)
    where `decodes` is sampled sequences with shape
    [batch_size, num_decodes, max_decode_len] sorted by `log_prob`, which is log
    probability of each of the sampled sequences.
  """
  assert decoder_inputs.ndim == 2
  assert predicted_lengths.ndim == 2
  assert predicted_lengths.shape[0] == decoder_inputs.shape[0]
  top_k_lengths = predicted_lengths.shape[1]
  assert top_k_lengths * beam_length * parallel_decodes == num_decodes

  if decode_rng is None:
    decode_rng = jax.random.PRNGKey(0)  # Could make input dependent.

  if (max_decode_steps_hard_limit is not None and
      max_decode_steps_hard_limit > 0 and max_decode_steps is not None):
    max_decode_steps = jnp.minimum(max_decode_steps,
                                   max_decode_steps_hard_limit)

  # [batch, len] -> [batch * num_decodes, len]
  expanded_decoder_inputs = padir_utils.initialize_decoder_input(
      (decoder_inputs.shape[0] * num_decodes, decoder_inputs.shape[1]),
      decoder_input_scheme=decoder_input_scheme,
      vocab=output_vocab,
      key=decode_rng,
  )

  eos_pos = predicted_lengths - 1

  if parallel_decodes > 1:
    assert (
        decoder_input_scheme == config_options.DecoderInputScheme.RANDOM
    ), 'parallel_decodes > 1 only meaningful for diffusion models.'
  expanded_decoder_inputs = padir_utils.restore_eos(
      expanded_decoder_inputs,
      eos_pos,
      beam_length=beam_length,
      parallel_decodes=parallel_decodes,
      eos_id=output_vocab.eos_id,
  )

  if num_decodes > 1:
    expanded_cache = cache_map(
        functools.partial(
            padir_utils.flat_batch_beam_expand,
            beam_size=num_decodes,
            offset=cache_offset,
        ),
        cache,
        # When we start with a prefilled cache, the cache index is no longer a
        # scalar that will broadcast across multiple decodes, it is a vector and
        # needs to be updated to handle the multiple decodes.
        apply_to_index=initial_index is not None,
    )
    if initial_index is not None:
      initial_index = padir_utils.flat_batch_beam_expand(
          initial_index, num_decodes
      )
  else:
    expanded_cache = cache

  # expanded_decodes: [batch * num_decodes, len]
  # expanded_log_prob: [batch * num_decodes, len]
  single_trial_output = _temperature_sample_single_trial(
      expanded_decoder_inputs,
      expanded_cache,
      tokens_to_decoder_outputs,
      tokens_to_rejecter_outputs,
      output_vocab.eos_id,
      num_decode_iterations,
      decode_rng,
      num_decodes,
      temperature,
      topk,
      topp,
      initial_index=initial_index,
      max_decode_steps=max_decode_steps,
      rescale_log_probs=rescale_log_probs,
      state_callback_fn=state_callback_fn,
      logit_callback_fn=logit_callback_fn,
      mask_id=vocab_utils.get_mask_id(output_vocab),
      remask_stutter=remask_stutter,
      remasking_scheme=remasking_scheme,
      output_self_attentions=output_self_attentions,
      output_cross_attentions=output_cross_attentions,
      num_decoder_layers=num_decoder_layers,
      num_decoder_heads=num_decoder_heads,
      encoder_input_len=encoder_input_len,
  )
  expanded_decodes = single_trial_output['expanded_decodes']
  expanded_log_prob = single_trial_output['expanded_log_prob']
  expanded_decoder_self_attentions = single_trial_output[
      'expanded_decoder_self_attentions'
  ]
  expanded_cross_attentions = single_trial_output['expanded_cross_attentions']

  batch_size = decoder_inputs.shape[0]
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  decodes = padir_utils.unflatten_beam_dim(
      expanded_decodes, batch_size, num_decodes
  )
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  log_prob = padir_utils.unflatten_beam_dim(
      expanded_log_prob, batch_size, num_decodes
  )

  # [batch, num_decodes, 1]
  lengths = jnp.sum(decodes > 0, axis=-1, keepdims=True)

  # Sort `decodes` and `log_prob` by increasing (log) probabilities of the
  # sampled sequence.
  # [batch, num_decodes, 1]
  log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
  if brevity_alpha is not None:
    log_prob /= brevity_penalty(brevity_alpha, lengths)
  idxs = jnp.argsort(log_prob, axis=1)

  # returns [batch, num_decodes, len], [batch, num_decodes] in sorted order.
  decodes = jnp.take_along_axis(decodes, idxs, axis=1)
  log_prob = jnp.take_along_axis(log_prob, idxs, axis=1)

  # [batch * num_decodes, num_decoder_layers, num_decoder_heads, len, len]
  # -> [batch, num_decodes, num_decoder_layers, num_decoder_heads, len, len]
  if output_self_attentions:
    decoder_self_attentions = padir_utils.unflatten_beam_dim(
        expanded_decoder_self_attentions, batch_size, num_decodes
    )
    decoder_self_attentions = jnp.take_along_axis(
        decoder_self_attentions,
        jnp.expand_dims(
            idxs, axis=(3, 4, 5)
        ),  # do not change trailing dimensions
        axis=1,
    )
  else:
    decoder_self_attentions = jnp.zeros(0, dtype=jnp.float32)

  # [batch * num_decodes, num_decoder_layers, num_decoder_heads, len, len]
  # -> [batch, num_decodes, num_decoder_layers, num_decoder_heads, len, len]
  if output_cross_attentions:
    cross_attentions = padir_utils.unflatten_beam_dim(
        expanded_cross_attentions, batch_size, num_decodes
    )
    cross_attentions = jnp.take_along_axis(
        cross_attentions,
        jnp.expand_dims(
            idxs, axis=(3, 4, 5)
        ),  # do not change trailing dimensions
        axis=1,
    )
  else:
    cross_attentions = jnp.zeros(0, dtype=jnp.float32)

  return {
      'decodes': decodes,
      'log_prob': jnp.squeeze(log_prob, axis=-1),
      'decoder_self_attentions': decoder_self_attentions,
      'cross_attentions': cross_attentions,
  }


def _temperature_sample_single_trial(
    inputs,
    cache,
    tokens_to_decoder_outputs,
    tokens_to_rejecter_outputs,
    eos_id,
    num_decode_iterations,
    prng_key,
    num_decodes = 1,
    temperature = 1.0,
    topk = 20,
    topp = 0.0,
    initial_index = None,
    max_decode_steps = None,
    rescale_log_probs = True,
    state_callback_fn = None,
    logit_callback_fn = None,
    mask_id = None,
    remask_stutter = False,
    remasking_scheme = config_options.RemaskingScheme.PREVIOUS_TOKENS,
    output_self_attentions = False,
    output_cross_attentions = False,
    num_decoder_layers = 1,
    num_decoder_heads = 1,
    encoder_input_len = 1,
):
  """A helper function for `temperature_sample`."""

  del eos_id  # Unused in this version, keeping it to be consistent with the
  # standard temperature_sample function signature.
  if mask_id is None:
    raise ValueError('mask_id cannot be None.')

  # We can check the values of topp and topk only if they are not dynamic.
  if not (_is_tracer(topp) or _is_tracer(topk)) and topp and topk:
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

  extra_input_tokens = 0  # original: 2
  expanded_prompt_inputs = jnp.append(
      inputs,
      jnp.zeros((batch_size, extra_input_tokens), dtype=inputs.dtype),
      axis=1)

  temperature = jnp.asarray(temperature)

  # Initialize sampling loop state.
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
  # (batch, length) array containing prefix prompt tokens for sampling loop
  # as well as the generated output of newly sampled tokens.
  sequences0 = expanded_prompt_inputs
  assert sequences0.shape == (batch_size, max_decode_len)
  log_prob0 = jnp.zeros((batch_size, max_decode_len), dtype=jnp.float32)
  vocab_size = gin.query_parameter('network.PadirModelConfig.vocab_size')
  sum0 = jnp.zeros((batch_size, max_decode_len, vocab_size), dtype=jnp.float32)
  if output_self_attentions:
    decoder_self_attentions0 = jnp.zeros(
        (
            batch_size,
            num_decoder_layers,
            num_decoder_heads,
            max_decode_len,
            max_decode_len,
        ),
        dtype=jnp.float32,
    )
  else:
    decoder_self_attentions0 = jnp.zeros(0, dtype=jnp.float32)
  if output_cross_attentions:
    cross_attentions0 = jnp.zeros(
        (
            batch_size,
            num_decoder_layers,
            num_decoder_heads,
            max_decode_len,
            encoder_input_len,
        ),
        dtype=jnp.float32,
    )
  else:
    cross_attentions0 = jnp.zeros(0, dtype=jnp.float32)
  sampling_loop_init_state = SamplingLoopState(
      step,
      None,
      sequences0,
      cache,
      token0,
      ended0,
      rng0,
      log_prob0,
      sum0,
      decoder_self_attentions0,
      cross_attentions0,
  )
  mask0 = (sequences0 > 0).astype(jnp.int32)

  def sampling_loop_cond_fn(
      state,
  ):
    """Sampling loop termination condition."""
    # Have all sampled sequences reached an end marker?
    # Different elements in the batch can be at different loop indices, if any
    # of our examples are not at the end, keep going.
    all_sequences_ended = jnp.all(state.ended)
    return ~all_sequences_ended

  def sampling_loop_body_fn(state):
    """Sampling loop state update."""

    if state_callback_fn is not None:
      state = state_callback_fn(state)

    # Split RNG for sampling.
    rng1, rng2 = random.split(state.rng)
    # Call fast-decoder model on current tokens to get next-position logits.
    decoding_state = DecodingState(
        cur_index=state.cur_index,
        sequences=state.sequences,
        cur_token=state.cur_token,
        cache=state.cache,
    )
    decoder_outputs, _ = tokens_to_decoder_outputs(decoding_state)
    logits = decoder_outputs['logits']
    decoder_self_attentions = decoder_outputs['self_attentions']
    cross_attentions = decoder_outputs['cross_attentions']
    # Sample next token from logits.

    if logit_callback_fn is not None:
      logits = logit_callback_fn(logits, state)

    def sample_logits_with_nonzero_temperature(logits, temperature):
      scaled_logits = logits / jnp.maximum(temperature, MIN_TEMPERATURE)
      if _is_tracer(topk) or topk:
        scaled_logits = jax.lax.cond(
            topk > 0,
            lambda: binary_search.topk_mask(scaled_logits, topk, NEG_INF),  # pytype: disable=wrong-arg-types  # jax-ndarray
            lambda: scaled_logits,
        )

      # When topp is dynamic, we always use it since we cannot check
      # non-zeroness (but it will have no effect if topp is 0.0).
      if _is_tracer(topp) or topp:
        scaled_logits = binary_search.topp_mask(scaled_logits, topp, NEG_INF)  # pytype: disable=wrong-arg-types  # jax-ndarray

      # [batch, len]
      next_tokens = random.categorical(rng1, scaled_logits).astype(jnp.int32)

      # log probability of the current token conditioned on the previously
      # sampled and prefix tokens.
      # [batch, len, vocab] -> [batch, len, vocab]
      if rescale_log_probs:
        log_probs = jax.nn.log_softmax(scaled_logits)
      else:
        log_probs = jax.nn.log_softmax(logits)
      # [batch, len, vocab] -> [batch, len]
      next_log_probs = jnp.squeeze(
          jnp.take_along_axis(
              log_probs, jnp.expand_dims(next_tokens, axis=-1), axis=-1
          ),
          axis=-1,
      )

      return (next_tokens, next_log_probs)

    def sample_logits_with_zero_temperature(logits, temperature):  # pylint: disable=unused-argument
      # For zero temperature, we always want the greedy output, regardless
      # of the values of topk and topp.
      # [batch, len]
      next_tokens = jnp.argmax(logits, -1).astype(jnp.int32)

      if rescale_log_probs:
        next_log_probs = jnp.zeros_like(next_tokens, dtype=jnp.float32)
      else:
        log_probs = jax.nn.log_softmax(logits)
        # [batch, len, vocab] -> [batch, len]
        next_log_probs = jnp.squeeze(
            jnp.take_along_axis(
                log_probs, jnp.expand_dims(next_tokens, axis=-1), axis=-1
            ),
            axis=-1,
        )

      return (next_tokens, next_log_probs)

    # Perform sampling with temperature
    if len(temperature.shape) == 1:
      # Each batch item can have different temperatures.
      def map_logits_with_different_temperatures(logits_batch_item,
                                                 temperature_batch_item):
        return lax.cond(temperature_batch_item > MIN_TEMPERATURE,
                        sample_logits_with_nonzero_temperature,
                        sample_logits_with_zero_temperature,
                        jnp.expand_dims(logits_batch_item,
                                        axis=0), temperature_batch_item)

      (next_tokens, next_log_probs) = jax.vmap(
          map_logits_with_different_temperatures
      )(logits, jnp.repeat(temperature, num_decodes))
    else:
      # Single temperature value is applied to all batch items.
      (next_tokens, next_log_probs) = lax.cond(
          temperature > MIN_TEMPERATURE,
          sample_logits_with_nonzero_temperature,
          sample_logits_with_zero_temperature,
          logits,
          temperature,
      )
    # Mask everything outside the decoder attention mask.
    next_tokens *= mask0
    next_log_probs *= mask0

    ended = jax.lax.full(
        [batch_size, 1],
        state.step >= num_decode_iterations - 1,
        dtype=jnp.bool_,
    )

    rejecter_state = RejecterState(
        step=state.step,
        sequences=next_tokens,
        decoder_outputs=decoder_outputs,
        cache={},
    )
    rejecter_outputs, _ = tokens_to_rejecter_outputs(rejecter_state)
    assert isinstance(rejecter_outputs, Mapping)
    accepted_mask = rejecter_outputs['approved_mask']

    logits_sum = state.logits_sum + logits
    replacements = remasking_scheme.get_replacements(
        sequences0, state.sequences, logits_sum
    )
    next_tokens = padir_utils.replace_rejected_predictions(
        accepted_mask,
        predictions=next_tokens,
        replacements=replacements,
        ended=ended,
        remask_stutter=remask_stutter,
    )

    unchanged = jnp.all(
        jnp.equal(next_tokens, state.sequences), axis=-1, keepdims=True)
    ended = jnp.logical_or(ended, unchanged)

    return SamplingLoopState(
        state.step + 1,
        None,
        next_tokens,
        {},
        state.cur_token,
        ended,
        rng2,
        next_log_probs,
        logits_sum,
        decoder_self_attentions,
        cross_attentions,
    )

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                               sampling_loop_init_state)

  if state_callback_fn is not None:
    final_state = state_callback_fn(final_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state.sequences
  log_prob = final_state.log_prob
  decoder_self_attentions = final_state.decoder_self_attentions
  unchanged = jnp.equal(final_sequences, sequences0)
  final_sequences = jnp.where(unchanged, jnp.zeros_like(final_sequences),
                              final_sequences)

  log_prob *= (final_sequences > 0).astype(jnp.float32)
  return {
      'expanded_decodes': final_sequences,
      'expanded_log_prob': log_prob,
      'expanded_decoder_self_attentions': decoder_self_attentions,
      'expanded_cross_attentions': final_state.cross_attentions,
  }


# ------------------------------------------------------------------------------
# BEAM Sampling
# ------------------------------------------------------------------------------


def brevity_penalty(
    alpha, length
):
  """Brevity penalty function for beam search penalizing short sequences.

  Args:
    alpha: float: brevity-penalty scaling parameter.
    length: int: length of considered sequence.

  Returns:
    Brevity penalty score as jax scalar.
  """
  return jnp.power(((5.0 + length) / 6.0), alpha)


# Beam handling utility functions:


def cache_map(fn, cache, apply_to_index = False):
  """Maps function over that caches, even multiple caches in various layers.

  Args:
    fn: The function to apply.
    cache: The cache to apply it to.
    apply_to_index: Whether to apply the function to the cache index.

  Returns:
    The result of applying `fn` to the cache.
  """
  frozen = isinstance(cache, flax.core.FrozenDict)
  if frozen:
    cache = flax.core.unfreeze(cache)
  flat_cache = traverse_util.flatten_dict(cache)
  if apply_to_index:
    keyvals = flat_cache
  else:
    keyvals = {k: v for k, v in flat_cache.items() if k[-1] != 'cache_index'}
  # Exclude cached relative position bias from beam expansion, etc.
  # Also excludes scalar index in absolute position embedder from expansion.
  # TODO(levskaya): generalize cache_map to accept a list of leaf names to
  #   map over, instead of doing this ad-hoc.
  exclusion_list = ['cached_bias', 'position_embedder_index']
  keyvals = {k: v for k, v in keyvals.items() if k[-1] not in exclusion_list}

  keyvals = jax.tree.map(fn, keyvals)
  flat_cache.update(keyvals)
  new_cache = traverse_util.unflatten_dict(flat_cache)
  if frozen:
    new_cache = flax.core.freeze(new_cache)
  return new_cache
