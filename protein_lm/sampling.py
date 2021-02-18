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

# Lint as: python3
"""Functions related to sampling from a model."""

from absl import logging
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.nn
import jax.numpy as jnp

LARGE_NEGATIVE = -1e12


def multinomial(rng, logits):
  """Draws samples from a multinomial distribution.

  Args:
    rng: A jax.random.PRNGKey.
    logits: An array of shape (..., num_categories) containing unnormalized
      log-probabilities.

  Returns:
    An array of shape (...) containing sampled category indices.
  """
  probs = jax.nn.softmax(logits)
  probs = jnp.cumsum(probs, axis=-1)
  a = jax.random.uniform(rng, logits.shape[:-1] + (1,))
  out = jnp.argmin(a > probs, axis=-1)
  return out


def apply_repetition_penalty(sequences,
                             logits,
                             i,
                             repetition_penalty,
                             repetition_window,
                             repetition_penalty_normalize):
  """Apply repetition penalty.

  Repetition penalty is introduced in Section 4.1. of the
  [CTRL paper](https://einstein.ai/presentations/ctrl.pdf).

  It involves reducing the logits corresponding to previously generated tokens.
  Given a list of generated tokens g, the next token probability is given by:

    p_{i + 1} propto exp(logit_i / (T * I_i))

  where I_i = repetition_penalty, if i in g,
              1, otherwise.

  Args:
    sequences: An array of shape (batch_size, max_seq_len) with the sequences.
    logits: An array of shape (batch_size, vocab_size) with next token logits.
    i: An array of shape () with the current sequence positions.
    repetition_penalty: A float indicating the repetition penalty.
    repetition_window: An int indicating the window size for repetition penalty.
      We reduce the probability of tokens that have been generated within
      repetition_window tokens prior to the token to be generated.
    repetition_penalty_normalize: A bool indicating whether to normalize
      the logits (log_softmax) before applying the repetition penalty. This
      ensures that all logits have the same sign.

  Returns:
    An array of shape (batch_size, vocab_size) with updated next token logits.
  """
  max_i = i  # We are currently generating a token for position i + 1.
  min_i = i - repetition_window + 1
  batch_size, vocab_size = logits.shape
  positions = jnp.arange(sequences.shape[1])
  positions = jnp.tile(positions[jnp.newaxis, :, jnp.newaxis],
                       [batch_size, 1, vocab_size])
  sequences_onehot = jnp.eye(vocab_size)[sequences]
  sequences_onehot = jnp.where((positions >= min_i) * (positions <= max_i),
                               sequences_onehot,
                               jnp.zeros_like(sequences_onehot))
  # Compute the indicator that a token appeared at least once in the
  # repetition window. Output shape: (batch_size, vocab_size).
  indicator = jnp.max(sequences_onehot, axis=1)
  # Compute a penalty tensor. The logits are divided by the penalty tensor.
  penalty_tensor = jnp.where(indicator,
                             jnp.ones_like(logits) * repetition_penalty,
                             jnp.ones_like(logits))
  if repetition_penalty_normalize:
    logits = jax.nn.log_softmax(logits)
  # Dividing a negative logit by the penalty tensor actually increases the
  # resulting probability. Take the inverse for negative logits.
  penalty_tensor = jnp.where(logits > 0,
                             penalty_tensor,
                             1 / penalty_tensor)

  logits = logits / penalty_tensor
  return logits


def temperature_sample(prompt,
                       init_cache,
                       tokens_to_logits,
                       temperature=1.0,
                       top_k=None,
                       repetition_penalty=1,
                       repetition_window=4,
                       repetition_penalty_normalize=False,
                       max_decode_len=512,
                       rng=None,
                       eos_token=None,
                       pad_token=None,
                       masked_tokens=None,
                       use_lax_while_loop=True):
  """Temperature sampling for sequence models.

  Args:
    prompt: An array of shape (batch_size, prompt_length) containing the input
      prompt (the model consumes these tokens and starts generation after). For
      generic sampling, the prompt must be a single BOS token.
    init_cache: A flax.nn.attention.Cache object.
    tokens_to_logits: A fast autoregressive decoder function taking single token
      slices and the cache and returning next-token logits and updated cache.
    temperature: A float with the sampling temperature factor. As it approaches
      zero this sampling procedure becomes equivalent to greedy sampling.
    top_k: An int with the number of high probability tokens to keep for
      sampling. If None, keep all.
    repetition_penalty: A float with the repetition penalty. Values smaller
      (greater) than 1 encourage (discourage) repetition; 1 disables penalties.
    repetition_window: An int indicating the window size for repetition penalty.
      We reduce the probability of tokens that have been generated within
      repetition_window tokens prior to the token to be generated.
    repetition_penalty_normalize: A bool indicating whether to normalize the
      logits (log_softmax) before applying the repetition penalty. This ensures
      that all logits have the same sign.
    max_decode_len: An int indicating the maximum sequence length.
    rng: A jax.random.PRNGKey.
    eos_token: An int token id. If not None, we stop decoding a sequence after
      the first instance.
    pad_token: An int token used to pad sequences after the eos token. If none,
      we set pad_token to eos_token.
    masked_tokens: A list of int token id. If not None, we mask these token
      before sampling.
    use_lax_while_loop: A bool; whether to use a lax while loop or python loop.

  Returns:
    An array of shape (batch_size, max_decode_len) containing sequences. If
      variable-length, the sequences are right-padded with the EOS token.
  """
  batch_size = prompt.shape[0]
  eos_token = eos_token if eos_token is not None else -1
  if pad_token is None:
    logging.warn('Pad token is not provided. Using the EOS token.')
  pad_token = pad_token if pad_token is not None else eos_token
  end_marker = jnp.array(eos_token)
  temperature = jnp.array(temperature)

  # Initialize sampling loop state.
  rng0 = rng if rng is not None else random.PRNGKey(0)
  i0 = jnp.array(0)
  ended0 = jnp.zeros((batch_size, 1)).astype(jnp.bool_)

  # Initialize sequences with the prompt followed by the out_of_prompt_marker to
  # indicate when we can start generation.
  out_of_prompt_marker = jnp.array(-2)
  pad = jnp.ones((batch_size, max_decode_len - prompt.shape[1]),
                 dtype=jnp.int32) * out_of_prompt_marker
  sequences0 = jnp.concatenate([prompt, pad], axis=1)
  token0 = sequences0[:, 0:1]

  # Sampling loop state is stored in a simple tuple.
  tokens_to_logits_state = None

  sampling_loop_init_state = (i0, sequences0, init_cache, token0, ended0, rng0,
                              tokens_to_logits_state)

  def sampling_loop_cond_fn(state):
    """Sampling loop termination condition."""
    (i, _, _, _, ended, _, _) = state
    # Have we reached max decoding length?
    not_at_end = (i <= max_decode_len)
    # Have all sampled sequences reached an end marker?
    all_sequences_ended = jnp.all(ended)
    return not_at_end & (~all_sequences_ended)

  def sampling_loop_body_fn(state):
    """Sampling loop state update."""
    i, sequences, cache, cur_token, ended, rng, tokens_to_logits_state = state

    # Split RNG for sampling.
    rng1, rng2 = random.split(rng)

    # Call fast-decoder model on current tokens to get raw next-position logits.
    logits, new_cache, new_tokens_to_logits_state = tokens_to_logits(
        cur_token, cache, internal_state=tokens_to_logits_state)
    logits = logits / temperature

    # Mask out the BOS token.
    if masked_tokens is not None:
      mask = common_utils.onehot(
          jnp.array(masked_tokens),
          num_classes=logits.shape[-1],
          on_value=LARGE_NEGATIVE)
      mask = jnp.sum(mask, axis=0)[None, :]  # Combine multiple masks together
      logits = logits + mask

    # Apply the repetition penalty.
    if repetition_penalty != 1:
      logits = apply_repetition_penalty(
          sequences, logits, i,
          repetition_penalty=repetition_penalty,
          repetition_window=repetition_window,
          repetition_penalty_normalize=repetition_penalty_normalize)

    # Mask out everything but the top-k entries.
    if top_k is not None:
      # Compute top_k_index and top_k_threshold with shapes (batch_size, 1).
      top_k_index = jnp.argsort(logits, axis=-1)[:, ::-1][:, top_k-1:top_k]
      top_k_threshold = jnp.take_along_axis(logits, top_k_index, axis=-1)
      logits = jnp.where(logits < top_k_threshold,
                         jnp.full_like(logits, LARGE_NEGATIVE),
                         logits)
    # Sample next token from logits.
    sample = multinomial(rng1, logits)
    next_token = sample.astype(jnp.int32)
    # Only use sampled tokens if we have past the out_of_prompt_marker.
    out_of_prompt = (sequences[:, i + 1] == out_of_prompt_marker)
    next_token = (next_token * out_of_prompt +
                  sequences[:, i + 1] * ~out_of_prompt)
    # If end-marker reached for batch item, only emit padding tokens.
    next_token = next_token[:, None]
    next_token_or_endpad = jnp.where(ended,
                                     jnp.full_like(next_token,
                                                   pad_token), next_token)
    ended |= (next_token_or_endpad == end_marker)
    # Add current sampled tokens to recorded sequences.
    new_sequences = lax.dynamic_update_slice(sequences, next_token_or_endpad,
                                             (0, i + 1))
    return (i + 1, new_sequences, new_cache, next_token_or_endpad, ended, rng2,
            new_tokens_to_logits_state)

  # Run sampling loop and collect final state.
  if use_lax_while_loop:
    final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                                 sampling_loop_init_state)
  else:
    final_state = sampling_loop_init_state
    while sampling_loop_cond_fn(final_state):
      final_state = sampling_loop_body_fn(final_state)

  final_sequences = final_state[1]
  # If generation ended early (all sampled sequences reached an end marker)
  # replace all remaining out_of_prompt_marker instances with the pad_token.
  final_sequences = jnp.where(final_sequences == out_of_prompt_marker,
                              jnp.full_like(final_sequences, pad_token),
                              final_sequences)
  return final_sequences
