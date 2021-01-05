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

"""Fast decoding routines for inference from a trained model."""

from typing import Callable, Mapping, Optional, Tuple
import flax
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

PyTreeDef = type(jax.tree_structure(None))
SamplingLoopState = Tuple[int, jnp.ndarray, Mapping[str, jnp.ndarray],
                          jnp.ndarray, jnp.ndarray, jnp.ndarray]

# Constants
# We assume the default End-of-Sentence token id is 1 (T5 convention).
EOS_ID = 1
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = np.array(-1.0e7)

#------------------------------------------------------------------------------
# Temperature Sampling
#------------------------------------------------------------------------------


def temperature_sample(prompt_inputs,
                       init_cache,
                       tokens_to_logits,
                       prng_key,
                       temperature = 1.0,
                       topk = 20,
                       eos_id = EOS_ID):
  """Temperature sampling for language model generation.

  Args:
    prompt_inputs: array: [batch_size, max_decode_len] int32 sequence of tokens.
    init_cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    prng_key: JAX PRNGKey.
    temperature: float: sampling temperature factor. As it approaches zero this
      becomes equivalent to greedy sampling.
    topk: integer: if nonzero only use the top-k logits to sample next token, if
      zero don't use any cutoff and sample from full logits over vocabulary.
    eos_id: int: end-of-sentence token for target vocabulary.

  Returns:
    Array of sampled sequences: [batch_size, max_decode_len]
  """
  batch_size = prompt_inputs.shape[0]
  max_decode_len = prompt_inputs.shape[1]
  end_marker = jnp.array(eos_id)
  temperature = jnp.array(temperature)

  # Initialize sampling loop state.
  # initial loop PRNGKey
  rng0 = prng_key
  # loop position counter.
  i0 = jnp.array(0)
  # per batch-item holding current token in loop.
  token0 = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  # per batch-item state bit indicating if sentence has finished.
  ended0 = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
  # (batch, length) array containing prefix prompt tokens for sampling loop
  # as well as the generated output of newly sampled tokens.
  sequences0 = prompt_inputs
  # Sampling loop state is stored in a simple tuple.
  sampling_loop_init_state = (i0, sequences0, init_cache, token0, ended0, rng0)

  def sampling_loop_cond_fn(state):
    """Sampling loop termination condition."""
    (i, _, _, _, ended, _) = state
    # Have we reached max decoding length?
    not_at_end = (i <= max_decode_len)
    # Have all sampled sequences reached an end marker?
    all_sequences_ended = jnp.all(ended)
    return not_at_end & (~all_sequences_ended)

  def sampling_loop_body_fn(state):
    """Sampling loop state update."""
    i, sequences, cache, cur_token, ended, rng = state
    # Split RNG for sampling.
    rng1, rng2 = random.split(rng)
    # Call fast-decoder model on current tokens to get next-position logits.
    logits, new_cache = tokens_to_logits(cur_token, cache)
    # Sample next token from logits.
    # TODO(levskaya): add top-p "nucleus" sampling option.
    if topk:
      # Get top-k logits and their indices, sample within these top-k tokens.
      topk_logits, topk_idxs = lax.top_k(logits, topk)
      topk_token = jnp.expand_dims(
          random.categorical(rng1, topk_logits / temperature).astype(jnp.int32),
          axis=-1)
      # Return the original indices corresponding to the sampled top-k tokens.
      next_token = jnp.squeeze(
          jnp.take_along_axis(topk_idxs, topk_token, axis=-1), axis=-1)
    else:
      next_token = random.categorical(rng1,
                                      logits / temperature).astype(jnp.int32)
    # Only use sampled tokens if we're past provided prefix tokens.
    out_of_prompt = (sequences[:, i + 1] == 0)
    next_token = (
        next_token * out_of_prompt + sequences[:, i + 1] * ~out_of_prompt)
    # If end-marker reached for batch item, only emit padding tokens.
    next_token_or_endpad = next_token * ~ended
    ended |= (next_token_or_endpad == end_marker)
    # Add current sampled tokens to recorded sequences.
    new_sequences = lax.dynamic_update_slice(sequences, next_token_or_endpad,
                                             (0, i + 1))
    return (i + 1, new_sequences, new_cache, next_token_or_endpad, ended, rng2)

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                               sampling_loop_init_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state[1]
  return final_sequences


#------------------------------------------------------------------------------
# BEAM Sampling
#------------------------------------------------------------------------------


def brevity_penalty(alpha, length):
  """Brevity penalty function for beam search penalizing short sequences.

  Args:
    alpha: float: brevity-penalty scaling parameter.
    length: int: length of considered sequence.

  Returns:
    Brevity penalty score as jax scalar.
  """
  return jnp.power(((5.0 + length) / 6.0), alpha)


# Beam handling utility functions:


def add_beam_dim(x, beam_size):
  """Creates new beam dimension in non-scalar array and tiles into it."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  x = jnp.expand_dims(x, axis=1)
  tile_dims = [1] * x.ndim
  tile_dims[1] = beam_size
  return jnp.tile(x, tile_dims)


def flatten_beam_dim(x):
  """Flattens the first two dimensions of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unflatten_beam_dim(x, batch_size,
                       beam_size):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  assert batch_size * beam_size == x.shape[0]
  return x.reshape((batch_size, beam_size) + x.shape[1:])


def flat_batch_beam_expand(x, beam_size):
  """Expands the each batch item by beam_size in batch_dimension."""
  return flatten_beam_dim(add_beam_dim(x, beam_size))


def gather_beams(nested, beam_indices, batch_size,
                 new_beam_size):
  """Gathers the beam slices indexed by beam_indices into new beam array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    beam_indices: array of beam_indices
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ beam dimension.

  Returns:
    New pytree with new beam arrays.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  batch_indices = jnp.reshape(
      jnp.arange(batch_size * new_beam_size) // new_beam_size,
      (batch_size, new_beam_size))

  def gather_fn(x):
    if x.ndim == 0:  # ignore scalars (e.g. cache index)
      return x
    else:
      return x[batch_indices, beam_indices]

  return jax.tree_map(gather_fn, nested)


def gather_topk_beams(nested, score_or_log_prob,
                      batch_size, new_beam_size):
  """Gathers the top-k beam slices given by score_or_log_prob array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    score_or_log_prob: [batch_size, old_beam_size] array of values to sort by
      for top-k selection of beam slices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ top-k selected beam dimension

  Returns:
    New pytree with new beam arrays containing top k new_beam_size slices.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  _, topk_indices = lax.top_k(score_or_log_prob, k=new_beam_size)
  topk_indices = jnp.flip(topk_indices, axis=1)
  return gather_beams(nested, topk_indices, batch_size, new_beam_size)


# Beam search state:


@flax.struct.dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.DeviceArray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: jnp.DeviceArray  # float32: [batch_size, beam_size]
  finished_scores: jnp.DeviceArray  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size, max_decode_len]
  finished_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jnp.DeviceArray  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  cache: PyTreeDef  # Any pytree of arrays, e.g. flax attention Cache object


def beam_init(batch_size, beam_size, max_decode_len,
              cache):
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (beam_size - 1)), [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * NEG_INF
  live_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  beam_cache0 = jax.tree_map(lambda x: add_beam_dim(x, beam_size), cache)
  return BeamState(
      cur_index=cur_index0,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      cache=beam_cache0)


# Beam search routine:


def beam_search(
    inputs,
    cache,
    tokens_to_logits,
    beam_size = 4,
    alpha = 0.6,
    eos_id = EOS_ID,
    max_decode_len = None):
  """Beam search for transformer machine translation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    beam_size: int: number of beams to use in beam search.
    alpha: float: scaling factor for brevity penalty.
    eos_id: int: id of end-of-sentence token for target vocabulary.
    max_decode_len: int: maximum length of decoded translations.

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  # We liberally annotate shape information for clarity below.

  batch_size = inputs.shape[0]
  if max_decode_len is None:
    max_decode_len = inputs.shape[1]
  end_marker = jnp.array(eos_id)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size, beam_size, max_decode_len,
                                     cache)

  def beam_search_loop_cond_fn(state):
    """Beam search loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = (state.cur_index < max_decode_len - 1)

    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_decode_len)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores = jnp.min(
        state.finished_scores, axis=1, keepdims=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = jnp.where(state.finished_flags,
                                      worst_finished_scores, NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = jnp.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)

  def beam_search_loop_body_fn(state):
    """Beam search loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the beam dimension into batch
    # dimension for feeding into the model.
    # --> [batch * beam, 1]
    flat_ids = flatten_beam_dim(
        lax.dynamic_slice(state.live_seqs, (0, 0, state.cur_index),
                          (batch_size, beam_size, 1)))
    # Flatten beam dimension into batch to be compatible with model.
    # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
    flat_cache = jax.tree_map(flatten_beam_dim, state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * beam, vocab]
    flat_logits, new_flat_cache = tokens_to_logits(flat_ids, flat_cache)

    # unflatten beam dimension
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    # Unflatten beam dimension in attention cache arrays
    # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
    new_cache = jax.tree_map(
        lambda x: unflatten_beam_dim(x, batch_size, beam_size), new_flat_cache)

    # Gather log probabilities from logits
    candidate_log_probs = jax.nn.log_softmax(logits)
    # Add new logprobs to existing prefix logprobs.
    # --> [batch, beam, vocab]
    log_probs = (
        candidate_log_probs + jnp.expand_dims(state.live_logprobs, axis=2))

    # We'll need the vocab size, gather it from the log probability dimension.
    vocab_size = log_probs.shape[2]

    # Each item in batch has beam_size * vocab_size candidate sequences.
    # For each item, get the top 2*k candidates with the highest log-
    # probabilities. We gather the top 2*K beams here so that even if the best
    # K sequences reach EOS simultaneously, we have another K sequences
    # remaining to continue the live beam search.
    beams_to_keep = 2 * beam_size
    # Flatten beam and vocab dimensions.
    flat_log_probs = log_probs.reshape((batch_size, beam_size * vocab_size))
    # Gather the top 2*K scores from _all_ beams.
    # --> [batch, 2*beams], [batch, 2*beams]
    topk_log_probs, topk_indices = lax.top_k(flat_log_probs, k=beams_to_keep)
    # Recover the beam index by floor division.
    topk_beam_indices = topk_indices // vocab_size
    # Gather 2*k top beams.
    # --> [batch, 2*beams, length]
    topk_seq = gather_beams(state.live_seqs, topk_beam_indices, batch_size,
                            beams_to_keep)

    # Append the most probable 2*K token IDs to the top 2*K sequences
    # Recover token id by modulo division and expand Id array for broadcasting.
    # --> [batch, 2*beams, 1]
    topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(topk_seq, topk_ids,
                                        (0, 0, state.cur_index + 1))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index + 1] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # Determine the top k beam indices (from top 2*k beams) from log probs.
    # --> [batch, beams]
    _, new_topk_indices = lax.top_k(new_log_probs, k=beam_size)
    new_topk_indices = jnp.flip(new_topk_indices, axis=1)
    # Gather the top k beams (from top 2*k beams).
    # --> [batch, beams, length], [batch, beams]
    top_alive_seq, top_alive_log_probs = gather_beams([topk_seq, new_log_probs],
                                                      new_topk_indices,
                                                      batch_size, beam_size)

    # Determine the top k beam indices from the original set of all beams.
    # --> [batch, beams]
    top_alive_indices = gather_beams(topk_beam_indices, new_topk_indices,
                                     batch_size, beam_size)
    # With these, gather the top k beam-associated caches.
    # --> {[batch, beams, ...], ...}
    top_alive_cache = gather_beams(new_cache, top_alive_indices, batch_size,
                                   beam_size)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq],
        axis=1)
    finished_scores = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], axis=1)
    finished_flags = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], axis=1)
    # --> [batch, beams, length], [batch, beams], [batch, beams]
    top_finished_seq, top_finished_scores, top_finished_flags = (
        gather_topk_beams([finished_seqs, finished_scores, finished_flags],
                          finished_scores, batch_size, beam_size))

    return BeamState(
        cur_index=state.cur_index + 1,
        live_logprobs=top_alive_log_probs,
        finished_scores=top_finished_scores,
        live_seqs=top_alive_seq,
        finished_seqs=top_finished_seq,
        finished_flags=top_finished_flags,
        cache=top_alive_cache)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn, \
                               beam_search_loop_body_fn, \
                               beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(none_finished[:, None, None], \
                            final_state.finished_seqs, \
                            final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(none_finished[:, None], \
                              final_state.finished_scores, \
                              final_state.live_logprobs)

  return finished_seqs, finished_scores
