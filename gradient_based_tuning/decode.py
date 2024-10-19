# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

# Copyright 2022 The Google Research Authors.
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
"""Fast decoding routines for inference from a model or ensemble of models."""

import functools
from typing import Any

import flax
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# Constants
# The default End-of-Sentence token id is 1.
_EOS_ID = 1
# "Effective negative infinity" constant for masking in beam search.
_NEG_INF = np.array(-1.0e7)


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


def unflatten_beam_dim(x, batch_size, beam_size):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  assert batch_size * beam_size == x.shape[0]
  return x.reshape((batch_size, beam_size) + x.shape[1:])


def flat_batch_beam_expand(x, beam_size):
  """Expands the each batch item by beam_size in batch_dimension."""
  return flatten_beam_dim(add_beam_dim(x, beam_size))


def gather_beams(nested, beam_indices, batch_size, new_beam_size):
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

  return jax.tree.map(gather_fn, nested)


def gather_topk_beams(nested, score_or_log_prob, batch_size, new_beam_size):
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
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: jax.Array  # float32: [batch_size, beam_size]
  finished_scores: jax.Array  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: jax.Array  # int32: [batch_size, beam_size, max_decode_len]
  finished_seqs: jax.Array  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jax.Array  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  cache: Any  # Any pytree of arrays, e.g. flax attention Cache object


def beam_init(batch_size, beam_size, max_decode_len, cache):
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [_NEG_INF] * (beam_size - 1)), [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * _NEG_INF
  live_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  if isinstance(cache, list):
    beam_cache0 = [
        jax.tree.map(lambda x: add_beam_dim(x, beam_size), c) for c in cache
    ]
  else:
    beam_cache0 = jax.tree.map(lambda x: add_beam_dim(x, beam_size), cache)
  return BeamState(
      cur_index=cur_index0,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      cache=beam_cache0)


# Beam search routine for ensemble of models:
def beam_search_ens(inputs,
                    models,
                    caches,
                    tokens_to_logits,
                    ens_type,
                    beam_size=4,
                    alpha=0.6,
                    eos_id=_EOS_ID,
                    max_decode_len=None):
  """Beam search for transformer machine translation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    models: List[model]
    caches: List[flax attention cache]
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    ens_type: str: one of {'arithmetic', 'geometric'}, determines the kind of
      averaging applied to the logits
    beam_size: int: number of beams to use in beam search.
    alpha: float: scaling factor for brevity penalty.
    eos_id: int: if of end-of-sentence token for target vocabulary.
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
                                     caches)

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
                                      worst_finished_scores, _NEG_INF)
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

    new_caches = []
    log_probs_list = []
    for i, model in enumerate(models):
      # Flatten beam dimension into batch to be compatible with model.
      # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
      flat_cache = jax.tree.map(flatten_beam_dim, state.cache[i])

      # Call fast-decoder model on current tokens to get next-position logits.
      # --> [batch * beam, vocab]
      flat_logits, new_flat_cache = tokens_to_logits(model, flat_ids,
                                                     flat_cache, i)
      # unflatten beam dimension
      # [batch * beam, vocab] --> [batch, beam, vocab]
      current_logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
      # Gather log probabilities from logits
      candidate_log_probs = jax.nn.log_softmax(current_logits)
      log_probs_list.append(candidate_log_probs)
      # Unflatten beam dimension in attention cache arrays
      # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
      new_caches.append(
          jax.tree.map(lambda x: unflatten_beam_dim(x, batch_size, beam_size),
                       new_flat_cache))

    if len(models) > 1:
      if ens_type == 'arithmetic':
        candidate_log_probs = functools.reduce(lambda x, y: x + y,
                                               log_probs_list)
        candidate_log_probs /= len(models)
      elif ens_type == 'geometric':
        candidate_log_probs = functools.reduce(lambda x, y: x * y,
                                               log_probs_list)
        candidate_log_probs = -1 * (
            jnp.abs(candidate_log_probs)**(1 / len(models)))
      else:
        raise ValueError('Unrecognized ens_type: %s' % ens_type)
    else:
      candidate_log_probs = log_probs_list[0]

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
    new_log_probs = topk_log_probs + newly_finished * _NEG_INF
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
    top_alive_caches = []
    for i in range(len(new_caches)):
      top_alive_caches.append(
          gather_beams(new_caches[i], top_alive_indices, batch_size, beam_size))

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * _NEG_INF

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
        cache=top_alive_caches)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn,
                               beam_search_loop_body_fn, beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(none_finished[:, None, None],
                            final_state.finished_seqs, final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(none_finished[:,
                                            None], final_state.finished_scores,
                              final_state.live_logprobs)

  return finished_seqs, finished_scores
