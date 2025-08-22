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

"""Utilities for PaDIR training and inference."""

from typing import Optional, Union

import jax
import jax.numpy as jnp
import seqio

from padir.padir import config_options
from padir.padir.utils import vocab_utils


def token_lengths(token_ids):
  """Returns the number of unpadded tokens in each batch element.

  Args:
    token_ids: [B, L] int32 token ids.

  Returns:
    [B, 1] int32 array with lengths
  """
  return jnp.sum((token_ids > 0).astype(jnp.int32), axis=-1, keepdims=True)


def extend_ones(mask):
  """Extend a sequence of 1's by an additional 1.

    E.g. [[1, 1, 0, 0],[0, 0, 0, 0]] -> [[1, 1, 1, 0], [1, 0, 0, 0]]

  Args:
    mask: 2-d tensor of contiguous 1's followed by 0's
  Returns:
    2-d tensor with 1's extended by one 1.
  """
  return mask + jnp.equal(jnp.cumsum(1 - mask, axis=1), 1).astype(jnp.int32)


def eos_mask(tokens, eos_id = None):
  """Returns a mask of 1's until the first eos_id (inclusive) and 0's after."""
  if eos_id is None:
    eos_id = 1
  eq_eos = jnp.equal(tokens, eos_id).astype(jnp.int32)
  past_eos = jnp.cumsum(eq_eos, axis=-1)
  before_eos = 1 - (past_eos > 0).astype(jnp.int32)
  return extend_ones(before_eos)


def stutter_mask(tokens):
  """Returns mask indicating stutter tokens: the the -> 0 1."""
  return jnp.logical_and(
      jnp.equal(tokens, jnp.roll(tokens, 1, axis=-1)), tokens > 0
  )


def replace_id(
    tokens, mask, new_id
):
  """Replaces elements of tokens with new_id where indicated by mask."""
  mask = mask.astype(jnp.int32)
  return tokens * (1 - mask) + new_id * mask


def lowest_scores_mask(
    scores,
    padding_mask,
    num_to_mask,
):
  """Returns mask indicating num_to_mask lowest-scoring unpadded positions.

  Args:
    scores: [B, L] float32 score of the top prediction at each decoder position.
    padding_mask: [B, L] int32 mask indicating unpadded decoder positions.
    num_to_mask: int or [B]

  Returns:
    int32 with 1 in positions with the lowest scores, 0 otherwise.
  """
  assert scores.ndim == 2
  assert padding_mask.shape == scores.shape

  scores *= padding_mask
  scores += (1 - padding_mask) * 1e8

  if isinstance(num_to_mask, int):
    vals, _ = jax.lax.top_k(-scores, k=num_to_mask)
    thresholds = -vals[:, -1:]
  else:
    assert num_to_mask.ndim == 1
    asc_scores = jnp.sort(scores, axis=-1)
    # For simplicity, always mask at least one.
    idx_to_mask = jnp.clip(num_to_mask - 1, 0)
    thresholds = jax.vmap(lambda row, idx: row[idx])(asc_scores, idx_to_mask)
    thresholds = thresholds[:, jnp.newaxis]

  return jnp.less_equal(scores, thresholds).astype(jnp.int32) * padding_mask


def replace_rejected_predictions(
    approved_mask,
    predictions,
    replacements,
    ended = None,
    start_token = None,
    remask_stutter = False,
):
  """Replaces rejected predictions with values from `replacements`.

  Args:
    approved_mask: [B, L] with 1s in approved positions.
    predictions: [B, L] int32 input tokens
    replacements: [B, L] int32 replacement tokens
    ended: [B, 1] 1/0 indicating decoding ended or not.
    start_token: If not None, sets the first token to this.
    remask_stutter: Whether to replace stutter.

  Returns:
    [B, L] output tokens with rejected predictions updated.
  """
  assert approved_mask.ndim == 2
  assert predictions.shape == approved_mask.shape
  assert replacements.shape == approved_mask.shape
  if ended is not None:
    assert ended.ndim == 2

  # Note: predictions may possibly contain more than one EOS. We let the
  # approved_mask dictate which tokens, including EOS, should be preserved.
  predictions_mask = (predictions > 0).astype(jnp.int32)
  rejected_mask = jnp.logical_not(approved_mask) * predictions_mask
  if ended is not None:
    rejected_mask *= jnp.logical_not(ended).astype(jnp.int32)
  if remask_stutter:
    rejected_mask |= stutter_mask(predictions)
  predictions = jnp.where(rejected_mask, replacements, predictions)

  if start_token is not None:
    predictions = predictions.at[:, 0].set(start_token)
  return predictions


def scatter_2d(
    ids, indices, updates
):
  """Scatters updates into ids at indices."""

  def _scatter_one(row, index, update):
    return row.at[index].set(update)

  return jax.vmap(_scatter_one)(ids, indices, updates)


def add_beam_dim(
    x, beam_size, offset = 0
):
  """Creates new beam dimension in non-scalar array and tiles into it."""
  x = jnp.expand_dims(x, axis=offset + 1)
  tile_dims = [1] * x.ndim
  tile_dims[offset + 1] = beam_size
  return jnp.tile(x, tile_dims)


def flatten_beam_dim(x, offset = 0):
  """Flattens the first two dimensions of a non-scalar array."""
  xshape = list(x.shape)
  b_sz = xshape.pop(offset)
  xshape[offset] *= b_sz
  return x.reshape(xshape)


def unflatten_beam_dim(
    x, batch_size, beam_size, offset = 0
):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  assert batch_size * beam_size == x.shape[offset]
  xshape = list(x.shape)
  newshape = xshape[:offset] + [batch_size, beam_size] + xshape[offset + 1 :]
  return x.reshape(newshape)


def flat_batch_beam_expand(
    x, beam_size, offset = 0
):
  """Expands each batch item by beam_size in batch_dimension."""
  return flatten_beam_dim(add_beam_dim(x, beam_size, offset), offset)


def flat_batch_random_expand(
    x,
    beam_size,
    num_reserved_tokens,
    vocab_size,
    rng,
):
  """As flat_batch_beam_expand, but initializing with random token ids."""
  shape = list(x.shape)
  shape[0] *= beam_size
  _, subkey = jax.random.split(rng, 2)
  return jax.random.randint(
      subkey,
      shape=shape,
      minval=num_reserved_tokens,
      maxval=vocab_size,
      dtype=x.dtype,
  )


def pad_after_eos(x, eos_id):
  return x * eos_mask(x, eos_id)


def _restore_eos_from_pos(
    x,
    eos_pos,
    eos_id,
):
  """Restores EOS and PAD tokens in x given EOS positions (indices).

  Example:
  x = [
      [100,101,102,103,104],
      [110,111,112,113,114],
  ]
  eos_pos = [2, 3]

  _restore_eos_from_pos(x, eos_pos, eos_id=1) = [
      [100,101,1,0,0],
      [110,111,112,1,0],
  ]

  Args:
    x: [B, L] input tokens, typically fully random.
    eos_pos: [B] indices of EOS tokens.
    eos_id: EOS token id.

  Returns:
    x with EOS and PAD restored from eos_pos.
  """
  assert x.ndim == 2  # [B, L]
  assert eos_pos.ndim == 1  # [B]
  x = scatter_2d(
      x,
      eos_pos,
      eos_id * jnp.ones(x.shape[0], dtype=jnp.int32),
  )
  return pad_after_eos(x, eos_id)


def _restore_eos_uniform_length(
    expanded_x, x, parallel_decodes, eos_id
):
  """Restores EOS and PAD tokens from x inside expanded_x.

  Example:
  x = [
      [10,11,1,0,0],
      [20,21,22,23,1],
  ]
  expanded_x = [
      [100,101,102,103,104],
      [110,111,112,113,114],
      [200,201,202,203,204],
      [210,211,212,213,214],
  ]
  restore_eos_uniform_length(expanded_x, x, parallel_decodes==2, eos_id=1) = [
      # First row of 'x'.
      [100,101,1,0,0],
      [110,111,1,0,0],
      # Second row of 'x'.
      [200,201,202,203,1],
      [210,211,212,213,1],
  ]

  Args:
    expanded_x: [Batch*Parallel, L] input tokens, typically fully random.
    x: [Batch, L] input tokens with EOS and PAD.
    parallel_decodes: number of parallel decodes per row in 'x'.
    eos_id: EOS token id.

  Returns:
    expanded_x with EOS and PAD restored from x.
  """
  assert parallel_decodes == expanded_x.shape[0] // x.shape[0]
  eos_id_mask = flat_batch_beam_expand(
      jnp.equal(x, eos_id).astype(x.dtype), parallel_decodes
  )
  expanded_x = replace_id(expanded_x, eos_id_mask, eos_id)
  return pad_after_eos(expanded_x, eos_id)


def _restore_eos_beam_length(
    expanded_x, x, beam_length, eos_id = 1
):
  """Restores EOS and PAD tokens from x inside expanded_x, for beam length.

  Adjusts the EOS position using 'beam_length' length candidates,
  offsetting EOS by [-beam_length//2, ... -1, 0, 1, ..., beam_length//2].

  Note that EOS indices are clipped between 1 and L-1 (where L is the max
  sequence length), to ensure each row remains a valid decoder input.

  Example:
  x = [
      [10,11,1,0,0],
      [20,21,22,23,1],
  ]
  expanded_x = [
      [100,101,102,103,104],
      [110,111,112,113,114],
      [120,121,122,123,124],
      [200,201,202,203,204],
      [210,211,212,213,214],
      [220,221,222,223,224],
  ]
  restore_eos_beam_length(expanded_x, x, beam_length=3, eos_id=1) = [
      # First row of 'x'.
      [100,1,0,0,0],       # 1 token shorter than in 'x'.
      [110,111,1,0,0],     # EOS in same position as in 'x'.
      [120,121,122,1,0],   # 1 token longer than in 'x'.
      # Second row of 'x'.
      [200,201,202,1,0],   # 1 token shorter than in 'x'.
      [210,211,212,213,1], # EOS in same position as in 'x'.
      [220,221,222,223,1], # Clipped to last position.
  ]

  Args:
    expanded_x: [Batch*Beam, L] input tokens, typically fully random.
    x: [Batch, L] input tokens with EOS and PAD.
    beam_length: integer beam size. Must be an odd number.
    eos_id: EOS token id.

  Returns:
    expanded_x with EOS and PAD restored from x with beam length adjustments.
  """
  assert beam_length == expanded_x.shape[0] // x.shape[0]
  assert beam_length % 2 == 1
  unexpanded_batch_size, seq_len = x.shape
  half_beam = beam_length // 2

  # [expanded_batch_size]
  eos_pos = flat_batch_beam_expand(
      jnp.argwhere(x == eos_id, size=unexpanded_batch_size)[:, 1], beam_length
  )
  eos_offsets = jnp.tile(
      jnp.arange(-half_beam, half_beam + 1), reps=[unexpanded_batch_size]
  )
  shifted_eos_pos = jnp.clip(eos_pos + eos_offsets, min=1, max=seq_len - 1)

  return _restore_eos_from_pos(expanded_x, shifted_eos_pos, eos_id)


def restore_eos(
    expanded_x,
    eos_pos,
    beam_length,
    parallel_decodes,
    eos_id = 1,
):
  """Restores EOS and PAD tokens from x inside expanded_x.

  Handles top_k_lengths, beam_length and parallel decodes, each possibly 1.

  Args:
    expanded_x: [B, L] input tokens, typically fully random.
    eos_pos: [B, top_k_lengths] indices of EOS tokens.
    beam_length: integer beam size. Must be an odd number.
    parallel_decodes: number of parallel decodes per candidate length.
    eos_id: EOS token id.

  Returns:
    expanded_x with EOS and PAD restored.
  """
  assert eos_pos.ndim == 2
  unexpanded_batch_size, top_k_lengths = eos_pos.shape
  expanded_batch_size = expanded_x.shape[0]
  num_decodes = top_k_lengths * beam_length * parallel_decodes
  assert num_decodes == expanded_batch_size // unexpanded_batch_size

  eos_pos = jnp.ravel(eos_pos)
  top_k_batch_size = eos_pos.shape[0]
  top_k_x = expanded_x[:top_k_batch_size, :]
  top_k_x = _restore_eos_from_pos(top_k_x, eos_pos, eos_id)

  beam_batch_size = top_k_batch_size * beam_length
  beam_x = expanded_x[:beam_batch_size, :]
  beam_x = _restore_eos_beam_length(
      beam_x,
      top_k_x,
      beam_length,
      eos_id,
  )

  return _restore_eos_uniform_length(
      expanded_x, beam_x, parallel_decodes, eos_id
  )


def initialize_decoder_input(
    decoder_shape,
    decoder_input_scheme,
    vocab,
    key = None,
):
  """Creates the decoder input tensor for the first inference step."""

  if decoder_input_scheme == config_options.DecoderInputScheme.RANDOM:
    num_reserved_tokens = 3
    return jax.random.randint(
        key,
        shape=decoder_shape,
        minval=num_reserved_tokens,
        maxval=vocab.vocab_size,
        dtype=jnp.int32,
    )
  elif decoder_input_scheme == config_options.DecoderInputScheme.MASK_RANGE:
    batch_size, max_target_len = decoder_shape
    start_id, end_id = vocab_utils.get_mask_id_range(vocab, max_target_len)
    assert max_target_len == end_id - start_id
    decoder_input = jnp.arange(start_id, end_id, dtype=jnp.int32)[
        jnp.newaxis, :
    ]
    return jnp.repeat(decoder_input, batch_size, axis=0)
  elif decoder_input_scheme == config_options.DecoderInputScheme.MASK:
    return vocab_utils.get_mask_id(vocab) * jnp.ones(
        decoder_shape, dtype=jnp.int32
    )
  else:
    assert False, 'Unsupported decoder input scheme'
