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

"""Functions used to manipulate alignments and smith-waterman parameters."""

from typing import Sequence, Tuple, Union

import tensorflow as tf

# Type aliases
PackedSWParams = tf.Tensor
UnpackedSWParams = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
SWParams = Union[PackedSWParams, UnpackedSWParams]


# SW dynamic program edge types, grouped by associated edge weight kind. See
MATCH_STATES = [0, 1, 2, 3]
GAP_OPEN_STATES = [4, 6, 7]
GAP_EXTEND_STATES = [5, 8]
STATES = {
    'match': MATCH_STATES,
    'gap_open': GAP_OPEN_STATES,
    'gap_extend': GAP_EXTEND_STATES,
}


def large_compatible_positive(tensor_type):
  """Large positive number as Tensor.

  This function is necessary because the standard value for "inf" in this module
  (1e9) cannot be represented using tf.float16.

  NOTE(fllinares): Borrowed from
  tensorflow/python/keras/layers/advanced_activations.py
  which is used already in this codebase indirectly (e.g. in self-attention
  layers).

  Args:
    tensor_type: a dtype to determine the type.

  Returns:
    A large positive number.
  """
  if tensor_type == tf.dtypes.float16:
    return tf.dtypes.float16.max
  return tf.convert_to_tensor(1e9, dtype=tensor_type)


def top_pad(t, v):
  """Pads tf.Tensor `t` by prepending `v` along the leading dimension."""
  return tf.pad(t, [[1, 0], [0, 0], [0, 0]], constant_values=v)


def left_pad(t, v):
  """Pads tf.Tensor `t` by prepending `v` along the second leading dimension."""
  return tf.pad(t, [[0, 0], [1, 0], [0, 0]], constant_values=v)


def right_pad(t, v):
  """Pads tf.Tensor `t` by appending `v` along the second leading dimension."""
  return tf.pad(t, [[0, 0], [0, 1], [0, 0]], constant_values=v)


def alignments_to_paths(
    alignments, len_x, len_y):
  """Converts sparse representation of alignments into dense paths tensor.

  Args:
    alignments: A tf.Tensor<int>[batch, 3, align_len] = tf.stack([pos_x, pos_y,
      enc_trans], 1) such that
        (pos_x[b][i], pos_y[b][i], enc_trans[b][i]) represents the i-th
      transition in the alignment for the b-th sequence pair in the minibatch.
      Both pos_x and pos_y are assumed to use one-based indexing and enc_trans
      follows the (categorical) 9-state encoding of edge types used throughout
      alignment/smith_waterman.py.
    len_x: The (padded) length of "X"/"query" sequences in the minibatch.
    len_y: The (padded) length of "Y"/"subject" sequences in the minibatch.

  Returns:
    A tf.Tensor of type tf.float32 and shape (batch_size, len_x, len_y, 9) with
    binary entries, representing the trajectory of the indices along the
    alignment path, by having a one along the taken edges, with nine possible
    edges for each i,j.
  """
  batch_size = tf.shape(alignments)[0]
  align_len = tf.shape(alignments)[-1]

  # Tensor with the same shape as pos_x, pos_y and enc_trans such that
  #   seq_indicators[b][l] = b for all l in [0, align_len).
  seq_indicators = tf.multiply(tf.expand_dims(tf.range(batch_size), -1),
                               tf.ones((1, align_len), dtype=tf.int32))
  # Prepares inputs to scatter_nd.
  indices = tf.concat([seq_indicators[Ellipsis, None],
                       tf.transpose(alignments, (0, 2, 1))], -1)
  indices = tf.reshape(indices, (-1, 4))
  updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32)
  shape = (batch_size, len_x + 1, len_y + 1, 9)

  # Note(fllinares): this is a (fairly ugly) hack to deal with padding.
  #  - pos_x, pos_y must use one-based indexing instead of zero-based indexing.
  #  - we use the (b, 0, 0, 0) entries of paths as "padding dumps".
  #  - the resulting tensor will be sliced to remove these starting row/col.
  paths = tf.scatter_nd(indices, updates, shape)
  return paths[:, 1:, 1:, :]


def alignments_to_state_indices(
    alignments,
    states,
    zero_based_idx = True,
):
  """Retrieves indices of MATCH/GAP OPEN/GAP EXTEND states in alignments.

  Args:
    alignments: A tf.Tensor<int>[batch, 3, align_len] = tf.stack([pos_x, pos_y,
      enc_trans], 1) such that
        (pos_x[b][i], pos_y[b][i], enc_trans[b][i]) represents the i-th
      transition in the alignment for the b-th sequence pair in the minibatch.
      Both pos_x and pos_y are assumed to use one-based indexing and enc_trans
      follows the (categorical) 9-state encoding of edge types used throughout
      alignment/smith_waterman.py.
    states: A Python list of integers in [0, 9), representing an arbitrary
      subset of (encoded) edge types. Can also be set to 'match', 'gap_open' or
      'gap_extend' to query the set of edge types associated with each of those
      conditions.
    zero_based_idx: Whether to use zero-based (True) or one-based (False)
      indexing for the function's output. Note that, however, alignment must use
      one-based indexing regardless of the value of this argument.

  Returns:
    A tf.Tensor `state_indices` of type tf.int32 and shape (n_entries, 3) such
    that, for a tf.Tensor `sim_mat` of shape (batch_size, len_x, len_y),
      tf.gather_nd(sim_mat, state_indices)
    returns the set of entries in `sim_mat` along the alignments described by
    `alignment` that correspond to one of the states in `states`.

    Note(fllinares): this function aims to provide a way to avoid materializing
    weights in the crf_loss function in alignment/smith_waterman.py, as
    suggested by @mblondel. Some extra care might be needed to keep per-example
    losses, as tf.gather_nd will flatten the output by default. For
    position-independent gap penalties, only the total number of entries per
    example in state_indices would be needed. See `score_from_alignment` below
    for extra details.
  """
  pos_x, pos_y, enc_trans = alignments[:, 0], alignments[:, 1], alignments[:, 2]
  states = STATES.get(states, states)

  # Note(fllinares): another ugly "hack", here we assume one-based idx to encode
  # the padding mask implicitly.
  padding_mask = tf.logical_and(pos_x > 0, pos_y > 0)
  hits = enc_trans == states[0]
  for state in states[1:]:
    hits = tf.logical_or(hits, enc_trans == state)
  hits = tf.logical_and(hits, padding_mask)
  indices = tf.cast(tf.where(hits), tf.int32)

  batch_indices = indices[:, 0]
  x_indices = tf.gather_nd(pos_x, indices) - int(zero_based_idx)
  y_indices = tf.gather_nd(pos_y, indices) - int(zero_based_idx)
  state_indices = tf.stack([batch_indices, x_indices, y_indices], axis=0)
  return tf.transpose(state_indices, (1, 0))


def paths_to_state_indicators(
    paths,
    states,
):
  """Computes (batch_size, len_x, len_y) tensor of binary state indicators.

  Args:
    paths: A tf.Tensor of type tf.float32 and shape (batch_size, len_x, len_y,
      9) with binary entries, representing the trajectory of the indices along
      the alignment path, by having a one along the taken edges, with nine
      possible edges for each i,j.
    states: A Python list of integers in [0, 9), representing an arbitrary
      subset of (encoded) edge types. Can also be set to 'match', 'gap_open' or
      'gap_extend' to query the set of edge types associated with each of those
      conditions.

  Returns:
    A tf.Tensor `state_indicators` of type tf.float32 and shape (batch_size,
    len_x, len_y) with binary entries such that
      state_indicators[b][i][j] = 1.0
    iff the trajectory of the alignment for the b-th sequence pair passes by
    character pair (i, j) under one of the states in `states`.
  """
  states = STATES.get(states, states)
  return tf.reduce_max(tf.gather(paths, indices=states, axis=-1), axis=-1)


def sw_score_from_alignments(
    sw_params,
    alignments,
):
  """Computes SW score of `alignments` for DP parameterized by `sw_params`.

  Args:
    sw_params: The parameters (sim_mat, gap_open, gap_extend) for the dynamic
      program underlying the Smith-Waterman algorithm.
      These can be input either as a tuple of tf.Tensor objects or as a single
      "packed" tensor of rank 4. See class `SWParamsFromEmbeddings` in module
      `sw_params_from_embeddings.py` for additional details.
    alignments: A tf.Tensor<int>[batch, 3, align_len] = tf.stack([pos_x, pos_y,
      enc_trans], 1) such that
        (pos_x[b][i], pos_y[b][i], enc_trans[b][i]) represents the i-th
      transition in the alignment for the b-th sequence pair in the minibatch.
      Both pos_x and pos_y are assumed to use one-based indexing and enc_trans
      follows the (categorical) 9-state encoding of edge types used throughout
      alignment/smith_waterman.py.

  Returns:
    A tf.Tensor of type tf.float32 and shape (batch_size,) containing the SW
    score of each alignment in the batch.
  """
  # Ensures SW params are in "unpacked" format.
  if isinstance(sw_params, Sequence):  # _UnpackedSWParams format
    sim_mat, gap_open, gap_extend = sw_params
    gap_open, gap_extend = -gap_open, -gap_extend
  else:  # _PackedSWParams format
    sim_mat = sw_params[Ellipsis, MATCH_STATES[0]]
    gap_open = sw_params[Ellipsis, GAP_OPEN_STATES[0]]
    gap_extend = sw_params[Ellipsis, GAP_EXTEND_STATES[0]]

  batch_size = tf.shape(sim_mat)[0]  # Assumed consistent with gap_open/extend.

  def dot_by_states(t, states):
    """Sums entries of t along alignments for queried states."""

    def pos_dep_dot(t, states):
      """Sums entries of t[b,l1,l2] along alignments for queried states."""
      state_indices = alignments_to_state_indices(alignments, states)
      batch_indices = state_indices[:, 0]
      state_entries_along_path = tf.gather_nd(t, state_indices)
      total_per_example = tf.math.unsorted_segment_sum(
          state_entries_along_path, batch_indices, batch_size)
      return total_per_example

    def pos_indep_dot(t, states):
      """Sums entries of t[b] along alignments for queried states."""
      state_indices = alignments_to_state_indices(alignments, states)
      batch_indices = state_indices[:, 0]
      # Note(fllinares): tf.math.bincount unsupported in TPU :(
      n_state_entries_along_path = tf.math.unsorted_segment_sum(
          tf.ones_like(batch_indices, tf.float32), batch_indices, batch_size)
      total_per_example = t * n_state_entries_along_path
      return total_per_example

    return (pos_dep_dot(t, states) if t.shape.rank == 3
            else pos_indep_dot(t, states))

  sim_per_example = dot_by_states(sim_mat, 'match')
  gap_open_per_example = dot_by_states(gap_open, 'gap_open')
  gap_extend_per_example = dot_by_states(gap_extend, 'gap_extend')

  return sim_per_example + gap_open_per_example + gap_extend_per_example


def sw_score_from_paths(sw_params, paths):
  """Computes SW score of `paths` for DP parameterized by `sw_params`.

  Args:
    sw_params: The parameters (sim_mat, gap_open, gap_extend) for the dynamic
      program underlying the Smith-Waterman algorithm.
      These can be input either as a tuple of tf.Tensor objects or as a single
      "packed" tensor of rank 4. See class `SWParamsFromEmbeddings` in module
      `sw_params_from_embeddings.py` for additional details.
    paths: A tf.Tensor of type tf.float32 and shape (batch_size, len_x, len_y,
      9) with binary entries, representing the trajectory of the indices along
      the alignment path, by having a one along the taken edges, with nine
      possible edges for each i,j.

  Returns:
    A tf.Tensor of type tf.float32 and shape (batch_size,) containing the SW
    score of each alignment in the batch.
  """
  if isinstance(sw_params, Sequence):  # _UnpackedSWParams format
    sw_params = weights_from_sim_mat(*sw_params)
  return tf.reduce_sum(sw_params * paths, axis=[1, 2, 3])


def sw_score(
    sw_params,
    alignments_or_paths,
):
  """Wraps over sw_score_from_paths and sw_score_from_alignments."""
  if alignments_or_paths.shape.rank == 3:  # Sparse format
    return sw_score_from_alignments(sw_params, alignments_or_paths)
  else:  # tf.Tensor format
    return sw_score_from_paths(sw_params, alignments_or_paths)


def mask_from_similarities(sim_mat,
                           dtype = tf.float32,
                           pad_penalty = 1e8):
  """Recovers padding / special token mask from a similarities tensor.

  Args:
    sim_mat: A tf.Tensor<float>[batch, len, len] of pairwise similarities. It is
      assumed that entries corresponding to padding / special tokens have been
      masked by being set to have magnitude greater than pad_penalty.
    dtype: The desired dtype for the output mask.
    pad_penalty: The magnitude above which entries are considered to have been
      masked.

  Returns:
    A tf.Tensor<dtype>[batch, len, len] with binary entries, with 1.0 signifying
    "real" tokens and 0.0 padding / special tokens.
  """
  mask = tf.logical_and(sim_mat > -pad_penalty, sim_mat < pad_penalty)
  return tf.cast(mask, dtype)


def _broadcast_to_rank(t, rank, axis = -1):
  """Appends dimensions to tf.Tensor `t` at axis `axis` to match rank `rank`."""
  rank_t = t.shape.rank  # Assumes ranks are known at compile time (static).
  for _ in range(rank - rank_t):
    t = tf.expand_dims(t, axis=axis)
  return t


def _broadcast_to_shape(
    t,
    shape
    ):
  """Appends dimensions to and tiles tf.Tensor t to match desired shape."""
  rank = len(shape)
  t = _broadcast_to_rank(t, rank, axis=-1)
  return tf.tile(t, shape // tf.shape(t))


def weights_from_sim_mat(
    sim_mat,
    gap_open,
    gap_extend,
):
  """Computes the edge weights for the Smith-Waterman LP.

  Args:
    sim_mat: a tf.Tensor<float>[batch, len1, len2] with the substitution values
      for pairs of sequences.
    gap_open: a tf.Tensor<float>[batch, len1, len2] or tf.Tensor<float>[batch]
      of penalties for opening a gap.
    gap_extend: a tf.Tensor<float>[batch, len1, len2] or tf.Tensor<float>[batch]
      of penalties for extending a gap.

  Returns:
    A single tf.Tensor<float>[batch, len1, len2, 9] of edge weights for nine
    edge types. These correspond to a (strict) subset of allowed (from, to)
    state transitions between four state types, namely, start, match, gap_in_x
    and gap_in_y. Along the last dimension:
    + The first four (0:4) indices form a tf.Tensor<float>[batch, len1, len2, 4]
      of weights for all edges leading into match states. That is, these
      represent transitions (start, match), (match, match), (gap_in_x, match)
      and (gap_in_y, match), respectively.
    + The next two (4:6) indices form a tf.Tensor<float>[batch, len1, len2, 2]
      of weights for all edges leading into gap_in_x states. These represent
      transitions (match, gap_in_x) and (gap_in_x, gap_in_x), respectively. Note
      that, by convention, (gap_in_y, gap_in_x) transitions are disallowed.
    + The last three (6:9) indices form a tf.Tensor<float>[batch, len1, len2, 3]
      of weights for all edges leading into gap_in_y states. These represent
      transitions (match, gap_in_y) and (gap_in_x, gap_in_y) and, finally,
      (gap_in_y, gap_in_y), respectively.
  """
  l1, l2 = sim_mat.shape[1:3]

  sim_mat = sim_mat[Ellipsis, None]
  sim_mat = tf.tile(sim_mat, [1, 1, 1, 4])
  if gap_open.shape.rank == 3:
    gap_open = gap_open[Ellipsis, None]
    gap_extend = gap_extend[Ellipsis, None]
  else:
    gap_open = gap_open[Ellipsis, None, None, None]
    gap_open = tf.tile(gap_open, [1, l1, l2, 1])
    gap_extend = gap_extend[Ellipsis, None, None, None]
    gap_extend = tf.tile(gap_extend, [1, l1, l2, 1])

  weights_m = sim_mat
  weights_x = tf.concat([-gap_open, -gap_extend], axis=-1)
  weights_y = tf.concat([-gap_open, weights_x], axis=-1)

  return tf.concat([weights_m, weights_x, weights_y], axis=-1)


def adjoint_weights_from_sim_mat(
    weights,
    gap_open_shape,
    gap_extend_shape,
):
  """Computes the adjoint of `weights_from_sim_mat`.

  Viewing `weights_from_sim_mat` as a linear map weights = A sw_params, this
  function implements the linear map A^{T} weights. Primarily to be used when
  implementing custom_gradients in functions downstream.

  Args:
    weights: a tf.Tensor<float>[batch, len1, len2, 9].
    gap_open_shape: a tf.TensorShape representing the shape of gap_open in
      sw_params.
    gap_extend_shape: a tf.TensorShape representing the shape of gap_extend in
      sw_params.

  Returns:
    A tuple (sim_mat_out, gap_open_out, gap_extend_out) such that
      + sim_mat_out is a tf.Tensor<float>[batch, len1, len2] representing the
        elements of A^{T} weights corresponding to sim_mat.
      + gap_open_out is a tf.Tensor<float>[gap_open_shape] representing the
        elements of A^{T} weights corresponding to gap_open_shape.
      + gap_extend_out is a tf.Tensor<float>[gap_extend_shape] representing the
        elements of A^{T} weights corresponding to gap_extend_out.
  """
  sim_mat_out = tf.reduce_sum(weights[Ellipsis, :4], axis=-1)

  # Aggregates output across positions / examples too when appropriate.
  gap_open_out = - (weights[Ellipsis, 4] + weights[Ellipsis, 6] + weights[Ellipsis, 7])
  if gap_open_shape.rank == 1:
    gap_open_out = tf.reduce_sum(gap_open_out, axis=[1, 2])
  elif gap_open_shape.rank == 0:
    gap_open_out = tf.reduce_sum(gap_open_out)

  gap_extend_out = - (weights[Ellipsis, 5] + weights[Ellipsis, 8])
  if gap_extend_shape.rank == 1:
    gap_extend_out = tf.reduce_sum(gap_extend_out, axis=[1, 2])
  elif gap_extend_shape.rank == 0:
    gap_extend_out = tf.reduce_sum(gap_extend_out)

  return sim_mat_out, gap_open_out, gap_extend_out


def length(alignments_or_paths):
  """Computes the lengths in batch of sparse / dense alignments."""
  if alignments_or_paths.shape.rank == 3:  # Sparse format.
    pos_x, pos_y = alignments_or_paths[:, 0], alignments_or_paths[:, 1]
    padding_mask = tf.logical_and(pos_x > 0, pos_y > 0)
    return tf.reduce_sum(tf.cast(padding_mask, tf.float32), axis=-1)
  else:  # Dense format.
    return tf.reduce_sum(alignments_or_paths, axis=[1, 2, 3])


def state_count(alignments_or_paths, states):
  """Counts match/gap_open/gap_extend in batch of sparse / dense alignments."""
  if alignments_or_paths.shape.rank == 3:  # Sparse format.
    batch_size = tf.shape(alignments_or_paths)[0]
    state_indices = alignments_to_state_indices(alignments_or_paths, states)
    batch_indicators = state_indices[:, 0]
    ones = tf.ones_like(batch_indicators, tf.float32)
    return tf.math.unsorted_segment_sum(ones, batch_indicators, batch_size)
  else:  # Dense format.
    state_indicators = paths_to_state_indicators(alignments_or_paths, states)
    return tf.reduce_sum(state_indicators, axis=[1, 2])


def endpoints(alignments_or_paths, start = True):
  """Computes the endpoints in batch of sparse / dense alignments."""
  if alignments_or_paths.shape.rank == 3:  # Sparse format.
    pos = alignments_or_paths[:, :2]
    return pos[Ellipsis, 0] if start else tf.reduce_max(pos, axis=-1)
  else:  # Dense format.
    shape = tf.shape(alignments_or_paths)
    batch_size = shape[0]
    len_x, len_y = shape[1], shape[2]
    matches = paths_to_state_indicators(alignments_or_paths, 'match')
    matches = tf.reshape(matches, [batch_size, -1])
    matches = matches if start else matches[:, ::-1]
    raveled_indices = tf.cast(tf.argmax(matches, axis=-1), tf.int32)
    start_x = tf.cast(tf.math.floor(raveled_indices / len_x), tf.int32)
    start_y = raveled_indices - start_x * len_x
    # Uses one-based indexing for consistency with sparse format.
    endpoint_x = start_x + 1 if start else len_x - start_x
    endpoint_y = start_y + 1 if start else len_y - start_y
    return tf.stack([endpoint_x, endpoint_y])
