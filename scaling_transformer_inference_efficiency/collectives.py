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

"""Manual collectives which use bidirectional ICI and fully overlap compute."""

from typing import Optional, Union, Tuple
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy
import numpy as np


# numpy helper functions for weight pre-shuffling routines
# ------------------------------------------------------------------------------


def interleave(a, b):
  """Interleave two 1D arrays."""
  return jnp.dstack((a, b)).flatten()  # pytype: disable=wrong-arg-types  # jnp-type


def split_apart_axis(x, num_splits, axis):
  """Split an array at axis into num_split chunks."""
  return x.reshape(x.shape[:axis] + (num_splits, x.shape[axis] // num_splits) +
                   x.shape[axis + 1:])


def reflatten_axis(x, axis):
  """Reflatten an array previously split at axis."""
  return x.reshape(x.shape[:axis] + (x.shape[axis] * x.shape[axis + 1],) +
                   x.shape[axis + 2:])


def update_slice(x, update, index, axis):
  """Functional update of array at index on given axis."""
  indices = [slice(None) for _ in range(x.ndim)]
  indices[axis] = index
  return x.at[tuple(indices)].set(update)


def gather_axis(x, index, axis):
  """Gather slice of array at index on given axis."""
  indices = [slice(None) for _ in range(x.ndim)]
  indices[axis] = index
  return x[tuple(indices)]


# dynamic slice helper
# ------------------------------------------------------------------------------


def dynamic_index_and_slice(
    index_axis,
    index,
    slice_axis,
    slice_start,
    slice_length,
    x,
):
  """Helper for layer-indexing and slicing out chunks of layer-stacked weights.

  Args:
    index_axis: the stacked layer axis
    index: the stacked layer index. If not provided we just slice without
      indexing.
    slice_axis: the axis to slice a chunk from
    slice_start: the chunk index
    slice_length: the chunk size
    x: the layer-stacked weight to be sliced.

  Returns:
    The squashed layer slice with a chunk extracted.
  """
  assert (
      index is None or index_axis != slice_axis
  ), f'{index_axis} != {slice_axis}'
  sizes = list(x.shape)
  starts = [0] * len(sizes)
  if index is not None:
    starts[index_axis] = index
  starts[slice_axis] = slice_start
  if index is not None:
    sizes[index_axis] = 1
  sizes[slice_axis] = slice_length
  x = lax.dynamic_slice(x, starts, sizes)
  if index is not None:
    x = lax.squeeze(x, [index_axis])
  return x


# allgather-matmul fusion routines
# ------------------------------------------------------------------------------


def matmul_allgather_no_collective(
    einsum_spec,
    lhs,
    rhs,
    rhs_split_axis,
    axis_name,
    layer,
    layer_axis=0,
):
  """Non-overlapped allgather matmul using default allgather."""
  if layer is not None:
    rhs = lax.dynamic_index_in_dim(rhs, layer, layer_axis, keepdims=False)
  lhs = lax.all_gather(lhs, axis_name, axis=rhs_split_axis, tiled=True)
  return jnp.einsum(einsum_spec, lhs, rhs)


def allgather_matmul_one_way(
    einsum_spec,
    lhs,
    rhs,
    rhs_split_axis,
    axis_name,
    layer,
    layer_axis=0,
):
  """Uses a single ICI direction, overlapped all gather -> matmul.

  Example usage:
    [batch, len, heads.YZX, o_wo_per_head] @ [heads.YZ, o_wo_per_head, dmodel.X]
    allgather LHS over X: List([batch, len, heads.YZ/X, o_wo_per_head] * X)
    split RHS over heads by X: List([heads.YZ/X, o_wo_per_head, dmodel.X ]) * X
    -> (matmul) X times, overlap with compute
    -> X partial sums [batch, maxlen, dmodel.X]{YZ unreduced} -> sum
    -> Follow this function with: ( lax.reducescatter(Y,Z))
    -> [batch, maxlen, dmodel.XYZ]

  Args:
    einsum_spec: matmul specification
    lhs: activations: [batch, len, heads.YZX, o_wo_per_head]
    rhs: weights: [layer, heads.YZ, o_wo_per_head, dmodel.X]
    rhs_split_axis: The axis of the rhs to split
    axis_name: Which axis is being gathered along
    layer: Which layer on the rhs to use (integer index in)
    layer_axis: The dimension on the rhs which represents layers

  Returns:
    activations: new activations
  """
  axis_size = lax.psum(1, axis_name)  # along X
  axis_index = lax.axis_index(axis_name)
  chunk_size = rhs.shape[rhs_split_axis] // axis_size
  first_chunk = lax.dynamic_index_in_dim(rhs, layer, layer_axis, keepdims=False)
  chunk_size = first_chunk.shape[rhs_split_axis] // axis_size
  if rhs_split_axis >= layer_axis:
    rhs_split_axis += 1

  def indexed_computation(i, lhs):
    chunk_index = (axis_index - i) % axis_size
    c = dynamic_index_and_slice(layer_axis, layer, rhs_split_axis,
                                chunk_index * chunk_size, chunk_size, rhs)
    p = jnp.einsum(einsum_spec, lhs, c)
    return p

  accum_shape = jax.eval_shape(indexed_computation, 0, lhs)
  accum = jnp.zeros(accum_shape.shape, dtype=lhs.dtype)

  # all gather collective_matmul is as follows:
  # you chunk along a dimension of the weights, you then shift the acts around
  # and multiply that with it's corresponding index in the weights, and sum
  # partials
  def collective_matmul(i, carrys):
    accum, lhs = carrys
    accum = accum + indexed_computation(i, lhs)
    # in parallel, we shift the lhs around the next one
    lhs = lax.ppermute(
        lhs,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    return accum, lhs

  accum, lhs = jax.lax.fori_loop(
      0, axis_size - 1, collective_matmul, (accum, lhs))

  return accum + indexed_computation(axis_size - 1, lhs)


# TODO(levskaya): clarify relationship between pre-shuffle arguments and the
# target function args - I think the layer axis may implictly require a shifted
# axis argument in the preshuffle functions?
def preshuffle_for_allgather_matmul_throughput(
    x,
    shuffle_axis,
    axis_name):
  """Pre-shuffle weights for allgather_matmul_throughput.

  Function acts in a per-device view.

  Args:
    x: array to preshuffle. Intended to be used within shard map (or hardxmap).
    shuffle_axis: int axis along which to shuffle
    axis_name: name of mesh dimension sharded over.

  Returns:
    Weight array pre-shuffled for use with
    `allgather_matmul_throughput`.

  """
  axis_size = lax.psum(1, axis_name)
  axis_index = lax.axis_index(axis_name)

  def permutation_fn(i):
    iota = jnp.arange(axis_size, dtype=np.int32)
    flipped_evens = jnp.flip(np.roll(2 * iota, -i - 1))
    rolled_odds = jnp.roll(2 * iota + 1, -i)
    return interleave(flipped_evens, rolled_odds)

  shard = split_apart_axis(x, num_splits=2 * axis_size, axis=shuffle_axis)
  shard = gather_axis(
      shard, index=permutation_fn(axis_index), axis=shuffle_axis)
  shard = reflatten_axis(shard, axis=shuffle_axis)

  return shard


def allgather_matmul_throughput(
    einsum_spec,
    lhs,
    rhs,
    rhs_split_axis,
    axis_name,
    layer,
    subsplit_axis,
    layer_axis=0,
):
  """Uses a two ICI directions, overlapped all gather -> matmul.

  Example usage:
    [batch, maxlen, heads.YZX, o_wo_per_head]
              @ [heads.YZ, o_wo_per_head, dmodel.X]
    allgather LHS over X: List([batch, maxlen, heads.YZ/X, o_wo_per_head] * X)
    split RHS over heads by X: List([heads.YZ/X, o_wo_per_head, dmodel.X ]) * X
    -> ('bthd,hde->bte') X times, overlap with compute
    -> X partial sums [[batch, maxlen, dmodel.X]{YZ unreduced}] -> sum
    -> Later on: (unfused reducescatter)
    -> [batch, maxlen, dmodel.XYZ]
  Args:
    einsum_spec: matmul specification
    lhs: activations: [batch, maxlen, heads.YZX, o_wo_per_head]
    rhs: weights: [layer, heads.YZ, o_wo_per_head, dmodel.X]
    rhs_split_axis: The axis of the rhs to split
    axis_name: Which axis is being gathered along
    layer: Which layer on the rhs to use (integer index in)
    subsplit_axis: Axis to split the lhs along to utilise both ICI directions
    layer_axis: The dimension on the rhs which represents layers

  Returns:
    activations: new activations
  """
  axis_size = lax.psum(1, axis_name)  # along X
  chunk_size = rhs.shape[rhs_split_axis] // axis_size
  first_chunk = lax.dynamic_index_in_dim(rhs, layer, layer_axis, keepdims=False)
  chunk_size = first_chunk.shape[rhs_split_axis] // axis_size
  if rhs_split_axis >= layer_axis:
    rhs_split_axis += 1

  def indexed_computation(chunk_index, lhs):
    c = dynamic_index_and_slice(layer_axis, layer, rhs_split_axis,
                                chunk_index * chunk_size, chunk_size, rhs)
    return jnp.einsum(einsum_spec, lhs, c)

  accum_shape = jax.eval_shape(indexed_computation, 0, lhs)
  accum = jnp.zeros(accum_shape.shape, dtype=lhs.dtype)
  lhs_top, lhs_bottom = jnp.split(lhs, 2, subsplit_axis)

  # all gather collective_matmul is as follows:
  # you chunk along a dimension of the weights, you then shift the acts around
  # and multiply that with it's corresponding index in the weights, and sum
  # partials
  def collective_matmul(i, carrys):
    accum, lhs_top, lhs_bottom = carrys
    # do matmul on the concatenated lhs streams
    lhs = jnp.concatenate([lhs_top, lhs_bottom], axis=subsplit_axis)
    accum = accum + indexed_computation(i, lhs)
    # in parallel, we roll the two split lhs streams
    lhs_top = lax.ppermute(
        lhs_top,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    lhs_bottom = lax.ppermute(
        lhs_bottom,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
    return accum, lhs_top, lhs_bottom

  accum, lhs_top, lhs_bottom = jax.lax.fori_loop(0, axis_size - 1,
                                                 collective_matmul,
                                                 (accum, lhs_top, lhs_bottom))

  # loop epilogue: perform a final matmul on the concatenated lhs streams
  lhs = jnp.concatenate([lhs_top, lhs_bottom], axis=subsplit_axis)
  return accum + indexed_computation(axis_size - 1, lhs)


def preshuffle_for_allgather_matmul_latency(
    x,
    shuffle_axis,
    axis_name,
):
  """Pre-shuffle weights for allgather_matmul_latency.

    Function acts at a per-device view.
  Args:
    x: array to preshuffle. Intended to be used within shard map (or hardxmap).
    shuffle_axis: int axis along which to shuffle
    axis_name: name of mesh dimension sharded over.

  Returns:
    Weight array pre-shuffled for use with
    `allgather_matmul_latency`.
  """
  axis_size = lax.psum(1, axis_name)

  def permutation_fn(i):
    evens = [(i - j - 1) % axis_size for j in range(axis_size // 2)]
    odds = [(i + j) % axis_size for j in range(axis_size // 2)]
    block_perm = interleave(evens, odds)
    return interleave(2 * block_perm, 2 * block_perm + 1)

  shard = split_apart_axis(x, num_splits=2 * axis_size, axis=shuffle_axis)
  shard = gather_axis(
      shard,
      index=permutation_fn(lax.axis_index(axis_name)),
      axis=shuffle_axis)
  shard = reflatten_axis(shard, axis=shuffle_axis)

  return shard


def allgather_matmul_latency(
    einsum_spec,
    lhs,
    rhs,
    rhs_split_axis,
    axis_name,
    layer,
    subsplit_axis,
    layer_axis=0,
):
  """Uses a two ICI directions, overlapped all gather -> matmul.

  Example usage:
    [batch, maxlen, heads.YZX, o_wo_per_head]
                                      @ [heads.YZ, o_wo_per_head, dmodel.X]
    first allgather the next shard so that we can multiply with the whole thing
    gives
    allgather over X: List([batch, maxlen, 2 * heads.YZ/X, o_wo_per_head] * X)
    split RHS over heads by X: List([heads.YZ/X, o_wo_per_head, dmodel.X ]) * X
    -> ('bthd,hde->bte') X times, overlap with compute
    -> X partial sums [[batch, maxlen, dmodel.X]{YZ unreduced}] -> sum
    -> Later on: (unfused reducescatter)
    -> [batch, maxlen, dmodel.XYZ]
  Args:
    einsum_spec: matmul specification
    lhs: activations: [batch, maxlen, heads.YZX, o_wo_per_head]
    rhs: weights: [layer, heads.YZ, o_wo_per_head, dmodel.X]
    rhs_split_axis: The axis of the rhs to split, rhs contracting dimension,
      along which we split for steps
    axis_name: Which axis is being gathered along
    layer: Which layer on the rhs to use (integer index in)
    subsplit_axis: Axis to split the lhs along to utilise both ICI directions,
      lhs contracting dimension, along which we concatenate + and -
      directions
    layer_axis: The dimension on the rhs which represents layers

  Returns:
    activations: new activations
  """

  axis_size = lax.psum(1, axis_name)  # along X
  matmul_steps = axis_size // 2
  chunk_size = rhs.shape[rhs_split_axis] // axis_size
  first_chunk = lax.dynamic_index_in_dim(rhs, layer, layer_axis, keepdims=False)
  chunk_size = first_chunk.shape[rhs_split_axis] // matmul_steps
  if rhs_split_axis >= layer_axis:
    rhs_split_axis += 1

  def indexed_computation(chunk_index, lhs):
    c = dynamic_index_and_slice(layer_axis, layer, rhs_split_axis,
                                chunk_index * chunk_size, chunk_size, rhs)
    return jnp.einsum(einsum_spec, lhs, c)

  # loop prologue: get the current and next lhs on the same device
  lhs_bwd = lhs
  lhs_fwd = lax.ppermute(
      lhs, axis_name, perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

  working_lhs = jnp.concatenate([lhs_fwd, lhs_bwd], axis=subsplit_axis)
  accum_shape = jax.eval_shape(indexed_computation, 0, working_lhs)
  accum = jnp.zeros(accum_shape.shape, dtype=lhs.dtype)

  def collective_matmul(i, carrys):
    accum, lhs_fwd, lhs_bwd = carrys
    # do matmul on the concatenated lhs streams
    lhs = jnp.concatenate([lhs_fwd, lhs_bwd], axis=subsplit_axis)
    accum = accum + indexed_computation(i, lhs)
    # in parallel, we roll the two split lhs streams
    lhs_fwd = lax.ppermute(
        lhs_fwd,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    lhs_bwd = lax.ppermute(
        lhs_bwd,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
    return accum, lhs_fwd, lhs_bwd

  accum, lhs_fwd, lhs_bwd = jax.lax.fori_loop(
      0, matmul_steps - 1, collective_matmul, (accum, lhs_fwd, lhs_bwd))

  # loop epilogue: perform a final matmul on the concatenated lhs streams
  lhs = jnp.concatenate([lhs_fwd, lhs_bwd], axis=subsplit_axis)
  return accum + indexed_computation(axis_size - 1, lhs)


# overlapped matmul-reducescatter routines
# ------------------------------------------------------------------------------


def matmul_reducescatter_no_collective(
    einsum_spec,
    lhs,
    rhs,
    scatter_axis,
    axis_name,
    layer,
    layer_axis=0,
):
  """Non overlapped matmul reduce scatter using default psum_scatter."""
  if layer is not None:
    rhs = lax.dynamic_index_in_dim(rhs, layer, layer_axis, keepdims=False)
  tmp = jnp.einsum(einsum_spec, lhs, rhs)
  result = lax.psum_scatter(
      tmp, axis_name, scatter_dimension=scatter_axis, tiled=True)
  return result


def matmul_reducescatter_oneway(
    einsum_spec,
    lhs,
    rhs,
    scatter_axis,
    axis_name,
    layer,
    layer_axis=0,
):
  """Uses a single ICI direction, overlapped weight stationary reduce scatter.

  Usage:
    [batch, maxlen, dmodel.X] @ [heads.YZ, dmodel.X, q_wi_per_head]
      -> (matmul)
      -> [batch, maxlen, heads.YZ, q_wi_per_head]{X unreduced}
      -> (reducescatter over X into heads)
      -> [batch, maxlen, heads.YZX, q_wi_per_head]

  Args:
    einsum_spec: Spec for the einsum
    lhs: Typically activations
    rhs: Typically weights
    scatter_axis: The rhs scatter axis.
    axis_name: The hardware axis along which we are reducing
    layer: Weights are stored with layer as the first dimension, index of the
      layer. If not provided we skip indexing and use the weights as is.
    layer_axis: Which axis is the layer dimension

  Returns:
    Result of a matmul and reduce_scatter
  """

  # we want to maintain an accumulator, then permute and add to said accumulator
  axis_size = lax.psum(1, axis_name)
  axis_index = lax.axis_index(axis_name)
  rhs_scatter_axis = scatter_axis
  if rhs_scatter_axis >= layer_axis and layer is not None:
    rhs_scatter_axis += 1

  permutes_remaining = axis_size - 1

  chunk_index = (axis_index + permutes_remaining) % axis_size
  chunk_size = rhs.shape[rhs_scatter_axis] // axis_size
  if chunk_size == 0:
    # Scatter axis size was smaller than axis size.
    # Can't scatter so all reduce.
    out = jnp.einsum(einsum_spec, lhs, rhs)
    return lax.psum(out, axis_name)
  first_chunk = dynamic_index_and_slice(layer_axis, layer, rhs_scatter_axis,
                                        chunk_index * chunk_size, chunk_size,
                                        rhs)

  p = jnp.einsum(einsum_spec, lhs, first_chunk)
  accum = jnp.zeros(p.shape, dtype=lhs.dtype)

  # collective_matmul reduce scatter is as follows:
  # you chunk along a dimension separate from the partitioned one you
  # are multiplying along
  # to reduce, you sum chunk partial sums per index, therefore you get
  # chunk sized full sums partitioned along the scatter axis
  def collective_matmul(i, carrys):
    chunk_index = (axis_index + (axis_size - 1) - i) % axis_size
    # matmul
    accum, p = carrys
    accum = accum + p
    c = dynamic_index_and_slice(layer_axis, layer, rhs_scatter_axis,
                                chunk_index * chunk_size, chunk_size, rhs)
    p = jnp.einsum(einsum_spec, lhs, c)
    accum = lax.ppermute(
        accum,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    return accum, p

  accum, p = jax.lax.fori_loop(
      1, axis_size, collective_matmul, (accum, p))

  return accum + p


# bidirectional forms


def preshuffle_for_reducescatter_throughput(
    x,
    scatter_axis,
    subsplit_axis,
    axis_name):
  """Pre-shuffles input arrays for bidirectional matmul-reduce-scatters.

  Args:
    x: array to preshuffle. Assumes the array has already been reshaped
      appropriately as an input for xmap, with materialized sharding dims.
    scatter_axis: array dim to scatter into.
    subsplit_axis: array dim to split along for bidirectional split.
    axis_name: name of mesh dimension sharded over.

  Returns:
    Weight array pre-shuffled for use with
    `matmul_reducescatter_throughput`.
  """
  axis_size = lax.psum(1, axis_name)
  axis_index = lax.axis_index(axis_name)

  half = x.shape[subsplit_axis] // 2
  pos_perm = lambda idx: jnp.roll(jnp.flip(np.arange(axis_size)), idx)
  neg_perm = lambda idx: jnp.roll(np.arange(axis_size), -idx - 1)
  new_x = jnp.zeros_like(x)

  # Handle shifts in each half of the subsplits:
  for pos, perm_fn in ((0, pos_perm), (half, neg_perm)):
    split = gather_axis(x, index=slice(pos, pos + half), axis=subsplit_axis)
    chunked_split = split_apart_axis(split, axis_size, axis=scatter_axis)
    rolled = gather_axis(
        chunked_split, index=perm_fn(axis_index), axis=scatter_axis)
    rolled = reflatten_axis(rolled, scatter_axis)
    new_x = update_slice(
        new_x, rolled, slice(pos, pos + half), axis=subsplit_axis)
  return new_x


def matmul_reducescatter_throughput(einsum_spec,
                                    lhs,
                                    rhs,
                                    scatter_axis,
                                    axis_name,
                                    layer,
                                    subsplit_axis,
                                    layer_axis=0):
  """Uses a two ICI directions, overlapped weight stationary reduce scatter.

  Usage:
    [batch, maxlen, dmodel.X] @ [heads.YZ, dmodel.X, q_wi_per_head]
    -> (matmul)
    -> [batch, maxlen, heads.YZ, q_wi_per_head]{X unreduced}
    -> (reducescatter over X into heads)
    -> [batch, maxlen, heads.YZX, q_wi_per_head]
    we want to maintain an accumulator, then permute and add to said accumulator

  Args:
    einsum_spec: Spec for the einsum
    lhs: Typically activations
    rhs: Typically weights
    scatter_axis: The rhs scatter axis
    axis_name: The hardware axis along which we are reducing
    layer: Weights are stored with layer as the first dimension, index of the
      layer
    subsplit_axis: Axis to split the accumulator along to utilise both ICI links
    layer_axis: Which axis is the layer dimension

  Returns:
    Result of a matmul and reduce_scatter
  """
  axis_size = lax.psum(1, axis_name)  # two for both directions
  rhs_scatter_axis = scatter_axis
  if rhs_scatter_axis >= layer_axis:
    rhs_scatter_axis += 1

  chunk_size = rhs.shape[rhs_scatter_axis] // axis_size
  chunk_index = 0
  first_chunk = dynamic_index_and_slice(layer_axis, layer, rhs_scatter_axis,
                                        chunk_index * chunk_size, chunk_size,
                                        rhs)
  p = jnp.einsum(einsum_spec, lhs, first_chunk)
  accum_shape = list(p.shape)
  accum_shape[subsplit_axis] = accum_shape[subsplit_axis] // 2
  accum_pos = jnp.zeros(accum_shape, dtype=lhs.dtype)
  accum_neg = jnp.zeros(accum_shape, dtype=lhs.dtype)

  # collective_matmul reduce scatter is as follows:
  # you chunk along a different dimension to the partitioned one you
  # are multiplying along
  # to reduce, you sum chunk partial sums per index, therefore you get
  # chunk sized full sums partitioned along the scatter axis
  def collective_matmul(i, carrys):
    # matmul
    accum_pos, accum_neg, p = carrys
    p_pos, p_neg = jnp.split(p, 2, subsplit_axis)
    accum_pos += p_pos
    accum_neg += p_neg
    c = dynamic_index_and_slice(layer_axis, layer, rhs_scatter_axis,
                                i * chunk_size, chunk_size, rhs)
    p = jnp.einsum(einsum_spec, lhs, c)

    accum_pos = lax.ppermute(
        accum_pos,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    accum_neg = lax.ppermute(
        accum_neg,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])

    return accum_pos, accum_neg, p

  accum_pos, accum_neg, p = jax.lax.fori_loop(1, axis_size, collective_matmul,
                                              (accum_pos, accum_neg, p))

  result = jnp.concatenate((accum_pos, accum_neg), subsplit_axis) + p
  return result


def preshuffle_for_reducescatter_latency(
    x,
    scatter_axis,
    axis_name,
):
  """Pre-shuffles input arrays for bidirectional matmul-reduce-scatters.

  Function acts at a per-device view.
  Args:
    x: array to preshuffle. Intended to be used within shard map (or hardxmap).
    scatter_axis: array dim to scatter into.
    axis_name: name of mesh dimension sharded over.

  Returns:
    Weight array pre-shuffled for use with
    `matmul_reducescatter_latency`.
  """
  axis_size = lax.psum(1, axis_name)
  axis_index = lax.axis_index(axis_name)

  # closed form for modelling the data movement of
  # each nth chunk over n iterations of the latency-optimized
  # pincer data movement.
  def permutation_fn(idx):
    iota = jnp.arange(axis_size // 2) + axis_size // 2
    evens = (idx - iota) % axis_size
    odds = (idx + iota + 1) % axis_size
    return interleave(evens, odds)

  chunked_shard = split_apart_axis(x, axis_size, axis=scatter_axis)
  permuted = gather_axis(
      chunked_shard, permutation_fn(axis_index), axis=scatter_axis)
  permuted = reflatten_axis(permuted, scatter_axis)

  return permuted


def matmul_reducescatter_latency(
    einsum_spec,
    lhs,
    rhs,
    scatter_axis,
    axis_name,
    layer,
    subsplit_axis,
    layer_axis=0,
):
  """Uses a two ICI directions, pre-communicate to halve the steps.

  Usage:
    [batch, maxlen, dmodel.X] @ [heads.YZ, dmodel.X, q_wi_per_head]
    -> (matmul)
    -> [batch, maxlen, heads.YZ, q_wi_per_head]{X unreduced}
    -> (reducescatter over X into heads)
    -> [batch, maxlen, heads.YZX, q_wi_per_head]
    we want to maintain an accumulator, then permute and add to said accumulator

  Args:
    einsum_spec: Spec for the einsum
    lhs: Typically activations
    rhs: Typically weights
    scatter_axis: the rhs scatter axis
    axis_name: The hardware axis along which we are reducing
    layer: Weights are stored with layer as the first dimension, index of the
      layer
    subsplit_axis: axis to split the accumulator along to utilise both ICI links
    layer_axis: Which axis is the layer dimension

  Returns:
    Result of a matmul and reduce_scatter
  """
  axis_size = lax.psum(1, axis_name)  # two for both directions
  if scatter_axis >= layer_axis:
    scatter_axis += 1
  matmul_steps = axis_size // 2

  chunk_size = rhs.shape[scatter_axis] // matmul_steps
  first_chunk = dynamic_index_and_slice(
      layer_axis, layer, scatter_axis, 0, chunk_size, rhs)

  p = jnp.einsum(einsum_spec, lhs, first_chunk)
  accum_shape = jax.eval_shape(
      lambda p: jnp.split(p, 2, subsplit_axis), p)[0].shape
  accum_pos = jnp.zeros(accum_shape, dtype=lhs.dtype)
  accum_neg = jnp.zeros(accum_shape, dtype=lhs.dtype)

  # collective_matmul reduce scatter is as follows:
  # you chunk along a different dimension to the partitioned one you
  # are multiplying along
  # to reduce, you sum chunk partial sums per index, therefore you get
  # chunk sized full sums partitioned along the scatter axis
  def collective_matmul(i, carrys):
    # matmul
    accum_pos, accum_neg, p = carrys
    p_pos, p_neg = jnp.split(p, 2, subsplit_axis)
    accum_pos += p_pos
    accum_neg += p_neg

    c = dynamic_index_and_slice(layer_axis, layer, scatter_axis, i * chunk_size,
                                chunk_size, rhs)
    p = jnp.einsum(einsum_spec, lhs, c)
    accum_pos = lax.ppermute(
        accum_pos,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    accum_neg = lax.ppermute(
        accum_neg,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])

    return accum_pos, accum_neg, p

  accum_pos, accum_neg, p = jax.lax.fori_loop(1, matmul_steps,
                                              collective_matmul,
                                              (accum_pos, accum_neg, p))

  p_pos, p_neg = jnp.split(p, 2, subsplit_axis)
  accum_pos = lax.ppermute(
      accum_pos + p_pos,
      axis_name,
      perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

  return accum_pos + accum_neg + p_neg


# non-weight-stationary fused matmul routines
# ------------------------------------------------------------------------------


def matmul_collective_weights_gather_q_wi(
    einsum_spec,
    lhs,
    rhs,
    lhs_split_axis):
  """Designed for prefill, moves the weights through x,y and z instead of lhs.

  Pseudocode:

    [batch.XYZ, t, e] @ [heads.YZ, e.X, q_wi_per_head]
    result = zeros((Y, Z, ...))
    for y in Y:
      for z in Z:
        local_accum = zeros(...)
        for x in X:
          next_weights = ppermute(weights, 'x', plus_1)
          local_accum += jnp.einsum(lhs, next_weights)
        result[Y, Z] = local_accum

  Args:
    einsum_spec: the computation
    lhs: activations, sharded over batch
    rhs: weights, to be moved through all chips in a space filling curve
    lhs_split_axis: the lhs axis along which to split so we can overlap the math
      in the innermost loop

  Returns:
    new_activations: result of computation
  """

  x_size, y_size, z_size = lax.psum(1, 'x'), lax.psum(1, 'y'), lax.psum(1, 'z')
  x_ai, y_ai, z_ai = (
      lax.axis_index('x'), lax.axis_index('y'), lax.axis_index('z'))

  # chunking specific
  chunk_size = lhs.shape[lhs_split_axis] // x_size

  # slice into the weights along the axis we will scatter along
  # all others are just in a chain as partitioned
  def indexed_computation(x_i, y_i, z_i, rhs):
    chunk_index = (x_ai -
                   y_i * (z_size * (x_size - 1)) +
                   z_i * (x_size - 1) + x_i) % x_size
    c = jax.lax.dynamic_slice_in_dim(
        lhs, chunk_index * chunk_size, chunk_size, lhs_split_axis)
    return jnp.einsum(einsum_spec, c, rhs)

  # this should be sized as though normally partitoned by the einsum,
  # but also partitioned along the chunk_split axis
  # [b.YZ, t, h.YZ, q]
  local_accum_shape = jax.eval_shape(indexed_computation, 0, 0, 0, rhs).shape
  # we will concatenate in the loops over heads, so that we are sharded
  # only over batch
  head_idx = 2
  final_heads_dim = local_accum_shape[head_idx] * y_size * z_size
  final_accum_shape = list(local_accum_shape)
  final_accum_shape[head_idx] = final_heads_dim
  final_accum = jnp.zeros(final_accum_shape, dtype=lhs.dtype)

  def indexed_update(y, z, update, final_accum):
    """Stacks alongs the dimension sharded by yz."""
    indices = [0] * len(final_accum.shape)
    yidx = (y_ai - y) % y_size
    zidx = (z_ai - (y * (z_size - 1) + z)) % z_size
    indices[head_idx] = (yidx * z_size + zidx) * update.shape[head_idx]
    return jax.lax.dynamic_update_slice(final_accum, update, indices)

  # overlap chunk computation with x
  def x_loop_and_matmul(x_i, carrys):
    with jax.named_scope('x'):
      accum, rhs, y_i, z_i = carrys
      accum += indexed_computation(x_i, y_i, z_i, rhs)
      next_rhs = lax.ppermute(
          rhs, 'x', perm=[(j, (j + 1) % x_size) for j in range(x_size)])
      return (accum, next_rhs, y_i, z_i)

  def collect_x(final_accum, rhs, z_i, y_i):
    # [b.YZ, t, h.YZ, q]
    local_accum = jnp.zeros(local_accum_shape, dtype=lhs.dtype)
    local_accum, rhs, _, _ = jax.lax.fori_loop(
        0, x_size - 1, x_loop_and_matmul, (local_accum, rhs, y_i, z_i))
    # do one less loop to skip an uneeded permute
    local_accum += indexed_computation(x_size - 1, y_i, z_i, rhs)
    # concatenate and unshard by heads
    final_accum = indexed_update(y_i, z_i, local_accum, final_accum)
    return final_accum, rhs

  # overlap z movement with entire x computation
  def z_loop(z_i, carrys):
    with jax.named_scope('z'):
      final_accum, rhs, y_i = carrys
      final_accum, rhs = collect_x(final_accum, rhs, z_i, y_i)
      # ideally don't want to do this y extra times
      next_rhs = lax.ppermute(
          rhs, 'z', perm=[(j, (j + 1) % z_size) for j in range(z_size)])
      return (final_accum, next_rhs, y_i)

  # overlap y movement with all z computation
  def y_loop(y_i, carrys):
    with jax.named_scope('y'):
      final_accum, rhs = carrys
      final_accum, rhs, _ = jax.lax.fori_loop(0, z_size - 1, z_loop,
                                              (final_accum, rhs, y_i))
      # we do the z loop one less time than we need to, as the weights
      # shift Z-1 times, then we do an x loop with the final set of z weights
      # [b.YZ, t, h.YZ, q]
      final_accum, rhs = collect_x(final_accum, rhs, z_size - 1, y_i)
      # not so costly to do this one extra time
      next_rhs = lax.ppermute(
          rhs, 'y', perm=[(j, (j + 1) % y_size) for j in range(y_size)])
      return (final_accum, next_rhs)

  final_accum, rhs = jax.lax.fori_loop(
      0, y_size - 1, y_loop, (final_accum, rhs))
  # we do one less y loop than we need, and instead finish off with the z loop
  # that would have been (to avoid a final permute)
  final_accum, rhs, _ = jax.lax.fori_loop(
      0, z_size - 1, z_loop, (final_accum, rhs, y_size - 1))

  # and also for x
  final_accum, rhs = collect_x(final_accum, rhs, z_size - 1, y_size - 1)

  return final_accum


def matmul_collective_weights_gather_o_wo(
    einsum_spec,
    lhs,
    rhs,
    lhs_split_axis):
  """Designed for prefill, moves the weights through x,y and z instead of lhs.

  Pseudocode:
    result = zeros((X, ...)) for x in X:
      local_accum = zeros(...)
      for y in Y:
        for z in Z:
          next_weights = ppermute(weights, 'x', plus_1)
          local_accum += jnp.einsum(lhs, next_weights)
      result[X] = local_accum

  Args:
    einsum_spec: the computation
    lhs: activations, sharded over batch
    rhs: weights, to be moved through all chips in a space filling curve
    lhs_split_axis: the lhs axis along which to split so we can overlap the math
      in the innermost loop

  Returns:
    new_activations: result of computation
  """
  x_size, y_size, z_size = lax.psum(1, 'x'), lax.psum(1, 'y'), lax.psum(1, 'z')
  x_ai, y_ai, z_ai = (
      lax.axis_index('x'), lax.axis_index('y'), lax.axis_index('z'))
  axis_size = y_size * z_size
  # chunking specific - should be nh dim on lhs
  chunk_size = lhs.shape[lhs_split_axis] // axis_size

  # slice into the weights along the axis we will scatter along
  # all others are just in a chain as partitioned
  def indexed_computation(x_i, y_i, z_i, rhs):
    zidx = (z_ai - (x_i * y_size * (z_size - 1) +
                    y_i * (z_size - 1) + z_i)) % z_size
    yidx = (y_ai - (x_i * (y_size - 1) + y_i)) % y_size
    chunk_index = yidx * z_size + zidx
    c = jax.lax.dynamic_slice_in_dim(
        lhs, chunk_index * chunk_size, chunk_size, lhs_split_axis)
    return jnp.einsum(einsum_spec, c, rhs)

  # this should be sized as though normally partitoned by the einsum,
  # but also partitioned along the chunk_split axis
  # [b.YZ, t, h.YZ, q]

  local_accum_shape = jax.eval_shape(indexed_computation, 0, 0, 0, rhs).shape
  # we will concatenate in the loops over heads, so that we are sharded
  # only over batch
  embed_idx = 2
  final_embed_dim = local_accum_shape[embed_idx] * x_size
  final_accum_shape = list(local_accum_shape)
  final_accum_shape[embed_idx] = final_embed_dim
  final_accum = jnp.zeros(final_accum_shape, dtype=lhs.dtype)

  def indexed_update(x_i, update, final_accum):
    """Stacks alongs the dimension sharded by xy."""
    indices = [0] * len(final_accum.shape)
    indices[embed_idx] = ((x_ai - x_i) % x_size) * update.shape[lhs_split_axis]
    return jax.lax.dynamic_update_slice(final_accum, update, indices)

  # overlap chunk computation with x
  def z_loop_and_matmul(z_i, carrys):
    with jax.named_scope('z'):
      accum, rhs, x_i, y_i = carrys
      accum += indexed_computation(x_i, y_i, z_i, rhs)
      next_rhs = lax.ppermute(
          rhs, 'z', perm=[(j, (j + 1) % z_size) for j in range(z_size)])
      return accum, next_rhs, x_i, y_i

  # overlap z movement with entire x computation
  def y_loop(y_i, carrys):
    with jax.named_scope('y'):
      local_accum, rhs, x_i = carrys
      local_accum, rhs, _, _ = jax.lax.fori_loop(
          0, z_size - 1, z_loop_and_matmul, (local_accum, rhs, x_i, y_i))
      local_accum += indexed_computation(x_i, y_i, z_size - 1, rhs)
      # ideally don't want to do this y extra times
      next_rhs = lax.ppermute(
          rhs, 'y', perm=[(j, (j + 1) % y_size) for j in range(y_size)])
      return local_accum, next_rhs, x_i

  def collect_yz(final_accum, rhs, x_i):
    local_accum = jnp.zeros(local_accum_shape, dtype=lhs.dtype)
    local_accum, rhs, _ = jax.lax.fori_loop(
        0, y_size - 1, y_loop, (local_accum, rhs, x_i))
    local_accum, rhs, _, _ = jax.lax.fori_loop(
        0, z_size - 1, z_loop_and_matmul, (local_accum, rhs, x_i, y_size - 1))
    # do one less loop to skip an uneeded permute
    local_accum += indexed_computation(x_i, y_size - 1, z_size - 1, rhs)
    # concatenate and unshard by heads
    final_accum = indexed_update(x_i, local_accum, final_accum)
    return final_accum, rhs

  # overlap y movement with all z computation
  def x_loop(x_i, carrys):
    with jax.named_scope('x'):
      final_accum, rhs = carrys
      # we do the z loop one less time than we need to, as the weights
      # shift Z-1 times, then we do an x loop with the final set of z weights
      # [b.YZ, t, h.YZ, q]
      final_accum, rhs = collect_yz(final_accum, rhs, x_i)
      # not so costly to do this one extra time
      next_rhs = lax.ppermute(
          rhs, 'x', perm=[(j, (j + 1) % x_size) for j in range(x_size)])
      return (final_accum, next_rhs)

  final_accum, rhs = jax.lax.fori_loop(
      0, x_size - 1, x_loop, (final_accum, rhs))
  # we do one less y loop than we need, and instead finish off with the z loop
  # that would have been (to avoid a final permute)
  final_accum, rhs = collect_yz(final_accum, rhs, x_size - 1)

  return final_accum


# raw reduce-scatter collectives
# ------------------------------------------------------------------------------

# unused in paper, but presented here as isolated lowerings
# of reduce-scatter to collective permutes.


# pylint: disable = unused-argument
def plain_reducescatter(
    val,
    scatter_dimension,
    axis_name,
    subsplit_axis=None):
  return lax.psum_scatter(
      val, axis_name, scatter_dimension=scatter_dimension, tiled=True)


def reducescatter_oneway(
    val,
    scatter_dimension,
    axis_name,
    subsplit_axis=None):
  """Uses one ICI direction, overlapped weight stationary reduce scatter."""
  # [A, B, C] -> [A, B, C.X]
  axis_size = lax.psum(1, axis_name)
  axis_index = lax.axis_index(axis_name)
  scatter_axis = scatter_dimension
  permutes_remaining = axis_size - 1
  chunk_index = (axis_index + permutes_remaining) % axis_size
  chunk_size = val.shape[scatter_axis] // axis_size
  p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                               scatter_axis)
  accum = jnp.zeros(p.shape, dtype=val.dtype)

  def collective(i, carrys):
    permutes_remaining_after = (axis_size - 1) - i
    chunk_index = (axis_index + permutes_remaining_after) % axis_size
    accum, p = carrys
    accum = accum + p
    p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                                 scatter_axis)
    accum = lax.ppermute(
        accum,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

    return accum, p
  accum, p = jax.lax.fori_loop(1, axis_size, collective, (accum, p))
  return accum + p


def reducescatter_throughput(
    val,
    scatter_dimension,
    axis_name,
    subsplit_axis):
  """Using two ICI directions, manual reduce scatter."""
  axis_size = lax.psum(1, axis_name)
  chunk_size = val.shape[scatter_dimension] // axis_size
  chunk_index = 0
  p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                               scatter_dimension)
  accum_shape = list(p.shape)
  accum_shape[subsplit_axis] = accum_shape[subsplit_axis] // 2
  accum_pos = jnp.zeros(accum_shape, dtype=val.dtype)
  accum_neg = jnp.zeros(accum_shape, dtype=val.dtype)
  def collective(i, carrys):
    # matmul
    accum_pos, accum_neg, p = carrys
    p_pos, p_neg = jnp.split(p, 2, subsplit_axis)
    accum_pos += p_pos
    accum_neg += p_neg
    p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                                 scatter_dimension)
    accum_pos = lax.ppermute(
        accum_pos,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    accum_neg = lax.ppermute(
        accum_neg,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
    return accum_pos, accum_neg, p
  accum_pos, accum_neg, p = jax.lax.fori_loop(1, axis_size, collective,
                                              (accum_pos, accum_neg, p))
  return jnp.concatenate((accum_pos, accum_neg), subsplit_axis) + p


def reducescatter_latency(
    val,
    scatter_dimension,
    axis_name,
    subsplit_axis=None):
  """Using two ICI directions, manual reduce scatter."""
  axis_size = lax.psum(1, axis_name)  # two for both directions
  matmul_steps = axis_size // 2
  chunk_size = val.shape[scatter_dimension] // matmul_steps
  chunk_index = 0
  p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                               scatter_dimension)

  p_pos, p_neg = jnp.split(p, 2, scatter_dimension)
  accum_size = p_pos.shape
  accum_pos = jnp.zeros(accum_size, dtype=val.dtype)
  accum_neg = jnp.zeros(accum_size, dtype=val.dtype)

  def collective_matmul(i, carrys):
    # matmul
    accum_pos, accum_neg, p = carrys
    p_pos, p_neg = jnp.split(p, 2, scatter_dimension)
    accum_pos += p_pos
    accum_neg += p_neg
    p = lax.dynamic_slice_in_dim(val, chunk_index * chunk_size, chunk_size,
                                 scatter_dimension)
    accum_pos = lax.ppermute(
        accum_pos,
        axis_name,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
    accum_neg = lax.ppermute(
        accum_neg,
        axis_name,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
    return accum_pos, accum_neg, p

  accum_pos, accum_neg, p = jax.lax.fori_loop(1, matmul_steps,
                                              collective_matmul,
                                              (accum_pos, accum_neg, p))
  p_pos, p_neg = jnp.split(p, 2, scatter_dimension)
  accum_pos = lax.ppermute(
      accum_pos + p_pos,
      axis_name,
      perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

  return accum_pos + accum_neg + p_neg
