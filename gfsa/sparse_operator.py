# coding=utf-8
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

# Lint as: python3
"""A sparse operator abstraction for JAX."""

from typing import Any, Optional, Sequence
import dataclasses

import jax

from gfsa import jax_util


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class SparseCoordOperator:
  """Sparse multidimensional linear operator stored in coordinate format.

  This operator is a multidimensional generalization of the COO format for a
  sparse matrix, which specifies row indices, column indices, and values for
  each of its nonzero elements. For a SparseCoordOperator, each "row" and
  "column" is a full NDArray instead of a flat vector, and thus the input and
  output indices have one component for each axis of the input and output
  NDArrays.

  It is allowed for multiple input-output index pairs to be the same. This will
  sum the two corresponding values in the values array when applied.

  Attributes:
    input_indices: <int32[nonzeros, num_in_dims]> array indexing into input.
    output_indices: <int32[nonzeros, num_out_dims]> array indexing into output.
    values: <float[nonzeros]> array containing values to scale by.
  """
  input_indices: Any
  output_indices: Any
  values: Any

  def transpose(self):
    """Returns a transposed linear operator (swapping inputs and outputs)."""
    return SparseCoordOperator(
        input_indices=self.output_indices,
        output_indices=self.input_indices,
        values=self.values,
    )

  def apply_add(self,
                in_array,
                out_array,
                in_dims = None,
                out_dims = None):
    """Applies the operator to an input, and adds the result to the output.

    In other words, computes `out_array + A(in_array)` where A is the linear
    operator represented by this object. To just compute `A(in_array)`, pass
    out_array=jnp.zeros(out_shape).

    Optionally, in_array and out_array may have additional "batch" dimensions
    that are not processed by the linear operator. In this case, `in_dims`
    should be a list of axis indices to reduce over, and `out_dims` should be
    a list of axis indices to insert the results into. For instance, the
    equivalent of the einsum transformation

      out_array + einsum("abxyz,uavb->zuxvy", A, in_array)

    would be

      A.apply_add(in_array, out_array, [1,3], [2,4,0]).

    Args:
      in_array: Input array.
      out_array: Destination array to apply the sparse update to.
      in_dims: Sequence of dimensions in `in_array` that correspond to the
        indices in `input_indices`. By default, uses `range(in_array.ndim)`.
      out_dims: Sequence of dimensions in `out_array` that correspond to the
        indices in `output_indices`. By default, uses `range(out_array.ndim)`.

    Returns:
      Array of same shape as `out_array`, with contributions from the operator
      added in.
    """
    if in_dims is None:
      in_dims = range(in_array.ndim)
    in_dims = tuple(in_dims)

    if out_dims is None:
      out_dims = range(out_array.ndim)
    out_dims = tuple(out_dims)

    # Check that our sizes are correct before we call the low-level XLA scatter
    # and gather operations (to avoid confusing XLA error messages).
    domain_ndim = self.input_indices.shape[-1]
    if len(in_dims) != domain_ndim:
      raise ValueError(f"Expected {domain_ndim} input dimensions, "
                       f"got {len(in_dims)}")

    if len(in_dims) != len(set(in_dims)):
      raise ValueError(f"Duplicate entries in in_dims: {in_dims}")

    range_ndim = self.output_indices.shape[-1]
    if len(out_dims) != range_ndim:
      raise ValueError(f"Expected {range_ndim} output dimensions, "
                       f"got {len(out_dims)}")

    if len(out_dims) != len(set(out_dims)):
      raise ValueError(f"Duplicate entries in out_dims: {out_dims}")

    batched_ndim = in_array.ndim - domain_ndim
    in_batch_sizes = [
        s for i, s in enumerate(in_array.shape) if i not in in_dims
    ]
    out_batch_sizes = [
        s for i, s in enumerate(out_array.shape) if i not in out_dims
    ]
    if in_batch_sizes != out_batch_sizes:
      raise ValueError("Input and output must have the same batch sizes: "
                       f"got {in_batch_sizes} and {out_batch_sizes}")

    # Gather from in_array (indexing into the dimensions in `in_dims`, and
    # taking full slices of the other dimensions).
    slice_sizes = tuple(
        1 if i in in_dims else s for i, s in enumerate(in_array.shape))
    extracted_slices = jax.lax.gather(
        operand=in_array,
        start_indices=self.input_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=tuple(range(batched_ndim)),
            collapsed_slice_dims=tuple(sorted(in_dims)),
            start_index_map=in_dims),
        slice_sizes=slice_sizes)

    # `extracted_slices` has one axis for each input axis not in `in_dims`,
    # followed by a new axis corresponding to the operator nonzero entries.

    # Apply weights to our slices.
    weighted_slices = self.values * extracted_slices

    # Scatter into out_array (indexing into the dimensions in `out_dims`, and
    # taking full slices of the other dimensions).
    result = jax.lax.scatter_add(
        operand=out_array,
        scatter_indices=self.output_indices,
        updates=weighted_slices,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(
            update_window_dims=tuple(range(batched_ndim)),
            inserted_window_dims=tuple(sorted(out_dims)),
            scatter_dims_to_operand_dims=out_dims))
    return result

  def pad_nonzeros(self, nonzeros_axis_size):
    """Pad the number of entries in the operator's `nonzero` axis.

    We can always expand an operator by adding new entries with value 0 without
    changing its meaning. This is useful for batching examples together, for
    instance.

    Args:
      nonzeros_axis_size: Size of the nonzero axis after padding.

    Returns:
      Operator that is equivalent to `self` (in the sense that `apply_add`
      behaves identically) but has `nonzeros_axis_size` as the size of the
      first axis.

    Raises:
      ValueError: If this operator has too many nonzero entries to fit in the
      requested size.
    """
    return SparseCoordOperator(
        input_indices=jax_util.pad_to(self.input_indices, nonzeros_axis_size),
        output_indices=jax_util.pad_to(self.output_indices, nonzeros_axis_size),
        values=jax_util.pad_to(self.values, nonzeros_axis_size),
    )
