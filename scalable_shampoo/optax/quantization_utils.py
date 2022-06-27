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

"""Helper routines for quantization."""

from typing import Any

import chex
from flax import struct
import jax.numpy as jnp


# pylint:disable=no-value-for-parameter
@struct.dataclass
class QuantizedValue:
  """State associated with quantized value."""
  quantized: chex.Array
  diagonal: chex.Array  # Diagonal (if extract_diagonal is set)
  bucket_size: chex.Array
  quantized_dtype: jnp.dtype = struct.field(
      pytree_node=False)  # Dtype for the quantized value.
  extract_diagonal: bool = struct.field(
      pytree_node=False)  # In case its centered.
  shape: Any = struct.field(pytree_node=False)  # Shape of the tensor.

  @classmethod
  def from_float_value(cls, fvalue, quantized_dtype, extract_diagonal=False):
    if isinstance(fvalue, list) and not fvalue:
      return QuantizedValue([], [], [], quantized_dtype, extract_diagonal, [])
    quantized, diagonal_fvalue, bucket_size = QuantizedValue.quantize(
        fvalue, quantized_dtype, extract_diagonal)
    return QuantizedValue(quantized, diagonal_fvalue, bucket_size,
                          quantized_dtype, extract_diagonal,
                          list(quantized.shape))

  # Quantization is from Lingvo JAX optimizers.
  # We extend it for int16 quantization of PSD matrices.
  @classmethod
  def quantize(cls, fvalue, quantized_dtype, extract_diagonal=False):
    """Returns quantized value and the bucket."""
    if quantized_dtype == jnp.float32:
      return fvalue, [], []
    elif quantized_dtype == jnp.bfloat16:
      return fvalue.astype(jnp.bfloat16), [], []

    float_dtype = fvalue.dtype
    if quantized_dtype == jnp.int8:
      # value -128 is not used.
      num_buckets = jnp.array(127.0, dtype=float_dtype)
    elif quantized_dtype == jnp.int16:
      # value -32768 is not used.
      num_buckets = jnp.array(32767.0, dtype=float_dtype)
    else:
      raise ValueError(f'Quantized dtype {quantized_dtype} not supported.')
    # max value is mapped to num_buckets

    if extract_diagonal and fvalue.ndim != 2:
      raise ValueError(
          f'Input array {fvalue} must be 2D to work with extract_diagonal.')

    diagonal_fvalue = []
    if extract_diagonal:
      diagonal_fvalue = jnp.diag(fvalue)
      # Remove the diagonal entries.
      fvalue = fvalue - jnp.diag(diagonal_fvalue)

    # TODO(rohananil): Extend this by making use of information about the blocks
    # SM3 style which will be useful for diagonal statistics
    # We first decide the scale.
    if fvalue.ndim < 1:
      raise ValueError(
          f'Input array {fvalue} must have a strictly positive number of '
          'dimensions.')

    max_abs = jnp.max(jnp.abs(fvalue), axis=0)
    bucket_size = max_abs / num_buckets
    bs_expanded = bucket_size[jnp.newaxis, Ellipsis]
    # To avoid divide by 0.0
    bs_nonzero = jnp.where(bs_expanded > 0.0, bs_expanded,
                           jnp.ones_like(bs_expanded))
    ratio = fvalue / bs_nonzero
    # We use rounding to remove bias.
    quantized = jnp.round(ratio)
    return quantized.astype(quantized_dtype), diagonal_fvalue, bucket_size

  def to_float(self):
    """Returns the float value."""
    if isinstance(self.quantized, list) and not self.quantized:
      return self.quantized

    if self.quantized_dtype == jnp.float32:
      return self.quantized

    if self.quantized_dtype == jnp.bfloat16:
      return self.quantized.astype(jnp.float32)

    float_dtype = self.bucket_size.dtype
    bucket_size = self.bucket_size[jnp.newaxis, Ellipsis]
    val = self.quantized.astype(float_dtype) * bucket_size
    if self.extract_diagonal:
      val += jnp.diag(self.diagonal)
    return val

