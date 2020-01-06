# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Common weight initializers used in the sparse transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf

from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import init_ops  # pylint: disable=g-direct-tensorflow-import


class SparseGlorotUniform(init_ops.Initializer):
  """Re-weighted glorot uniform initializer based on sparsity."""

  def __init__(self, sparsity, seed=None, dtype=tf.float32):
    if sparsity < 0.0 or sparsity >= 1.0:
      raise ValueError("sparsity must be in range [0.0, 1.0).")

    self.sparsity = sparsity
    self.seed = seed
    self.dtype = init_ops._assert_float_dtype(  # pylint: disable=protected-access
        dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    if partition_info is not None:
      raise ValueError("partition_info not supported.")
    if dtype is None:
      dtype = self.dtype

    if len(shape) != 2:
      raise ValueError("Weights must be 2-dimensional.")

    fan_in, fan_out = init_ops._compute_fans(shape)  # pylint: disable=protected-access

    # Calculate the number of non-zero weights in the weight matrix
    nnz = 1.0
    for d in shape:
      nnz *= d
    nnz *= 1 - self.sparsity

    limit = math.sqrt(6.0 / (nnz / fan_out + nnz / fan_in))
    return tf.random_uniform(
        shape,
        -limit,
        limit,
        dtype,
        seed=self.seed)

  def get_config(self):
    return {
        "seed": self.seed,
        "dtype": self.dtype.name
    }
