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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Euclidean utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


def euc_sq_distance(x, y, eval_mode=False, rhs_dep_lhs=False):
  """Computes Euclidean squared distance.

  Args:
    x: Tensor of size B1 x d
    y: Tensor of size (B1 x) B2 x d if rhs_dep_lhs = False (True)
    eval_mode: boolean indicating whether to compute all pairwise distances or
      not. If eval_mode=False, must have B1=B2.
    rhs_dep_lhs: boolean indicating the shape of y.

  Returns:
    Tensor of size B1 x B2 if eval_mode=True, otherwise Tensor of size B1 x 1.
  """
  x2 = tf.math.reduce_sum(x * x, axis=-1, keepdims=True)
  y2 = tf.math.reduce_sum(y * y, axis=-1, keepdims=True)
  if eval_mode:
    if rhs_dep_lhs:
      y2 = tf.squeeze(y2)
      xy = tf.squeeze(
          tf.matmul(tf.expand_dims(x, 1), tf.transpose(y, perm=[0, 2, 1])))
    else:
      y2 = tf.transpose(y2)
      xy = tf.linalg.matmul(x, y, transpose_b=True)
  else:
    xy = tf.math.reduce_sum(x * y, axis=-1, keepdims=True)
  return x2 + y2 - 2 * xy
