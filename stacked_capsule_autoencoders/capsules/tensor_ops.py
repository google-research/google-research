# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tensor ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


def expand_dims(tensor, axis, n_dims=1):

  for _ in range(n_dims):
    tensor = tf.expand_dims(tensor, axis)
  return tensor


def make_brodcastable(tensor, against_tensor):
  n_dim_diff = against_tensor.shape.ndims - tensor.shape.ndims
  assert n_dim_diff >= 0
  return expand_dims(tensor, axis=-1, n_dims=n_dim_diff)


def py_func_metric(func, inputs, output_dtype=tf.float32):
  res = tf.py_func(func, inputs, [output_dtype], stateful=False)
  res = tf.reshape(res, [])
  return res


def ensure_length(x, length):
  """Enusres that the input is an array of a given length.

  Args:
    x: tensor or a list of tensors.
    length: int.
  Returns:
    list of tensors of a given length.
  """
  x = nest.flatten(x)
  if len(x) == 1:
    x *= length

  return x
