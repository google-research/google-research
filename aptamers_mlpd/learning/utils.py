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

# Lint as: python3
"""Utility functions for learning code."""
import functools

import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor


@functools.partial(lt.define_reduce_op, 'reduce_nanmean')
def reduce_nanmean(tensor, axes=None, keepdims=False, name=None):
  """Take the mean of a tensor, skipping NaNs.

  Args:
    tensor: tensor to reduce.
    axes: optional list of axes to reduce.
    keepdims: optional boolean indicating whether to keep dimensions or not.
    name: optional op name.

  Returns:
    tf.Tensor with reduce values.
  """
  masked = tf.is_nan(tensor)
  valid_tensor = tf.where(masked, tf.zeros_like(tensor), tensor)
  total = tf.reduce_sum(valid_tensor, axes, keepdims=keepdims)
  counts = tf.reduce_sum(tf.cast(tf.logical_not(masked), tensor.dtype),
                         axes, keepdims=keepdims)
  return tf.div(total, counts, name=name)
