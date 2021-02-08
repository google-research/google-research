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

"""Util function to pad tensor shapes."""

import tensorflow as tf


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.
  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor

