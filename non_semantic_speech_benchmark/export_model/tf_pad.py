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

"""A TensorFlow pad function that works like numpy.

Specifically, tf.pad can't more than double the length of a Tensor, while
numpy's can. For example:

x = np.array(range(3))
np.pad(x, [0, 5], mode='symmetric')
-> array([0, 1, 2, 2, 1, 0, 0, 1])

x = tf.constant(range(3))
tf.pad(x, [(0, 5)], mode='symmetric')
-> fails
"""

from typing import Union
import tensorflow as tf


def tf_pad(samples, padding,
           mode):
  if samples.shape.ndims != 2:
    raise ValueError(f'tensor must be rank 2: {samples.shape}')
  if mode == 'SYMMETRIC':
    return tf_pad_symmetric(samples, padding)
  else:
    return tf.pad(samples, [(0, 0), (0, padding)], mode=mode)


def tf_pad_symmetric(tensor, padding):
  """Symmetric pad a 2D Tensor."""
  if tensor.shape.ndims != 2:
    raise ValueError(f'tensor must be rank 2: {tensor.shape}')
  t_len = tf.shape(tensor)[1]
  return tf.cond(
      padding > t_len,
      lambda: _repeat_n_times_with_extra(tensor, padding, t_len),
      lambda: tf.pad(tensor, [(0, 0), (0, padding)], mode='SYMMETRIC'))


def _repeat_n_times_with_extra(tensor, padding,
                               t_len):
  """Pad symmetric longer than the original tensor."""
  assert tensor.shape.ndims == 2, tensor.ndims
  num_copies = tf.math.floordiv(padding, t_len)
  r = tf.reverse(tensor, axis=[1])
  f = tf.concat([r, tensor], axis=1)

  copies = tf.tile(f, [1, tf.math.floordiv(num_copies, 2)])
  copies = tf.cond(
      tf.math.mod(num_copies, 2) == 0,
      lambda: copies,
      lambda: tf.concat([copies, r], axis=1),
  )

  pre_pad_tensor = tf.concat([tensor, copies], axis=1)
  extra = tf.math.mod(padding, t_len)
  return tf.pad(pre_pad_tensor, paddings=[(0, 0), (0, extra)], mode='SYMMETRIC')
