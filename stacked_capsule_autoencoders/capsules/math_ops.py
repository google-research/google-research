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

"""Math ops.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import sonnet as snt
import tensorflow.compat.v1 as tf


def relu1(x):
  return tf.nn.relu6(x * 6.) / 6.


def safe_log(tensor, eps=1e-16):
  is_zero = tf.less(tensor, eps)
  tensor = tf.where(is_zero, tf.ones_like(tensor), tensor)
  tensor = tf.where(is_zero, tf.zeros_like(tensor) - 1e8, tf.log(tensor))
  return tensor


def safe_ce(labels, probs, axis=-1):
  return tf.reduce_mean(-tf.reduce_sum(labels * safe_log(probs), axis=axis))


def flat_reduce(tensor, reduce_type='sum', final_reduce='mean'):
  """Flattens the tensor and reduces it."""

  def _reduce(tensor, how, *args):
    return getattr(tf, 'reduce_{}'.format(how))(tensor, *args)  # pylint:disable=not-callable

  tensor = snt.BatchFlatten()(tensor)
  tensor = _reduce(tensor, reduce_type, -1)
  if final_reduce is not None:
    tensor = _reduce(tensor, final_reduce)

  return tensor


def to_homogenous(tensor):
  one = tf.ones_like(tensor[Ellipsis, :1])
  return tf.concat([tensor, one], -1)


def from_homogenous(tensor):
  tensor = tensor[Ellipsis, :-1] / (tensor[Ellipsis, -1:] + 1e-8)
  return tensor


def apply_transform(transform, tensor=None, affine=True):
  """Applies a linear transform to a tensor.

  Returns the translation components of the transform if tensor=None.

  Args:
    transform: [..., d+1, d+1] tensor.
    tensor: [..., d] tensor or None.
    affine: boolean; assumes affine transformation if True and does a smaller
      matmul + offset instead of matmul.

  Returns:
    [..., d] tensor.
  """

  if tensor is None:
    # extract translation
    tensor = transform[Ellipsis, :-1, -1]

  elif affine:
    tensor = tf.matmul(tensor, transform[Ellipsis, :-1, :-1], transpose_b=True)
    tensor = (tensor + transform[Ellipsis, :-1, -1])

  else:
    tensor = to_homogenous(tensor)
    tensor = tf.matmul(tensor, transform, transpose_b=True)
    tensor = from_homogenous(tensor)

  return tensor


def geometric_transform(pose_tensor, similarity=False, nonlinear=True,
                        as_matrix=False):
  """Convers paramer tensor into an affine or similarity transform.

  Args:
    pose_tensor: [..., 6] tensor.
    similarity: bool.
    nonlinear: bool; applies nonlinearities to pose params if True.
    as_matrix: bool; convers the transform to a matrix if True.

  Returns:
    [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
  """

  scale_x, scale_y, theta, shear, trans_x, trans_y = tf.split(
      pose_tensor, 6, -1)

  if nonlinear:
    scale_x, scale_y = (tf.nn.sigmoid(i) + 1e-2
                        for i in (scale_x, scale_y))

    trans_x, trans_y, shear = (
        tf.nn.tanh(i * 5.) for i in (trans_x, trans_y, shear))

    theta *= 2. * math.pi

  else:
    scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

  c, s = tf.cos(theta), tf.sin(theta)

  if similarity:
    scale = scale_x
    pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]

  else:
    pose = [
        scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
        trans_x, scale_y * s, scale_y * c, trans_y
    ]

  pose = tf.concat(pose, -1)

  # convert to a matrix
  if as_matrix:
    shape = pose.shape[:-1].as_list()
    shape += [2, 3]
    pose = tf.reshape(pose, shape)
    zeros = tf.zeros_like(pose[Ellipsis, :1, 0])
    last = tf.stack([zeros, zeros, zeros + 1], -1)
    pose = tf.concat([pose, last], -2)

  return pose


def normalize(tensor, axis):
  return tensor / (tf.reduce_sum(tensor, axis, keepdims=True) + 1e-8)
