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

"""Loss functions to train DirectionNet."""
import tensorflow.compat.v1 as tf
import util


def direction_loss(v_pred, v_true):
  """The direction loss measures the negative cosine similarity between vectors.

  Args:
    v_pred: [BATCH, 3] predicted unit vectors.
    v_true: [BATCH, 3] ground truth unit vectors.

  Returns:
    A float scalar.

  Raises:
    InvalidArgumentError: If 'v_pred' or 'v_true' is not a unit vector.
  """
  norm1 = tf.norm(v_pred, axis=-1)
  norm2 = tf.norm(v_true, axis=-1)
  one = tf.ones_like(norm1)
  with tf.control_dependencies([
      tf.compat.v1.assert_near(norm1, one),
      tf.compat.v1.assert_near(norm2, one)]):
    return -tf.reduce_mean(tf.reduce_sum(v_pred * v_true, -1))


def distribution_loss(p_pred, p_true):
  """The distribution loss measures the MSE between spherical distributions.

  Args:
    p_pred: [BATCH, HEIGHT, WIDTH, N] predicted spherical distributions.
    p_true: [BATCH, HEIGHT, WIDTH, N] ground truth spherical distributions.

  Returns:
    A float scalar.
  """
  height = p_pred.shape.as_list()[1]
  return tf.reduce_mean(
      util.equirectangular_area_weights(height) * (p_pred - p_true)**2)


def spread_loss(v_pred):
  """The spread loss penalizes the spherical “variance".

  Args:
    v_pred: [BATCH, 3] predicted unnormalized vectors.

  Returns:
    A float scalar.
  """
  return 1 - tf.reduce_mean(tf.norm(v_pred, axis=-1))
