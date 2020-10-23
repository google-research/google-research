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

"""Custom SPADE (Park et al.) library for running factorize_city."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from factorize_a_city.libs import sn_ops
from factorize_a_city.libs import utils


def diff_resize_area(tensor, new_height_width):
  """Performs a resize op that passes gradients evenly.

  The tensor goes through a resize and pool where the resize and pool
  operations are determined by the Least Common Multiplier. Since resize with
  nearest_neighbors and avg_pool distributes the gradients from the
  output to input evenly, there's less of a chance of learning artifacts. First
  we resize to the LCM then avg_pool to new_height_width. This resize operation
  is only efficient in cases where LCM is small. This is typically the case when
  upsampling or downsampling by a factor of 2 (e.g H = 0.5 * new_H).

  Args:
    tensor: a tensor of shape [B, H, W, D]
    new_height_width: A tuple of length two which specifies new height, width
      respectively.

  Returns:
    The resize area tensor [B, H_new, W_new, D].

  Raises:
    RuntimeError: If the LCM is larger than 10 x new_height_width, then
      raise an error to prevent inefficient memory usage.
  """
  new_h, new_w = new_height_width
  unused_b, curr_h, curr_w, unused_d = tensor.shape.as_list()
  # The least common multiplier used to determine the intermediate resize
  # operation.
  l_h = np.lcm(curr_h, new_h)
  l_w = np.lcm(curr_w, new_w)
  if l_h == curr_h and l_w == curr_w:
    im = tensor
  elif (l_h < (10 * new_h) and l_w < (10 * new_w)):
    im = tf.compat.v1.image.resize_bilinear(
        tensor, [l_h, l_w], half_pixel_centers=True)
  else:
    raise RuntimeError("DifferentiableResizeArea is memory inefficient"
                       "for resizing from (%d, %d) -> (%d, %d)" %
                       (curr_h, curr_w, new_h, new_w))
  lh_factor = l_h // new_h
  lw_factor = l_w // new_w
  if lh_factor == lw_factor == 1:
    return im
  return tf.nn.avg_pool2d(
      im, [lh_factor, lw_factor], [lh_factor, lw_factor], padding="VALID")


def sn_spade_normalize(tensor,
                       condition,
                       second_condition=None,
                       scope="spade",
                       is_training=False):
  """A spectral normalized version of SPADE.

  Performs SPADE normalization (Park et al.) on a tensor based on condition. If
  second_condition is defined, concatenate second_condition to condition.
  These inputs are separated because they encode conditioning of different
  things.

  Args:
    tensor: [B, H, W, D] a tensor to apply SPADE normalization to
    condition: [B, H', W', D'] A tensor used to predict SPADE's normalization
      parameters.
    second_condition: [B, H'', W'', D''] A tensor used to encode another kind of
      conditioning. second_condition is provided in case its dimensions do not
      natively match condition's dimension.
    scope: (str) The scope of the SPADE convolutions.
    is_training: (bool) used to control the spectral normalization update
      schedule. When true apply an update, else freeze updates.

  Returns:
    A SPADE normalized tensor of shape [B, H, W, D].
  """
  # resize condition to match input spatial
  n_tensor = layers.instance_norm(
      tensor, center=False, scale=False, trainable=False, epsilon=1e-4)
  unused_b, h_tensor, w_tensor, feature_dim = n_tensor.shape.as_list()
  with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):
    resize_condition = diff_resize_area(condition, [h_tensor, w_tensor])
    if second_condition is not None:
      second_condition = diff_resize_area(second_condition,
                                          [h_tensor, w_tensor])
      resize_condition = tf.concat([resize_condition, second_condition],
                                   axis=-1)
    resize_condition = utils.pad_panorama_for_convolutions(
        resize_condition, 3, "symmetric")
    condition_net = tf.nn.relu(
        sn_ops.snconv2d(
            resize_condition,
            32,
            3,
            1,
            is_training=is_training,
            name="intermediate_spade"))

    condition_net = utils.pad_panorama_for_convolutions(condition_net, 3,
                                                        "symmetric")
    gamma_act = sn_ops.snconv2d(
        condition_net,
        feature_dim,
        3,
        1,
        is_training=is_training,
        name="g_spade")
    mu_act = sn_ops.snconv2d(
        condition_net,
        feature_dim,
        3,
        1,
        is_training=is_training,
        name="b_spade")

    return n_tensor * (1 + gamma_act) - mu_act
