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

"""Library for defining SPADE (Park et al) components."""
import numpy as np
import ops
import tensorflow as tf


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


def spade(x,
          condition,
          num_hidden=128,
          use_spectral_norm=False,
          scope="spade"):
  """Spatially Adaptive Instance Norm implementation.

  Given x, applies a normalization that is conditioned on condition.

  Args:
    x: [B, H, W, C] A tensor to apply normalization
    condition: [B, H', W', C'] A tensor to condition the normalization
      parameters
    num_hidden: (int) The number of intermediate channels to create the SPADE
      layer with
    use_spectral_norm: (bool) If true, creates convolutions with spectral
      normalization applied to its weights
    scope: (str) The variable scope

  Returns:
    A tensor that has been normalized by parameters estimated by cond.
  """
  channel = x.shape[-1]
  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x_normed = ops.instance_norm(x)

    # Produce affine parameters from conditioning image.
    # First resize.
    height, width = x.get_shape().as_list()[1:3]

    condition = diff_resize_area(condition, [height, width])
    condition = ops.sn_conv(
        condition,
        num_hidden,
        kernel_size=3,
        use_spectral_norm=use_spectral_norm,
        scope="conv_cond")
    condition = tf.nn.relu(condition)
    gamma = ops.sn_conv(condition, channel, kernel_size=3,
                        use_spectral_norm=use_spectral_norm, scope="gamma",
                        pad_type="CONSTANT")
    beta = ops.sn_conv(condition, channel, kernel_size=3,
                       use_spectral_norm=use_spectral_norm, scope="beta",
                       pad_type="CONSTANT")

    out = x_normed * (1 + gamma) + beta
    return out


def spade_resblock(tensor,
                   condition,
                   channel_out,
                   use_spectral_norm=False,
                   scope="spade_resblock"):
  """A SPADE resblock.

  Args:
    tensor: [B, H, W, C] image to be generated
    condition: [B, H, W, D] conditioning image to compute affine
      normalization parameters.
    channel_out: (int) The number of channels of the output tensor
    use_spectral_norm: (bool) If true, use spectral normalization in conv layers
    scope: (str) The variable scope

  Returns:
    The output of a spade residual block
  """

  channel_in = tensor.get_shape().as_list()[-1]
  channel_middle = min(channel_in, channel_out)

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = spade(tensor, condition,
              use_spectral_norm=use_spectral_norm, scope="spade_0")
    x = ops.leaky_relu(x, 0.2)
    # This one always uses spectral norm.
    x = ops.sn_conv(x, channel_middle, kernel_size=3,
                    use_spectral_norm=True, scope="conv_0")

    x = spade(x, condition,
              use_spectral_norm=use_spectral_norm, scope="spade_1")
    x = ops.leaky_relu(x, 0.2)
    x = ops.sn_conv(x, channel_out, kernel_size=3,
                    use_spectral_norm=True, scope="conv_1")

    if channel_in != channel_out:
      x_in = spade(tensor, condition,
                   use_spectral_norm=use_spectral_norm, scope="shortcut_spade")
      x_in = ops.sn_conv(x_in, channel_out, kernel_size=1, stride=1,
                         use_bias=False, use_spectral_norm=True,
                         scope="shortcut_conv")
    else:
      x_in = tensor

    out = x_in + x

  return out
