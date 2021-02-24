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

"""Defines the initial image encoder and refinement network."""
import config
import ops
import spade
import tensorflow as tf


def reparameterize(mu, logvar):
  """Sample a random variable.

  Args:
    mu: Mean of normal noise to sample
    logvar: log variance of normal noise to sample

  Returns:
    Random Gaussian sampled from mu and logvar.
  """
  with tf.name_scope("reparameterization"):
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(std.get_shape())

    return eps * std + mu


def encoder(x, scope="spade_encoder"):
  """Encoder that outputs global N(mu, sig) parameters.

  Args:
    x: [B, H, W, 4] an RGBD image (usually the initial image) which is used to
      sample noise from a distirbution to feed into the refinement
      network. Range [0, 1].
    scope: (str) variable scope

  Returns:
    (mu, logvar) are [B, 256] tensors of parameters defining a normal
      distribution to sample from.
  """

  x = 2 * x - 1
  num_channel = 16

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = ops.sn_conv(x, num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_0")
    x = ops.instance_norm(x, scope="inst_norm_0")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 2 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_1")
    x = ops.instance_norm(x, scope="inst_norm_1")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 4 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_2")
    x = ops.instance_norm(x, scope="inst_norm_2")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_3")
    x = ops.instance_norm(x, scope="inst_norm_3")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_4")
    x = ops.instance_norm(x, scope="inst_norm_4")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_5")
    x = ops.instance_norm(x, scope="inst_norm_5")
    x = ops.leaky_relu(x, 0.2)

    mu = ops.fully_connected(x, config.DIM_OF_STYLE_EMBEDDING,
                             scope="linear_mu")
    logvar = ops.fully_connected(x, config.DIM_OF_STYLE_EMBEDDING,
                                 scope="linear_logvar")
  return mu, logvar


def refinement_network(rgbd, mask, z, scope="spade_generator"):
  """Refines rgbd, mask based on noise z.

  H, W should be divisible by 2 ** num_up_layers

  Args:
    rgbd: [B, H, W, 4] the rendered view to be refined
    mask: [B, H, W, 1] binary mask of unknown regions. 1 where known and 0 where
      unknown
    z: [B, D] a noise vector to be used as noise for the generator
    scope: (str) variable scope

  Returns:
    [B, H, W, 4] refined rgbd image.
  """
  img = 2 * rgbd - 1
  img = tf.concat([img, mask], axis=-1)

  num_channel = 32

  num_up_layers = 5
  out_channels = 4  # For RGBD

  batch_size, im_height, im_width, unused_c = rgbd.get_shape().as_list()

  init_h = im_height // (2 ** num_up_layers)
  init_w = im_width // (2 ** num_up_layers)

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = ops.fully_connected(z, 16 * num_channel * init_h * init_w,
                            "fc_expand_z")
    x = tf.reshape(x, [batch_size, init_h, init_w, 16 * num_channel])
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="head")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="middle_0")
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="middle_1")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 8 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_0")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 4 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_1")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 2 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_2")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 1 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_3")
    x = ops.leaky_relu(x, 0.2)
    # Pre-trained checkpoint uses default conv scoping.
    x = ops.sn_conv(x, out_channels, kernel_size=3)
    x = tf.tanh(x)
    return 0.5 * (x + 1)


