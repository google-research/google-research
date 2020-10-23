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

"""Defines a factorize encoder-decoder model for panoramas."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from factorize_a_city.libs import pano_transformer
from factorize_a_city.libs import sn_ops
from factorize_a_city.libs import spade
from factorize_a_city.libs import utils


def recomposite_from_log_components(log_reflectance, log_shading):
  """Combines log_reflectance and log_shading to produce an rgb image.

  I = R x S = e^(log_reflectance + log_shading)

  Args:
    log_reflectance: [B, H, W, 3] Log-reflectance image
    log_shading: [B, H, W, 1 or 3] Log-shading image

  Returns:
    An rgb image ranging from [0, 255] of shape [B, H, W, 3]
  """
  return tf.clip_by_value(tf.exp(log_reflectance + log_shading), 0., 255.)


def compute_log_reflectance(image, log_shading):
  """Computes the reflectance for each frame of an image given its shading.

  Args:
    image: [B, H, W, 3] rgb image from [0, 1] ranged
    log_shading: [B, H, W, 3] log-shading image

  Returns:
    A log-reflectance image [B, H, W, 3]
  """
  # Compute the reflectance which is done by:
  # log(image) - log(shading) = log(reflectance)
  return tf.math.log(tf.maximum(255 * image, 1e-3)) - log_shading


def reparameterize(mu, logvar):
  """Samples a latent representation from a multivariate distribution.

  Args:
    mu: [B, D] the mean of the multivariate distribution
    logvar: [B, D] the logvar of the multivariate distribution

  Returns:
    A tensor of shape [B, D] containing batch-wise samples that take advantage
    of the reparameterization trick.
  """
  std = tf.exp(0.5 * logvar)
  eps = tf.random.truncated_normal(tf.shape(mu))
  return eps * std + mu


class FactorizeEncoderDecoder():
  """Class for constructing encoder-decoder used for relighting scenes."""

  def __init__(self, params, is_training):
    # if is_training is true, disable spectral normalization updates
    self.is_training = is_training

    # Network configuration
    self.permanent_dim = params["permanent_dim"]
    self.lighting_dim = params["lighting_dim"]
    self.kernel_size = params.get("kernel_size", 3)
    self.normalization_permanent = utils.instance_normalization
    self.normalization_illumination = utils.instance_normalization
    self.intial_reshape = params.get("intial_spatial_reshape", [10, 30])

  def compute_decomposition(self,
                            aligned_stack,
                            single_image_decomposition=False,
                            average_stack=True):
    """Compute factors and intrinsic image components for aligned_stack.

    Run permanent and illumination encoders on aligned_stack and then generate a
    log shading image decoded from the encoded factors and a log reflectance
    image by subtracting the shading image. The resulting components correspond
    to the predicted intrinsic images of aligned_stack.

    Args:
      aligned_stack: [B, H, W, 3] a stack of panoramas that have been aligned.
      single_image_decomposition: (bool) if true, treats each frame of
        aligned_stack as an individual stack.
      average_stack: (bool) if true, the shared stack factors are computed by
        averaging the factors computed from each frame. When false, the shared
        factors are randomly sampled from a random individual frame.

    Returns:
      A dictionary containing the encoded stack factors:
        permanent_factor [B or 1, H // 8, W // 8, permanet_dim]
        individual_permanent_factor [B, H // 8, W // 8, permanet_dim]
        lighting_context_factor [B, lighting_dim]
        azimuth_factor [B, 40] where 40 is the bins over azimuth angles.
      and intrinsic image components:
        log_reflectance [1, H, W, 3] A stack-reduced representation for
          reflectance
        log_shading [B, H, W, 3] A shading image for each input image.
        individual_log_reflectance [B, H, W, 3] A reflectance image computed
          from each individual image.
    """
    individual_permanent_factors = self.extract_permanent(aligned_stack)
    batch_size = individual_permanent_factors.shape.as_list()[0]

    if not single_image_decomposition:
      if average_stack:
        # Average individual permanent_factor to get align_stack's
        # permanent_factor.
        shared_permanent_factor = tf.reduce_mean(
            individual_permanent_factors, axis=0, keep_dims=True)
      else:
        # Sample a random different frame's permanent_factor to be
        # aligned_stack's permanent_factor
        random_index = tf.random.uniform([1],
                                         minval=0,
                                         maxval=batch_size,
                                         dtype=tf.int32)
        shared_permanent_factor = tf.gather(individual_permanent_factors,
                                            random_index)
      permanent_factors = tf.tile(shared_permanent_factor,
                                  [batch_size, 1, 1, 1])
    else:
      permanent_factors = individual_permanent_factors

    temporal_factors = self.extract_illumination(aligned_stack)

    # Sample from lighting_context distribution and compute
    # expected average of the azimuth distribution.
    lighting_context = reparameterize(temporal_factors["mu"],
                                      temporal_factors["logvar"])
    azimuth_dist = temporal_factors["azimuth_distribution"]
    yaw_orientation = utils.compute_circular_average(azimuth_dist)

    # Generate log shading image for aligned_stack.
    shading_image = self.generate_shading_image(permanent_factors,
                                                lighting_context,
                                                yaw_orientation)
    # Compute log reflectance from log shading and aligned_stack.
    individual_log_reflectance = compute_log_reflectance(
        aligned_stack, shading_image)
    if not single_image_decomposition:
      log_reflectance = utils.reduce_median(
          individual_log_reflectance, keep_dims=True)
    else:
      log_reflectance = individual_log_reflectance[:1]
    return {
        "permanent_factor": permanent_factors,
        "individual_permanent_factor": individual_permanent_factors,
        "lighting_context_factor": lighting_context,
        "azimuth_factor": yaw_orientation,
        "log_reflectance": log_reflectance,
        "individual_log_reflectance": individual_log_reflectance,
        "log_shading": shading_image,
    }

  def extract_illumination(self, aligned_stack):
    """Predicts a style vector and sun azimuth distribution for aligned_stack.

    The illumination factors are predicted from individual panoramas. For each
    panorama we predict a distribution over style vectors using a VAE and a
    distribution over sun positions in the panorama with a fully-convolutional
    encoder.

    Args:
      aligned_stack: [B, H, W, 3] An aligned panorama stack.

    Returns:
      A dictionary containing (mu, logvar) style vectors of shape [B, D], where
      D is self.lighting_dim, and azimuth_distribution of shape [B, 40].
    """
    postconvfn_actnorm = lambda x, y: tf.nn.leaky_relu(
        self.normalization_illumination(x, y))
    pad_fn = lambda x, y: utils.pad_panorama_for_convolutions(x, y, "reflect")

    with tf.compat.v1.variable_scope("decomp_internal", reuse=tf.AUTO_REUSE):
      with tf.compat.v1.variable_scope("shading"):
        net = layers.conv2d(
            pad_fn(aligned_stack, self.kernel_size),
            16,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        net = pad_fn(postconvfn_actnorm(net, "enc1"), self.kernel_size)
        net = layers.conv2d(
            net,
            32,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        net = pad_fn(postconvfn_actnorm(net, "enc2"), self.kernel_size)
        net = layers.conv2d(
            net,
            64,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        net = pad_fn(postconvfn_actnorm(net, "enc3"), self.kernel_size)
        net = layers.conv2d(
            net,
            128,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)

        # Compute horizontal azimuth bins from a feature map by learning a
        # convolution layer that reduces over the height of the panorama. This
        # distribution is fully convolutional along the width of the panoramas.
        feature_map_height = net.shape.as_list()[1]
        with tf.compat.v1.variable_scope("azimuth"):
          hm_net = layers.conv2d(
              tf.nn.leaky_relu(net),
              1, [feature_map_height, 1],
              1,
              padding="VALID",
              activation_fn=None,
              normalizer_fn=None)
          azimuth_distribution = tf.nn.softmax(hm_net, axis=-2)
          azimuth_distribution = azimuth_distribution[:, 0, :, 0]
        net = pad_fn(postconvfn_actnorm(net, "enc4"), self.kernel_size)
        net = layers.conv2d(
            net,
            256,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        # Transform the spatial map into a global style vector.
        mean_net = tf.reduce_mean(net, axis=[1, 2])
        net = layers.fully_connected(mean_net, 128, activation_fn=None)

        net = layers.fully_connected(
            tf.nn.leaky_relu(net), 64, activation_fn=None)
        x = tf.nn.leaky_relu(net)

        # Predict parameters of a multivariate normal distribution.
        mu = layers.fully_connected(x, self.lighting_dim, activation_fn=None)
        logvar = layers.fully_connected(
            x, self.lighting_dim, activation_fn=None)
    return {
        "mu": mu,
        "logvar": logvar,
        "azimuth_distribution": azimuth_distribution
    }

  def extract_permanent(self, aligned_stack):
    """Encode aligned_stack to individual permanent factors.

    While a stack is expected to have a shared permanent factor, the permanent
    encoder predicts a permanent factor from each individual frame. The choice
    of extracting a single shared representation from multiple permanent factors
    is determined outside of this function.

    Args:
      aligned_stack: [B, H, W, 3] An aligned panorama stack.

    Returns:
      A [B, H//8, W//8, D] feature map where D is self.permanent_dim.

    """
    postconvfn_actnorm = lambda x, y: tf.nn.leaky_relu(
        self.normalization_permanent(x, y))
    pad_fn = lambda x, y: utils.pad_panorama_for_convolutions(x, y, "reflect")
    with tf.compat.v1.variable_scope("decomp_internal", reuse=tf.AUTO_REUSE):
      with tf.compat.v1.variable_scope("geometry"):
        net = layers.conv2d(
            pad_fn(aligned_stack, self.kernel_size),
            16,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        net = pad_fn(postconvfn_actnorm(net, "enc1"), self.kernel_size)
        net = layers.conv2d(
            net,
            32,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        net = pad_fn(postconvfn_actnorm(net, "enc2"), self.kernel_size)
        net = layers.conv2d(
            net,
            64,
            self.kernel_size,
            2,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        for i in range(2):
          net_pad = pad_fn(
              postconvfn_actnorm(net, "res1_%d" % i), self.kernel_size)
          res_net = layers.conv2d(
              net_pad,
              128,
              self.kernel_size,
              1,
              padding="VALID",
              activation_fn=None,
              normalizer_fn=None)

          res_net_pad = pad_fn(
              postconvfn_actnorm(res_net, "res2_%d" % i), self.kernel_size)
          res_net = layers.conv2d(
              res_net_pad,
              128,
              self.kernel_size,
              1,
              padding="VALID",
              activation_fn=None,
              normalizer_fn=None)

          skip_pad = pad_fn(
              postconvfn_actnorm(net, "skip_%d" % i), self.kernel_size)
          skip = layers.conv2d(
              skip_pad,
              128,
              self.kernel_size,
              1,
              padding="VALID",
              activation_fn=None,
              normalizer_fn=None)

          net = res_net + skip
        old_net = pad_fn(net, self.kernel_size)
        net = layers.conv2d(
            old_net,
            self.permanent_dim,
            self.kernel_size,
            1,
            padding="VALID",
            activation_fn=None,
            normalizer_fn=None)
        return net

  def generate_shading_image(self, permanent_factor, lighting_factor,
                             azimuth_factor):
    """Generate log shading images from the input factors.

    Given a set of input factors, decode a log shading image that can be used
    to relight a given scene. The generated shading image is comprised of a
    standard shading image as well as predictions of two global illuminations.

    The system can model lighter and darker areas as well as mixed exposure to
    different global illuminants such as being illuminated by the yellow
    sun and the blue sky.

    Args:
      permanent_factor: [B, H // 8, W // 8, D] A representation that encodes the
        scene property that we wish to relight. H and W are dimensions of the
        panorama stack used to encode permanent_factor.
      lighting_factor: [B, lighting_dim] A style vector sampled from the
        multivariate normal distribution predicted by the illumination encoder.
      azimuth_factor: [B] An angle in radians that describe the offset of the
        sun's position from the center of the image.

    Returns:
      A log shading image, of shape [B, H, W, D], of a scene encoded by
      permanent_factor illuminated by lighting_factor and azimuth_factor.
    """
    UPSAMPLING_FACTOR = 8
    factor_h, factor_w = permanent_factor.shape.as_list()[1:3]
    output_h = UPSAMPLING_FACTOR * factor_h
    output_w = UPSAMPLING_FACTOR * factor_w

    pad_fn = lambda x, y: utils.pad_panorama_for_convolutions(x, y, "reflect")
    filters_up = [256, 128, 128, 64]
    bsz = tf.shape(permanent_factor)[0]

    # Rotate the permanent_factor by azimuth_factor such that the generator's
    # input is azimuth-normalized representation.
    rot_permanent = pano_transformer.rotate_pano_horizontally(
        permanent_factor, azimuth_factor)

    # Positional encoding to break the generator's shift invariance.
    radian_vec = tf.linspace(-np.pi, np.pi,
                             output_w + 1)[tf.newaxis, :-1] + tf.zeros([bsz, 1])
    cos_vec = tf.cos(radian_vec)
    sin_vec = tf.sin(radian_vec)
    circular_embed = tf.stack([cos_vec, sin_vec], axis=-1)[:, None, :, :]
    circular_embed = tf.tile(circular_embed, [1, output_h, 1, 1])

    spade_normalization = lambda inp, scope: spade.sn_spade_normalize(
        inp, rot_permanent, circular_embed, scope, is_training=self.is_training)
    postconvfn_actnorm = lambda inp, scope: tf.nn.leaky_relu(
        spade_normalization(inp, scope))
    with tf.compat.v1.variable_scope("decomp_internal", reuse=tf.AUTO_REUSE):
      with tf.compat.v1.variable_scope("0_shade"):
        net = sn_ops.snlinear(
            lighting_factor,
            2 * self.lighting_dim,
            is_training=self.is_training,
            name="g_sn_enc0")
        net = tf.nn.leaky_relu(net)
        net = sn_ops.snlinear(
            net,
            4 * self.lighting_dim,
            is_training=self.is_training,
            name="g_sn_enc1")
        net = tf.nn.leaky_relu(net)
        net = sn_ops.snlinear(
            net,
            8 * self.lighting_dim,
            is_training=self.is_training,
            name="g_sn_enc2")
        net = tf.nn.leaky_relu(net)

        net = sn_ops.snlinear(
            net, (self.intial_reshape[0] * self.intial_reshape[1] *
                  self.lighting_dim),
            is_training=self.is_training,
            name="g_sn_enc3")

        # Decode global illuminations from the style vector.
        rgb1 = sn_ops.snlinear(
            tf.nn.leaky_relu(net),
            3,
            is_training=self.is_training,
            name="g_sn_rgb")
        rgb2 = sn_ops.snlinear(
            tf.nn.leaky_relu(net),
            3,
            is_training=self.is_training,
            name="g_sn_rgb2")
        rgb1 = tf.reshape(rgb1, [bsz, 1, 1, 3])
        rgb2 = tf.reshape(rgb2, [bsz, 1, 1, 3])

        # Reshape the vector to a spatial representation.
        netreshape = tf.reshape(net,
                                (bsz, self.intial_reshape[0],
                                 self.intial_reshape[1], self.lighting_dim))
        net = sn_ops.snconv2d(
            postconvfn_actnorm(netreshape, "enc"),
            256,
            1,
            1,
            is_training=self.is_training,
            name="g_sn_conv0")
        net = utils.upsample(net)

        for i in range(len(filters_up)):
          # Because k is different at each block, we need to learn skip.
          skip = postconvfn_actnorm(net, "skiplayer_%d_0" % i)
          skip = pad_fn(
              skip,
              self.kernel_size,
          )
          skip = sn_ops.snconv2d(
              skip,
              filters_up[i],
              self.kernel_size,
              1,
              is_training=self.is_training,
              name="g_skip_sn_conv%d_0" % i)

          net = postconvfn_actnorm(net, "layer_%d_0" % i)
          net = pad_fn(
              net,
              self.kernel_size,
          )
          net = sn_ops.snconv2d(
              net,
              filters_up[i],
              self.kernel_size,
              1,
              is_training=self.is_training,
              name="g_sn_conv%d_0" % i)

          net = postconvfn_actnorm(net, "layer_%d_1" % i)
          # net = pad_fn(net, self.kernel_size,)
          net = sn_ops.snconv2d(
              net,
              filters_up[i],
              1,
              1,
              is_training=self.is_training,
              name="g_sn_conv%d_1" % i)
          net = utils.upsample(net + skip)
        net = pad_fn(net, 5)
        # Predict a standard gray-scale log shading.
        monochannel_shading = sn_ops.snconv2d(
            net,
            1,
            5,
            1,
            name="output1",
            is_training=self.is_training,
            use_bias=False)
        # Predicts the influence of each global color illuminant at every pixel.
        mask_shading = tf.nn.sigmoid(
            sn_ops.snconv2d(
                net, 1, 5, 1, is_training=self.is_training, name="output2"))
        mixed_lights = mask_shading * rgb1 + (1 - mask_shading) * rgb2
        # Restore the original orientation of the sun position.
        log_shading = pano_transformer.rotate_pano_horizontally(
            monochannel_shading + mixed_lights, -azimuth_factor)
    return log_shading

  def generate_sun_rotation(self, permanent_factor, lighting_factor,
                            frame_rate):
    """Generate shading for a scene under a full rotation of sun positions.

    Given a scene described by permanent_factor and a desired style defined by
    lighting_factor, generate frame_rate sun positions interpolated over a
    full rotation.

    Args:
      permanent_factor: [1, H // 8, W // 8, D] A representation that encodes the
        scene property that we wish to relight. H and W are dimensions of the
        panorama stack used to encode permanent_factor and D is permanent_dim.
      lighting_factor: [1, lighting_dim] A style vector sampled from the
        multivariate normal distribution predicted by the illumination encoder.
      frame_rate: (int) The number of sun positions to generate.

    Returns:
      [frame_rate, H, W, 3] log shading images.
    """
    # Interpolate frame_rate azimuth factors sampled between -np.pi to np.pi.
    azimuth_factor_sample = tf.linspace(-np.pi, np.pi,
                                        frame_rate + 1)[:frame_rate]
    # Tile the permanent and lighting factor.
    geometry_factors = tf.tile(permanent_factor, [frame_rate, 1, 1, 1])
    lighting_factors = tf.tile(lighting_factor, [frame_rate, 1])

    return self.generate_shading_image(geometry_factors, lighting_factors,
                                       azimuth_factor_sample)
