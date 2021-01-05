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

"""Glimpse network.

Glimpse network class builds feature representations based on glimpses of an
image and locations of glimpses.

The glimpse network receives two inputs image and location on the image. Image
locations are in the interval of [-1, 1] where points:
(-1, -1): upper left corner.
(-1, 1): upper right corner.
(1, 1): lower right corner.
(1, -1): lower left corner.
Glimpses are extracted with different resolutions at the provided location.
Then, glimpses are processed by convolutional layers.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from saccader import model_utils
from saccader import utils
from tensorflow_models.slim.nets import nets_factory


class GlimpseNetwork(object):
  """Glimpse network.

  Network that receives an image and location and extracts representation from
  a glimpse of the image at that location.

  Attributes:
    network_type: (String) Network type.
    glimpse_shape: 2-D tuple of integers indicating glimpse shape.
    num_resolutions: (Integer) Number of image resolutions used.
    output_dims: (Integer) indicating network output dimensionality.
    regularizer: Regularizer for network.
    normalization_type: Normalization for layers either None, 'batch', or
      'layer'.
    residual_blocks_per_group: (Integer) length of one residual block.
    number_groups: (Integer) number of residual groups.
    num_units_fc_layers: List with number of units for fully connected layers
      (None or []: no fully connect layers are used).
    init_conv_channels: Starting size of convolutional channels.
    widening_factor: Network widening factor from init_conv_channels (k variable
      in the wide resnet paper).
    dropout_rate: Number [0, 1] indicating drop out rate.
    activation: Activation function (None: for linear network).
    global_average_pool: (boolean) Use global average pooling.
    zero_pad: (boolean) Skip connection by zero padding or 1x1 convolution.
    var_list: list of network variables.
    init_op: initialization operations for model variables.
    residual_blocks_per_group: Number of residual blacks per group (N variable
      in the wide resnet paper arXiv:1605.07146).
    number_groups: Number of groups in the network.
    conv_size: (integer) Convolution filter size.
    apply_stop_gradient: (boolean) If True, prevent gradient from propagating
      through the glimpse extraction procedure.
  """

  def __init__(self, config):
    """Init.

    Args:
      config: ConfigDict object with model parameters (see dram_config.py).
    """
    self.network_type = config.network_type
    if self.network_type == "wrn":
      self.regularizer = config.regularizer
      self.normalization_type = config.normalization_type
      self.residual_blocks_per_group = config.residual_blocks_per_group
      self.number_groups = config.number_groups
      self.init_conv_channels = config.init_conv_channels
      self.widening_factor = config.widening_factor
      self.dropout_rate = config.dropout_rate
      self.zero_pad = config.zero_pad
      self.activation = config.activation
      self.global_average_pool = config.global_average_pool

    self.glimpse_shape = config.glimpse_shape
    self.num_resolutions = config.num_resolutions  # number of resolutions.
    self.output_dims = config.output_dims
    self.apply_stop_gradient = config.apply_stop_gradient

    # Call once to create network variables. Then reuse variables later.
    self.var_list = []
    self.init_op = []

  def collect_variables(self):
    """Collects model variables.

    Populates self.var_list with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.

    """
    self.var_list = [
        v for v in tf.global_variables() if "glimpse_network" in v.name
    ]
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def extract_glimpses(self, images, locations):
    """Extracts fovea-like glimpses.

    Args:
      images: 4-D Tensor of shape [batch, height, width, channels].
      locations: 2D Tensor of shape [batch, 2] with glimpse locations. Locations
        are in the interval of [-1, 1] where points:
        (-1, -1): upper left corner.
        (-1, 1): upper right corner.
        (1, 1): lower right corner.
        (1, -1): lower left corner.

    Returns:
      glimpses: 5D tensor of size [batch, # glimpses, height, width, channels].
    """
    # Get multi resolution fields of view (first is full resolution)
    image_shape = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
    start = tf.cast(self.glimpse_shape[0], dtype=tf.float32) / image_shape[0]
    fields_of_view = tf.cast(tf.lin_space(start, 1., self.num_resolutions),
                             dtype=tf.float32)
    receptive_fields = [self.glimpse_shape] + [
        tf.cast(fields_of_view[i] * image_shape, dtype=tf.int32)
        for i in range(1, self.num_resolutions)
    ]
    images_glimpses_list = []
    for field in receptive_fields:
      # Extract a glimpse with specific shape and scale.
      images_glimpse = utils.extract_glimpse(
          images, size=field, offsets=locations)
      # Bigger receptive fields have lower resolution.
      images_glimpse = tf.image.resize_images(
          images_glimpse, size=self.glimpse_shape)
      # Stop gradient
      if self.apply_stop_gradient:
        images_glimpse = tf.stop_gradient(images_glimpse)
      images_glimpses_list.append(images_glimpse)
    return images_glimpses_list

  def __call__(self, images, locations, is_training, use_resolution):
    """Builds glimpse network.

    Args:
      images: 4-D Tensor of shape [batch, height, width, channels].
      locations: 2D Tensor of shape [batch, 2] with glimpse locations.
      is_training: (Boolean) training or inference mode.
      use_resolution: (List of Boolean of size num_resolutions) Indicates which
        resolutions to use from high (small receptive field)
        to low (wide receptive field).

    Returns:
      output: Network output reflecting representation learned from glimpses
        and locations.
      endpoints: Dictionary with activations at different layers.
    """
    if self.var_list:
      reuse = True
    else:
      reuse = False

    tf.logging.info("Build Glimpse Network")
    endpoints = {}

    # Append position channels.
    images_with_position = tf.concat(
        [images, utils.position_channels(images)], axis=3)

    images_glimpses_list = self.extract_glimpses(images_with_position,
                                                 locations)
    endpoints["images_glimpses_list"] = [
        g[:, :, :, 0:3] for g in images_glimpses_list
    ]
    endpoints["model_input_list"] = images_glimpses_list
    # Concatenate along channels axis.
    images_glimpses_list_ = []
    for use, g in zip(use_resolution, images_glimpses_list):
      if not use:
        # If masking is required, use the spatial mean per channel.
        images_glimpses_list_.append(0. * g + tf.stop_gradient(
            tf.reduce_mean(g, axis=[1, 2], keepdims=True)))
      else:
        images_glimpses_list_.append(g)

    images_glimpses_list = images_glimpses_list_

    images_glimpses = tf.concat(images_glimpses_list, axis=3)
    net = images_glimpses

    if self.network_type == "wrn":
      with tf.variable_scope("glimpse_network", reuse=reuse):
        output, endpoints_ = model_utils.build_wide_residual_network(
            net,
            self.output_dims,
            residual_blocks_per_group=self.residual_blocks_per_group,
            number_groups=self.number_groups,
            init_conv_channels=self.init_conv_channels,
            widening_factor=self.widening_factor,
            dropout_rate=self.dropout_rate,
            expand_rate=2,
            conv_size=3,
            is_training=is_training,
            activation=self.activation,
            regularizer=self.regularizer,
            normalization_type=self.normalization_type,
            zero_pad=self.zero_pad,
            global_average_pool=self.global_average_pool)
    else:
      network = nets_factory.get_network_fn(
          self.network_type, num_classes=self.output_dims,
          is_training=is_training)
      output, endpoints_ = network(net, scope="glimpse_network", reuse=reuse)
      if self.output_dims is None:
        # Global average of activations.
        output = tf.reduce_mean(output, [1, 2])

    endpoints.update(endpoints_)

    if not reuse:
      self.collect_variables()

    return output, endpoints
