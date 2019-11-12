# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility methods for models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers

MOMENTUM = 0.9
EPS = 1e-5


def pool2d_layer(inputs, pool_type, pool_size=2, pool_stride=2):
  """Pooling layer.

  Args:
    inputs: Tensor of size [batch, H, W, channels].
    pool_type: String ("max", or "average"), specifying pooling type.
    pool_size: Integer > 1 pooling size.
    pool_stride: Integer > 1 pooling stride.

  Returns:
    Pooling result.
  """
  if pool_type == "max":
    # Max pooling layer
    return tf.layers.max_pooling2d(
        inputs, pool_size=[pool_size] * 2, strides=pool_stride)

  elif pool_type == "average":
    # Average pooling layer
    return tf.layers.average_pooling2d(
        inputs, pool_size=[pool_size] * 2, strides=pool_stride)


def build_conv_pool_layers(inputs,
                           filter_sizes_conv_layers,
                           activation=None,
                           pool_params=None,
                           batch_norm=False,
                           regularizer=None,
                           is_training=False):
  """Builds convolution and pooling layers.

  Args:
    inputs: Tensor of size [batch, H, W, channels].
    filter_sizes_conv_layers: List of tuples of length number of convolutional
      layers. Each tuple is of two elements; the first is an integer
      representing 2D convolution kernel size and the second is an integer
      representing the number of convolutional channels.
    activation: Activation function.
    pool_params: Dictionary of three elements {"type": string 'max' or
      'average', "size": integer pooling size, "stride": integer pooling
        stride}. If None no pooling used.
    batch_norm: (boolean) It true add batch normalization layers.
    regularizer: Regularize function for the convolutional kernels.
    is_training: (boolean) Indication training or inference mode.

  Returns:
    net: Output after applying the layers.
    endpoints: Dictionary with activations at different points of the model.
  """
  tf.logging.info("==========================================================")
  endpoints = {}
  net = inputs
  for i, filter_size in enumerate(filter_sizes_conv_layers):
    layer_suffix = "_layer%d" % i
    # Convolution layer.
    tf.logging.info("________________________________________________________")
    tf.logging.info("                       Convolution layer %d" % i)

    if batch_norm:
      net = tf.layers.conv2d(
          net,
          filters=filter_size[1],
          kernel_size=filter_size[0],
          strides=(1, 1),
          padding="same",
          activation=None,
          kernel_regularizer=regularizer,
          use_bias=False,
          name="conv" + layer_suffix)
      net = tf.layers.batch_normalization(
          net,
          training=is_training,
          momentum=MOMENTUM,
          epsilon=EPS,
          name="conv_batch_norm" + layer_suffix)
      if activation:
        net = activation(net)
    else:
      net = tf.layers.conv2d(
          net,
          filters=filter_size[1],
          kernel_size=filter_size[0],
          strides=(1, 1),
          padding="same",
          activation=activation,
          kernel_regularizer=regularizer,
          name="conv" + layer_suffix)

    endpoints["conv" + layer_suffix] = net

    tf.logging.info(net.shape)
    tf.logging.info("____________________________________________________")

    # Pooling layer.
    if pool_params:
      tf.logging.info("__________________________________________________")
      tf.logging.info("         %s Pooling layer %d" % (pool_params["type"], i))
      net = pool2d_layer(
          net,
          pool_type=pool_params["type"],
          pool_size=pool_params["size"],
          pool_stride=pool_params["stride"])
      endpoints["pool" + layer_suffix] = net
      tf.logging.info(net.shape)
      tf.logging.info("__________________________________________________")

  tf.logging.info("==========================================================")
  return net, endpoints


def build_fc_layers(inputs,
                    num_units_fc_layers,
                    activation=None,
                    batch_norm=False,
                    regularizer=None,
                    is_training=False):
  """Builds fully connected layers.

  Args:
    inputs: Tensor of size [batch, channels].
    num_units_fc_layers: List of integers of length number of layers. Each
      element is the number of units for each fully connect layer.
    activation: Activation function.
    batch_norm: (boolean) It true add batch normalization layers.
    regularizer: Regularize function for the layer weights.
    is_training: (boolean) Indication training or inference mode.

  Returns:
    net: Output after applying the layers.
    endpoints: Dictionary with activations at different points of the model.


  """
  tf.logging.info("=========================================================")
  endpoints = {}
  net = inputs
  for i, num_units in enumerate(num_units_fc_layers):
    layer_suffix = "_layer%d" % i
    tf.logging.info("_______________________________________________________")
    tf.logging.info("                     Fully connected layer %d" % i)
    tf.logging.info("_____________________________________________________")

    if batch_norm:
      net = tf.layers.dense(
          net,
          num_units,
          activation=None,
          name="fc" + layer_suffix,
          kernel_regularizer=regularizer,
          use_bias=False)
      net = tf.layers.batch_normalization(
          net,
          momentum=MOMENTUM,
          epsilon=EPS,
          training=is_training,
          name="fc_batch_norm" + layer_suffix)
      if activation:
        net = activation(net)
    else:
      net = tf.layers.dense(
          net,
          num_units,
          activation=activation,
          name="fc" + layer_suffix,
          kernel_regularizer=regularizer)

    endpoints["fc" + layer_suffix] = net
    tf.logging.info(net.shape)
    tf.logging.info("________________________________________________________")

  tf.logging.info("==========================================================")
  return net, endpoints


def normalize_layer(net, normalization_type=None, is_training=False):
  """Applies normalization to network layer."""
  if normalization_type:
    if normalization_type == "batch":
      # Batch normalization
      net = tf.layers.batch_normalization(
          net, momentum=MOMENTUM, epsilon=EPS, training=is_training)
    elif normalization_type == "layer":
      # Layer normalization
      net = contrib_layers.layer_norm(net)
    else:
      raise ValueError(
          "normalization_type can be either None, 'batch', or 'layer'")
  return net


def residual_layer(x,
                   conv_size,
                   conv_channels,
                   stride=1,
                   dropout_rate=0,
                   regularizer=None,
                   activation=None,
                   normalization_type=None,
                   is_training=False):
  """Builds a residual layer.


  Args:
    x: Input tensor of size batch x height width x channels.
    conv_size: Convolution filter size.
    conv_channels: Feature channels.
    stride: Convolution stride.
    dropout_rate: Number [0, 1] indicating drop out rate.
    regularizer: Passing a regularizer to the weights.
    activation: Activation function to use.
    normalization_type: If None no normalization is used. This can be 'batch',
      'layer', or None.
    is_training: (Boolean) training or inference mode.

  Returns:
    net: Network activation after the residual block.
  """

  # Assumes channels last.
  num_input_channels = x.shape.as_list()[-1]
  net = x
  skip = x
  # Convolution
  net = tf.layers.conv2d(
      net,
      filters=conv_channels,
      kernel_size=(conv_size, conv_size),
      strides=(stride, stride),
      padding="same",
      activation=None,
      kernel_regularizer=regularizer,
      use_bias=normalization_type is None)

  # Drop out.
  if dropout_rate > 0:
    net = tf.layers.dropout(
        net,
        rate=dropout_rate,
        noise_shape=None,
        training=is_training,
    )

  # Normalize layer.
  net = normalize_layer(
      net, normalization_type=normalization_type, is_training=is_training)

  # Activation function.
  if activation:
    net = activation(net)

  # Convolution layer.
  net = tf.layers.conv2d(
      net,
      filters=conv_channels,
      kernel_size=(conv_size, conv_size),
      strides=(1, 1),
      padding="same",
      activation=None,
      kernel_regularizer=regularizer,
      use_bias=normalization_type is None)

  # Normalize layer.
  net = normalize_layer(
      net, normalization_type=normalization_type, is_training=is_training)

  # Merge with skip connection.
  if num_input_channels != conv_channels:
    skip = tf.layers.conv2d(
        skip,
        filters=conv_channels,
        kernel_size=(1, 1),
        strides=(stride, stride),
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
        use_bias=normalization_type is None)

  net += skip

  # Activation function.
  if activation:
    net = activation(net)
  return net


def add_residual(transformed_inputs, original_inputs, zero_pad=True):
  """Adds a skip branch to residual block to the output."""
  original_shape = original_inputs.shape.as_list()
  transformed_shape = transformed_inputs.shape.as_list()

  delta = transformed_shape[3] - original_shape[3]
  stride = int(np.ceil(original_shape[1] / transformed_shape[1]))
  if stride > 1:
    original_inputs = tf.layers.average_pooling2d(
        original_inputs, pool_size=[stride] * 2, strides=stride, padding="same")

  if delta != 0:
    if zero_pad:
      # Pad channels with zeros at the beginning and end.
      if delta > 0:
        original_inputs = tf.pad(
            original_inputs, [[0, 0], [0, 0], [0, 0], [delta // 2, delta // 2]],
            mode="CONSTANT",
            constant_values=0)
      else:
        transformed_inputs = tf.pad(
            transformed_inputs, [
                [0, 0], [0, 0], [0, 0], [-delta // 2, -delta // 2]],
            mode="CONSTANT",
            constant_values=0)
    else:
      # Convolution
      original_inputs = tf.layers.conv2d(
          original_inputs,
          filters=transformed_shape[3],
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="same",
          activation=None,
          use_bias=False)
  net = original_inputs + transformed_inputs
  return net, original_inputs


def wide_residual_layer(x,
                        conv_size,
                        conv_channels,
                        stride=1,
                        dropout_rate=0,
                        regularizer=None,
                        activation=None,
                        normalization_type=None,
                        is_training=False,
                        activation_before_residual=False,
                        zero_pad=True,
                        conv_scope="",
                        normalization_scope=""):
  """Builds a wide residual layer.


  Args:
    x: Input tensor of shape [batch_size, height, width, channels].
    conv_size: Convolution filter size.
    conv_channels: Feature channels.
    stride: Convolution stride.
    dropout_rate: Number in the interval [0, 1] indicating drop out rate.
    regularizer: Passing a regularizer to the weights.
    activation: Activation function to use.
    normalization_type: If None no normalization is used. This can be 'batch',
      'layer', or None.
    is_training: (Boolean) training or inference mode.
    activation_before_residual: Whether to place activation before the residual
      branch or not.
    zero_pad: (boolean) Skip connection by zero padding or 1x1 convolution.
    conv_scope: Convolution layers variable scope.
    normalization_scope: Normalization layers variable scope.

  Returns:
    A pair (net, skip) where net is the network activation after the
    residual block, and skip is the residual skip connection.
  """
  net = x
  with tf.variable_scope(normalization_scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("a"):
      net = normalize_layer(
          net, normalization_type=normalization_type, is_training=is_training)

  if activation:
    net = activation(net)

  skip = net if activation_before_residual else x

  # Convolution
  with tf.variable_scope(conv_scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("a"):
      net = tf.layers.conv2d(
          net,
          filters=conv_channels,
          kernel_size=(conv_size, conv_size),
          strides=(stride, stride),
          padding="same",
          activation=None,
          kernel_regularizer=regularizer,
          use_bias=normalization_type is None)

    # Drop out.
    if dropout_rate > 0:
      net = tf.layers.dropout(
          net,
          rate=dropout_rate,
          noise_shape=None,
          training=is_training,
      )

  with tf.variable_scope(normalization_scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("b"):
      # Normalize layer.
      net = normalize_layer(
          net, normalization_type=normalization_type, is_training=is_training)

  # Activation function.
  if activation:
    net = activation(net)

  with tf.variable_scope(conv_scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("b"):
      # Convolution layer.
      net = tf.layers.conv2d(
          net,
          filters=conv_channels,
          kernel_size=(conv_size, conv_size),
          strides=(1, 1),
          padding="same",
          activation=None,
          kernel_regularizer=regularizer,
          use_bias=normalization_type is None)

  # Merge with skip connection.
  net, skip = add_residual(net, skip, zero_pad)
  return net, skip


def build_wide_residual_network(x,
                                num_classes,
                                residual_blocks_per_group=6,
                                number_groups=3,
                                conv_size=3,
                                init_conv_channels=16,
                                widening_factor=10,
                                expand_rate=2,
                                dropout_rate=0,
                                regularizer=None,
                                activation=None,
                                normalization_type=None,
                                is_training=False,
                                zero_pad=True,
                                global_average_pool=True):
  """Builds a wide residual network.

  Each group consists of residual_blocks_per_group residual blocks.
  Depth = 6 * residual_blocks_per_group + 4
  For example, WRN-40-10, has 40 layers, 6 residual_blocks_per_group, and 10
  widening_factor.

  The conv_scope and normalization_scope here allows the reuse of convolution
  and the normalization parameters, if needed.

  Args:
    x: Input tensor of shape [batch_size, height, width, channels].
    num_classes: (integer) Number of classes, or None to output the network
      activity before the final linear layer.
    residual_blocks_per_group: Number of residual blacks per group (N variable
      in the wide resnet paper arXiv:1605.07146).
    number_groups: Number of groups in the network.
    conv_size: (integer) Convolution filter size.
    init_conv_channels: Starting size of convolutional channels.
    widening_factor: Network widening factor from init_conv_channels (k variable
      in the wide resnet paper).
    expand_rate: Expansion rate for convolutional channel from one group to the
      next.
    dropout_rate: Number [0, 1] indicating drop out rate.
    regularizer: A tensorflow regularizer for weights (such as tf.nn.l2_loss).
    activation: Activation function to use.
    normalization_type: If None no normalization is used. This can be 'batch',
      'layer', or None.
    is_training: (boolean) Training or inference mode.
    zero_pad: (boolean) Skip connection by zero padding or 1x1 convolution.
    global_average_pool: (boolean) Use global average pooling at the prelogits.

  Returns:
    A pair (logits, endpoints) where logits is network output logits, and
    endpoints is a dictionary with different parts of the model.
  """
  # Image batch x h x w x 3 (h and w are assumed to be the same).
  image_size = x.shape.as_list()[1]
  endpoints = {}
  endpoints["images"] = x
  # Initial convolution.
  with tf.variable_scope("init_conv"):
    conv_layer_index = 0
    while image_size > 32:
      x = tf.layers.conv2d(
          x,
          filters=init_conv_channels,
          kernel_size=conv_size,
          strides=(2, 2),
          padding="same",
          activation=None,
          kernel_regularizer=regularizer,
          name="init_conv_%d" % conv_layer_index)
      tf.logging.info(x)
      conv_layer_index += 1
      image_size = image_size // 2

    net = x
    net = tf.layers.conv2d(
        net,
        filters=init_conv_channels,
        kernel_size=conv_size,
        strides=(1, 1),
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer,
        name="init_conv",
        use_bias=normalization_type is None)

  endpoints["init_conv"] = net

  net_init = net
  net_prev_block = net  # Res from previous block
  conv_channels = init_conv_channels * widening_factor
  stride = 1

  for i in range(number_groups):
    tf.logging.info("Residual group %d" % i)
    for j in range(residual_blocks_per_group):
      layer_name = "residual_block_%d_%d" % (i, j)
      net, _ = wide_residual_layer(
          net,
          conv_size=conv_size,
          conv_channels=conv_channels,
          stride=stride if j == 0 else 1,
          dropout_rate=dropout_rate,
          regularizer=regularizer,
          activation=activation,
          normalization_type=normalization_type,
          is_training=is_training,
          activation_before_residual=(i == 0) and (j == 0),
          zero_pad=zero_pad,
          conv_scope=layer_name + "/conv",
          normalization_scope=layer_name + "/norm")
      endpoints[layer_name] = net
      tf.logging.info(layer_name)
      tf.logging.info(net)

    net, net_prev_block = add_residual(net, net_prev_block, zero_pad)
    conv_channels *= int(expand_rate)
    stride = int(expand_rate)

  net, _ = add_residual(net, net_init, zero_pad)
  # Output layer.
  with tf.variable_scope("output_layer"):
    # Normalize layer.
    net = normalize_layer(
        net, normalization_type=normalization_type, is_training=is_training)
    # Activation function.
    if activation:
      net = activation(net)

    # Global average_pooling.
    if global_average_pool:
      net = tf.reduce_mean(net, axis=[1, 2], name="global_average_pool")
      endpoints["global_average_pool"] = net
    else:
      net = tf.layers.flatten(net)
      endpoints["prelogits"] = net

    if num_classes:
      # Linear layer.
      output = tf.layers.dense(
          net,
          num_classes,
          activation=None,
          name="logits",
          kernel_regularizer=regularizer)
    else:
      # Prelogits output.
      output = net

  return output, endpoints


def build_recurrent_wide_residual_network(x,
                                          num_classes,
                                          num_times=5,
                                          number_groups=3,
                                          conv_size=3,
                                          init_conv_channels=16,
                                          widening_factor=10,
                                          expand_rate=2,
                                          dropout_rate=0,
                                          regularizer=None,
                                          activation=tf.nn.relu,
                                          normalization_type="batch",
                                          is_training=False,
                                          zero_pad=True):
  """Builds a recurrent wide residual network.

  Each group consists of (num_times+1) residual blocks.
  Depth = 6 * (num_times+1) + 4
  For example, RWRN-40-10, has 40 layers, 6 residual blocks per group, and 10
  widening_factor. For each group, the weights starting the second residual
  block are shared (except normalization layer params).

  Args:
    x: Input tensor of shape [batch, height, width, channels].
    num_classes: (integer) Number of classes, or None to output the network
      activity before the final linear layer.
    num_times: Number of time unrolling of residual blacks per group (N-1
      variable in the wide resnet paper arXiv:1605.07146).
    number_groups: Number of groups in the network.
    conv_size: (Integer) with convolution filter size.
    init_conv_channels: Starting size of convolutional channels.
    widening_factor: Network widening factor from init_conv_channels (k variable
      in the wide resnet paper).
    expand_rate: Expansion rate for convolutional channel from one group to the
      next.
    dropout_rate: Number in the interval [0, 1] indicating drop out rate.
    regularizer: A tensorflow regularizer for weights (such as tf.nn.l2_loss).
    activation: Activation function to use.
    normalization_type: If None no normalization is used. This can be 'batch',
      'layer', or None.
    is_training: (Boolean) training or inference mode.
    zero_pad: (boolean) Skip connection by zero padding or 1x1 convolution.

  Returns:
    A pair (logits, endpoints) where logits is network output logits, and
    endpoints is a dictionary with different parts of the model.
  """
  # Image batch x h x w x 3.
  net = x
  endpoints = {}
  # Initial convolution.
  with tf.variable_scope("init_conv"):
    net = tf.layers.conv2d(
        net,
        filters=init_conv_channels,
        kernel_size=(conv_size, conv_size),
        strides=(1, 1),
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer,
        use_bias=normalization_type is None)
  net_init = net
  net_prev_block = net  # Res from previous block
  conv_channels = init_conv_channels * widening_factor
  stride = 1

  for i in range(number_groups):
    tf.logging.info("Residual group %d" % i)
    for j in range(num_times + 1):
      layer_name = "residual_block_%d_%d" % (i, j)
      normalization_scope = layer_name + "/norm"
      if j == 0:
        conv_scope = layer_name + "/conv"
      else:
        # Reuse conv scope.
        conv_scope = "residual_block_%d_t" % (i) + "/conv"

      net, _ = wide_residual_layer(
          net,
          conv_size=conv_size,
          conv_channels=conv_channels,
          stride=stride if j == 0 else 1,
          dropout_rate=dropout_rate,
          regularizer=regularizer,
          activation=activation,
          normalization_type=normalization_type,
          is_training=is_training,
          activation_before_residual=(i == 0) and (j == 0),
          zero_pad=zero_pad,
          conv_scope=conv_scope,
          normalization_scope=normalization_scope)
      endpoints[layer_name] = net
      tf.logging.info(layer_name)
      tf.logging.info(net)

    net, net_prev_block = add_residual(net, net_prev_block, zero_pad)
    conv_channels *= int(expand_rate)
    stride = int(expand_rate)

  net, _ = add_residual(net, net_init, zero_pad)
  with tf.variable_scope("output_layer"):
    # Normalize layer.
    net = normalize_layer(
        net, normalization_type=normalization_type, is_training=is_training)
  # Activation function.
  if activation:
    net = activation(net)

  # Global average_pooling.
  net = tf.reduce_mean(net, axis=[1, 2])
  endpoints["global_average_pool"] = net

  if num_classes:
    # Output layer.
    with tf.variable_scope("output_layer"):
      output = tf.layers.dense(
          net,
          num_classes,
          activation=None,
          name="logits",
          kernel_regularizer=regularizer)
  else:
    # Prelogits output.
    output = net

  return output, endpoints


def cifar10_generator(z, is_training, activation=tf.nn.relu):
  """Generator for CIFAR10.

  Args:
    z: 2D Tensor of shape [batch, latent dimension].
    is_training: (boolean) Training or inference mode.
    activation: Tensorflow activation function.

  Returns:
    Reconstructed images of size [batch, 32, 32, 3].
  """

  start_shape = (8, 8, 128)
  with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
    layer_suffix = "_layer0"
    net = tf.layers.dense(
        z,
        np.prod(start_shape),
        activation=None,
        name="fc" + layer_suffix,
        use_bias=False)
    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="fc_batch_norm" + layer_suffix)
    net = activation(net)

    net = tf.reshape(net, (-1,) + start_shape)

    # Layer 1.
    layer_suffix = "_layer1"
    net = tf.layers.conv2d(
        net,
        filters=128,
        kernel_size=4,
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=False,
        name="conv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="conv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 2.
    layer_suffix = "_layer2"
    net = tf.layers.conv2d_transpose(
        net,
        filters=128,
        kernel_size=4,
        strides=(2, 2),
        padding="same",
        activation=None,
        use_bias=False,
        name="deconv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="deconv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 3.
    layer_suffix = "_layer3"
    net = tf.layers.conv2d(
        net,
        filters=128,
        kernel_size=5,
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=False,
        name="conv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="conv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 4.
    layer_suffix = "_layer4"
    net = tf.layers.conv2d_transpose(
        net,
        filters=128,
        kernel_size=4,
        strides=(2, 2),
        padding="same",
        activation=None,
        use_bias=False,
        name="deconv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="deconv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 5.
    layer_suffix = "_layer5"
    net = tf.layers.conv2d(
        net,
        filters=128,
        kernel_size=5,
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=False,
        name="conv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="conv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 6.
    layer_suffix = "_layer6"
    net = tf.layers.conv2d(
        net,
        filters=128,
        kernel_size=5,
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=False,
        name="conv" + layer_suffix)

    net = tf.layers.batch_normalization(
        net,
        momentum=MOMENTUM,
        epsilon=EPS,
        training=is_training,
        name="conv_batch_norm" + layer_suffix)
    net = activation(net)

    # Layer 7.
    layer_suffix = "_layer7"
    images = tf.layers.conv2d(
        net,
        filters=3,
        kernel_size=5,
        strides=(1, 1),
        padding="same",
        activation=tf.nn.tanh,
        use_bias=False,
        name="conv" + layer_suffix)

    biases = tf.get_variable(
        name="channel_biases",
        shape=(1, 1, 3),
        initializer=tf.zeros_initializer())
    scales = tf.get_variable(
        name="channel_scales",
        shape=(1, 1, 3),
        initializer=tf.zeros_initializer())

  images = images * scales
  images = images - biases
  return images
