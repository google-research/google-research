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

"""Bag of local features network.

Model similar to that from https://openreview.net/pdf?id=SkfMWhAqYQ.
The model has an optional bottleneck before the linear logits layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow.compat.v1 as tf


ConvLayer = collections.namedtuple(
    "ConvLayer",
    ["kernel_size", "stride", "filters"])


def receptive_field_size(conv_ops_list, input_size=None):
  """Computes receptive field and output size for valid convolutions.

  Args:
    conv_ops_list: List of named tuples in the order of convolutions operation
      with the format (kernel_size, stride, filters).
    input_size: (Integer) If not None the function also computes output size.
  Returns:
    output_rf: (Integer) Size of receptive field
    output_size: (Integer) Spatial size of network tensor at the output.
  """
  output_rf = 1
  output_size = input_size
  stride_jumps = 1
  for conv_op in conv_ops_list:
    kernel_size, stride, _ = conv_op
    output_rf += stride_jumps * (kernel_size-1)
    stride_jumps *= stride
    if input_size:
      output_size = int(np.ceil((output_size - kernel_size + 1) / stride))
  return output_rf, output_size


def bottleneck(x, filters, kernel_size, stride, activation, expansion,
               batch_norm_config, is_training):
  """Creates a bottleneck layer.

  conv(kernel:1, stride:1) -> conv(kernel, stride) -> conv(kernel:1, stride:1)
  Args:
    x: input tensor.
    filters: (Integer) Number of filters for first two convolutions.
    kernel_size: (Integer) Bottleneck kernel size.
    stride: (Integer) Bottleneck stride size.
    activation: Tensorflow activation function.
    expansion: (Integer) Expansion of feature channels at the last convolution.
    batch_norm_config: (Configuration object) with batch normalization params.
    is_training: (Boolean) Whether training on inference mode.

  Returns:
    Returns bottleneck output tensor.
  """

  residual = x
  with tf.variable_scope("a"):
    net = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=1,
        strides=(1, 1),
        use_bias=not batch_norm_config.enable,
        padding="valid")

    tf.logging.info("Constructing layer: %s", net)
    if batch_norm_config.enable:
      net = tf.layers.batch_normalization(
          net, training=is_training,
          momentum=batch_norm_config.momentum,
          epsilon=batch_norm_config.epsilon)

    tf.logging.info("Constructing layer: %s", net)
    net = activation(net)

  with tf.variable_scope("b"):
    net = tf.layers.conv2d(
        net,
        filters=filters,
        kernel_size=kernel_size,
        strides=(stride, stride),
        use_bias=not batch_norm_config.enable,
        padding="valid")

    tf.logging.info("Constructing layer: %s", net)
    if batch_norm_config.enable:
      net = tf.layers.batch_normalization(
          net, training=is_training,
          momentum=batch_norm_config.momentum,
          epsilon=batch_norm_config.epsilon)

    tf.logging.info("Constructing layer: %s", net)

    net = activation(net)

  with tf.variable_scope("c"):
    net = tf.layers.conv2d(
        net,
        filters=filters * expansion,
        kernel_size=1,
        strides=(1, 1),
        use_bias=not batch_norm_config.enable,
        padding="valid")
    if batch_norm_config.enable:
      net = tf.layers.batch_normalization(
          net, training=is_training,
          momentum=batch_norm_config.momentum,
          epsilon=batch_norm_config.epsilon,
          gamma_initializer=tf.zeros_initializer())

    tf.logging.info("Constructing layer: %s", net)

  if kernel_size == 3:
    residual = residual[:, 1:-1, 1:-1, :]

  with tf.variable_scope("downsample"):
    if stride != 1 or residual.shape.as_list()[-1] != filters * expansion:
      residual = tf.layers.conv2d(
          residual,
          filters=filters * expansion,
          kernel_size=1,
          strides=(stride, stride),
          use_bias=not batch_norm_config.enable,
          padding="valid")
      if batch_norm_config.enable:
        net = tf.layers.batch_normalization(
            net, training=is_training,
            momentum=batch_norm_config.momentum,
            epsilon=batch_norm_config.epsilon)
  with tf.variable_scope("add_residual"):
    net += residual
    out = activation(net)
  return out


class BagNet(object):
  """Bag of local fetures network (BagNet).

  Attributes:
    infilters: (Integer) Base filters dimensionality.
    expansion: (Integer) Filter expansion multiplier.
    config: (Configuration object) With parameters:
      blocks: (List of integers) Number of bottleneck blocks per group.
      strides: (List of integers) Stride size per group.
      num_classes: (Integer) Number of output classes.
      kernel3: (List of integers) Number 3x3 kernels per group.
      activation: Tensorflow activation function.
      num_classes: (Integer) Number of output classes.
      planes: (List of integer) Base filters size per group.
      final_bottleneck: (Boolean) Use a final features bottleneck.
      batch_norm: (Configuration object) Batch normalization parameters.
    conv_ops_list: (List of named tuples) Settings of the convolutions.
    receptive_field: (Tuple of integers) Receptive field shape.
    variable_scope: (string) Name of variable scope.
    var_list: List of network variables.
    init_op: Initialization operations for model variables.
  """

  def __init__(self, config, variable_scope="bagnet"):
    self.infilters = 64
    self.config = config
    self.variable_scope = variable_scope
    self.receptive_field = None
    self.conv_ops_list = []
    # Call once to create network variables. Then reuse variables later.
    self.var_list = []
    self.init_op = []

  def _collect_variables(self, vs):
    """Collects model variables.

    Populates self.var_list with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.

    Args:
      vs: Variables list to be added to self.var_list.
    """
    self.var_list.extend(vs)
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def _make_group(self,
                  x,
                  filters,
                  blocks,
                  is_training,
                  activation,
                  expansion,
                  stride=1,
                  kernel3=0,
                  prefix=""):
    """Makes network group of layers.

    Args:
      x: Input tensor.
      filters: (Integer) Number of filters for first two convolutions.
      blocks: (Integer) Number of bottleneck blocks.
      is_training: (Boolean) Whether training on inference mode.
      activation: Tensorflow activation function.
      expansion: (Integer) Expansion of feature multiplier.
      stride: (Integer) Group stride size.
      kernel3: (Integer) Number of 3x3 convolutional layers.
      prefix: (String) Prefix of variable scope.

    Returns:
      Group output tensor.
    """
    net = x
    with tf.variable_scope(prefix):
      for i in range(blocks):
        with tf.variable_scope("block%d" % i):
          kernel_size = 3 if i < kernel3 else 1
          stride_size = stride if i == 0 else 1
          net = bottleneck(
              net,
              filters,
              kernel_size=kernel_size,
              stride=stride_size,
              activation=activation,
              expansion=expansion,
              batch_norm_config=self.config.batch_norm,
              is_training=is_training)
          self.conv_ops_list.extend(
              [
                  ConvLayer(1, 1, filters),
                  ConvLayer(kernel_size, stride_size, filters),
                  ConvLayer(1, 1, filters * expansion)
              ]
              )
    self.infilters = filters * expansion
    return net

  def _build_model_graph(self, x, is_training):
    """Builds model graph."""
    endpoints = {}

    with tf.variable_scope("pre_groups"):
      net = tf.layers.conv2d(
          x,
          filters=self.config.init_conv_channels,
          kernel_size=3,
          strides=(1, 1),
          use_bias=not self.config.batch_norm.enable,
          padding="valid")
      self.conv_ops_list.append(ConvLayer(3, 1, self.config.init_conv_channels))
      if self.config.batch_norm.enable:
        net = tf.layers.batch_normalization(
            net, training=is_training,
            momentum=self.config.batch_norm.momentum,
            epsilon=self.config.batch_norm.epsilon)

      net = self.config.activation(net)
    number_groups = len(self.config.blocks)
    for i in range(number_groups):
      net = self._make_group(
          net,
          filters=self.config.planes[i],
          blocks=self.config.blocks[i],
          is_training=is_training,
          activation=self.config.activation,
          expansion=self.config.expansion,
          stride=self.config.strides[i],
          kernel3=self.config.kernel3[i],
          prefix="group%d" % i)

    tf.logging.info("Constructing layer: %s", net)
    endpoints["features2d"] = net
    if self.config.final_bottleneck:
      with tf.variable_scope("final_bottleneck"):
        channels = net.shape.as_list()[-1]
        net = tf.layers.conv2d(
            net,
            filters=channels // 4,
            kernel_size=1,
            strides=(1, 1),
            use_bias=not self.config.batch_norm.enable,
            padding="valid")
        self.conv_ops_list.append(ConvLayer(1, 1, channels // 4))
        if self.config.batch_norm.enable:
          net = tf.layers.batch_normalization(
              net, training=is_training,
              momentum=self.config.batch_norm.momentum,
              epsilon=self.config.batch_norm.epsilon)
        net = self.config.activation(net)
      endpoints["features2d_lowd"] = net

    with tf.variable_scope("logits2d"):
      net = tf.layers.conv2d(
          net,
          filters=self.config.num_classes,
          kernel_size=1,
          strides=(1, 1),
          use_bias=True,
          padding="valid")
      self.conv_ops_list.append(ConvLayer(1, 1, self.config.num_classes))

    tf.logging.info("Constructing layer: %s", net)
    endpoints["logits2d"] = net
    logits = tf.reduce_mean(net, axis=[1, 2])
    tf.logging.info("Constructing layer: %s", logits)

    return logits, endpoints

  def __call__(self, x, is_training):
    """Builds network.

    Args:
      x: 4-D Tensor of shape [batch, height, width, channels].
      is_training: (Boolean) Training or inference mode.

    Returns:
      logits: Network output.
      endpoints: Dictionary with activations at different layers.
    """
    variables_before = set(tf.global_variables())
    reuse = bool(self.var_list)
    tf.logging.info("Build bagnet.")
    with tf.variable_scope(self.variable_scope, reuse=reuse):
      logits, endpoints = self._build_model_graph(x, is_training)
    variables_after = set(tf.global_variables())
    if not reuse:
      self._collect_variables(list(variables_after - variables_before))
      self.receptive_field = tuple([
          receptive_field_size(self.conv_ops_list)[0]] * 2)
    return logits, endpoints
