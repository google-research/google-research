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

"""CIFAR10 ResNet architecture, with customized layers for first PC regularization

Adapted from
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
"""
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras import layers
from absl import logging


def power_iterate(x, u_var, num_iter=1):
  """Perform power iteration.

  Args:
    x: A matrix.
    u_var: Stored eigenvector.
    num_iter: The number of iterations to perform power iteration for.

  Returns:
    Amount of variance explained by the first principal component.
  """
  u = u_var
  for _ in range(num_iter):
    u = tf.stop_gradient(u)
    u = tf.matmul(x, tf.matmul(x, u), transpose_a=True)
    u_norm = tf.linalg.norm(u)
    u /= u_norm
  u_var.assign(u)
  return u_norm


class PCRegularizationLayer(layers.Layer):

  def __init__(self, reg_strength, threshold):
    super(PCRegularizationLayer, self).__init__()
    self.reg_strength = reg_strength
    self.threshold = threshold

  def build(self, input_shape):
    self.u_var = self.add_weight(
        name='stored_eigenvector',
        shape=(np.prod(input_shape[1:]), 1),
        initializer=tf.keras.initializers.VarianceScaling,
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

  def call(self, inputs):
    reshaped_inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
    reshaped_inputs = (
        reshaped_inputs - tf.reduce_mean(reshaped_inputs, 0, keepdims=True))
    s = power_iterate(reshaped_inputs, self.u_var)
    variance_explained_ratio = s / tf.einsum('nc,nc->', reshaped_inputs,
                                             reshaped_inputs)
    self.add_loss(self.reg_strength *
                  tf.nn.relu(variance_explained_ratio - self.threshold))
    return inputs  # Pass-through layer.


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 weight_decay=0,
                 sync_bn=False,
                 reg_strength=0,
                 residual_reg_only=False,
                 relu_reg_only=False,
                 threshold=0):
  """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
        sync_bn (bool): Whether to use synchronous batch normalization.
    # Returns
        x (tensor): tensor as input to the next layer
    """
  conv = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
  if batch_normalization:
    if sync_bn:
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization()
    else:
      bn_layer = tf.keras.layers.BatchNormalization()
  else:
    bn_layer = lambda x: x
  x = inputs
  if conv_first:
    x = conv(x)
    if not residual_reg_only and not relu_reg_only:
      x = PCRegularizationLayer(reg_strength, threshold)(x)
    x = bn_layer(x)
    if not residual_reg_only and not relu_reg_only:
      x = PCRegularizationLayer(reg_strength, threshold)(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
      if not residual_reg_only:
        x = PCRegularizationLayer(reg_strength, threshold)(x)
  else:
    x = bn_layer(x)
    if not residual_reg_only and not relu_reg_only:
      x = PCRegularizationLayer(reg_strength, threshold)(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
      if not residual_reg_only:
        x = PCRegularizationLayer(reg_strength, threshold)(x)
    x = conv(x)
    if not residual_reg_only and not relu_reg_only:
      x = PCRegularizationLayer(reg_strength, threshold)(x)
  return x


def ResNet_CIFAR(depth,
                 width_multiplier,
                 weight_decay,
                 num_classes,
                 input_shape=(32, 32, 3),
                 sync_bn=False,
                 use_residual=True,
                 reg_strength=0,
                 residual_reg_only=False,
                 relu_reg_only=False,
                 threshold=0,
                 last_2_stages=False):
  """ResNet Version 1 Model builder

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
        sync_bn (bool): Whether to use synchronous batch normalization.
    # Returns
        model (Model): Keras model instance
    """
  #tf.config.experimental_run_functions_eagerly(True)
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_filters = int(round(16 * width_multiplier))
  num_res_blocks = int((depth - 2) / 6)
  inputs = tf.keras.Input(shape=input_shape)
  if last_2_stages:
    stage_threshold = 1.0
  else:
    stage_threshold = threshold

  x = resnet_layer(
      inputs=inputs,
      num_filters=num_filters,
      weight_decay=weight_decay,
      sync_bn=sync_bn,
      reg_strength=reg_strength,
      residual_reg_only=residual_reg_only,
      relu_reg_only=relu_reg_only,
      threshold=stage_threshold)
  # Instantiate the stack of residual units
  for stack in range(3):
    if last_2_stages and stack == 0:
      stage_threshold = 1.0
    else:
      stage_threshold = threshold
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          weight_decay=weight_decay,
          sync_bn=sync_bn,
          reg_strength=reg_strength,
          residual_reg_only=residual_reg_only,
          relu_reg_only=relu_reg_only,
          threshold=stage_threshold)
      y = resnet_layer(
          inputs=y,
          num_filters=num_filters,
          activation=None,
          weight_decay=weight_decay,
          sync_bn=sync_bn,
          reg_strength=reg_strength,
          residual_reg_only=residual_reg_only,
          relu_reg_only=relu_reg_only,
          threshold=stage_threshold)
      if stack > 0 and res_block == 0:  # first block but not first stack
        # linear projection residual shortcut connection to match
        # changed dims
        x = resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            activation=None,
            batch_normalization=False,
            weight_decay=weight_decay,
            reg_strength=reg_strength,
            residual_reg_only=residual_reg_only,
            relu_reg_only=relu_reg_only,
            threshold=stage_threshold)
      if use_residual:
        x = tf.keras.layers.add([x, y])
        x = PCRegularizationLayer(reg_strength, stage_threshold)(x)
      else:
        x = y
      x = tf.keras.layers.Activation('relu')(x)
      if not residual_reg_only:
        x = PCRegularizationLayer(reg_strength, stage_threshold)(x)
    num_filters *= 2
  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  if not residual_reg_only and not relu_reg_only:
    x = PCRegularizationLayer(reg_strength, stage_threshold)(x)
  y = tf.keras.layers.Flatten()(x)
  if not residual_reg_only and not relu_reg_only:
    y = PCRegularizationLayer(reg_strength, stage_threshold)(y)
  outputs = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          y)
  if not residual_reg_only and not relu_reg_only:
    outputs = PCRegularizationLayer(reg_strength, stage_threshold)(outputs)
  # Instantiate model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model
