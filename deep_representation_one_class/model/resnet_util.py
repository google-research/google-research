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

# Lint as: python3
"""Resnet utilities."""

import tensorflow as tf

BN_MOM = 0.9
BN_EPS = 1e-05
NN_AXIS = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1


def convnxn(x,
            filters=64,
            kernel_size=3,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  """Conv NxN with SAME padding."""
  pad_size = kernel_size // 2
  if pad_size > 0:
    x = tf.keras.layers.ZeroPadding2D(
        ((pad_size, pad_size), (pad_size, pad_size)), name=name + '_pad')(
            x)
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      name=name)(
          x)


def conv3x3(x,
            filters=64,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  """Conv 3x3."""
  return convnxn(x, filters, 3, strides, use_bias, kernel_initializer, name)


def conv1x1(x,
            filters=64,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  """Conv 1x1."""
  return convnxn(x, filters, 1, strides, use_bias, kernel_initializer, name)


def normalization_fn(x,
                     normalization='bn',
                     norm_axis=NN_AXIS,
                     bn_mom=BN_MOM,
                     bn_eps=BN_EPS,
                     name=None):
  """Normalization layers."""
  if normalization == 'bn':
    return tf.keras.layers.BatchNormalization(
        axis=norm_axis, momentum=bn_mom, epsilon=bn_eps, name=name)(
            x)


def nonlinearity(x, activation='relu', name=None):
  """Nonlinearity layers."""
  if activation in ['relu', 'sigmoid']:
    return tf.keras.layers.Activation(activation, name=name)(x)
  elif activation == 'leaky_relu':
    return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
  else:
    return x


def get_head(x, dims=None, num_class=2):
  """Gets projection head."""
  proj = x
  print(dims)
  if dims is not None:
    for i, d in enumerate(dims[:-1]):
      proj = tf.keras.layers.Dense(
          units=d, use_bias=False, name='proj_%d' % i)(
              proj)
      proj = normalization_fn(proj, norm_axis=-1, name='proj_%d_bn' % i)
      proj = nonlinearity(proj, activation='relu', name='proj_%d_relu' % i)
    embeds = tf.keras.layers.Dense(
        units=dims[-1],
        activation='linear',
        use_bias=num_class > 0,
        name='embeds')(
            proj)
  else:
    embeds = proj
  if num_class > 0:
    logits = tf.keras.layers.Dense(
        units=num_class, activation='linear', name='logits')(
            embeds)
  else:
    logits = embeds
  return {'logits': logits, 'embeds': embeds, 'pools': x}


# pylint: disable=invalid-name
def ResNet(stack_fn,
           preact,
           model_name='resnet',
           head_dims=None,
           input_shape=None,
           pooling=None,
           normalization='bn',
           activation='relu',
           num_class=4):
  """Instantiates the ResNet.

  Reference paper:
  - [Deep Residual Learning for Image Recognition]
      (https://arxiv.org/abs/1512.03385) (CVPR 2015)

  Args:
    stack_fn: a function that returns output tensor for the stacked residual
      blocks.
    preact: whether to use pre-activation or not (True for resnetV2).
    model_name: string, model name.
    head_dims: list indicating number of hidden units for each layer of
      projection head.
    input_shape: optional shape tuple. It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction. - `avg` means that
      global average pooling will be applied to the output of the last
      convolutional layer, and thus the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    normalization: the layer normalization. by default, batch normalization.
    activation: the activation function. by default, relu.
    num_class: optional number of classes to classify images into, only to be
      specified if `head` is True, and if no `weights` argument is specified.

  Returns:
    A `keras.Model` instance.
  """

  # Input layer.
  inputs = img_input = tf.keras.layers.Input(shape=(None, None, 3))

  # Conv1 block.
  if input_shape[0] in [128, 256]:
    kernel_size, stride, maxpool = 7, 2, True
  elif input_shape[0] in [32, 64]:
    kernel_size, stride, maxpool = 3, 1, False
  else:
    raise NotImplementedError
  x = convnxn(
      img_input,
      filters=64,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=preact,
      name='conv1_conv')
  if not preact:
    x = normalization_fn(
        x, normalization=normalization, name='conv1_' + normalization)
    x = nonlinearity(x, activation=activation, name='conv1_' + activation)
  if maxpool:
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  # Conv2 to Conv5 blocks.
  x = stack_fn(x)
  if preact:
    x = normalization_fn(
        x, normalization=normalization, name='post_' + normalization)
    x = nonlinearity(x, activation=activation, name='post_' + activation)

  # Pooling layer.
  if pooling in ['avg']:
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  elif pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Projection head or classifier.
  outputs = get_head(x, dims=head_dims, num_class=num_class)

  # Creates model.
  model_name += '_{}'.format(pooling)
  if head_dims:
    model_name += '_mlp' + '_'.join(['%d' % d for d in head_dims])
  if num_class > 0:
    model_name += '_cls{}'.format(num_class)
  return tf.keras.models.Model(inputs, outputs, name=model_name)


def block1(x,
           filters,
           bottleneck=False,
           stride=1,
           expansion=1,
           normalization='bn',
           activation='relu',
           name=None):
  """Applies a basic residual block.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    bottleneck: use bottleneck if True.
    stride: default 1, stride of the first layer.
    expansion: default 4, the expansion.
    normalization: default batch normalization.
    activation: default relu, the activation function.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  conv_shortcut = (stride != 1) or (expansion * filters != x.shape[3])
  if conv_shortcut:
    shortcut = conv1x1(
        x, filters=expansion * filters, strides=stride, name=name + '_0_conv')
    shortcut = normalization_fn(
        shortcut,
        normalization=normalization,
        name=name + '_0_' + normalization)
  else:
    shortcut = x
  # First conv.
  if bottleneck:
    x = conv1x1(x, filters=filters, strides=1, name=name + '_1_conv')
    x = normalization_fn(
        x, normalization=normalization, name=name + '_1_' + normalization)
    x = nonlinearity(x, activation=activation, name=name + '_1_' + activation)
  # Second conv.
  idx = 2 if bottleneck else 1
  x = conv3x3(x, filters=filters, strides=stride, name=name + '_%d_conv' % idx)
  x = normalization_fn(
      x,
      normalization=normalization,
      name=name + '_%d_%s' % (idx, normalization))
  x = nonlinearity(
      x, activation=activation, name=name + '_%d_%s' % (idx, activation))
  # Last conv.
  last_conv = conv1x1 if bottleneck else conv3x3
  x = last_conv(
      x,
      filters=expansion * filters,
      strides=1,
      name=name + '_%d_conv' % (idx + 1))
  x = normalization_fn(
      x,
      normalization=normalization,
      name=name + '_%d_%s' % (idx + 1, normalization))
  # Skip connection.
  x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
  x = nonlinearity(x, activation=activation, name=name + '_out_' + activation)
  return x


def stack_v1(x,
             filters,
             blocks,
             bottleneck=False,
             stride1=2,
             expansion=4,
             normalization='bn',
             activation='relu',
             name=None):
  """Applies a set of stacked residual blocks.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    blocks: integer, blocks in the stacked blocks.
    bottleneck: use bottleneck if True.
    stride1: default 2, stride of the first layer in the first block.
    expansion: default 4, the expansion.
    normalization: default batch normalization.
    activation: default relu, the activation function.
    name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  x = block1(
      x,
      filters,
      bottleneck=bottleneck,
      stride=stride1,
      expansion=expansion,
      normalization=normalization,
      activation=activation,
      name=name + '_block1')
  for i in range(1, blocks):
    x = block1(
        x,
        filters,
        bottleneck=bottleneck,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name=name + '_block' + str(i + 1))
  return x


def basic_stack1(x, filters, blocks, **kwargs):
  return stack_v1(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack1(x, filters, blocks, **kwargs):
  return stack_v1(x, filters, blocks, bottleneck=True, **kwargs)
