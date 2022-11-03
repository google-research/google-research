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

"""Define networks for behavioral cloning."""

from absl import flags
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf

flags.DEFINE_boolean('layer_norm_policy', False,
                     'If True, a Sonnet LayerNormMLP architecture for fully '
                     'connected layers.')
flags.DEFINE_boolean('data_smaller', False,
                     'If True, scale hand_vil_net output layer by 0.01 to '
                     'match original implementation.')
FLAGS = flags.FLAGS

keras = tf.keras
tfl = keras.layers

BATCH_NORM_DECAY_50 = 0.9
BATCH_NORM_EPSILON = 1e-5

v1_conv2d_init = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out')
v1_linear_init = keras.initializers.VarianceScaling(
    scale=1.0, distribution='uniform')
v1_linear_bias_init = keras.initializers.VarianceScaling(
    scale=1/3., distribution='uniform')

# Matches default Conv2d kernel initialization in PyTorch.
torch_conv2d_init = keras.initializers.VarianceScaling(
    scale=1/3, distribution='uniform')
# Matches default Conv2d bias initialization in PyTorch.
torch_conv2d_bias_init = keras.initializers.VarianceScaling(
    scale=1/3., distribution='uniform')
# Matches default Dense kernel initialization in PyTorch.
torch_linear_init = keras.initializers.VarianceScaling(
    scale=1/3, distribution='uniform')
# Matches default Dense bias initialization in PyTorch.
torch_linear_bias_init = keras.initializers.VarianceScaling(
    scale=1/3., distribution='uniform')


class FeedForwardNet(keras.Model):
  """Fully connected network with custom layer sizes and weight decay."""

  def __init__(self,
               n_classes=None,
               last_activation=None,
               fc_layer_sizes=(),
               weight_decay=5e-4):
    super(FeedForwardNet, self).__init__()
    self.fcs = []
    for size in fc_layer_sizes:
      self.fcs.append(
          tfl.Dense(
              size,
              activation=tf.nn.relu,
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(weight_decay),
              bias_regularizer=keras.regularizers.l2(weight_decay)))
    if n_classes is not None:
      self.linear = tfl.Dense(
          n_classes,
          activation=last_activation,
          kernel_initializer='he_normal',
          kernel_regularizer=keras.regularizers.l2(weight_decay),
          bias_regularizer=keras.regularizers.l2(weight_decay),
          name='fc%d' % n_classes)
    self.n_classes = n_classes

  def call(self, unused_input_tensor, input_scalars=None, unused_training=False,
           return_feats=False):
    x = input_scalars
    for layer in self.fcs:
      x = layer(x)
    feats = x
    if self.n_classes is None:
      return x
    else:
      x = self.linear(feats)
      if return_feats:
        return x, feats
      else:
        return x


class SimpleCNN(keras.Model):
  """Basic CNN with custom layer sizes and max pooling between layers."""

  def __init__(self,
               n_classes=None,
               last_activation=None,
               fc_layer_sizes=(64,),
               weight_decay=5e-4):
    super(SimpleCNN, self).__init__()
    self.conv1 = tfl.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(weight_decay))
    self.max_pool1 = tfl.MaxPooling2D((2, 2))
    self.conv2 = tfl.Conv2D(
        64, (3, 3),
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(weight_decay))
    self.max_pool2 = tfl.MaxPooling2D((2, 2))
    self.conv3 = tfl.Conv2D(
        64, (3, 3),
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(weight_decay))
    self.flatten = tfl.Flatten()
    self.fcs = []
    for size in fc_layer_sizes:
      self.fcs.append(
          tfl.Dense(
              size,
              activation='relu',
              kernel_regularizer=keras.regularizers.l2(weight_decay),
              bias_regularizer=keras.regularizers.l2(weight_decay)))
    if n_classes is not None:
      self.linear = tfl.Dense(
          n_classes,
          activation=last_activation,
          kernel_regularizer=keras.regularizers.l2(weight_decay),
          bias_regularizer=keras.regularizers.l2(weight_decay))
    self.n_classes = n_classes

  def call(self, input_tensor, input_scalars=None, unused_training=False,
           return_feats=False):
    x = self.conv1(input_tensor)
    x = self.max_pool1(x)
    x = self.conv2(x)
    x = self.max_pool2(x)
    x = self.conv3(x)
    feats = self.flatten(x)
    if input_scalars is None:
      x = feats
    else:
      x = tf.concat([feats, input_scalars], axis=1)
    for layer in self.fcs:
      x = layer(x)
    if self.n_classes is None:
      return x
    else:
      x = self.linear(x)
      if return_feats:
        return x, feats
      else:
        return x


class ResnetProjBlock(keras.Model):
  """Resnet projection block."""

  def __init__(self, kernel_size, filters, stage, block, strides, weight_decay,
               batch_norm_decay, conv2d_init):
    super(ResnetProjBlock, self).__init__(name='')
    filters1, filters2 = filters
    if keras.backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.strides = strides
    first_padding = 'same'
    if self.strides[0] > 1:
      self.zero_pad = tfl.ZeroPadding2D(
          padding=(1, 1), name=conv_name_base + 'zero_pad')
      first_padding = 'valid'
    self.conv1 = tfl.Conv2D(
        filters1,
        kernel_size,
        strides=strides,
        padding=first_padding,  # 'same',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2a')
    self.bn1 = tfl.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

    self.conv2 = tfl.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2b')
    self.bn2 = tfl.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

    self.shortcut_conv = tfl.Conv2D(
        filters2, (1, 1),
        strides=strides,
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '1')
    self.shortcut_bn = tfl.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '1',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

  def call(self, input_tensor, training=False):
    x = input_tensor
    if self.strides[0] > 1:
      x = self.zero_pad(x)
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    shortcut_x = self.shortcut_conv(input_tensor)
    shortcut_x = self.shortcut_bn(shortcut_x, training=training)

    x += shortcut_x
    return tf.nn.relu(x)


class ResnetIdBlock(keras.Model):
  """Resnet identity block."""

  def __init__(self, kernel_size, filters, stage, block, weight_decay,
               batch_norm_decay, conv2d_init):
    super(ResnetIdBlock, self).__init__(name='')
    filters1, filters2 = filters
    if keras.backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv1 = tfl.Conv2D(
        filters1,
        kernel_size,
        padding='same',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2a')
    self.bn1 = tfl.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

    self.conv2 = tfl.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2b')
    self.bn2 = tfl.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class Resnet18Block(keras.Model):
  """Resnet-18 block."""

  def __init__(self,
               kernel_size,
               input_planes,
               output_planes,
               stage,
               batch_norm_decay,
               conv2d_init,
               strides=(2, 2),
               weight_decay=5e-4):
    super(Resnet18Block, self).__init__(name='')
    if output_planes != input_planes:
      self.first_block = ResnetProjBlock(
          kernel_size, (output_planes, output_planes),
          stage=stage,
          strides=strides,
          block='block_0',
          weight_decay=weight_decay,
          batch_norm_decay=batch_norm_decay,
          conv2d_init=conv2d_init)
    else:
      self.first_block = ResnetIdBlock(
          kernel_size, (output_planes, output_planes),
          stage=stage,
          block='block_0',
          weight_decay=weight_decay,
          batch_norm_decay=batch_norm_decay,
          conv2d_init=conv2d_init)
    self.second_block = ResnetIdBlock(
        kernel_size, (output_planes, output_planes),
        stage=stage,
        block='block_1',
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        conv2d_init=conv2d_init)

  def call(self, input_tensor, training=False):
    x = self.first_block(input_tensor, training=training)
    x = self.second_block(x, training=training)
    return x


class Resnet18Narrow32(keras.Model):
  """Resnet-18 with filter sizes halved."""

  def __init__(self,
               n_classes=None,
               last_activation=None,
               fc_layer_sizes=(),
               weight_decay=5e-4,
               bn_axis=3,
               batch_norm_decay=0.1,
               init_scheme='v1'):
    super(Resnet18Narrow32, self).__init__(name='')

    if init_scheme == 'v1':
      print('Using v1 weight init')
      conv2d_init = v1_conv2d_init
      # Bias is not used in conv layers.
      linear_init = v1_linear_init
      linear_bias_init = v1_linear_bias_init
    else:
      print('Using v2 weight init')
      conv2d_init = keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_out', distribution='untruncated_normal')
      linear_init = torch_linear_init
      linear_bias_init = torch_linear_bias_init

    # Why is this separate instead of padding='same' in tfl.Conv2D?
    self.zero_pad = tfl.ZeroPadding2D(
        padding=(3, 3), input_shape=(32, 32, 3), name='conv1_pad')
    self.conv1 = tfl.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        use_bias=False,
        name='conv1')
    self.bn1 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv1',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)
    self.zero_pad2 = tfl.ZeroPadding2D(padding=(1, 1), name='max_pool_pad')
    self.max_pool = tfl.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid')

    self.resblock1 = Resnet18Block(
        kernel_size=3,
        input_planes=64,
        output_planes=32,
        stage=2,
        strides=(1, 1),
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        conv2d_init=conv2d_init)

    self.resblock2 = Resnet18Block(
        kernel_size=3,
        input_planes=32,
        output_planes=64,
        stage=3,
        strides=(2, 2),
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        conv2d_init=conv2d_init)

    self.resblock3 = Resnet18Block(
        kernel_size=3,
        input_planes=64,
        output_planes=128,
        stage=4,
        strides=(2, 2),
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        conv2d_init=conv2d_init)

    self.resblock4 = Resnet18Block(
        kernel_size=3,
        input_planes=128,
        output_planes=256,
        stage=4,
        strides=(2, 2),
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        conv2d_init=conv2d_init)

    self.pool = tfl.GlobalAveragePooling2D(name='avg_pool')
    self.bn2 = tfl.BatchNormalization(
        axis=-1,
        name='bn_conv2',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)
    self.fcs = []
    if FLAGS.layer_norm_policy:
      self.linear = snt.Sequential([
          networks.LayerNormMLP(fc_layer_sizes),
          networks.MultivariateNormalDiagHead(n_classes),
          networks.StochasticMeanHead()
      ])
    else:
      for size in fc_layer_sizes:
        self.fcs.append(
            tfl.Dense(
                size,
                activation=tf.nn.relu,
                kernel_initializer=linear_init,
                bias_initializer=linear_bias_init,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                bias_regularizer=keras.regularizers.l2(weight_decay)))
      if n_classes is not None:
        self.linear = tfl.Dense(
            n_classes,
            activation=last_activation,
            kernel_initializer=linear_init,
            bias_initializer=linear_bias_init,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='fc%d' % n_classes)
    self.n_classes = n_classes
    if n_classes is not None:
      self.log_std = tf.Variable(tf.zeros(n_classes), trainable=True,
                                 name='log_std')
    self.first_forward_pass = FLAGS.data_smaller

  def call(self, input_tensor, input_scalars=None, training=False,
           return_feats=False):
    x = self.zero_pad(input_tensor)
    x = self.conv1(x)
    x = self.bn1(x, training)
    x = tf.nn.relu(x)
    x = self.zero_pad2(x)  # Added in Resnet18Narrow32.
    x = self.max_pool(x)  # Added in Resnet18Narrow32.
    x = self.resblock1(x, training=training)
    x = self.resblock2(x, training=training)
    x = self.resblock3(x, training=training)
    x = self.resblock4(x, training=training)
    feats = self.pool(x)
    if FLAGS.bn_before_concat:
      feats = self.bn2(feats, training)
    if input_scalars is None:
      x = feats
    else:
      x = tf.concat([feats, input_scalars], axis=1)
    for layer in self.fcs:
      x = layer(x)
    if self.n_classes is None:
      return x
    else:
      if self.first_forward_pass:
        first_x = self.linear(x)
        self.first_forward_pass = False
        self.linear.weights[0].assign(0.01 * self.linear.weights[0])
        self.linear.weights[1].assign(0.01 * self.linear.weights[1])
        x = self.linear(x)
        print('pred changed after scaling', not np.array_equal(first_x, x))
      else:
        x = self.linear(x)
      if return_feats:
        return x, feats
      else:
        return x


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   weight_decay=1e-4,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    weight_decay: L2 regularization weight.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tfl.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2a')(
          input_tensor)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = tfl.Activation('relu')(x)

  x = tfl.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2b')(
          x)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = tfl.Activation('relu')(x)

  x = tfl.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2c')(
          x)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  x = tfl.add([x, input_tensor])
  x = tfl.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               weight_decay=1e-4,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    weight_decay: L2 regularization weight.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tfl.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2a')(
          input_tensor)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = tfl.Activation('relu')(x)

  x = tfl.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2b')(
          x)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = tfl.Activation('relu')(x)

  x = tfl.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '2c')(
          x)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  shortcut = tfl.Conv2D(
      filters3, (1, 1),
      strides=strides,
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name=conv_name_base + '1')(
          input_tensor)
  shortcut = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '1')(
          shortcut)

  x = tfl.add([x, shortcut])
  x = tfl.Activation('relu')(x)
  return x


def resnet50(n_classes=None,
             input_shape=(240, 240, 1),
             last_activation=None,
             fc_layer_sizes=(),
             batch_size=None,
             weight_decay=1e-4):
  """Instantiates the ResNet50 architecture.

  Forked from
  tensorflow_models/official/vision/image_classification/resnet_model.py.

  Args:
    n_classes: `int` number of regression outputs.
    input_shape: static input shape.
    last_activation: activation to apply on the output, if any.
    fc_layer_sizes: sizes of fully connected layers to add at the end of the
        network, if any.
    batch_size: Size of the batches for each step.
    weight_decay: L2 regularization weight.

  Returns:
    keras Model for the network.
  """

  img_input = tfl.Input(shape=input_shape, batch_size=batch_size)
  x = img_input

  if keras.backend.image_data_format() == 'channels_first':
    x = tfl.Lambda(
        lambda x: keras.backend.permute_dimensions(x, (0, 3, 1, 2)),
        name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  x = tfl.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  x = tfl.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=keras.regularizers.l2(weight_decay),
      name='conv1')(
          x)
  x = tfl.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY_50,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(
          x)
  x = tfl.Activation('relu')(x)
  x = tfl.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = conv_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='a',
      strides=(1, 1),
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='b',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='c',
      weight_decay=weight_decay)

  x = conv_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='a',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='b',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='c',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='d',
      weight_decay=weight_decay)

  x = conv_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='a',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='b',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='c',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='d',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='e',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='f',
      weight_decay=weight_decay)

  x = conv_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='a',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='b',
      weight_decay=weight_decay)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='c',
      weight_decay=weight_decay)

  rm_axes = ([1, 2] if keras.backend.image_data_format() == 'channels_last'
             else [2, 3])
  x = tfl.Lambda(
      lambda x: keras.backend.mean(x, rm_axes), name='reduce_mean')(x)
  for size in fc_layer_sizes:
    x = tfl.Dense(
        size,
        activation=tf.nn.relu,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay))(
            x)
  if n_classes is not None:
    x = tfl.Dense(
        n_classes,
        activation=last_activation,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay),
        name='fc1000')(
            x)

  return keras.models.Model(img_input, x, name='resnet50')


class HandVilNet(keras.Model):
  """Network architecture used by Jain et al.

  Jain et al. "Learning Deep Visuomotor Policies for Dexterous Hand
  Manipulation". ICRA 2019.
  """

  def __init__(self,
               n_classes=None,
               last_activation=None,
               fc_layer_sizes=(200, 128),
               weight_decay=5e-4,
               bn_axis=-1,
               batch_norm_decay=0.1,
               late_fusion=True,
               init_scheme='v1'):
    super().__init__(name='hand_vil_net')

    if init_scheme == 'v1':
      print('Using v1 weight init')
      conv2d_init = v1_conv2d_init
      conv2d_bias_init = 'zeros'
      linear_init = v1_linear_init
      linear_bias_init = v1_linear_bias_init
    else:
      print('Using v2 weight init')
      conv2d_init = torch_conv2d_init
      conv2d_bias_init = torch_conv2d_bias_init
      linear_init = torch_linear_init
      linear_bias_init = torch_linear_bias_init

    self.conv1 = tfl.Conv2D(
        16, (3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_initializer=conv2d_bias_init,
        input_shape=(32, 32, 3),
        name='conv1')
    self.bn1 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv1',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)
    self.max_pool = tfl.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid')

    self.conv2 = tfl.Conv2D(
        32, (3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_initializer=conv2d_bias_init,
        name='conv2')
    self.bn2 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv2',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)
    self.dropout = tfl.Dropout(0.2)

    self.conv3 = tfl.Conv2D(
        32, (3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_initializer=conv2d_bias_init,
        name='conv3')
    self.bn3 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv3',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

    self.conv4 = tfl.Conv2D(
        32, (3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=conv2d_init,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_initializer=conv2d_bias_init,
        name='conv4')
    self.bn4 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv4',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)

    self.flatten = tfl.Flatten()
    self.fcs = []
    for l in range(len(fc_layer_sizes)):
      self.fcs.append(
          tfl.Dense(
              fc_layer_sizes[l],
              activation=tf.nn.tanh,
              kernel_initializer=linear_init,
              bias_initializer=linear_bias_init,
              kernel_regularizer=keras.regularizers.l2(weight_decay),
              bias_regularizer=keras.regularizers.l2(weight_decay),
              name=f'fc{l + 1}'))
    self.bn5 = tfl.BatchNormalization(
        axis=bn_axis,
        name='bn_conv5',
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON)
    # TODO(minttu): output layer weights *= 0.01?
    if n_classes is not None:
      self.fc_out = tfl.Dense(
          n_classes,
          activation=last_activation,
          kernel_initializer=linear_init,
          bias_initializer=linear_bias_init,
          kernel_regularizer=keras.regularizers.l2(weight_decay),
          bias_regularizer=keras.regularizers.l2(weight_decay),
          name='fc_out')
    self.n_classes = n_classes
    self.log_std = None
    if n_classes is not None:
      self.log_std = tf.Variable(tf.zeros(n_classes), trainable=True,
                                 name='log_std')
    self.late_fusion = late_fusion
    self.first_forward_pass = FLAGS.data_smaller
    assert late_fusion == FLAGS.late_fusion

  def call(self, input_tensor, input_scalars=None, training=False,
           return_feats=False):
    """Forward pass the network.

    Args:
      input_tensor: [batch, history, h, w, c] if using late fusion,
          [batch, h, w, c * history] otherwise
      input_scalars: scalar (1d) features to concatenate to flattened output of
          convolutional layers.
      training: whether network is called in training of inference
      return_feats: whether to return penultimate layer activations in addition
          to network output

    Returns:
      x: final layer output
      feats: penultimate layer activations
    """
    x = input_tensor
    if len(x.shape) < 5:
      x = tf.expand_dims(x, -1)
    if FLAGS.late_fusion:
      # Merge batch and history dimensions.
      #
      # torch: 128, 3, 3, 128, 128
      # torch: 0 * 1, 2, 3
      # tf: 128, 3, 128, 128, 3
      # tf: 0 * 1, 2, 3, 4

      # Batch size can be None in placeholders.
      if input_tensor.shape[0] is None:
        batch_size = tf.shape(input_tensor)[0]
        history_size = tf.shape(input_tensor)[1]
        x = tf.reshape(x, [batch_size * history_size, *x.shape[2:]])
      else:
        x = tf.reshape(x, [np.prod(x.shape[:2]), *x.shape[2:]])

    x = self.conv1(x)
    x = self.bn1(x, training)
    x = tf.nn.relu(x)
    x = self.max_pool(x)

    x = self.conv2(x)
    x = self.bn2(x, training)
    x = tf.nn.relu(x)
    x = self.max_pool(x)
    x = self.dropout(x, training)

    x = self.conv3(x)
    x = self.bn3(x, training)
    x = tf.nn.relu(x)
    x = self.max_pool(x)

    x = self.conv4(x)
    x = self.bn4(x, training)
    x = tf.nn.relu(x)

    if FLAGS.late_fusion:
      if input_tensor.shape[0] is None:
        x = tf.reshape(x, [batch_size, history_size, *x.shape[1:]])
      else:
        x = tf.reshape(
            x, [input_tensor.shape[0], input_tensor.shape[1], *x.shape[1:]])
    if FLAGS.policy_init_path is not None:
      # Permute before flatten in order to match torch order.
      x = tf.transpose(x, [0, 1, 4, 2, 3])
    x = self.flatten(x)
    if self.fcs:
      x = self.fcs[0](x)
    feats = x
    if FLAGS.bn_before_concat:
      feats = self.bn5(feats, training)
    if input_scalars is None:
      x = feats
    else:
      x = tf.concat([feats, input_scalars], axis=1)
    for layer in self.fcs[1:]:
      x = layer(x)

    if self.n_classes is None:
      return x
    else:
      if self.first_forward_pass:
        first_x = self.fc_out(x)
        self.first_forward_pass = False
        self.fc_out.weights[0].assign(0.01 * self.fc_out.weights[0])
        self.fc_out.weights[1].assign(0.01 * self.fc_out.weights[1])
        x = self.fc_out(x)
        print('pred changed after scaling', not np.array_equal(first_x, x))
      else:
        x = self.fc_out(x)
      if return_feats:
        return x, feats
      else:
        return x
