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
"""Model factory."""

import functools
import gin
import tensorflow.compat.v2 as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class ProjectLayer(tf.keras.layers.Layer):

  def build(self, input_shape):
    self.w = self.add_weight('weights', (input_shape[-1],), trainable=True)

  def call(self, inputs, training=True):
    return tf.linalg.matvec(inputs, self.w[tf.newaxis, :])


@gin.configurable
def projection(input_shape=None, output_shape=None, dtype=tf.float32):
  del input_shape, output_shape, dtype
  return tf.keras.Sequential([ProjectLayer()])


@gin.configurable
def mlp(input_shape,
        output_shape,
        dtype=tf.float32,
        output_activation='linear',
        hidden=()):
  """Create a multi-layers perceptron."""
  layers = [tf.keras.layers.Input(shape=input_shape)]
  for num_neurons in hidden:
    layers.extend([
        tf.keras.layers.Dense(num_neurons, dtype=dtype),
        tf.keras.layers.Activation('relu', dtype=dtype)
    ])
  layers.extend([
      tf.keras.layers.Dense(output_shape[0], dtype=dtype),
      tf.keras.layers.Activation(output_activation, dtype=dtype)
  ])
  return tf.keras.Sequential(layers)


@gin.configurable
def vanilla_cnn(
    input_shape, output_shape, dtype=tf.float32, output_activation='linear'):
  return tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          32, (3, 3), padding='same', dtype=dtype, input_shape=input_shape),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.Conv2D(32, (3, 3), dtype=dtype),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), dtype=dtype),

      tf.keras.layers.Conv2D(64, (3, 3), padding='same', dtype=dtype),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.Conv2D(64, (3, 3), dtype=dtype),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), dtype=dtype),

      tf.keras.layers.Flatten(dtype=dtype),
      tf.keras.layers.Dense(512, dtype=dtype),
      tf.keras.layers.Activation('relu', dtype=dtype),

      tf.keras.layers.Dense(output_shape[0], dtype=dtype),
      tf.keras.layers.Activation(output_activation, dtype=dtype),
  ])


@gin.configurable
def pure_conv_cnn(
    input_shape, output_shape, dtype=tf.float32, output_activation='linear'):
  return tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          32, (4, 4), padding='same', dtype=dtype, input_shape=input_shape),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.MaxPooling2D(pool_size=(4, 4), dtype=dtype),
      tf.keras.layers.Conv2D(32, (2, 2), dtype=dtype, padding='same',),
      tf.keras.layers.Activation('relu', dtype=dtype),
      tf.keras.layers.AveragePooling2D(pool_size=(2, 2), dtype=dtype),
      tf.keras.layers.Conv2D(1, (2, 2), dtype=dtype, padding='same',),
      tf.keras.layers.Activation(output_activation, dtype=dtype),
      tf.keras.layers.Reshape(output_shape)
  ])


class ResnetProjBlock(tf.keras.Model):
  """One of the Resnet block."""

  def __init__(self, kernel_size, filters, stage, block, strides, weight_decay):
    super(ResnetProjBlock, self).__init__(name='')
    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv1 = tf.keras.layers.Conv2D(
        filters1,
        kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2a')
    self.bn1 = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

    self.conv2 = tf.keras.layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2b')
    self.bn2 = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

    self.shortcut_conv = tf.keras.layers.Conv2D(
        filters2, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '1')
    self.shortcut_bn = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '1',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    shortcut_x = self.shortcut_conv(input_tensor)
    shortcut_x = self.shortcut_bn(shortcut_x, training=training)

    x += shortcut_x
    return tf.nn.relu(x)


class ResnetIdBlock(tf.keras.Model):
  """Resetnet identity block."""

  def __init__(self, kernel_size, filters, stage, block, weight_decay):
    super(ResnetIdBlock, self).__init__(name='')
    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv1 = tf.keras.layers.Conv2D(
        filters1,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2a')
    self.bn1 = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

    self.conv2 = tf.keras.layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name=conv_name_base + '2b')
    self.bn2 = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class Resnet18Block(tf.keras.Model):
  """Resnet block."""

  def __init__(self,
               kernel_size,
               input_planes,
               output_planes,
               stage,
               strides=(2, 2),
               weight_decay=5e-4):
    super(Resnet18Block, self).__init__(name='')
    if output_planes > input_planes:
      self.first_block = ResnetProjBlock(
          kernel_size, (output_planes, output_planes),
          stage=stage,
          strides=strides,
          block='block_0',
          weight_decay=weight_decay)
    else:
      self.first_block = ResnetIdBlock(
          kernel_size, (output_planes, output_planes),
          stage=stage,
          block='block_0',
          weight_decay=weight_decay)
    self.second_block = ResnetIdBlock(
        kernel_size, (output_planes, output_planes),
        stage=stage,
        block='block_1',
        weight_decay=weight_decay)

  def call(self, input_tensor, training=False):
    x = self.first_block(input_tensor, training=training)
    x = self.second_block(x, training=training)
    return x


@gin.configurable
class Resnet18(tf.keras.Model):
  """The resnet model."""

  NAME = 'resnet18'

  def __init__(self,
               output_size,
               output_activation=None,
               weight_decay=5e-4,
               bn_axis=3,
               dtype=tf.float32):
    super(Resnet18, self).__init__(name='')
    del dtype   # we use it only for the sake of having a single interface.
    self._dtype = tf.float32
    self.output_size = output_size
    self.zero_pad = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), input_shape=(32, 32, 3), name='conv1_pad')
    self.conv1 = tf.keras.layers.Conv2D(
        64, (3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        name='conv1')
    self.bn1 = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        name='bn_conv1',
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)

    self.resblock1 = Resnet18Block(
        kernel_size=3,
        input_planes=64,
        output_planes=64,
        stage=2,
        strides=(1, 1),
        weight_decay=weight_decay)

    self.resblock2 = Resnet18Block(
        kernel_size=3,
        input_planes=64,
        output_planes=128,
        stage=3,
        strides=(2, 2),
        weight_decay=weight_decay)

    self.resblock3 = Resnet18Block(
        kernel_size=3,
        input_planes=128,
        output_planes=256,
        stage=4,
        strides=(2, 2),
        weight_decay=weight_decay)

    self.resblock4 = Resnet18Block(
        kernel_size=3,
        input_planes=256,
        output_planes=512,
        stage=4,
        strides=(2, 2),
        weight_decay=weight_decay)

    self.pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
    self.linear = tf.keras.layers.Dense(
        output_size,
        activation=output_activation,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='fc{}'.format(output_size))

  @property
  def dtype(self):
    return self._dtype

  def call(self, input_tensor, training=False):
    x = self.zero_pad(input_tensor)
    x = self.conv1(x)
    x = self.bn1(x, training)
    x = tf.nn.relu(x)
    x = self.resblock1(x, training=training)
    x = self.resblock2(x, training=training)
    x = self.resblock3(x, training=training)
    x = self.resblock4(x, training=training)
    x = self.pool(x)
    x = self.linear(x)
    return x


@gin.configurable
class PseudoResnet18(tf.keras.Model):
  """What they are using."""

  def __init__(self):
    super().__init__()
    resnet18 = Resnet18(output_size=144)
    resnet18.conv1 = tf.keras.layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        use_bias=False,
        name='conv1')

    self.lays = [resnet18.zero_pad,
                 resnet18.conv1,
                 resnet18.bn1,
                 tf.keras.layers.ReLU(),
                 tf.keras.layers.MaxPool2D(),
                 resnet18.resblock1,
                 tf.keras.layers.MaxPool2D(),
                 tf.keras.layers.Lambda(
                     functools.partial(tf.math.reduce_mean, axis=-1)),
                 ]

  def call(self, x):
    for lay in self.lays:
      x = lay(x)
    return x


@gin.configurable
def make_pseudo_resnet(input_shape=None, output_shape=None, dtype=None):
  del input_shape, output_shape, dtype
  return PseudoResnet18()
