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

"""CIFAR ResNet-32 model definition."""

import tensorflow as tf

_L2_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True):
  return tf.keras.regularizers.l2(
      _L2_WEIGHT_DECAY) if use_l2_regularizer else None


def _identity_block(input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    use_l2_regularizer=True,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=1e-5):
  """The identity block is the block that has no conv layer at shortcut.

  Forked from
  tensorflow_models.official.legacy.image_classification.resnet.resnet_model.

  Args:
    input_tensor: input tensor.
    kernel_size: default 3, the kernel size of middle conv layer at main path.
    filters: list of integers, the filters of 3 conv layer at main path.
    stage: integer, current stage label, used for generating layer names.
    block: 'a','b'..., current block label, used for generating layer names.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def _conv_block(input_tensor,
                kernel_size,
                filters,
                stage,
                block,
                strides=(2, 2),
                use_l2_regularizer=True,
                batch_norm_decay=0.9,
                batch_norm_epsilon=1e-5):
  """A block that has a conv layer at shortcut.

  Forked from
  tensorflow_models.official.legacy.image_classification.resnet.resnet_model.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor.
    kernel_size: default 3, the kernel size of middle conv layer at main path.
    filters: list of integers, the filters of 3 conv layer at main path.
    stage: integer, current stage label, used for generating layer names.
    block: 'a','b'..., current block label, used for generating layer names.
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  shortcut = layers.Conv2D(
      filters3, (1, 1),
      strides=strides,
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '1')(
          input_tensor)
  shortcut = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '1')(
          shortcut)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def _cifar_resnet(input_shape, num_classes, config, use_l2_regularizer=True):
  """Instantiates the ResNet architecture.

  Args:
    input_shape: Tensor size of the image input.
    num_classes: Number of classes for image classification.
    config: Config of the network.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.

  Returns:
    A Keras model instance.
  """

  img_input = layers.Input(shape=input_shape)
  x = img_input

  # channels_last
  bn_axis = 3

  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  x = layers.Conv2D(
      16, (3, 3),
      strides=(1, 1),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      name='bn_conv1')(
          x)
  x = layers.Activation('relu')(x)

  for stage, (n_layers, n_filters, stride) in enumerate(config):
    x = _conv_block(
        x,
        3, [n_filters, n_filters, n_filters],
        stage=stage + 2,
        block='a',
        strides=(stride, stride),
        use_l2_regularizer=use_l2_regularizer)
    for i in range(n_layers - 1):
      x = _identity_block(
          x,
          3, [n_filters, n_filters, n_filters],
          stage=stage + 2,
          block='bcdefghijklm'[i],
          use_l2_regularizer=use_l2_regularizer)

  x = layers.BatchNormalization(
      axis=3, momentum=0.9, epsilon=1e-5, name='final_bn')(
          x)

  rm_axes = [1, 2]
  x = layers.Lambda(
      lambda x: tf.keras.backend.mean(x, rm_axes), name='reduce_mean')(
          x)
  x = layers.Dense(
      num_classes,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc10')(
          x)

  return tf.keras.models.Model(img_input, x, name='resnet')


def cifar_resnet32(num_classes):
  """Returns CIFAR ResNet-32 for the given number of classes."""
  return _cifar_resnet(
      input_shape=(32, 32, 3),
      num_classes=num_classes,
      config=[(5, 16, 1), (5, 32, 2), (5, 64, 2)],
      use_l2_regularizer=True)
