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

"""ResNet v2 model for Keras using Batch or Group Normalization.

Related papers/blogs:
- http://arxiv.org/pdf/1603.05027v2.pdf

"""

import tensorflow as tf
import tensorflow_addons.layers.normalizations as tfa_norms
import edward2 as ed

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 1e-4


def _norm_relu(input_tensor, norm='group', training=True):
  """Helper function to make a Norm -> ReLU block."""
  if tf.keras.backend.image_data_format() == 'channels_last':
    channel_axis = 3
  else:
    channel_axis = 1

  if norm == 'group':
    x = tfa_norms.GroupNormalization(axis=channel_axis)(input_tensor)
  else:
    x = tf.keras.layers.BatchNormalization(
        axis=channel_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON)(input_tensor, training)
  return tf.keras.layers.Activation('relu')(x)


def _conv_norm_relu(input_tensor,
                    filters,
                    kernel_size,
                    strides=(1, 1),
                    norm='group',
                    training=True):
  """Helper function to make a Conv -> Norm -> ReLU block."""
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
          input_tensor)
  return _norm_relu(x, norm=norm, training=training)


def _norm_relu_conv(input_tensor,
                    filters,
                    kernel_size,
                    strides=(1, 1),
                    norm='group',
                    training=True):
  """Helper function to make a Norm -> ReLU -> Conv block."""
  x = _norm_relu(input_tensor, norm=norm, training=training)
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
          x)
  return x


def _shortcut(input_tensor, residual, norm='group', training=True):
  """Adds a shortcut between input and the residual."""
  input_shape = tf.keras.backend.int_shape(input_tensor)
  residual_shape = tf.keras.backend.int_shape(residual)

  if tf.keras.backend.image_data_format() == 'channels_last':
    row_axis = 1
    col_axis = 2
    channel_axis = 3
  else:
    channel_axis = 1
    row_axis = 2
    col_axis = 3

  stride_width = int(round(input_shape[row_axis] / residual_shape[row_axis]))
  stride_height = int(round(input_shape[col_axis] / residual_shape[col_axis]))
  equal_channels = input_shape[channel_axis] == residual_shape[channel_axis]

  shortcut = input_tensor
  # 1 X 1 conv if shape is different. Else identity.
  if stride_width > 1 or stride_height > 1 or not equal_channels:
    shortcut = tf.keras.layers.Conv2D(
        filters=residual_shape[channel_axis],
        kernel_size=(1, 1),
        strides=(stride_width, stride_height),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
            shortcut)

    if norm == 'group':
      shortcut = tfa_norms.GroupNormalization(axis=channel_axis)(shortcut)
    else:
      shortcut = tf.keras.layers.BatchNormalization(
          axis=channel_axis,
          momentum=BATCH_NORM_DECAY,
          epsilon=BATCH_NORM_EPSILON)(shortcut, training)

  return tf.keras.layers.add([shortcut, residual])


def _basic_block(input_tensor,
                 filters,
                 strides=(1, 1),
                 avoid_norm=False,
                 norm='group',
                 training=True):
  """Basic convolutional block for use on resnets with <= 34 layers."""
  if avoid_norm:
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
            input_tensor)
  else:
    x = _norm_relu_conv(
        input_tensor,
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        norm=norm,
        training=training)

  x = _norm_relu_conv(
      x,
      filters=filters,
      kernel_size=(3, 3),
      strides=strides,
      norm=norm,
      training=training)
  return _shortcut(input_tensor, x, norm=norm)


def _bottleneck_block(input_tensor,
                      filters,
                      strides=(1, 1),
                      avoid_norm=False,
                      norm='group',
                      training=True):
  """Bottleneck convolutional block for use on resnets with > 34 layers."""
  if avoid_norm:
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
            input_tensor)
  else:
    x = _norm_relu_conv(
        input_tensor,
        filters=filters,
        kernel_size=(1, 1),
        strides=strides,
        norm=norm,
        training=training)

  x = _norm_relu_conv(
      x,
      filters=filters,
      kernel_size=(3, 3),
      strides=strides,
      norm=norm,
      training=training)

  x = _norm_relu_conv(
      x,
      filters=filters * 4,
      kernel_size=(1, 1),
      strides=strides,
      norm=norm,
      training=training)
  return _shortcut(input_tensor, x, norm=norm, training=training)


def _preact_block(input_tensor,
                  filters,
                  strides=(1, 1),
                  avoid_norm=False,
                  norm='group',
                  training=True):
  """Basic convolutional block for use on resnets with <= 34 layers."""
  in_channels = input_tensor.get_shape()[-1]
  x = _norm_relu(input_tensor, norm=norm, training=training)
  is_irregular_shortcut = strides[0] != 1 or in_channels != filters
  if is_irregular_shortcut:
    shortcut = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
            x)
  else:
    shortcut = x

  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size=(3, 3),
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
          x)
  x = _norm_relu_conv(
      x,
      filters=filters,
      kernel_size=(3, 3),
      strides=(1, 1),
      norm=norm,
      training=training)
  x += shortcut
  return x


def _residual_block(input_tensor,
                    block_function,
                    filters,
                    num_blocks,
                    strides=(1, 1),
                    is_first_layer=False,
                    norm='group',
                    training=True):
  """Builds a residual block with repeating bottleneck or basic blocks."""
  strides = [strides] + [(1, 1)] * (num_blocks - 1)
  x = input_tensor
  for i in range(num_blocks):
    # avoid_norm = is_first_layer and i == 0
    avoid_norm = 0
    x = block_function(
        x,
        filters=filters,
        strides=strides[i],
        avoid_norm=avoid_norm,
        norm=norm,
        training=training)
  return x


def create_resnet(input_shape,
                  num_classes=10,
                  block='bottleneck',
                  repetitions=None,
                  initial_filters=64,
                  initial_strides=(2, 2),
                  initial_kernel_size=(7, 7),
                  initial_pooling='max',
                  norm='group',
                  dense_layer='dense',
                  dense_layer_args=None):
  """Instantiates a ResNet v2 model with Group Normalization.

  Instantiates the architecture from http://arxiv.org/pdf/1603.05027v2.pdf.
  The ResNet contains stages of residual blocks. Each residual block contains
  some number of...

  Args:
    input_shape: A tuple of length 3 describing the number of rows, columns, and
      channels of an input. Can be in channel-first or channel-last format.
    num_classes: Number of output classes.
    block: Whether to use a bottleneck or basic block within each stage.
    repetitions: A list of integers describing the number of blocks within each
      stage. If None, defaults to the resnet50 repetitions of [3, 4, 6, 3].
    initial_filters: The number of filters in the initial conv layer.
    initial_strides: The strides in the initial conv layer.
    initial_kernel_size: The kernel size for the initial conv layer.
    initial_pooling: The type of pooling after the initial conv layer.
    norm: Type of normalization to be used. Can be 'group' or 'batch'.
    dense_layer: Type of last fully connected layer Can be 'dense',
      'dense_diagcov' or 'dense_fullcov'.
    dense_layer_args: Args for the last fully connected layer.

  Returns:
    A `tf.keras.Model`.

  Raises:
    Exception: Input shape should be a tuple of length 3.
  """

  training = tf.keras.Input(shape=[], name='training')
  if len(input_shape) != 3:
    raise Exception('Input shape should be a tuple of length 3.')
  if repetitions is None:
    repetitions = [3, 4, 6, 3]

  if block == 'basic':
    block_fn = _basic_block
  elif block == 'bottleneck':
    block_fn = _bottleneck_block
  elif block == 'preact':
    block_fn = _preact_block

  img_input = tf.keras.layers.Input(shape=input_shape)
  x = _conv_norm_relu(
      img_input,
      filters=initial_filters,
      kernel_size=initial_kernel_size,
      strides=initial_strides,
      norm=norm,
      training=training)

  if initial_pooling == 'max':
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=initial_strides, padding='same')(
            x)

  filters = initial_filters

  strides = [1, 2, 2, 2]
  for i, r in enumerate(repetitions):
    x = _residual_block(
        x,
        block_fn,
        filters=filters,
        num_blocks=r,
        strides=(strides[i], strides[i]),
        is_first_layer=(i == 0),
        norm=norm,
        training=training)
    filters *= 2

  # Final activation in the residual blocks
  x = _norm_relu(x, norm=norm, training=training)

  # Classification block
  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  if dense_layer == 'dense':
    x = tf.keras.layers.Dense(
        num_classes,
        # activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        # kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=None)(
            x)
  elif dense_layer == 'dense_diagcov':
    x = ed.layers.MCSoftmaxDense(**dense_layer_args)(x)
  elif dense_layer == 'dense_fullcov':
    x = ed.layers.MCSoftmaxDenseFA(**dense_layer_args)(x)
  else:
    raise ValueError('Unkown last layer type: {}'.format(dense_layer))

  model = tf.keras.models.Model(inputs=[img_input, training], outputs=x)
  return model


def create_resnet18(input_shape,
                    num_classes,
                    norm='group',
                    dense_layer='dense',
                    dense_layer_args=None):
  """ResNet with 18 layers and basic residual blocks."""
  return create_resnet(
      input_shape,
      num_classes,
      'preact',
      repetitions=[2, 2, 2, 2],
      norm=norm,
      initial_strides=(1, 1),
      initial_kernel_size=(3, 3),
      initial_pooling='none',
      dense_layer=dense_layer,
      dense_layer_args=dense_layer_args)
