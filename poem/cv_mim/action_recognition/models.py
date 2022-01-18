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

"""Defines model architectures."""

import tensorflow as tf

layers = tf.keras.layers

# Classifier type.
TYPE_CLASSIFIER_CONVNET = 'CONVNET'
TYPE_CLASSIFIER_RESNET = 'RESNET'
TYPE_CLASSIFIER_RESTCN = 'RESTCN'
SUPPORTED_CLASSIFIER_TYPES = [
    TYPE_CLASSIFIER_CONVNET, TYPE_CLASSIFIER_RESNET, TYPE_CLASSIFIER_RESTCN
]


def get_temporal_classifier(classifier_type, **kwargs):
  """Gets classifier.

  Args:
    classifier_type: A string for classifier type.
    **kwargs: A dictionary for additional arguments.

  Returns:
    An classifier instance.
  """
  if classifier_type == TYPE_CLASSIFIER_CONVNET:
    return build_simple_temporal_model(**kwargs)
  elif classifier_type == TYPE_CLASSIFIER_RESNET:
    return build_residual_temporal_model(**kwargs)
  elif classifier_type == TYPE_CLASSIFIER_RESTCN:
    return build_residual_temporal_convolutional_model(**kwargs)
  else:
    raise ValueError('Unknown classifier: {}'.format(classifier_type))


def build_residual_block(input_layer,
                         feature_dim,
                         stride,
                         activation='relu',
                         dropout_rate=0.5,
                         **layer_kwargs):
  """Builds a residual block.

  Args:
    input_layer: A `tf.keras.layers.Layer` object for the input layer.
    feature_dim: An integer for the feature dimension of all layers.
    stride: An integer for the stride.
    activation: A string for the activation function.
    dropout_rate: A float between 0 and 1 for the dropout rate.
    **layer_kwargs: A dictionary for additional layer arguments.

  Returns:
    A configured model.
  """
  layer_kwargs.update(dict(padding='same'))

  conv_x = layers.Conv1D(
      filters=feature_dim,
      kernel_size=7,
      strides=stride,
      **layer_kwargs)(input_layer)
  conv_x = layers.BatchNormalization()(conv_x)
  conv_x = layers.Activation(activation)(conv_x)
  conv_x = layers.Dropout(dropout_rate)(conv_x)

  conv_y = layers.Conv1D(
      filters=feature_dim, kernel_size=5, **layer_kwargs)(conv_x)
  conv_y = layers.BatchNormalization()(conv_y)
  conv_y = layers.Activation(activation)(conv_y)
  conv_y = layers.Dropout(dropout_rate)(conv_y)

  conv_z = layers.Conv1D(
      filters=feature_dim, kernel_size=3, **layer_kwargs)(conv_y)
  conv_z = layers.BatchNormalization()(conv_z)

  shortcut_y = layers.Conv1D(
      filters=feature_dim,
      kernel_size=1,
      strides=stride,
      **layer_kwargs)(input_layer)
  shortcut_y = layers.BatchNormalization()(shortcut_y)

  output_layer = layers.add([shortcut_y, conv_z])
  output_layer = layers.Activation(activation)(output_layer)
  return output_layer


def build_residual_temporal_model(input_shape,
                                  num_classes,
                                  channel_depths=(64, 128, 256),
                                  activation='relu',
                                  temporal_stride=2,
                                  dropout_rate=0.5):
  """Builds a residual temporal model for classifier.

  Reference:
    Fawaz et al. Deep learning for time series classification: a review.
    https://arxiv.org/pdf/1809.04356.pdf.

  Args:
    input_shape: A tuple for the shape of inputs.
    num_classes: An integer for the number of classes.
    channel_depths: A tuple for the feature dimension of all layers.
    activation: A string for the activation function.
    temporal_stride: An integer for the stride of temporal dimension.
    dropout_rate: A float between 0 and 1 for the dropout rate.

  Returns:
    A configured model.
  """
  layer_kwargs = dict(
      kernel_initializer='he_normal', bias_initializer='he_normal')

  input_layer = tf.keras.Input(input_shape)
  output_layer = layers.Conv1D(
      filters=channel_depths[0], kernel_size=7, padding='same', **layer_kwargs)(
          input_layer)
  output_layer = layers.BatchNormalization()(output_layer)
  output_layer = layers.Activation(activation)(output_layer)

  stride_layout = [temporal_stride] * (len(channel_depths) - 1) + [1]
  for feature_dim, stride in zip(channel_depths, stride_layout):
    output_layer = build_residual_block(
        output_layer,
        feature_dim,
        stride,
        activation=activation,
        dropout_rate=dropout_rate,
        **layer_kwargs)

  output_layer = layers.GlobalAveragePooling1D()(output_layer)
  output_layer = layers.Dense(num_classes, **layer_kwargs)(output_layer)
  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


def build_simple_temporal_model(input_shape,
                                num_classes,
                                channel_depths=(64, 128, 256),
                                activation='relu',
                                temporal_stride=2,
                                kernel_size=7,
                                dropout_rate=0.5):
  """Builds a simple temporal model for classifier.

  Args:
    input_shape: A tuple for the shape of inputs.
    num_classes: An integer for the number of classes.
    channel_depths: A tuple for the feature dimension of all layers.
    activation: A string for the activation function.
    temporal_stride: An integer for the stride of temporal dimension.
    kernel_size: An integer for the kernel size of the 1D Conv layers.
    dropout_rate: A float between 0 and 1 for the dropout rate.

  Returns:
    A configured model.
  """
  layer_kwargs = dict(
      kernel_initializer='he_normal', bias_initializer='he_normal')

  input_layer = tf.keras.Input(input_shape)
  output_layer = input_layer
  for feature_dim in channel_depths:
    output_layer = layers.Conv1D(
        filters=feature_dim,
        kernel_size=kernel_size,
        strides=temporal_stride,
        padding='same',
        **layer_kwargs)(output_layer)
    output_layer = layers.BatchNormalization()(output_layer)
    output_layer = layers.Activation(activation)(output_layer)
    output_layer = layers.Dropout(dropout_rate)(output_layer)

  output_layer = layers.GlobalAveragePooling1D()(output_layer)
  output_layer = layers.Dense(num_classes, **layer_kwargs)(output_layer)
  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


def build_residual_temporal_convolutional_model(
    input_shape,
    num_classes,
    pooling=True,
    activation='relu',
    dropout_rate=0.5,
    kernel_regularizer=tf.keras.regularizers.l1(1.e-4)):
  """Builds a residual temporal convolutional model for classifier (Res-TCN).

  Reference:
    Kim et al. Interpretable 3D Human Action Analysis with Temporal
    Convolutional Networks. https://arxiv.org/pdf/1704.04516.pdf.

  Args:
    input_shape: A tuple for the shape of inputs.
    num_classes: An integer for the number of classes.
    pooling: A boolean for whether to use average pooling.
    activation: A string for the activation function.
    dropout_rate: A float between 0 and 1 for the dropout rate.
    kernel_regularizer: A `tf.keras.regularizers` instance for regularizer.

  Returns:
    A configured model.
  """
  row_axis = 1
  channel_axis = 2

  # Each tuple in config represents (stride, kernel_size, feature_dim).
  config = [[(1, 8, 64)], [(1, 8, 64)], [(1, 8, 64)], [(2, 8, 128)],
            [(1, 8, 128)], [(1, 8, 128)], [(2, 8, 256)], [(1, 8, 256)],
            [(1, 8, 256)]]
  initial_stride = 1
  initial_kernel_size = 8
  initial_num = 64

  model_input = tf.keras.Input(shape=input_shape)
  model = layers.Conv1D(
      initial_num,
      kernel_size=initial_kernel_size,
      strides=initial_stride,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=kernel_regularizer)(
          model_input)

  for depth in range(len(config)):
    for stride, kernel_size, feature_dim in config[depth]:
      bn = layers.BatchNormalization(axis=channel_axis)(model)
      relu = layers.Activation(activation)(bn)
      dr = layers.Dropout(dropout_rate)(relu)
      res = layers.Conv1D(
          feature_dim,
          kernel_size=kernel_size,
          strides=stride,
          padding='same',
          kernel_initializer='he_normal',
          kernel_regularizer=kernel_regularizer)(
              dr)

      res_shape = tf.keras.backend.int_shape(res)
      model_shape = tf.keras.backend.int_shape(model)
      if res_shape[channel_axis] != model_shape[channel_axis]:
        model = layers.Conv1D(
            feature_dim,
            kernel_size=1,
            strides=stride,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer)(
                model)
      model = layers.add([model, res])

  bn = layers.BatchNormalization(axis=channel_axis)(model)
  model = layers.Activation(activation)(bn)

  if pooling:
    pool_window_shape = tf.keras.backend.int_shape(model)
    gap = layers.AveragePooling1D(pool_window_shape[row_axis], strides=1)(model)
    flatten = layers.Flatten()(gap)
  else:
    flatten = layers.Flatten()(model)
  dense = layers.Dense(
      units=num_classes, activation='softmax', kernel_initializer='he_normal')(
          flatten)

  return tf.keras.models.Model(inputs=model_input, outputs=dense)
