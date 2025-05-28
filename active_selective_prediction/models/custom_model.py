# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Custom Model Classes."""

from typing import Tuple
import tensorflow as tf


class SimpleMLP(tf.keras.Model):
  """Simple MLP model."""

  def __init__(
      self,
      input_shape,
      num_classes,
      num_layers = 2,
      name = 'simple_mlp',
  ):
    super().__init__(name=name)
    self.feature_dim = 128
    self.feature_layers = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            input_shape=input_shape,
        ),
        tf.keras.layers.BatchNormalization(),
    ])
    for _ in range(num_layers):
      self.feature_layers.add(
          tf.keras.layers.Dense(
              256,
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
          )
      )
    self.feature_layers.add(tf.keras.layers.Dense(128, activation='relu'))
    self.feature_layers.add(tf.keras.layers.Dropout(0.5))
    self.feature_layers.add(
        tf.keras.layers.Dense(self.feature_dim, activation='relu')
    )
    self.feature_layers.add(tf.keras.layers.Dropout(0.5))
    self.fc = tf.keras.layers.Dense(num_classes, activation=None)

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = self.feature_layers(inputs, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out


class SimpleConvNet(tf.keras.Model):
  """Simple ConvNet model."""

  def __init__(
      self,
      input_shape,
      num_classes,
      name = 'simple_convnet',
  ):
    super().__init__(name=name)
    self.feature_dim = 128
    self.conv_1 = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        activation='relu',
        input_shape=input_shape,
    )
    self.conv_2 = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu'
    )
    self.pool_1 = tf.keras.layers.MaxPooling2D((2, 2))
    self.conv_3 = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu'
    )
    self.batchnorm_1 = tf.keras.layers.BatchNormalization()
    self.conv_4 = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu'
    )
    self.pool_2 = tf.keras.layers.MaxPooling2D((2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.dense_1 = tf.keras.layers.Dense(256, activation='relu')
    self.dropout_1 = tf.keras.layers.Dropout(0.5)
    self.dense_2 = tf.keras.layers.Dense(self.feature_dim, activation='relu')
    self.dropout_2 = tf.keras.layers.Dropout(0.5)
    self.fc = tf.keras.layers.Dense(num_classes, activation=None)

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = self.conv_1(inputs, training=training)
    out = self.conv_2(out, training=training)
    out = self.pool_1(out, training=training)
    out = self.conv_3(out, training=training)
    out = self.batchnorm_1(out, training=training)
    out = self.conv_4(out, training=training)
    out = self.pool_2(out, training=training)
    out = self.flatten(out, training=training)
    out = self.dense_1(out, training=training)
    out = self.dropout_1(out, training=training)
    out = self.dense_2(out, training=training)
    out = self.dropout_2(out, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out


class ResNetLayer(tf.keras.layers.Layer):
  """ResNet layer."""

  def __init__(
      self,
      num_filters = 16,
      kernel_size = 3,
      strides = 1,
      activation = 'relu',
      batch_normalization = True,
      conv_first = True,
  ):
    super().__init__()
    self.batch_normalization = batch_normalization
    self.activation = activation
    self.conv_first = conv_first
    self.conv = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training = False):
    """Gets model output."""
    x = inputs
    if self.conv_first:
      x = self.conv(x, training=training)
      if self.batch_normalization:
        x = self.bn(x, training=training)
      if self.activation is not None:
        x = tf.keras.layers.Activation(self.activation)(x)
    else:
      if self.batch_normalization:
        x = self.bn(x, training=training)
      if self.activation is not None:
        x = tf.keras.layers.Activation(self.activation)(x)
      x = self.conv(x, training=training)
    return x


class BasicBlock(tf.keras.Model):
  """ResNet Block."""

  def __init__(
      self,
      activation,
      batch_normalization,
      strides,
      num_filters_in,
      num_filters_out,
      first_block,
  ):
    super().__init__()
    self.first_block = first_block
    self.layer_1 = ResNetLayer(
        num_filters=num_filters_in,
        kernel_size=1,
        strides=strides,
        activation=activation,
        batch_normalization=batch_normalization,
        conv_first=False,
    )
    self.layer_2 = ResNetLayer(
        num_filters=num_filters_in,
        conv_first=False,
    )
    self.layer_3 = ResNetLayer(
        num_filters=num_filters_out,
        kernel_size=1,
        conv_first=False,
    )
    if self.first_block:
      # linear projection residual shortcut connection to match
      # changed dims
      self.shortcut_layer = ResNetLayer(
          num_filters=num_filters_out,
          kernel_size=1,
          strides=strides,
          activation=None,
          batch_normalization=False,
      )

  def call(self, inputs, training = False):
    """Gets model output."""
    x = inputs
    y = self.layer_1(x, training=training)
    y = self.layer_2(y, training=training)
    y = self.layer_3(y, training=training)
    if self.first_block:
      x = self.shortcut_layer(x, training=training)
    x = tf.keras.layers.add([x, y])
    return x


class CifarResNet(tf.keras.Model):
  """ResNet model for the CIFAR datasets."""

  def __init__(
      self,
      input_shape,
      num_classes,
      depth = 20,
      mean = (0.4915, 0.4823, 0.4468),
      variance = (0.2470, 0.2435, 0.2616),
      name = 'resnet20',
  ):
    super().__init__(name=name)
    self.feature_dim = 128
    if (depth - 2) % 9 != 0:
      raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)
    self.input_normalize = tf.keras.layers.Normalization(
        axis=-1,
        mean=tf.constant(mean, dtype=tf.float32),
        variance=tf.constant(variance, dtype=tf.float32),
    )
    self.conv = tf.keras.layers.Conv2D(
        num_filters_in,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        input_shape=input_shape,
    )
    self.bn_1 = tf.keras.layers.BatchNormalization()
    # Instantiate the stack of residual units
    self.res_units = []
    for stage in range(3):
      for res_block in range(num_res_blocks):
        activation = 'relu'
        batch_normalization = True
        strides = 1
        if stage == 0:
          num_filters_out = num_filters_in * 4
          if res_block == 0:  # first layer and first stage
            activation = None
            batch_normalization = False
        else:
          num_filters_out = num_filters_in * 2
          if res_block == 0:  # first layer but not first stage
            strides = 2  # downsample
        self.res_units.append(
            BasicBlock(
                activation=activation,
                batch_normalization=batch_normalization,
                strides=strides,
                num_filters_in=num_filters_in,
                num_filters_out=num_filters_out,
                first_block=(res_block == 0),
            )
        )
      num_filters_in = num_filters_out
    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    self.bn_2 = tf.keras.layers.BatchNormalization()
    self.avg_pool2d = tf.keras.layers.AveragePooling2D(pool_size=8)
    self.flatten = tf.keras.layers.Flatten()
    self.dense_1 = tf.keras.layers.Dense(256, activation='relu')
    self.dropout_1 = tf.keras.layers.Dropout(0.5)
    self.dense_2 = tf.keras.layers.Dense(self.feature_dim, activation='relu')
    self.dropout_2 = tf.keras.layers.Dropout(0.5)
    self.fc = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer='he_normal',
    )

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = self.input_normalize(inputs)
    out = self.conv(out, training=training)
    out = self.bn_1(out, training=training)
    out = tf.keras.layers.Activation('relu')(out)
    for unit in self.res_units:
      out = unit(out, training=training)
    out = self.bn_2(out, training=training)
    out = tf.keras.layers.Activation('relu')(out)
    out = self.avg_pool2d(out)
    out = self.flatten(out)
    out = self.dense_1(out, training=training)
    out = self.dropout_1(out, training=training)
    out = self.dense_2(out, training=training)
    out = self.dropout_2(out, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out


class DenseNet(tf.keras.Model):
  """DenseNet model."""

  def __init__(
      self,
      input_shape,
      num_classes,
      weights = 'imagenet',
      densenet_name = 'DenseNet121',
      name = 'densenet',
  ):
    super().__init__(name=name)
    self.feature_dim = 256
    if densenet_name == 'DenseNet121':
      self.backbone = tf.keras.applications.DenseNet121(
          include_top=False,
          weights=weights,
          input_tensor=None,
          input_shape=input_shape,
          pooling=None,
          classes=1000,
          classifier_activation='softmax',
      )
    else:
      raise ValueError(f'Not supported DenseNet: {densenet_name}')
    self.flatten = tf.keras.layers.Flatten()
    self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
    self.dropout_1 = tf.keras.layers.Dropout(0.5)
    self.dense_2 = tf.keras.layers.Dense(self.feature_dim, activation='relu')
    self.dropout_2 = tf.keras.layers.Dropout(0.5)
    self.fc = tf.keras.layers.Dense(num_classes, activation=None)

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = tf.keras.applications.densenet.preprocess_input(inputs)
    out = self.backbone(out, training=training)
    out = self.flatten(out, training=training)
    out = self.dense_1(out, training=training)
    out = self.dropout_1(out, training=training)
    out = self.dense_2(out, training=training)
    out = self.dropout_2(out, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out


class ResNet(tf.keras.Model):
  """ResNet model."""

  def __init__(
      self,
      input_shape,
      num_classes,
      weights = 'imagenet',
      resnet_name = 'ResNet50',
      name = 'resnet',
  ):
    super().__init__(name=name)
    self.feature_dim = 512
    if resnet_name == 'ResNet50':
      self.backbone = tf.keras.applications.ResNet50(
          include_top=False,
          weights=weights,
          input_tensor=None,
          input_shape=input_shape,
          pooling=None,
          classes=1000,
          classifier_activation='softmax',
      )
    else:
      raise ValueError(f'Not supported ResNet: {resnet_name}')
    self.flatten = tf.keras.layers.Flatten()
    self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
    self.dropout_1 = tf.keras.layers.Dropout(0.5)
    self.dense_2 = tf.keras.layers.Dense(self.feature_dim, activation='relu')
    self.dropout_2 = tf.keras.layers.Dropout(0.5)
    self.fc = tf.keras.layers.Dense(num_classes, activation=None)

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = tf.keras.applications.resnet.preprocess_input(inputs)
    out = self.backbone(out, training=training)
    out = self.flatten(out, training=training)
    out = self.dense_1(out, training=training)
    out = self.dropout_1(out, training=training)
    out = self.dense_2(out, training=training)
    out = self.dropout_2(out, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out


class RoBertaMLP(tf.keras.Model):
  """RoBerta MLP model."""

  def __init__(
      self,
      input_shape,
      num_classes,
      num_layers = 4,
      name = 'roberta_mlp',
  ):
    super().__init__(name=name)
    self.feature_dim = 128
    self.feature_layers = tf.keras.Sequential([
        tf.keras.layers.Dense(
            768,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            input_shape=input_shape,
        ),
        tf.keras.layers.BatchNormalization(),
    ])
    for _ in range(num_layers):
      self.feature_layers.add(
          tf.keras.layers.Dense(
              256,
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-6),
          )
      )
    self.feature_layers.add(tf.keras.layers.Dense(128, activation='relu'))
    self.feature_layers.add(tf.keras.layers.Dropout(0.5))
    self.feature_layers.add(
        tf.keras.layers.Dense(self.feature_dim, activation='relu')
    )
    self.feature_layers.add(tf.keras.layers.Dropout(0.5))
    self.fc = tf.keras.layers.Dense(num_classes, activation=None)

  def get_output_and_feature(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output and feature."""
    feature = self.get_feature(inputs, training=training)
    cls_out = self.fc(feature, training=training)
    cls_out = cls_out / temperature
    cls_out = tf.nn.softmax(cls_out, axis=1)
    return cls_out, feature

  def get_feature(self, inputs, training = False):
    """Gets model feature."""
    out = self.feature_layers(inputs, training=training)
    return out

  def call(
      self, inputs, training = False, temperature = 1.0
  ):
    """Gets model output."""
    out = self.get_feature(inputs, training=training)
    out = self.fc(out, training=training)
    out = out / temperature
    out = tf.nn.softmax(out, axis=1)
    return out
