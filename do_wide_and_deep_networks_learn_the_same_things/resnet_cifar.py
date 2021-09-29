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

"""CIFAR10 ResNet architecture.
"""
import tensorflow.compat.v2 as tf


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 weight_decay=0,
                 sync_bn=False):
  """2D Convolution-Batch Normalization-Activation stack builder.

    Arguments:
        inputs: (tensor) input tensor from input image or previous layer
        num_filters: (int) Conv2D number of filters
        kernel_size: (int) Conv2D square kernel dimensions
        strides: (int) Conv2D square stride dimensions
        activation: (string) activation name
        batch_normalization: (bool) whether to include batch normalization
        conv_first: (bool) conv-bn-activation (True) or
            bn-activation-conv (False)
        sync_bn: (bool) Whether to use synchronous batch normalization.
    Returns:
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
    x = bn_layer(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
  else:
    x = bn_layer(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
    x = conv(x)
  return x


class CustomModel(tf.keras.Model):

  def __init__(self, inputs, outputs):
    super(CustomModel, self).__init__(inputs=inputs, outputs=outputs)
    self.all_ids = []

  def train_step(self, data):
    """Modify tf.keras.Model train_step() to save image ids in each minibatch."""
    id, x, y = data
    self.all_ids.append(id.numpy())
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      # Compute the loss value (the loss function is configured in `compile()`)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}


def ResNet_CIFAR(depth,
                 width_multiplier,
                 weight_decay,
                 num_classes,
                 input_shape=(32, 32, 3),
                 sync_bn=False,
                 use_residual=True,
                 save_image=False):
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
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_filters = int(round(16 * width_multiplier))
  num_res_blocks = int((depth - 2) / 6)

  inputs = tf.keras.Input(shape=input_shape)
  x = resnet_layer(
      inputs=inputs,
      num_filters=num_filters,
      weight_decay=weight_decay,
      sync_bn=sync_bn)
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          weight_decay=weight_decay,
          sync_bn=sync_bn)
      y = resnet_layer(
          inputs=y,
          num_filters=num_filters,
          activation=None,
          weight_decay=weight_decay,
          sync_bn=sync_bn)
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
            weight_decay=weight_decay)
      if use_residual:
        x = tf.keras.layers.add([x, y])
      else:
        x = y
      x = tf.keras.layers.Activation('relu')(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  y = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          y)

  # Instantiate model.
  if save_image:
    model = CustomModel(inputs=inputs, outputs=outputs)
  else:
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model
