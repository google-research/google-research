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

# Lint as: python3
"""MNIST example with compression op in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from graph_compression.compression_lib import compression_op as compression
from graph_compression.compression_lib import compression_wrapper
from graph_compression.compression_lib.keras_layers import layers as compression_layers

flags.DEFINE_boolean('use_lennet', True, 'If set use lennet model')
FLAGS = flags.FLAGS


class CompressedModelV2(tf.keras.Model):
  """A two layer compressed model that consists of two CompressedLinearLayer."""

  def __init__(self, num_hidden_nodes, num_classes, compression_obj,
               compression_flag=True):
    """Initializer.

    Args:
      num_hidden_nodes: int
      num_classes: int
      compression_obj: a matrix compression object obtained by calling
          compression_wrapper
      compression_flag: if True compressed model will be used
    """
    super().__init__()
    self.num_hidden_nodes = num_hidden_nodes
    self.num_classes = num_classes
    self.compression_obj = compression_obj

    compression_flag = True

    if compression_flag:
      self.layer_1 = compression_layers.CompressedDense(
          num_hidden_nodes, compression_obj=compression_obj, activation='relu')
      self.layer_2 = compression_layers.CompressedDense(
          num_classes, compression_obj=compression_obj)
    else:
      self.layer_1 = tf.keras.layers.Dense(num_hidden_nodes, activation='relu')
      self.layer_2 = tf.keras.layers.Dense(num_classes)

    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs):
    x = self.layer_1(inputs)
    x = self.softmax(self.layer_2(x))
    return x

  def run_alpha_update(self, step_number):
    """Run alpha update for all compressed layers.

    Args:
      step_number: training step number, int
    """
    self.layer_1.run_alpha_update(step_number)
    self.layer_2.run_alpha_update(step_number)


def compressed_lenet5(input_shape, num_classes, compression_obj):
  """Builds Compressed version of LeNet5."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = compression_layers.CompressedConv2D(
      6,
      kernel_size=5,
      padding='SAME',
      activation='relu',
      compression_obj=compression_obj)(
          inputs)
  pool1 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding='SAME')(
          conv1)
  conv2 = compression_layers.CompressedConv2D(
      16,
      kernel_size=5,
      padding='SAME',
      activation='relu',
      compression_obj=compression_obj)(
          pool1)
  pool2 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding='SAME')(
          conv2)
  conv3 = compression_layers.CompressedConv2D(
      120,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      compression_obj=compression_obj)(
          pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = compression_layers.CompressedDense(
      84, activation=tf.nn.relu, compression_obj=compression_obj)(
          flatten)
  logits = tf.keras.layers.Dense(num_classes)(dense1)
  outputs = tf.keras.layers.Softmax()(logits)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def lenet5(input_shape, num_classes):
  """Builds LeNet5."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = tf.keras.layers.Conv2D(
      6, kernel_size=5, padding='SAME', activation='relu')(
          inputs)
  pool1 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding='SAME')(
          conv1)
  conv2 = tf.keras.layers.Conv2D(
      16, kernel_size=5, padding='SAME', activation='relu')(
          pool1)
  pool2 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding='SAME')(
          conv2)
  conv3 = tf.keras.layers.Conv2D(
      120, kernel_size=5, padding='SAME', activation=tf.nn.relu)(
          pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)
  logits = tf.keras.layers.Dense(num_classes)(dense1)
  outputs = tf.keras.layers.Softmax()(logits)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def main(argv):
  del argv  # unused

  tf.enable_v2_behavior()

  # Load MNIST data.
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  if not FLAGS.use_lennet:
    x_train = x_train.reshape(60000, 784).astype('float32')

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

  # Define model.

  num_hidden_nodes = 64
  num_classes = 10

  hparams = ('name=mnist_compression,'
             'prune_option=compression,'
             'input_block_size=20,'
             'rank=2,'
             'compression_option=9')

  compression_hparams = compression.InputCompressionOp.get_default_hparams(
  ).parse(hparams)
  compression_obj = compression_wrapper.get_apply_compression(
      compression_hparams, global_step=0)

  if not FLAGS.use_lennet:
    compressed_model = CompressedModelV2(num_hidden_nodes, num_classes,
                                         compression_obj)
  else:
    compressed_model = compressed_lenet5([28, 28, 1], num_classes,
                                         compression_obj)

  optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  epochs = 10

  step_number = 0
  for epoch in range(epochs):
    for x, y in train_dataset:
      with tf.GradientTape() as t:
        loss_value = loss(y, compressed_model(x))
      grads = t.gradient(loss_value, compressed_model.trainable_variables)
      optimizer.apply_gradients(
          zip(grads, compressed_model.trainable_variables))

      # compressed_model.run_alpha_update(step_number)

      step_number += 1
    print('Training loss at epoch {} is {}.'.format(epoch, loss_value))


if __name__ == '__main__':
  app.run(main)
