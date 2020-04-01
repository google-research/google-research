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

FLAGS = flags.FLAGS


class CompressedLinearLayer(tf.keras.layers.Layer):
  """A Compressed linear layer with the compression op.

  In a CompressedLinearLayer, the W matrix is replaced by the compressed version
  of W, where specific form of the compressed version of W is determined by the
  compressor. For example, if compressor is
  compression_op.LowRankDecompMatrixCompressor, then W is replaced by
  alpha * W + (1 - alpha) * tf.matmul(B, C), see compression_op.py for more
  details.
  """

  def __init__(self, input_dim, num_hidden_nodes, compressor):
    """Initializer.

    Args:
      input_dim: int
      num_hidden_nodes: int
      compressor: a matrix compressor object (instance of a subclass of
        compression_op.MatrixCompressorInferface)
    """
    super(CompressedLinearLayer, self).__init__()
    self.num_hidden_nodes = num_hidden_nodes
    self.compressor = compressor
    self.w = self.add_weight(
        shape=(input_dim, self.num_hidden_nodes),
        initializer='random_normal',
        trainable=True)
    self.b = self.add_weight(
        shape=(self.num_hidden_nodes,),
        initializer='random_normal',
        trainable=True)

  def set_up_variables(self):
    """Set up variables used by compression_op."""
    self.compression_op = compression.CompressionOpEager()
    self.compression_op.set_up_variables(self.w, self.compressor)

  @tf.function
  def call(self, inputs):
    self.compressed_w = self.compression_op.get_apply_compression()
    return tf.matmul(inputs, self.compressed_w) + self.b

  def run_alpha_update(self, step_number):
    """Run alpha update for the alpha parameter in compression_op.

    Args:
      step_number: step number in the training process.
    Note: This method should only be called during training.
    """
    self.compression_op.run_update_step(step_number)


class CompressedModel(tf.keras.Model):
  """A two layer compressed model that consists of two CompressedLinearLayer."""

  def __init__(self, input_dim, num_hidden_nodes, num_classes, compressor):
    """Initializer.

    Args:
      input_dim: int
      num_hidden_nodes: int
      num_classes: int
      compressor: a matrix compressor object (instance of a subclass of
        compression_op.MatrixCompressorInferface)
    """
    super(CompressedModel, self).__init__()
    self.layer_1 = CompressedLinearLayer(input_dim, num_hidden_nodes,
                                         compressor)
    self.layer_1.set_up_variables()
    self.activation_1 = tf.keras.layers.ReLU()

    self.layer_2 = CompressedLinearLayer(num_hidden_nodes, num_classes,
                                         compressor)
    self.layer_2.set_up_variables()

    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs):
    x = self.activation_1(self.layer_1(inputs))
    x = self.softmax(self.layer_2(x))
    return x

  def run_alpha_update(self, step_number):
    """Run alpha update for all compressed layers.

    Args:
      step_number: training step number, int
    """
    self.layer_1.run_alpha_update(step_number)
    self.layer_2.run_alpha_update(step_number)


def main(argv):
  del argv  # unused

  tf.enable_v2_behavior()

  # Load MNIST data.
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0
  x_train = x_train.reshape(60000, 784).astype('float32')

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

  # Define model.
  input_dim = 28 * 28
  num_hidden_nodes = 50
  num_classes = 10

  lowrank_compressor = compression.LowRankDecompMatrixCompressor(
      compression.LowRankDecompMatrixCompressor.get_default_hparams())
  compressed_model = CompressedModel(input_dim, num_hidden_nodes, num_classes,
                                     lowrank_compressor)

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

      compressed_model.run_alpha_update(step_number)

      step_number += 1
    print('Training loss at epoch {} is {}.'.format(epoch, loss_value))


if __name__ == '__main__':
  app.run(main)
