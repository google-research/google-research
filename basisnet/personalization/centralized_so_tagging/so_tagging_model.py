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

"""Models for Stackoverflow dataset tag prediction task."""
from typing import Optional
import tensorflow as tf

CLIENT_NUM = 500000


# TODO(hongyouc): take it out to a common location.
# The BasisDense layer is duplicated to
# photos/vision/curation/research/groupwise_fed/centralized_emnist
class BasisDense(tf.keras.layers.Layer):
  """A Dense layer with Basis kernels for each example in a mini-batch."""

  def __init__(self,
               units,
               num_basis=1,
               kernel_initializer=None,
               use_bias=True,
               activation=None,
               ):

    super(BasisDense, self).__init__()

    self.units = units
    self._num_basis = num_basis
    self.kernel_initializer = kernel_initializer
    self.use_bias = use_bias
    self.activation = activation

  def build(self, input_shape):
    last_dim = input_shape[-1]

    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units, self._num_basis],
        initializer=self.kernel_initializer,
        trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          trainable=True)
    else:
      self.bias = None

  def call(self, x, c_prob):
    c_prob = tf.reshape(c_prob, [-1, 1, 1, self._num_basis])
    composed_kernel = tf.reshape(self.kernel,
                                 [1, -1, self.units, self._num_basis])
    composed_kernel = tf.keras.backend.sum(composed_kernel * c_prob, axis=-1)
    # [None, last_dim, self.units]

    y = tf.matmul(tf.expand_dims(x, 1), composed_kernel)
    if self.use_bias:
      y = tf.keras.backend.bias_add(y, self.bias)
    if self.activation is not None:
      y = tf.keras.activations.get(self.activation)(y)
    y = tf.squeeze(y, 1)
    return y

  def get_config(self):
    return {'units': self.units}


def create_logistic_basis_model(vocab_tokens_size,
                                vocab_tags_size,
                                num_basis,
                                seed = 0):
  """Logistic regression to predict tags of StackOverflow with a BasisNet.

  Args:
    vocab_tokens_size: Size of token vocabulary to use.
    vocab_tags_size: Size of tag vocabulary to use.
    num_basis: The number of the channels.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.
  Returns:
    A `tf.keras.Model`.
  """
  input_x = tf.keras.layers.Input(
      shape=(vocab_tokens_size,), dtype=tf.float32, name='input_x')
  input_id = tf.keras.layers.Input(
      shape=(), dtype=tf.int64, name='input_id')
  basis_vec = tf.keras.layers.Embedding(
      CLIENT_NUM, num_basis, name='embedding')(
          tf.expand_dims(input_id, -1))

  basis_vec = tf.reshape(basis_vec, shape=[-1, 1, 1, num_basis])
  basis_prob = tf.keras.layers.Softmax()(basis_vec)
  x = tf.keras.layers.Dense(
      100,
      activation='relu',
      input_shape=(vocab_tokens_size,),
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
          input_x)
  x = BasisDense(
      100,
      num_basis=num_basis,
      activation='relu',
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
          x, basis_prob)
  y = tf.keras.layers.Dense(
      vocab_tags_size,
      activation='sigmoid',
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(x)

  return tf.keras.Model(inputs=[input_x, input_id], outputs=[y])


def create_logistic_model(vocab_tokens_size,
                          vocab_tags_size,
                          seed = 0):
  """Logistic regression to predict tags of StackOverflow.

  Args:
    vocab_tokens_size: Size of token vocabulary to use.
    vocab_tags_size: Size of tag vocabulary to use.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.
  Returns:
    A `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          100,
          activation='relu',
          input_shape=(vocab_tokens_size,),
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
      tf.keras.layers.Dense(
          100,
          activation='relu',
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
      tf.keras.layers.Dense(
          vocab_tags_size,
          activation='sigmoid',
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
  ])
  return model


def create_original_logistic_model(vocab_tokens_size,
                                   vocab_tags_size,
                                   seed = 0):
  """A single layer model for logistic regression to predict tags of StackOverflow.

  Args:
    vocab_tokens_size: Size of token vocabulary to use.
    vocab_tags_size: Size of tag vocabulary to use.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.
  Returns:
    A `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          vocab_tags_size,
          activation='sigmoid',
          input_shape=(vocab_tokens_size,),
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
  ])
  return model
