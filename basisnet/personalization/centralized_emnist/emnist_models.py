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

"""Models for EMNIST dataset."""
import functools

from typing import Optional
import tensorflow as tf

NUM_CLIENTS = 3401
EMNIST_MODELS = ['cnn', '2nn', '1m_cnn']


def get_model_builder(task_spec,
                      only_digits,
                      model='cnn',
                      batch_size=256,
                      with_dist=False,
                      global_embedding_only=False):
  """Builds a model function for the EMNIST dataset.

  Args:
    task_spec: task_spec objects for flags.
    only_digits: the EMNIST dataset split.
    model: model type.
    batch_size: batch size for the dataset.
    with_dist: using label distributions as inputs.
    global_embedding_only: learn with a global embedding, aways use 0.

  Returns:
    A model_builder function
  """

  if model == 'cnn':
    if task_spec.num_basis > 1:
      model_builder = functools.partial(
          create_basis_conv_dropout_model,
          num_basis=task_spec.num_basis,
          num_filters_expand=task_spec.num_filters_expand,
          temp=task_spec.temp,
          batch_size=batch_size,
          global_embedding_only=global_embedding_only,
          with_dist=with_dist,
          only_digits=only_digits)
    else:
      model_builder = functools.partial(
          create_conv_dropout_model,
          num_filters_expand=task_spec.num_filters_expand,
          only_digits=only_digits)
  elif model == '2nn':
    model_builder = functools.partial(
        create_two_hidden_layer_model, only_digits=only_digits)
  elif model == '1m_cnn':
    model_builder = functools.partial(
        create_1m_cnn_model, only_digits=only_digits)
  else:
    raise ValueError(
        'Cannot handle model flag [{!s}], must be one of {!s}.'.format(
            model, EMNIST_MODELS))
  return model_builder


def create_basis_conv_dropout_model(only_digits = True,
                                    num_filters_expand = 1,
                                    num_basis = 4,
                                    temp = 1.0,
                                    global_embedding_only = False,
                                    with_dist = False,
                                    batch_size = 256,
                                    seed = 0):
  """Convolutional Basis model with droupout for EMNIST experiments.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    num_filters_expand: A factor to expand the number of the channels.
    num_basis: The number of the bases.
    temp: temperature of the Softmax for the client embedding.
    global_embedding_only: learn with a global embedding, aways use 0.
      Still create the whole embedding for consistency of loading checkpoints.
    with_dist: whether the inputs use label distributions or not.
    batch_size: specify batch size for inputs reshaping.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)
  input_x = tf.keras.layers.Input(
      shape=(28, 28, 1),
      dtype=tf.float32,
      batch_size=batch_size,
      name='input_x')
  input_id = tf.keras.layers.Input(
      shape=(1,), dtype=tf.int64, batch_size=batch_size, name='input_id')

  embeddings_initializer = tf.keras.initializers.RandomUniform()
  basis_embeddings = tf.keras.layers.Embedding(
      NUM_CLIENTS,
      num_basis,
      embeddings_initializer=embeddings_initializer,
      name='embedding')

  if global_embedding_only:
    basis_vec = basis_embeddings(tf.zeros_like(input_id))
  else:
    basis_vec = basis_embeddings(input_id)

  basis_vec = tf.reshape(basis_vec, shape=[-1, 1, 1, 1, num_basis])

  if with_dist:
    input_dist = tf.keras.layers.Input(
        shape=(62,), dtype=tf.float32, batch_size=batch_size, name='input_dist')
    dist_vec = tf.keras.layers.Dense(
        num_basis, activation='linear',
        kernel_initializer=initializer, input_shape=(62,))(input_dist)
    dist_vec = tf.reshape(dist_vec, shape=[-1, 1, 1, 1, num_basis])
    basis_vec = basis_vec + dist_vec

  if temp != 1.0:
    basis_vec = basis_vec / temp

  basis_prob = tf.keras.layers.Softmax()(basis_vec)

  x = tf.keras.layers.Conv2D(
      int(32 * num_filters_expand),
      kernel_size=(3, 3),
      activation='relu',
      data_format=data_format,
      kernel_initializer=initializer)(
          input_x)
  x = BasisConv2D(
      int(64 * num_filters_expand),
      kernel_size=(3, 3),
      activation='relu',
      data_format=data_format,
      num_basis=num_basis,
      input_channel=int(32 * num_filters_expand),
      kernel_initializer=initializer)(
          x, basis_prob)

  x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format)(x)
  x = tf.keras.layers.Dropout(0.25, seed=seed)(x)
  x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Dense(
      128,
      activation='relu',
      kernel_initializer=initializer,
      input_shape=(9216,))(
          x)
  x = tf.keras.layers.Dropout(0.5, seed=seed)(x)
  y = tf.keras.layers.Dense(
      10 if only_digits else 62,
      activation=tf.nn.softmax,
      kernel_initializer=initializer, name='classifier')(x)

  if with_dist:
    return tf.keras.Model(inputs=[input_x, input_id, input_dist], outputs=[y])
  else:
    return tf.keras.Model(inputs=[input_x, input_id], outputs=[y])


def create_conv_dropout_model(only_digits = True,
                              num_filters_expand = 1,
                              seed = 0):
  """Convolutional model with droupout for EMNIST experiments.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    num_filters_expand: A factor to expand the number of the channels.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)
  input_x = tf.keras.layers.Input(
      shape=(28, 28, 1), dtype=tf.float32, name='input_x')
  input_id = tf.keras.layers.Input(
      shape=(1,), dtype=tf.int64, name='input_id')
  x = tf.keras.layers.Conv2D(
      int(32 * num_filters_expand),
      kernel_size=(3, 3),
      activation='relu',
      data_format=data_format,
      input_shape=(28, 28, 1),
      kernel_initializer=initializer)(
          input_x)
  x = tf.keras.layers.Conv2D(
      int(64*num_filters_expand),
      kernel_size=(3, 3),
      activation='relu',
      data_format=data_format,
      kernel_initializer=initializer)(x)
  x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format)(x)
  x = tf.keras.layers.Dropout(0.25, seed=seed)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      128, activation='relu', kernel_initializer=initializer)(x)
  x = tf.keras.layers.Dropout(0.5, seed=seed)(x)
  y = tf.keras.layers.Dense(
      10 if only_digits else 62,
      activation=tf.nn.softmax,
      kernel_initializer=initializer)(
          x)

  return tf.keras.Model(inputs=[input_x, input_id], outputs=[y])


def create_original_fedavg_cnn_model(only_digits = True,
                                     num_filters_expand = 1,
                                     seed = 0):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    num_filters_expand: A factor to expand the number of the channels.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer)

  model = tf.keras.models.Sequential([
      conv2d(filters=32*num_filters_expand, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64*num_filters_expand),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          512, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def create_two_hidden_layer_model(only_digits = True,
                                  hidden_units = 200,
                                  seed = 0):
  """Create a two hidden-layer fully connected neural network.

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.
      We default to 200 units, which matches that in
      https://arxiv.org/abs/1602.05629.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      tf.keras.layers.Dense(
          hidden_units, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          hidden_units, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def create_1m_cnn_model(only_digits = True, seed = 0):
  """A CNN model with slightly under 2^20 (roughly 1 million) params.

  A simple CNN model for the EMNIST character recognition task that is very
  similar to the default recommended model from `create_conv_dropout_model`
  but has slightly under 2^20 parameters. This is useful if the downstream task
  involves randomized Hadamard transform, which requires the model weights /
  gradients / deltas concatednated as a single vector to be padded to the
  nearest power-of-2 dimensions.

  This model is used in https://arxiv.org/abs/2102.06387.

  When `only_digits=False`, the returned model has 1,018,174 trainable
  parameters. For `only_digits=True`, the last dense layer is slightly smaller.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


class BasisConv2D(tf.keras.layers.Layer):
  """A Conv2D layer with Basis kernels for each example in a mini-batch."""

  def __init__(self,
               filters,
               kernel_size,
               data_format,
               input_channel=1,
               num_basis=1,
               kernel_initializer=None,
               padding='valid',
               strides=(1, 1),
               dilation_rate=(1, 1),
               use_bias=False,
               activation=None,
               ):

    super().__init__()

    self.filters = filters
    self.kernel_size = kernel_size
    self.data_format = data_format
    self._num_basis = num_basis
    self.kernel_initializer = kernel_initializer
    self.padding = padding
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.use_bias = use_bias
    self.activation = activation
    self.input_channel = input_channel

  def fn_conv2d_one_example(self, elem):
    x = tf.expand_dims(elem[0], 0)
    w = elem[1]
    return tf.keras.backend.conv2d(
        x,
        w,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate
    )

  def build(self, input_shape):
    kernel_shape = self.kernel_size + (self.input_channel,
                                       self.filters * self._num_basis)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          trainable=True)
    else:
      self.bias = None

  def call(self, x, c_prob):
    y_group = tf.keras.backend.conv2d(
        x,
        self.kernel,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate
    )
    # [None, w, h, c*basis]
    group_shape = tf.shape(y_group)
    n, w, h = group_shape[0], group_shape[1], group_shape[2]
    y_group = tf.reshape(y_group, [n, w, h, self.filters, self._num_basis])
    y = tf.keras.backend.sum(y_group*c_prob, axis=-1)

    if self.use_bias:
      y = tf.keras.backend.bias_add(y, self.bias)
    if self.activation is not None:
      y = tf.keras.activations.get(self.activation)(y)
    return y

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'dilation_rate':
            self.dilation_rate,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
    }
    base_config = super(tf.keras.layers.Layer, self).get_config()  # pytype: disable=attribute-error  # typed-keras
    return dict(list(base_config.items()) + list(config.items()))


class BasisDense(tf.keras.layers.Layer):
  """A Dense layer with Basis kernels for each example in a mini-batch."""

  def __init__(self,
               units,
               num_basis=1,
               kernel_initializer=None,
               use_bias=True,
               activation=None,
               ):

    super().__init__()

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
    # [None, last_dim, self.units]
    composed_kernel = tf.keras.backend.sum(composed_kernel*c_prob, axis=-1)
    y = tf.matmul(tf.expand_dims(x, 1), composed_kernel)
    if self.use_bias:
      y = tf.keras.backend.bias_add(y, self.bias)
    if self.activation is not None:
      y = tf.keras.activations.get(self.activation)(y)
    y = tf.squeeze(y, 1)
    return y

  def get_config(self):
    return {'units': self.units}


class BasisCNNRegularizer(tf.keras.regularizers.Regularizer):
  """A regularizer on basis CNN kernels."""

  def __init__(self, strength, num_basis):
    self.strength = strength
    self.num_basis = num_basis

  def __call__(self, basis):
    basis_shape = tf.shape(basis)
    basis = tf.reshape(
        basis,
        [basis_shape[0], basis_shape[1], basis_shape[2], -1, self.num_basis])
    mean_basis = tf.reduce_mean(basis, axis=-1, keepdims=True)

    # to maximize the difference, add minus to the loss term
    return -self.strength * tf.reduce_sum(tf.square(basis - mean_basis))
