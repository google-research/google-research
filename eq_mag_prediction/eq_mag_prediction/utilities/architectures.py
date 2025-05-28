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

"""Architectural building blocks to construct larger models."""

from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import gin
import tensorflow as tf

# A function that takes the name of the model, the size of the input, and any
# additional arguments, and returns a Keras model with the given input size.
ModelConstructor = Callable[[str, int], tf.keras.Model]


class BiasPerCell(tf.keras.layers.Layer):
  """A layer that adds a trainable bias to an image, shared between channels."""

  def __init__(
      self,
      kernel_regularizer = None,
      **kwargs,
  ):
    super(BiasPerCell, self).__init__(**kwargs)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

  def build(self, input_shape):
    """Builds the layer - initialize the weights."""
    self.bias_weights = self.add_weight(
        name='bias_kernel',
        shape=(*input_shape[1:-1], 1),
        initializer='glorot_normal',
        dtype='float32',
        trainable=True,
        regularizer=self.kernel_regularizer,
    )

  def call(self, inputs):
    """Calls the layer - add the bias weights to the input."""
    return tf.math.add(self.bias_weights, inputs)

  def get_config(self):
    """Returns the config for the layer, used to store and load the model."""
    config = super().get_config().copy()
    config.update(
        {
            'kernel_regularizer': self.kernel_regularizer,
        }
    )
    return config


@gin.configurable
class EmbeddingPerCell(tf.keras.layers.Layer):
  """A layer that adds an embedding vector to feature channels of an image."""

  def __init__(
      self,
      kernel_regularizer = None,
      embedding_size = 1,
      **kwargs,
  ):
    super(EmbeddingPerCell, self).__init__(**kwargs)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.embedding_size = embedding_size

  def build(self, input_shape):
    """Builds the layer - initialize the weights."""
    self.embedding_matrix = self.add_weight(
        name='embedding_kernel',
        shape=(*input_shape[1:-1], self.embedding_size),
        initializer='glorot_normal',
        dtype='float32',
        trainable=True,
        regularizer=self.kernel_regularizer,
    )

  def call(self, inputs):
    """Calls the layer - concatenates the embeddings to the input."""
    repeated_embeddings = tf.repeat(
        tf.expand_dims(self.embedding_matrix, 0), tf.shape(inputs)[0], axis=0
    )
    return tf.concat([inputs, repeated_embeddings], axis=-1)

  def get_config(self):
    """Returns the config for the layer, used to store and load the model."""
    config = super().get_config().copy()
    config.update({
        'kernel_regularizer': self.kernel_regularizer,
        'embedding_size': self.embedding_size,
    })
    return config


class LayersOnEveryCell(tf.keras.layers.Layer):
  """A layer that applies a sequence of layers on every cell of a grid.

  If the input is (batch, x, y, ...), this layer is applying the same sequence
  of layers (with shared weights) to every slice shaped (batch, 1, 1, ...), and
  then concatenates the outputs to have the gridded shape (batch, x, y, ...).
  """

  def __init__(self, layers):
    """Initializes the layer, storing the sequence of layers to be applied."""
    super().__init__()
    self.layers = layers

  @tf.function
  def _apply_layers_to_cell(self, tensor):
    after_layer = tensor
    for layer in self.layers:
      after_layer = layer(after_layer)
    return after_layer

  @tf.function
  def _apply_layers_to_row(self, tensor):
    return tf.map_fn(
        self._apply_layers_to_cell,
        tensor,
        parallel_iterations=False,
        dtype='float32',
    )

  @tf.function
  def call(self, inputs):
    return tf.vectorized_map(self._apply_layers_to_row, inputs)


def _cnn_1d_layers(
    name,
    input_layer,
    input_shape,
    cnn_filters,
    activation = 'relu',
    kernel_regularization = None,
):
  """Applies a 1d CNN to an input, that represents a functions on the last axis.

  Assume that the input shape is (example, setups, feature), i.e. for every
  example (1st axis) there are several feature functions (3rd axis) that are
  calculated for different setups (2nd axis). For example `setups` can be one of
  the few past earthquakes, and `features` are functions that are calculated on
  each earthquake.
  This model starts with a few Conv layers, that are repeatedly applied to the
  entire last axis (thus, learning some combination of the features, that is
  shared between the different setups).

  Args:
    name: An identifier of the model.
    input_layer: The input layer.
    input_shape: The shape of the input layer. Expected to be 2-dimensional.
    cnn_filters: The number of filters per layer. The shape of every filter is
      equal to the number of filters in the previous layer (always taking the
      entire last axis).
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    The output layer after the CNN layers are applied.
  """
  kernel_shape = (1, input_shape[-1])
  output_layer = tf.keras.layers.Reshape((*input_shape, 1))(input_layer)
  for i, n_filters in enumerate(cnn_filters):
    output_layer = tf.keras.layers.Conv2D(
        name=f'{name}_{i}_cnn',
        dtype='float64',
        filters=n_filters,
        kernel_size=kernel_shape,
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(output_layer)
    kernel_shape = (1, n_filters)
    # Swap the 'channels' axis to the last feature axis.
    output_layer = tf.keras.layers.Permute((1, 3, 2), dtype='float64')(
        output_layer
    )

  return output_layer


@gin.configurable
def rnns(
    name,
    n_features,
    units,
    kernel_regularization = None,
    rnn_layer_type = 'LSTM',
):
  """Prepares a sequence of RNN models that are applied to every grid cell.

  That is, it's the same set of RNN models that are applied to every grid cell,
  with shared weights.

  Args:
    name: An identifier of the model.
    n_features: The number of features in the last axis of the input layer. The
      input layer will be 4-dimensional, the first two dimensions being the
      shape of the spatial grid, and the third being the history size. All 3 are
      unspecified, and the model is flexible enough to work with them.
    units: The number of unis per RNN layer.
    kernel_regularization: Regularizer function.
    rnn_layer_type: The type of RNN model - either GRU or LSTM.

  Returns:
    A model which applies a sequence of RNNs per grid cell.
  """
  input_layer = tf.keras.layers.Input(
      shape=(None, None, None, n_features), name=name, dtype='float64'
  )
  output_layer = input_layer
  rnn_layer_constructor = getattr(tf.keras.layers, rnn_layer_type)

  rnn_layers = []
  for i, layer_units in enumerate(units[:-1]):
    rnn_layers.append(
        tf.keras.layers.Bidirectional(
            rnn_layer_constructor(
                layer_units,
                name=f'{name}_{i}_{rnn_layer_type}',
                return_sequences=True,
                kernel_regularizer=kernel_regularization,
            )
        )
    )
  rnn_layers.append(
      rnn_layer_constructor(
          units[-1],
          name=f'{name}_{len(units) - 1}_{rnn_layer_type}',
          return_sequences=False,
          kernel_regularizer=kernel_regularization,
      )
  )

  output_layer = LayersOnEveryCell(rnn_layers)(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def same_output_size_cnn_model(
    name,
    input_shape,
    filter_sizes,
    filter_numbers,
    activation = 'relu',
    kernel_regularization = None,
):
  """Prepares a CNN model that has the same output size as the input.

  If the input is shaped (x, y, f), where (x, y) is the size of the image (or,
  in our cases, the size of spatial grid), then the output of this model is
  shaped (x, y, g) - where g is the last number of filters.

  Args:
    name: An identifier of the model.
    input_shape: The shape of the input layer. Expected to be 3-dimensional, the
      first two dimensions being the shape of the spatial grid.
    filter_sizes: The size of filter per layer. The shape of every filter is a
      square.
    filter_numbers: The number of filters per layer.
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    A CNN model that has the same output size as the input.
  """
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=name, dtype='float64'
  )
  output_layer = input_layer

  for i, (size, n_filters) in enumerate(zip(filter_sizes, filter_numbers)):
    output_layer = tf.keras.layers.Conv2D(
        name=f'{name}_{i}_cnn',
        dtype='float64',
        padding='same',
        filters=n_filters,
        kernel_size=(size, size),
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def same_output_size_3d_cnn_model(
    name,
    input_shape,
    filter_shapes,
    filter_numbers,
    activation = 'relu',
    kernel_regularization = None,
):
  """Prepares a 3D CNN model that has the same output size as the input.

  If the input is shaped (x, y, z, f), where (x, y, z) is the size of the image
  (or, in our cases, (x, y) is the shape of spatial grid and z is a secondary
  dimension, such as magnitude), then the output of this model is shaped
  (x, y, z * g) - where g is the last number of filters. The model applies a
  variable number of 3D convolutional layers, and then flattens the last axis.

  Args:
    name: An identifier of the model.
    input_shape: The shape of the input layer. Expected to be 2-dimensional.
    filter_shapes: The shape of the filter at every layer
    filter_numbers: The number of filters per layer.
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    A CNN followed by a fully connected model.
  """
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=name, dtype='float64'
  )
  output_layer = input_layer

  for i, (kernel, n_filters) in enumerate(zip(filter_shapes, filter_numbers)):
    output_layer = tf.keras.layers.Conv3D(
        name=f'{name}_{i}_cnn',
        dtype='float64',
        padding='same',
        filters=n_filters,
        kernel_size=kernel,
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(output_layer)

  output_layer = tf.keras.layers.Reshape((*input_shape[:2], -1))(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def order_invariant_cnn_model(
    name,
    input_shape,
    cnn_filters,
    activation = 'relu',
    kernel_regularization = None,
):
  """Prepares a CNN model that represents a function on the set of examples.

  Args:
    name: An identifier of the model.
    input_shape: The shape of the input layer. Expected to be 2-dimensional.
    cnn_filters: The number of filters per layer. The shape of every filter is
      equal to the number of filters in the previous layer (always taking the
      entire last axis).
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    A CNN followed by a sum (thus, being invariant to the order of examples).
  """
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=name, dtype='float64'
  )

  output_layer = _cnn_1d_layers(
      name,
      input_layer,
      input_shape,
      cnn_filters,
      activation,
      kernel_regularization,
  )

  output_layer = tf.keras.layers.Lambda(
      lambda x: tf.keras.backend.sum(x, axis=1)
  )(output_layer)
  output_layer = tf.keras.layers.Flatten()(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def spatial_order_invariant_cnn_model(
    name,
    input_shape,
    cnn_filters = (0,),
    activation = 'relu',
    kernel_regularization = None,
):
  """Prepares a CNN model that represents a function on the set of examples."""
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=name, dtype='float64'
  )

  output_layer = input_layer
  kernel_shape = (1, 1, 1)
  for i, n_filters in enumerate(cnn_filters):
    output_layer = tf.keras.layers.Conv3D(
        name=f'{name}_{i}_cnn',
        dtype='float64',
        filters=n_filters,
        kernel_size=kernel_shape,
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(output_layer)

  output_layer = tf.keras.layers.Lambda(
      lambda x: tf.keras.backend.sum(x, axis=3)
  )(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def cnn_1d_model(
    name,
    input_shape,
    cnn_filters,
    dense_units,
    activation = 'relu',
    kernel_regularization = None,
):
  """Prepares a CNN model that represents a functions on the last axis.

  Args:
    name: An identifier of the model.
    input_shape: The shape of the input layer. Expected to be 2-dimensional.
    cnn_filters: The number of filters per layer. The shape of every filter is
      equal to the number of filters in the previous layer (always taking the
      entire last axis).
    dense_units: A list of sizes of the hidden and output layers.
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    A CNN followed by a fully connected model.
  """
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=name, dtype='float64'
  )

  output_layer = _cnn_1d_layers(
      name,
      input_layer,
      input_shape,
      cnn_filters,
      activation,
      kernel_regularization,
  )

  output_layer = tf.keras.layers.Flatten()(output_layer)

  for i, units in enumerate(dense_units):
    output_layer = tf.keras.layers.Dense(
        units=units,
        name=f'{name}_{i}_dense',
        dtype='float64',
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(output_layer)
  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def fully_connected_model(
    name,
    input_size,
    layer_sizes,
    activation = 'relu',
    kernel_regularization = None,
):
  """Generates a keras model of stacked FC layers.

  Args:
    name: An identifier of the model.
    input_size: The number of units in the input layer.
    layer_sizes: A list of sizes of the hidden and output layers.
    activation: The Keras activation function to use.
    kernel_regularization: Regularizer function.

  Returns:
    A fully connecteed model.
  """
  return multi_layer_perceptron(
      n_filters=layer_sizes,
      input_shape=(input_size,),
      model_name=name,
      activation=activation,
      kernel_regularization=kernel_regularization,
      output_activation=activation,
      dtype='float64',  # pytype: disable=bad-return-type  # typed-keras
  )


def multi_layer_perceptron(
    n_filters,
    input_shape,
    model_name = None,
    activation = 'relu',
    kernel_regularization = None,
    output_activation = None,
    dtype = 'float32',
):
  """Generates a keras model of stacked FC layers.

  Args:
    n_filters: list of int. Specifies the dimension of each layer.
    input_shape: tuple of ints describing shape of input layer, excluding batch
      dimension (same convention as in keras.Layers.Input)
    model_name: Optional string. An identifier of the model that will be
      prepended to layer names.
    activation: string or keras.activation. Applied to output of each layer.
    kernel_regularization: keras.regularization. Applied to kernel of each
      layer.
    output_activation: Activation to apply on the last layer. If None, the
      linear activation will be applied.
    dtype: The type of all of the layers.

  Returns:
    A keras model.
  """
  if isinstance(input_shape, int):
    input_shape = (input_shape,)
  if len(input_shape) > 1:
    raise ValueError('Input shape must have only one dimension.')
  input_layer = tf.keras.layers.Input(
      shape=input_shape, name=model_name, dtype=dtype
  )
  output_layer = input_layer
  layer_activation = activation
  for i, n in enumerate(n_filters):
    if i == len(n_filters) - 1:
      layer_activation = output_activation
    output_layer = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        kernel_regularizer=kernel_regularization,
        activation=layer_activation,
        name=f'{model_name}_dense_{i}',
        dtype=dtype,
    )(output_layer)
  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


@gin.configurable
def identity_model(model_name, input_size):
  """Prepares a model that just outputs the input."""
  return tf.keras.models.Sequential(
      [
          tf.keras.layers.InputLayer(
              input_shape=(input_size,), name=model_name, dtype='float64'
          ),
      ]
  )


@gin.configurable
def cnn_model(
    name,
    grid_size,
    filters,
    kernels,
    activation = 'relu',
    kernel_regularization = None,
):
  """Generates a keras model of a sequence of convolutional layers."""
  input_layer = tf.keras.layers.Input(
      shape=grid_size, name=name, dtype='float64'
  )

  output = input_layer
  for i, units in enumerate(filters):
    kernel = kernels[i]
    output = tf.keras.layers.Conv2D(
        units,
        kernel,
        kernel_regularizer=kernel_regularization,
        name=f'{name}_{i}_cnn',
        activation=activation,
    )(output)

  output = tf.keras.layers.Flatten()(output)

  return tf.keras.models.Model(inputs=input_layer, outputs=output)


@gin.configurable
def rnn_model(
    name,
    n_features,
    units,
    kernel_regularization = None,
    rnn_layer_type = 'LSTM',
):
  """Generates a keras model of a sequence of RNN layers."""
  input_layer = tf.keras.layers.Input(
      shape=(None, n_features), name=name, dtype='float64'
  )
  rnn_layer_constructor = getattr(tf.keras.layers, rnn_layer_type)

  output_layer = input_layer
  for i, layer_units in enumerate(units[:-1]):
    output_layer = tf.keras.layers.Bidirectional(
        rnn_layer_constructor(
            layer_units,
            name=f'{name}_{i}_{rnn_layer_type}',
            return_sequences=True,
            kernel_regularizer=kernel_regularization,
        )
    )(output_layer)
  output_layer = rnn_layer_constructor(
      units[-1],
      name=f'{name}_{len(units) - 1}_{rnn_layer_type}',
      return_sequences=False,
      kernel_regularizer=kernel_regularization,
  )(output_layer)

  return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
