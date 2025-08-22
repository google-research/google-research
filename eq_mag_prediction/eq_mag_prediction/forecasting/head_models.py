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

"""Main methods to construct forecasting models."""

from typing import Iterable, Mapping, Optional

import gin
import tensorflow as tf
from eq_mag_prediction.forecasting import external_configurations


def input_order(
    *,
    spatially_dependent_model_names,
    spatially_independent_model_names,
):
  """Returns the expected order of inputs: spatially dependent, then independent."""
  return sorted(spatially_dependent_model_names), sorted(
      spatially_independent_model_names
  )


def _combined_encoders_model(
    spatially_dependent_models,
    spatially_independent_models,
    hidden_layer_sizes,
    output_size,
    hidden_activation,
    output_activation,
    kernel_regularization = None,
):
  """Returns the input and output layers for a model that combines encoders.

  Args:
    spatially_dependent_models: Have a single input tensor. This is a mapping
      between names (of the encoder, usually), and models.
    spatially_independent_models: Have two input tensors, one for spatially
      independent features, and one for location features. This is a mapping
      between names (of the encoder, usually), and models.
    hidden_layer_sizes: The sizes of hidden layers.
    output_size: The number of neurons in the last (output) layer.
    hidden_activation: The activation function for hidden layers.
    output_activation: The activation function for the output layer.
    kernel_regularization: Regularizer function.

  Returns:
    The input and output layers for a model that combined the input encoders.
    The inputs will be sorted by the name of the encoder - first the spatially
    dependent models, and then the spatially independent models.
  """
  model_outputs = [
      model.output for model in spatially_dependent_models.values()
  ] + [model.output for model in spatially_independent_models.values()]
  if len(model_outputs) == 1:
    combined_grid_layer = model_outputs[0]
  else:
    combined_grid_layer = tf.keras.layers.Concatenate(axis=1)(model_outputs)

  for layer_size, activation in zip(
      hidden_layer_sizes + [output_size],
      [hidden_activation] * len(hidden_layer_sizes) + [output_activation],
  ):
    combined_grid_layer = tf.keras.layers.Dense(
        layer_size,
        activation=activation,
        kernel_regularizer=kernel_regularization,
    )(combined_grid_layer)

  gridded_order, non_gridded_order = input_order(
      spatially_dependent_model_names=spatially_dependent_models.keys(),
      spatially_independent_model_names=spatially_independent_models.keys(),
  )
  inputs = [spatially_dependent_models[name].input for name in gridded_order]
  inputs += [
      spatially_independent_models[name].input for name in non_gridded_order
  ]

  return inputs, combined_grid_layer


@gin.configurable
def magnitude_prediction_model(
    spatially_dependent_models,
    spatially_independent_models,
    n_model_parameters,
    hidden_layer_sizes,
    hidden_activation = 'tanh',
    output_activation = 'softplus',
    kernel_regularization = None,
    output_shift = 1e-3,
):
  """Builds a model for magnitude prediction.

  Args:
    spatially_dependent_models: Have a single input tensor. This is a mapping
      between names (of the encoder, usually), and models.
    spatially_independent_models: Have two input tensors, one for spatially
      independent features, and one for location features. This is a mapping
      between names (of the encoder, usually), and models.
    n_model_parameters: number of parameters needed for the output probability
      density function.
    hidden_layer_sizes: The sizes of hidden layers.
    hidden_activation: The activation function for hidden layers.
    output_activation: The activation function for the output layer.
    kernel_regularization: Regularizer function.
    output_shift: Add a shift to the output's layer result. This is ueful for
      constricting the models outputs (i.e. probability distribution's
      parameters) to be >0.

  Returns:
    A Keras model that combines all encoders, and decodes from them a
    probability distribution of magnitudes per event. The inputs will be sorted
    by the name of the model - first the gridded models, and then the
    non-gridded models.
  """
  inputs, output = _combined_encoders_model(
      spatially_dependent_models=spatially_dependent_models,
      spatially_independent_models=spatially_independent_models,
      hidden_layer_sizes=hidden_layer_sizes,
      output_size=n_model_parameters,
      hidden_activation=hidden_activation,
      output_activation=output_activation,
      kernel_regularization=kernel_regularization,
  )
  combined_grid_layer = tf.keras.layers.Lambda(lambda x: x + output_shift)(
      output
  )

  return tf.keras.models.Model(inputs=inputs, outputs=combined_grid_layer)
