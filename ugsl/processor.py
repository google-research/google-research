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

"""Processor module of the GSL Layer."""
import tensorflow as tf

from ugsl import graph_structure


@tf.keras.utils.register_keras_serializable(package="GSL")
class Symmetrize(tf.keras.layers.Layer):
  """Symmetrize a full parameter adjacency."""

  def call(self, inputs):
    sources, targets, weights = inputs.sources, inputs.targets, inputs.weights
    symmetric_sources = tf.concat([sources, targets], 0)
    symmetric_targets = tf.concat([targets, sources], 0)
    symmetric_weights = tf.concat([weights / 2, weights / 2], 0)
    return graph_structure.GraphStructure(
        symmetric_sources, symmetric_targets, symmetric_weights
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class Activation(tf.keras.layers.Layer):
  """Applying an activation function on the weights of the adjacency."""

  def __init__(self, activation):
    super().__init__()
    self._activation = activation

  def call(self, inputs):
    weights = tf.keras.activations.get(self._activation)(inputs.weights)
    # elu non-linearity has range -1 to 1. Adding +1 to avoid non-negativity.
    if self._activation == "elu":
      weights += 1
    return graph_structure.GraphStructure(
        inputs.sources, inputs.targets, weights
    )

  def get_config(self):
    return dict(activation=self._activation, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GSL")
class ActivationSymmetrize(tf.keras.layers.Layer):
  """Applying an activation function on the weights of the adjacency."""

  def __init__(self, activation):
    super().__init__()
    self._activation = activation

  def call(self, inputs):
    sources, targets, weights = inputs.sources, inputs.targets, inputs.weights
    weights = tf.keras.activations.get(self._activation)(weights)
    if self._activation == "elu":
      weights += 1
    symmetric_sources = tf.concat([sources, targets], 0)
    symmetric_targets = tf.concat([targets, sources], 0)
    symmetric_weights = tf.concat([weights / 2, weights / 2], 0)
    return graph_structure.GraphStructure(
        symmetric_sources, symmetric_targets, symmetric_weights
    )

  def get_config(self):
    return dict(activation=self._activation, **super().get_config())


def get_processor(name, **kwargs):
  if name == "symmetrize":
    return Symmetrize()
  elif name == "activation":
    return Activation(**kwargs)
  elif name == "activation-symmetrize":
    return ActivationSymmetrize(**kwargs)
  elif name == "none":
    return tf.keras.layers.Layer()
  else:
    raise ValueError(f"Processor {name} is not defined.")
