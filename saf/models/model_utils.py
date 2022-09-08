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

"""Utilities for dealing with time-series forecasting models."""

import contextlib
import dataclasses
import random
from typing import Callable, Iterator, List, Tuple, Union

import numpy as np
import tensorflow as tf

WeightList = List[np.ndarray]


def set_seed(random_seed):
  random.seed(random_seed)
  np.random.seed(random_seed)
  tf.random.set_seed(random_seed)


@contextlib.contextmanager
def temporary_weights(
    *layers,
):
  """Uses temporary weights for this context.

  Calls the getters prior to the with statement and calls the setters after.

  Args:
    *layers: The keras layers to use temporary weights for. The weights will be
      gotten prior to the context block and reset to that value after the block.

  Yields:
    A function which will reset the weights to the initial value.
  """
  # Handle a single layer input
  starting_weights = freeze_layer_weights(*layers)

  def reset_fn():
    reset_layer_weights(starting_weights)

  yield reset_fn

  reset_fn()


@dataclasses.dataclass(frozen=True)
class KerasLayerWithWeights:
  """Helper class to keep a Keras layer paired with its weights.

  Frozen to make sure the association doesn't get mixed up.
  """
  layer: tf.keras.layers.Layer
  initial_weights: List[np.ndarray]

  def reset_weights(self):
    # When layers are first created there may not be any weights.
    if self.initial_weights:
      self.layer.set_weights(self.initial_weights)

  @classmethod
  def from_layer(cls, input_layer):
    return cls(input_layer, input_layer.get_weights())


def freeze_layer_weights(
    *layers,
):
  """Freeze the weights with the layer so they can be reset later.

  Args:
    *layers: The Keras layers whose weights will be frozen.

  Returns:
    A tuple of objects that pairs the layer with it's initial weights.
  """
  for current_layer in layers:
    if not isinstance(current_layer, tf.keras.layers.Layer):
      raise ValueError(
          f'All of the layers must be Keras layers: {current_layer}')

  return tuple(
      KerasLayerWithWeights.from_layer(current_layer)
      for current_layer in layers)


def reset_layer_weights(
    layers_and_weights):
  """Resets a Keras layer to it's paired weights.

  Args:
    layers_and_weights: A tuple of objects that pairs a Keras layer with it's
      initial weights. This is normally output from freeze_layer_weights.
  """
  for current_layer in layers_and_weights:
    current_layer.reset_weights()
