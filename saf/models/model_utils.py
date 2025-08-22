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

"""Utilities for dealing with time-series forecasting models."""

import contextlib
import dataclasses
import random
from typing import Any, Callable, Iterator, List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

WeightList = List[np.ndarray]


def set_seed(random_seed):
  # https://github.com/NVIDIA/framework-determinism
  random.seed(random_seed)
  np.random.seed(random_seed)
  tf.random.set_seed(random_seed)


@contextlib.contextmanager
def temporary_weights(
    *keras_objects,):
  """Resets objects weights to their initial values after the context.

  Calls the getters on entrance to the context and calling the yielded function
  resets the weights to this value. Regardless of if this function is called the
  weights will also be reset after the context.

  Args:
    *keras_objects: The keras objects to use temporary weights for. The weights
      will be gotten on entrance to the context block and reset to that value
      after the block.

  Yields:
    A function which will reset the weights to the initial value.
  """
  # Handle a single keras_object input
  starting_weights = freeze_weights(*keras_objects)

  def reset_fn():
    reset_object_weights(starting_weights)

  yield reset_fn

  reset_fn()


@dataclasses.dataclass(frozen=True)
class KerasObjectWithWeights:
  """Helper class to keep a Keras keras_object paired with its weights.

  Frozen to make sure the association doesn't get mixed up.
  """
  keras_object: Any
  initial_weights: List[np.ndarray]

  def reset_weights(self):
    # When layers are first created there may not be any weights.
    if self.initial_weights:
      self.keras_object.set_weights(self.initial_weights)

  @classmethod
  def from_object(cls, input_obj):
    return cls(input_obj, input_obj.get_weights())


def freeze_weights(
    *keras_objects,
    required_methods = ('get_weights', 'set_weights'),
):
  """Freeze the weights with the keras_object so they can be reset later.

  Args:
    *keras_objects: The Keras objects whose weights will be frozen.
    required_methods: The methods that each object must have.

  Returns:
    A tuple of objects that pairs the keras_object with its initial weights.
  """
  for obj in keras_objects:
    if not all(hasattr(obj, w_attr) for w_attr in required_methods):
      raise ValueError(
          f'All of the objects must have get and set weights: {obj}')

  return tuple(KerasObjectWithWeights.from_object(obj) for obj in keras_objects)


def reset_object_weights(
    layers_and_weights):
  """Resets a Keras keras_object to it's paired weights.

  Args:
    layers_and_weights: A tuple of objects that pairs a Keras keras_object with
      its initial weights. This is normally output from freeze_weights.
  """
  for current_layer in layers_and_weights:
    current_layer.reset_weights()
