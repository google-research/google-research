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

"""Implements scalers for data pre-processing in a JAX-compatible manner."""

import abc
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp


class Scaler(abc.ABC):
  """Abstract base class for scalers."""

  @abc.abstractmethod
  def fit(self, data):
    pass

  @abc.abstractmethod
  def transform(self, data):
    pass

  @abc.abstractmethod
  def inverse_transform(self, data):
    pass


@jax.tree_util.register_pytree_node_class
class IdentityScaler(Scaler):
  """Implements the identity scaler."""

  def fit(self, data):
    return IdentityScaler()

  def transform(self, data):
    return data

  def inverse_transform(self, data):
    return data

  def tree_flatten(self):
    return ((), None)

  @classmethod
  def tree_unflatten(cls, aux_data,
                     children):
    del aux_data, children
    return cls()


@jax.tree_util.register_pytree_node_class
class StandardScaler(Scaler):
  """Implements sklearn.preprocessing.StandardScaler."""

  def __init__(self,
               mean = None,
               std = None):
    super(StandardScaler, self).__init__()
    self._mean = mean
    self._std = std

  def fit(self, data):
    mean = jnp.mean(data, axis=0, keepdims=True)
    std = jnp.std(data, axis=0, keepdims=True)
    return StandardScaler(mean, std)

  def transform(self, data):
    return (data - self.mean()) / self.std()

  def inverse_transform(self, data):
    return data * self.std() + self.mean()

  def mean(self):
    assert self._mean is not None
    return self._mean

  def std(self):
    assert self._std is not None
    return self._std

  def tree_flatten(self):
    children = (self.mean(), self.std())
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
      cls, aux_data, children):
    del aux_data
    return cls(*children)
