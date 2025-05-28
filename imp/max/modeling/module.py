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

"""Base abstract modules used across MAX."""

from typing import Any, Set, Type, TypeVar

import flax.linen as nn
import jax

from imp.max.core import constants
from imp.max.utils import typing

ModelT = Type[TypeVar('_ModelT', bound='Model')]
DataFeatureType = constants.DataFeatureType


# TODO(b/228880098): automate manual specification of expected keys/inputs
class Model(nn.Module):
  """Abstract base class for models.

  Attributes:
    init_override: The post-param-initialization method. It will override some
      or all initialized parameters if provided. [Subject to deprecation.]
  """

  # TODO(b/234949870): deprecate this and merge with checkpointing pipeline
  init_override: str | None

  def get_rng_keys(self):
    """Returns keys of all rngs defined under this model."""

    raise NotImplementedError(
        'Please specify all associated keys with rgns in the model')

  def get_data_signature(self):
    """Returns the input signature to fully initialize this model."""

    raise NotImplementedError(
        'Please specify the exact expected input signature of the model.')

  @property
  def supported_modalities(self):
    """Returns the expected modality in the model."""

    expected_data = jax.eval_shape(self.get_data_signature)
    if not isinstance(expected_data, dict):
      raise ValueError(
          'It appears that input signature is not configured as a mapping. '
          'In this case, `supported_modalities` should be specifically defined.'
      )
    all_modalities = []
    all_routes = sorted(list(expected_data[DataFeatureType.INPUTS].keys()))
    for route in all_routes:
      route_modalities = sorted(
          list(expected_data[DataFeatureType.INPUTS][route].keys())
      )
      all_modalities.extend(route_modalities)
    return set(all_modalities)

  def __call__(self, data, deterministic):
    raise NotImplementedError(
        'This method should be specified by the downstream implementation.')
