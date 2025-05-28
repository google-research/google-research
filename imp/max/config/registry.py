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

"""Registry for experiment configs."""

import functools
from typing import Any

from imp.max.utils import typing

_CONFIG_TYPES = Any  # pylint: disable=invalid-name
_CLASS_TYPES = Any  # pylint: disable=invalid-name


class Registrar(object):
  """Base registry for configs and classes."""

  _CONFIG_REGISTRY: dict[str, Any] = {}
  _CLASS_REGISTRY: dict[str, Any] = {}
  _OVERWRITE: bool = False

  @classmethod
  def register(cls,
               config,
               bound = None,
               overwrite = False):
    """Adds the experiment to the registry.

    Args:
      config: The config class to be registered. This is expected to have
        the attribute `name` which will be used to register and refer to this
        very config.
      bound: The corresponding class whose instantiation args are the
        attributes defined in `config`.
      overwrite: Allows overwriting a pre-registered config/class.

    Returns:
      The given config. Hence, this method can be used as a decorator on
      any config definition.
    """

    if not hasattr(config, 'name'):
      raise ValueError(f'Config class {config} has no field `name`.')

    if not config.name:
      raise ValueError(f'Found empty `name` in {config}.')

    name = config.name
    if ((name in cls._CONFIG_REGISTRY or name in cls._CLASS_REGISTRY)
        and not overwrite and not cls._OVERWRITE):
      raise ValueError(f'Attempting to register duplicate experiment: {name}')

    cls._CONFIG_REGISTRY[name] = config
    if bound is not None:
      cls._CLASS_REGISTRY[name] = bound

    return config

  @classmethod
  def register_with_class(cls, bound):
    """Returns a callable register method, useful as a callable decorator."""
    return functools.partial(cls.register, bound=bound)

  @classmethod
  def get_class_by_name(cls, name):
    """Returns the fetched-by-name class from the registry."""

    if name not in cls._CLASS_REGISTRY:
      raise ValueError(
          f'Class name not registered: {name}. Available classes: '
          f'{Registrar.class_names()}')
    return cls._CLASS_REGISTRY[name]

  @classmethod
  def get_config_by_name(cls, name):
    """Returns the model class from the registry."""

    if name not in cls._CONFIG_REGISTRY:
      raise ValueError(
          f'Experiment name not registered: {name}. Available experiments: '
          f'{Registrar.config_names()}')
    return cls._CONFIG_REGISTRY[name]

  @classmethod
  def class_names(cls):
    """Returns all class names in registry."""
    return list(cls._CLASS_REGISTRY.keys())

  @classmethod
  def config_names(cls):
    """Returns all config names in registry."""
    return list(cls._CONFIG_REGISTRY.keys())

  @classmethod
  def reset(cls):
    """Removes all of the registered entries."""
    cls._CONFIG_REGISTRY = {}
    cls._CLASS_REGISTRY = {}

  @classmethod
  def overwrite_all_duplicates(cls, value = True):
    """If set, overwrite registry items with duplicate names.

    This is useful  where reloading modules is necessary, e.g., in a Colab.

    Args:
      value: the overwrite value to set.
    """
    cls._OVERWRITE = value
