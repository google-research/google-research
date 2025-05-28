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

"""A collection of configuration validators."""

import types
import typing
from typing import Any, Union


def _contains_type(base_type, value):
  """Returns True if the value is contained in the (union) base type."""
  if (typing.get_origin(base_type) is types.UnionType
      or typing.get_origin(base_type) is Union):
    return any(value == arg for arg in typing.get_args(base_type))
  return False


def _validate_config_is_instance_of_base_config(
    cls,
    base_cls,
    unwrap_types = True):
  """Validates that a config has the same attributes as its base config.

  Args:
    cls: the config to validate.
    base_cls: a base config that `cls` inherits from.
    unwrap_types: allow unwrapping of types in child configs. E.g.,
      a child config can specify a type of `int` when the base config specifies
      `int | None`.

  Raises:
    AttributeError: if the attribute names and types of the config and base
      config do not match.
  """
  config_types = typing.get_type_hints(cls)
  base_types = typing.get_type_hints(base_cls)

  new_attributes = set(config_types) - set(base_types)

  if new_attributes:
    raise AttributeError(
        'Encountered mismatched attributes of a locked class. New attributes: '
        f'{new_attributes}')

  for name in config_types:
    config_type = config_types[name]
    base_type = base_types[name]

    if config_type != base_type:
      # Unwrapped `Optional` or `Union` types are allowed.
      if not (unwrap_types and _contains_type(base_type, config_type)):
        raise AttributeError(
            'Encountered mismatched types of a locked class: expected '
            f'{base_types[name]} but got {config_types[name]} for attribute'
            f' {name}')


def lock(cls):
  """Sets a lock on this class to prevent any changes to attributes.

  See `validators.validate` for more info.

  Args:
    cls: the config class to lock.

  Returns:
    the config class.
  """
  cls.lock_attributes()
  return cls


def validate(cls,
             unwrap_types = True):
  """Validates that a config is an instance of its base config.

  This decorator is a useful check in situations where an inherited config
  only requires changes of the values of existing attributes. It avoids the
  pitfall of using dataclass inheritance for the sole purpose of setting
  existing attributes. This decorator effectively locks a config to have the
  same attributes and types as its base config. An error will be raised if the
  config is improperly redefining attributes.

  Usage:
  ```python
  @validators.lock
  @dataclasses.dataclass
  class MyBaseClass():
    myattr: int = 1
    myattr2: int = 2

  @validators.validate
  @dataclasses.dataclass
  MyClass(MyBaseClass):
    myattr: float = 1  # Will raise an error: types do not match
    mynewattr: int = 2  # Will raise an error: new attribute
  ```

  Args:
    cls: the class to validate.
    unwrap_types: allow unwrapping of types in child configs. E.g.,
      a child config can specify a type of `int` when the base config
      specifies `int | None`.

  Returns:
    the child config.

  Raises:
    AssertionError: if this function is called on a non-locked class.
    AttributeError: if the attribute names and types of the config and child
      config do not match.
  """
  base_cls = cls.get_attribute_lock()

  if base_cls is None:
    raise AssertionError(
        'Missing base class, nothing to validate. Make sure to add '
        '`@validators.lock on an inherited config.')

  _validate_config_is_instance_of_base_config(
      base_cls=base_cls,
      cls=cls,
      unwrap_types=unwrap_types)

  return cls
