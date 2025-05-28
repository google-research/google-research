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

"""Base configuration class."""

import copy
import dataclasses
# import functools
import os
from typing import Any, Optional

# import jax
from jax import numpy as jnp
# import numpy as np
import tensorflow as tf
import yaml

from imp.max.core import utils
from imp.max.utils import typing

# Constants for YAML serialization.
ALL_JAX_DTYPES = (
    jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64,
    jnp.int4, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
    jnp.uint4, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
    jnp.bool_,
)


# def _parse_jax_dtype(dumper: yaml.Dumper,
#                      data: Any,
#                      tag: str) -> yaml.Node:
#   """Converts jax dtypes to strings when parsing yaml."""
#   return dumper.represent_scalar(tag, str(data))


# def _parse_numpy_dtype(dumper: yaml.Dumper,
#                        data: Any,
#                        tag: str) -> yaml.Node:
#   """Converts numpy dtypes to strings when parsing yaml."""
#   return dumper.represent_scalar(tag, str(data))


# def _construct_dtype(loader: yaml.Loader,
#                      node: yaml.Node) -> jax.typing.DTypeLike:
#   """Converts strings to jax dtypes when parsing yaml."""
#   return jnp.dtype(loader.construct_scalar(node))

# # Add functions to serialize/deserialize jax dtypes with yaml.
# # Due to jnp.generic not being a subclass of all dtypes, we must loop over a
# # a list of all dtypes.
# for dtype in ALL_JAX_DTYPES:
#   dtype_tag = '!jax.numpy.dtype'
#   parse_fn = functools.partial(_parse_jax_dtype, tag=dtype_tag)
#   yaml.add_representer(type[dtype], parse_fn)
#   yaml.add_constructor(dtype_tag, _construct_dtype)

#   # Also support base numpy types
#   dtype_tag = '!numpy.dtype'
#   parse_fn = functools.partial(_parse_numpy_dtype, tag=dtype_tag)
#   yaml.add_representer(type[dtype], parse_fn)
#   yaml.add_constructor(dtype_tag, _construct_dtype)


@dataclasses.dataclass
class Config(object):
  """Constructs a base for creating and operating on configs."""

  _PARAM_TYPES = (int, float, str, bool, type(None),) + ALL_JAX_DTYPES

  # The base class used to validate config attributes against.
  # See config/validators.py for more information.
  _ATTRIBUTES_BASE_CLS = None

  def _is_private(self, attr_name):
    return attr_name.startswith('_')

  def _is_parameter(self, node):
    is_parameter = any([isinstance(node, type_) for type_ in self._PARAM_TYPES])
    return is_parameter

  def as_dict(self):
    """Traverses over the configs tree and returns it as a dictionary."""
    return dataclasses.asdict(self)

  def as_flat_dict(
      self,
      sep = None):
    """Traverses over the configs tree and returns it as a flat dictionary.

    Arguments are intended to be consistent with the API of
    flax.traverse_util.flatten_dict.

    Args:
      sep: if defined, joins the flattened keys with the separator string.
        Otherwise, keeps the keys as a tuple of strings.

    Returns:
      a dictionary representing the config parameters.
    """
    return utils.flatten_dict(self.as_dict(), sep=sep)

  def _get_key_and_indices(
      self,
      maybe_key_with_indices):
    """Extracts key_name and indices with format 'key_name[index0][index1]'."""
    patterns = maybe_key_with_indices.split('[')
    if len(patterns) == 1:
      return (maybe_key_with_indices, None)
    # For each index ensure that the brackets are closed and extract number
    indices = []
    for split_pattern in patterns[1:]:
      # Remove surrounding whitespace.
      split_pattern = split_pattern.strip()
      if split_pattern[-1] != ']':
        raise ValueError(
            'ParameterName {} has bad format. Supported format: key_name, '
            'key_name[index0], key_name[index0][index1], ...'.format(
                maybe_key_with_indices))
      try:
        indices.append(int(split_pattern[:-1]))
      except ValueError as e:
        raise ValueError(
            'Only integer indexing allowed for ParameterName. '
            'Faulty specification: {}'.format(maybe_key_with_indices)) from e
    return patterns[0], indices

  def override(self,
               params_dict,
               node = None):
    """Traverses over the params_dict and overrides the attributes."""

    if not isinstance(params_dict, dict):
      raise ValueError('Please provide a valid override mapping.')

    if node is None:
      node = self
    for key_with_indices, value in params_dict.items():
      key, indices = self._get_key_and_indices(key_with_indices)
      if not hasattr(node, key):
        raise ValueError(f'Key {key} not found in {node}')

      if indices is None:
        if self._is_parameter(value):
          setattr(node, key, value)
        else:
          child = getattr(node, key)
          if isinstance(child, dict):
            setattr(node, key, value)
          else:
            self.override(value, child)
      else:
        attribute = getattr(node, key)
        if not isinstance(attribute, list):
          raise ValueError(
              f'Key {key} specified with indices {indices}, but attribute '
              'is not iterable.')
        if len(indices) > 1:
          raise NotImplementedError(
              'Only flat lists supported, but nested list update requested for '
              f'key {key}.')
        index = indices[0]
        if index >= len(attribute):
          raise ValueError(f'Index {index} requested for key {key}, but only '
                           f'{len(attribute)} elements defined.')
        if self._is_parameter(value):
          attribute[index] = value
        else:
          if isinstance(attribute[index], dict):
            attribute[index] = value
          else:
            self.override(value, attribute[index])

  def copy(self):
    """Copies the config to a new object."""
    return copy.deepcopy(self)

  def copy_and_override(self,
                        params_dict,
                        node = None):
    """Same as override on copy of config."""
    copied = self.copy()
    copied.override(params_dict, node)
    return copied

  def override_from_file(self, path):
    """Loads a dictionary of configs and overrides."""

    with tf.io.gfile.GFile(path) as f:
      self.override_from_str(f)

  def override_from_str(self, dict_str):
    """Gets a safe-dumped dictionary of configs and overrides."""

    self.override(yaml.safe_load(dict_str))

  def export(self,
             path,
             filename = None):
    """Saves the config as yaml to to the given path.

    Args:
      path: the base directory to save the config file to.
      filename: the name of the file. If None, defaults to 'config.yaml'.
    """
    filename = filename or 'config.yaml'
    file_path = os.path.join(path, filename)
    utils.safe_write(
        file_path=file_path,
        content=yaml.dump(self.as_dict()))

  @classmethod
  def lock_attributes(cls):
    """Assigns this class to be the base class for attribute validation."""
    cls._ATTRIBUTES_BASE_CLS = cls

  @classmethod
  def get_attribute_lock(cls):
    """Gets the base class for attribute validation."""
    return cls._ATTRIBUTES_BASE_CLS
