# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Base configuration class."""

import dataclasses
from typing import Tuple

import tensorflow as tf
import yaml


@dataclasses.dataclass
class Config(object):
  """Constructs a base for creating and operating on configs."""

  _PARAM_TYPES = [int, float, str, bool, type(None)]

  def __init__(self):
    pass

  def _is_private(self, attr_name):
    return attr_name.startswith("__") or attr_name.startswith("_")

  def _is_parameter(self, node):
    is_parameter = any([isinstance(node, type_) for type_ in self._PARAM_TYPES])
    return is_parameter

  def _reconstruct_tuple(self, node):
    """Reconstructs the tuple structure but with dictionary as children."""

    tuple_param = ()
    for child in node:
      if self._is_parameter(child):
        tuple_param += (child,)
      elif isinstance(child, Tuple):
        tuple_param += self._reconstruct_tuple(child)
      elif isinstance(child, Config):
        tuple_param += (child.as_dict(),)

    return tuple_param

  def as_dict(self):
    """Traverses over the configs tree and returns it as a dictionary."""

    params_dict = {}
    for name in dir(self):
      node = getattr(self, name)
      if self._is_parameter(node) and not self._is_private(name):
        params_dict[name] = node

      elif isinstance(node, Tuple):
        params_dict[name] = self._reconstruct_tuple(node)

      elif isinstance(node, Config):
        params_dict[name] = node.as_dict()

    return params_dict

  def override(self, params_dict, node=None):
    """Traverses over the params_dict and overrides the attributes."""

    assert isinstance(params_dict,
                      dict), "Please provide a valid override mapping."
    if node is None:
      node = self
    for key, value in params_dict.items():
      assert hasattr(node, key), f"Key {key} not found in {node}"
      if self._is_parameter(value):
        setattr(node, key, value)
      else:
        self.override(value, getattr(node, key))

  def override_from_file(self, path):
    """Loads a dictionary of configs and overrides."""

    with tf.io.gfile.GFile(path) as f:
      self.override(yaml.load(f, Loader=yaml.FullLoader))

  def override_from_str(self, dict_str):
    """Gets a safe-dumped dictionary of configs and overrides."""

    self.override(yaml.load(dict_str, Loader=yaml.FullLoader))

