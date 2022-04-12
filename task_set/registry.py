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

"""Manage registration of individal tasks as well as task samplers/getters."""

import collections
import glob
import json
import os
from typing import Text, Dict, Tuple, Callable, TypeVar, Generic, List, Any
import tensorflow.compat.v1 as tf

Config = Dict[Text, Any]
T = TypeVar("T")


def _load_configs(config_path):
  """Load all configs from the configs folder."""
  configs = {}

  for t in glob.glob(os.path.join(config_path, "*.json")):
    with tf.gfile.GFile(t) as f:
      for k, v in json.load(f).items():
        if k in configs:
          raise ValueError("Duplicate name found for: %s" % k)
        configs[k] = v
  return configs


class _Registry(Generic[T]):
  """Manages a mapping from names to objects (e.g.

  tasks/optimizers).

  One can add objects via a fixed mapping from name to function, or when the
  object is configurable, via samplers that sample configurations and getters
  which convert these configurations to object instances. The sampled configs
  are first stored as json in config_path so that they are consistent across
  modifications to the code. See generate_samples.py to generate configs.
  When getting a task, this class manages loading configs from config_path.
  """

  def __init__(self, config_path):
    """Initializes a sampler from the given config_path."""
    self._config_path = config_path
    self._samplers = collections.OrderedDict()
    self._getters = collections.OrderedDict()
    self._fixed_config = {}

  def register_sampler(self, name):
    """Register a sampler function.

    This should be used as a decorator.
    Args:
      name: string Name of the sampler.

    Returns:
      the decorator function.
    """

    def decorator(fn):
      if name not in self._samplers:
        self._samplers[name] = fn
      else:
        raise ValueError("name [%s] already in registered as a sampler" % name)
      return fn

    return decorator

  def get_sampler(self, name):
    """Get a sampler function from the given name."""
    if name not in self._samplers:
      raise ValueError("Name [%s] not found in samplers!" % name)
    return self._samplers[name]

  def register_getter(self, name):
    """Register a getter function.

    This should be used as a decorator.
    Args:
      name: string Name of the getter.

    Returns:
      the decorator function.
    """

    def decorator(fn):
      if name not in self._getters:
        self._getters[name] = fn
      else:
        raise ValueError("Name [%s] already in registered as a getter" % name)
      return fn

    return decorator

  def get_getter(self, name):
    """Get a getter function from the given name."""
    if name not in self._getters:
      raise ValueError("Name [%s] not found in getters!" % name)
    return self._getters[name]

  def get_config_and_getter(self, name):
    """Get the config, and the getter name used for a given config name.

    Args:
      name: Name of the task.

    Returns:
      config: Represents all data needed to create the corresponding object.
      getter: function that maps config to an onject instance.
    """

    # Fixed functions have no config, just a function that returns an instance.
    if name in self._fixed_config:
      return {}, self._fixed_config[name]

    configs = _load_configs(self._config_path)
    if name not in configs:
      raise ValueError("Name [%s] not found in configs."
                       " (%d total configs found)" % (name, len(configs)))
    return configs[name]

  def get_instance(self, name, **kwargs):
    """Get an instance of a object corresponding to the given name.

    Args:
      name: name of instance to get.
      **kwargs: Arbitrary type forwarded to instance constructor.

    Returns:
      Instance of the object with the given name.
    """
    if name in self._fixed_config:
      obj = self._fixed_config[name](**kwargs)
    else:
      config, getter_name = self.get_config_and_getter(name)
      obj = self.get_getter(getter_name)(config, **kwargs)
    obj.name = name
    return obj

  def register_fixed(self, name):
    """Registers a fixed config.

    This should be used as a decorator.
    Args:
      name: string of config.

    Returns:
      the decorator function.
    """

    def decorator(fn):
      if name in self._fixed_config:
        raise ValueError("Duplicate fixed config found for name [%s]" % name)
      self._fixed_config[name] = fn
      return fn

    return decorator

  def get_all_fixed_config_names(self):
    """Return the list of fixed config names."""
    return list(self._fixed_config.keys())


_cur_dir = os.path.dirname(__file__)

_optimizers_path = os.path.join(_cur_dir, "optimizers", "configs")
optimizers_registry = _Registry(_optimizers_path)

_tasks_path = os.path.join(_cur_dir, "tasks", "configs")
task_registry = _Registry(_tasks_path)
