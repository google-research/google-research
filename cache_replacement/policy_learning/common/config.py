# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Simple nested configs."""

import ast
import copy
import json


class Config(object):
  """Collection of potentially nested configs."""

  def __init__(self, config=None):
    """Constructs from a (nested) dict.

    Likely, from_file should be used and not the constructor.

    Args:
      config (dict | None): specifies initial key-value mappings. If None, the
        config is empty. All keys in the dict must be strings.
    """
    if config is None:
      config = {}
    self._config_tree = config

  @classmethod
  def from_files_and_bindings(cls, paths, bindings):
    """Creates a merged Config from config files and overriding bindings.

    Args:
      paths (list[str]): list of paths to config files, merged front to back.
        Paths must be included as resource dependencies.
      bindings (list[str]): list of strings formatted as key=value, which
        override settings in the config files.

    Returns:
      Config
    """
    configs = []
    for path in paths:
      with open(path, "r") as config_file:
        configs.append(cls.from_file(config_file))

    merged_config = cls.merge(configs)
    for binding in bindings:
      key, value = binding.split("=", 1)
      merged_config.set(key, ast.literal_eval(value))
    return merged_config

  @classmethod
  def merge(cls, configs):
    """Merges multiple configs together.

    (key, value) pairs appearing in later configs overwrite the values appearing
    in earlier configs.

    Args:
      configs (list[Config]): list of configs to merge.

    Returns:
      config (Config): the merged config.
    """
    merged_config = copy.deepcopy(configs[0])
    for config in configs[1:]:
      for key in config.keys():
        current_value = merged_config.get(key)
        merging_value = config.get(key)
        if (not isinstance(current_value, Config) or
            not isinstance(merging_value, Config)):
          merged_config.set(key, merging_value)
        else:
          merged_config.set(key, cls.merge([current_value, merging_value]))
    return merged_config

  @classmethod
  def from_file(cls, f):
    """Loads from the provided file.

    Args:
       f (File): file to read from.

    Returns:
      config (Config): config loaded from file.
    """
    return cls(json.load(f))

  def to_file(self, f):
    """Serializes to the provided file.

    Args:
      f (File): file to write to.
    """
    # Dump in human-readable format
    json.dump(self._config_tree, f, indent=4, sort_keys=True)

  def get(self, key, default=None):
    """Returns value associated with key.

    Args:
      key (str): use periods to separate nested acccesses, e.g.,
        config.get("foo.bar") for config.foo.bar.
      default (Object): value to return if key is not in the config.

    Returns:
      value (Object): value associated with the key. Returns a sub-config
        (Config) if the config is nested and the key does not identify a leaf.
    """
    nested_keys = key.split(".", 1)

    # Base case
    if len(nested_keys) == 1:
      value = self._config_tree.get(key, default)
      if isinstance(value, dict):
        return self.__class__(value)
      return value

    sub_config = self.get(nested_keys[0])
    if not isinstance(sub_config, Config):
      return default
    return sub_config.get(nested_keys[1], default)

  def keys(self):
    """Returns iterator over the keys in this Config.

    Returns:
      iterator: doesn't include nested keys.
    """
    return self._config_tree.keys()

  def set(self, key, value):
    """Sets value to be associated with key, replacing prior value if it exists.

    Args:
      key (str): use periods to separate nested accesses, see get.
      value (Object): value to associate with the key.
    """
    nested_keys = key.split(".", 1)

    # Base case
    if len(nested_keys) == 1:
      if isinstance(value, Config):
        value = value._config_tree  # pylint: disable=protected-access
      self._config_tree[key] = value
      return

    current_value = self.get(nested_keys[0])
    if current_value is not None and not isinstance(current_value, Config):
      raise ValueError(("Trying to perform nested set with key: {}, "
                        "but value associated with {} is not nested.")
                       .format(key, nested_keys[0]))

    sub_config = self.__class__(
        self._config_tree.setdefault(nested_keys[0], {}))
    sub_config.set(nested_keys[1], value)

  def __eq__(self, other):
    if not isinstance(other, Config):
      return False

    if set(self.keys()) != set(other.keys()):
      return False

    for key in self._config_tree:
      if other.get(key) != self.get(key):
        return False
    return True

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return json.dumps(self._config_tree, indent=4, sort_keys=True)
