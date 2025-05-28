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
"""Functions for parsing args."""

import ast
import copy
import os

import yaml


class CfgNode(dict):
  """CfgNode represents an internal node in the configuration tree.

  It's a simple

    dict-like container that allows for attribute-based access to keys.
  """

  def __init__(self, init_dict=None, key_list=None):
    # Recursively convert nested dictionaries in init_dict into CfgNodes
    init_dict = {} if init_dict is None else init_dict
    key_list = [] if key_list is None else key_list
    for k, v in init_dict.items():
      if isinstance(v, dict):
        # Convert dict to CfgNode
        init_dict[k] = CfgNode(v, key_list=key_list + [k])
    super().__init__(init_dict)

  def __getattr__(self, name):
    if name in self:
      return self[name]

  def __setattr__(self, name, value):
    self[name] = value

  def __str__(self):

    def _indent(s_, num_spaces):
      s = s_.split("\n")
      if len(s) == 1:
        return s_
      first = s.pop(0)
      s = [(num_spaces * " ") + line for line in s]
      s = "\n".join(s)
      s = first + "\n" + s
      return s

    r = ""
    s = []
    for k, v in sorted(self.items()):
      seperator = "\n" if isinstance(v, CfgNode) else " "
      attr_str = "{}:{}{}".format(str(k), seperator, str(v))
      attr_str = _indent(attr_str, 2)
      s.append(attr_str)
    r += "\n".join(s)
    return r

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, super().__repr__())


def load_cfg_from_cfg_file(file):
  """Load configs from cfg file."""
  cfg = {}
  assert os.path.isfile(file)
  assert file.endswith(".yaml"), "{} is not a yaml file".format(file)

  with open(file, "r") as f:
    cfg_from_file = yaml.safe_load(f)

  for key in cfg_from_file:
    for k, v in cfg_from_file[key].items():
      cfg[k] = v

  cfg = CfgNode(cfg)
  return cfg


def merge_cfg_from_list(cfg, cfg_list):
  """Merge configs from a list."""
  new_cfg = copy.deepcopy(cfg)
  assert len(cfg_list) % 2 == 0
  for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
    subkey = full_key.split(".")[-1]
    assert subkey in cfg, "Non-existent key: {}".format(full_key)
    value = _decode_cfg_value(v)
    value = _check_and_coerce_cfg_value_type(value, cfg[subkey], full_key)
    setattr(new_cfg, subkey, value)

  return new_cfg


def _decode_cfg_value(v):
  """Decodes a raw config value."""
  # All remaining processing is only applied to strings
  if not isinstance(v, str):
    return v
  # Try to interpret `v` as a:
  #   string, number, tuple, list, dict, boolean, or None
  try:
    v = ast.literal_eval(v)
  except ValueError:
    pass
  except SyntaxError:
    pass
  return v


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
  """Checks matches exactly or is one of a few cases in which can be easily coerced.
  """
  original_type = type(original)
  replacement_type = type(replacement)

  # The types must match (with some exceptions)
  if replacement_type == original_type or original is None:
    return replacement

  # Cast replacement from from_type to to_type if the replacement and original
  # types match from_type and to_type
  def conditional_cast(from_type, to_type):
    if replacement_type == from_type and original_type == to_type:
      return True, to_type(replacement)
    else:
      return False, None

  # Conditionally casts
  # list <-> tuple
  casts = [(tuple, list), (list, tuple)]
  for (from_type, to_type) in casts:
    converted, converted_value = conditional_cast(from_type, to_type)
    if converted:
      return converted_value

  raise ValueError(
      "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
      "key: {}".format(original_type, replacement_type, original, replacement,
                       full_key))
