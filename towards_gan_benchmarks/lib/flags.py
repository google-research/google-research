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

"""For storing and accessing flags."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import absl.flags

absl.flags.DEFINE_string("nn_flags", None, "Flags dict as b64-encoded JSON")


class Flags(dict):
  """For storing and accessing flags."""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def print_values(self, indent=1):
    """Print the values in this flags object."""
    for k, v in sorted(self.items()):
      if isinstance(v, Flags):
        print("{}{}:".format("\t" * indent, k))
        v.print_values(indent=indent + 1)
      else:
        print("{}{}: {}".format("\t" * indent, k, v))

  def load(self, other):
    """Recursively copy values from another flags object into this one."""
    def recursive_update(flags, dict_):
      for k in dict_:
        if isinstance(dict_[k], dict):
          flags[k] = Flags()
          recursive_update(flags[k], dict_[k])
        else:
          flags[k] = dict_[k]

    recursive_update(self, other)

  def load_json(self, json_):
    """Copy values from a JSON string into this flags object."""
    if json_.startswith("b64"):
      json_ = base64.b64decode(json_[3:])
    other = json.loads(json_)
    self.load(other)

  def load_from_cmdline(self):
    self.load_json(absl.flags.FLAGS.nn_flags)

  def set_if_empty(self, key, val):
    """If there's no current value for the given key, assign the given value."""
    if key not in self:
      self[key] = val
