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

"""Overrides library for Learned Interpreters framework."""

import functools
import inspect

import six
from ipagnn.config import overrides
from ipagnn.config import overrides_common

OVERRIDES_PREFIX = 'overrides_'
OVERRIDES_MODULES = [
    overrides,
    overrides_common,
]


@functools.lru_cache()
def get_all_overrides():
  """Returns the names of all overrides defined in this module."""
  all_overrides = {}
  for module in OVERRIDES_MODULES:
    for member_name, member in inspect.getmembers(module):
      if member_name.startswith(OVERRIDES_PREFIX):
        override_name = member_name[len(OVERRIDES_PREFIX):]
        if override_name in all_overrides:
          raise RuntimeError('Duplicate override_name:', override_name)
        all_overrides[override_name] = member
  return all_overrides


def get_override_fn(override_name):
  """Returns the override with the supplied name."""
  return get_all_overrides().get(override_name)


def apply_overrides(config, override_names):
  """Apply any overrides specified in the config."""
  assert not isinstance(override_names, six.string_types)

  for override_name in override_names:
    if not override_name:
      continue

    # Allow parameterized overrides. E.g. eval:xid=1234.
    components = override_name.split(':')
    override_name = components[0]
    args = components[1:]
    args = dict([arg.split('=', 2) for arg in args])

    override_fn = get_override_fn(override_name)
    if override_fn:
      override_fn(config, **args)
    else:
      raise ValueError('Unexpected override_name:', override_name)
