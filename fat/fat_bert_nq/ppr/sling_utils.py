# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""This file holds all sling related utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sling


def get_kb(kb_filename):
  kb = sling.Store()
  kb.load(kb_filename)
  return kb


def is_subj(item, kb):
  return 'isa' in kb[item.id] and kb[item.id]['isa']['id'] == '/w/item'


def is_property(item, kb):  # pylint: disable=unused-argument
  # maintaining kb for consistency in is_subj, is_prop functions
  # extentions could use kb for this check
  return 'isa' in item and item['isa']['id'] == '/w/property'


def is_direct_object(item, kb):  # pylint: disable=unused-argument
  # maintaining kb for consistency in is_subj, is_prop functions
  # extentions could use kb for this check
  return isinstance(
      item) == sling.Frame and 'id' in item and item.id.startswith('Q')


def is_nested_object(item, kb):  # pylint: disable=unused-argument
  # maintaining kb for consistency in is_subj, is_prop functions
  # extentions could use kb for this check
  return isinstance(item) == sling.Frame and 'is' in item and isinstance(
      item['is']
  ) == sling.Frame and 'id' in item['is'] and item['is']['id'].startswith('Q')


def get_properties(item, kb):  # pylint: disable=unused-argument
  """Function to retrieve [(prop, obj)] lists for a subj item."""

  # maintaining kb for consistency in is_subj, is_prop functions
  # extentions could use kb for this check
  props = []
  for k, v in item:
    if is_property(k, kb):
      rel = k.id
      # Check if Target Sling Frame is entity
      if is_direct_object(v, kb):
        obj = v.id
      elif is_nested_object(v, kb):
        obj = v['is']['id']
      else:
        # Target frame is not entity
        # TODO(vidhisha) Extension : Add handling of dates & values here
        continue
      props.append((rel, obj))
  return props
