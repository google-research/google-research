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

"""Contains class for looking up features using three formats."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Lookup(object):
  """Class for mapping between TF Example keys, indices, and shorthands.

  The dictionary maps between a shorthand name, a full TF Example key, and an
  assigned index.
  """

  def __init__(self, feature_keys, matchers):
    """Assigns all feature keys a unique shorthand name and numerical index."""
    feature_keys.sort()
    self._index_to_shorthand = {}
    self._shorthand_to_key = {}
    self._key_to_index = {}
    for index, key in enumerate(feature_keys):
      for matcher in matchers:
        match = matcher.match(key)
        if match:
          shorthand = match.group(1).replace('seizure', 'sz').upper()
          if shorthand not in self._shorthand_to_key:
            self._InitTriple(key, index, shorthand)
            continue
      # TODO(rldavies): map full key to shorthand if no regex matches.

  def _InitTriple(self, key, index, shorthand):
    self._index_to_shorthand[str(index)] = shorthand
    self._shorthand_to_key[shorthand] = key
    self._key_to_index[key] = str(index)

  def GetKeyFromIndex(self, index):
    return (self._shorthand_to_key[self._index_to_shorthand[str(index)]]
            if str(index) in self._index_to_shorthand else None)

  def GetIndexFromShorthand(self, shorthand):
    return (self._key_to_index[self._shorthand_to_key[shorthand]]
            if shorthand in self._shorthand_to_key else None)

  def GetShorthandFromKey(self, key):
    return (self._index_to_shorthand[self._key_to_index[key]]
            if key in self._key_to_index else None)

  def GetShorthandFromIndex(self, index):
    return (self._index_to_shorthand[str(index)]
            if str(index) in self._index_to_shorthand else None)

  def GetIndexFromKey(self, key):
    return (self._key_to_index[str(key)]
            if str(key) in self._key_to_index else None)

  def GetIndexToShorthandDict(self):
    return self._index_to_shorthand
