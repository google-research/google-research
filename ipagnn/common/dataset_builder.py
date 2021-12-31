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

"""Learned interpreters shared DatasetBuilder base class."""


class DatasetBuilder(object):
  """The base class for all learned interpreter compatible datasets."""

  def set_representation(self, representation):
    # The default implementation just saves the representation.
    self.representation = representation

  def key(self, key):
    # The default implementation ignores the representation.
    return key

  def as_in_memory_dataset(self):
    raise NotImplementedError()

  def _shepherds(self):
    """Gets all shepherds from any feature connectors supplying shepherds."""
    shepherds = []
    roots = []
    for key, feature in self._features().items():
      if hasattr(feature, "get_shepherd_info"):
        feature_shepherds = feature.get_shepherd_info()
        if feature_shepherds:
          shepherds.extend(feature_shepherds)
          roots.extend([key] * len(feature_shepherds))
    return shepherds, roots
