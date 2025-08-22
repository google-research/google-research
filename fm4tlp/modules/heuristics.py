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

import numpy as np


class PersistantForecaster:

  def __init__(self, num_class):
    self.dict = {}
    self.num_class = num_class

  def update_dict(self, node_id, label):
    self.dict[node_id] = label

  def query_dict(self, node_id):
    r"""Parameters:

        node_id: the node to query
    Returns:
        returns the last seen label of the node if it exists, if not return zero
        vector
    """
    if node_id in self.dict:
      return self.dict[node_id]
    else:
      return np.zeros(self.num_class)


class MovingAverage:

  def __init__(self, num_class, window=7):
    self.dict = {}
    self.num_class = num_class
    self.window = window

  def update_dict(self, node_id, label):
    if node_id in self.dict:
      total = self.dict[node_id] * (self.window - 1) + label
      self.dict[node_id] = total / self.window
    else:
      self.dict[node_id] = label

  def query_dict(self, node_id):
    r"""Parameters:

        node_id: the node to query
    Returns:
        returns the last seen label of the node if it exists, if not return zero
        vector
    """
    if node_id in self.dict:
      return self.dict[node_id]
    else:
      return np.zeros(self.num_class)
