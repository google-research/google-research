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

"""Class that caches things in Pickle."""
# pylint: disable-all
import os
import pickle
from typing import Any


class PickleCache:
  """Class that caches URL --> Annotation.

  Attributes:
    cache_path: The file location of the cache.
    cache: The dictionary representing the cache.
  """

  def __init__(self, cache_path):
    self.cache_path = cache_path
    if os.path.exists(cache_path):
      with open(cache_path, 'rb') as f:
        self.cache = pickle.load(f)
    else:
      self.cache = {}

  def __getitem__(self, key):
    return self.cache[key]

  def __setitem__(self, key, value):
    self.cache[key] = value

  def __contains__(self, key):
    return key in self.cache

  def save(self):
    with open(self.cache_path, 'wb') as f:
      pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)
      print('Saved at: ', self.cache_path)
