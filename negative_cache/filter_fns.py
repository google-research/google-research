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

"""Functions for calculating masks of new cache items."""
import abc
from typing import Dict, Tuple

import tensorflow.compat.v2 as tf

from negative_cache import negative_cache


class CacheFilterFn(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(self, cache,
               new_items):
    pass


class IsInCacheFilterFn(CacheFilterFn):
  """Creates a mask for items that are already in the cache.

  Given a tuple of keys, this class is a function that checks if there is a
  cache element that matches exactly on all keys.
  """

  def __init__(self, keys):
    self.keys = keys

  def __call__(self, cache,
               new_items):
    datawise_matches = []
    for key in self.keys:
      cache_vals = cache.data[key]
      new_items_vals = new_items[key]
      if cache_vals.dtype.is_floating:
        raise NotImplementedError('Floating datatypes are not yet implemented.')
      cache_vals = tf.expand_dims(cache_vals, axis=0)
      new_items_vals = tf.expand_dims(new_items_vals, axis=1)
      elementwise = cache_vals == new_items_vals
      datawise = tf.reduce_all(elementwise, axis=range(2, tf.rank(elementwise)))
      datawise_matches.append(datawise)
    all_keys_datawise = tf.stack(datawise_matches, axis=2)
    all_keys_match = tf.reduce_all(all_keys_datawise, axis=2)
    in_cache = tf.reduce_any(all_keys_match, axis=1)
    return tf.logical_not(in_cache)
