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

# Lint as: python3
"""Tests for negative_cache.filter_fns."""

import tensorflow.compat.v2 as tf
from negative_cache import filter_fns
from negative_cache import negative_cache


class FilterFnsTest(tf.test.TestCase):

  def test_is_in_cache_filter_fn(self):

    data = {
        '1':
            tf.convert_to_tensor([[0, 0], [1, 1], [2, 2], [3, 3], [20, 20],
                                  [30, 30], [40, 40]]),
        '2':
            tf.convert_to_tensor([[4, 4], [5, 5], [6, 6], [7, 7], [50, 50],
                                  [60, 60], [70, 70]])
    }
    age = tf.zeros([4])
    cache = negative_cache.NegativeCache(data, age)
    new_items = {
        '1': tf.convert_to_tensor([[0, 0], [1, 1], [3, 3], [8, 8], [9, 9]]),
        '2': tf.convert_to_tensor([[4, 4], [7, 7], [7, 7], [10, 10], [11, 11]])
    }
    is_in_cache_filter_fn = filter_fns.IsInCacheFilterFn(keys=('1', '2'))
    mask = is_in_cache_filter_fn(cache, new_items)
    mask_expected = tf.convert_to_tensor([False, True, False, True, True])
    self.assertAllEqual(mask_expected, mask)

  def test_is_in_cache_filter_fn_with_missing_keys(self):

    data = {
        '1':
            tf.convert_to_tensor([[0, 0], [1, 1], [2, 2], [3, 3], [20, 20],
                                  [30, 30], [40, 40]]),
        '2':
            tf.convert_to_tensor([[4, 4], [5, 5], [6, 6], [7, 7], [50, 50],
                                  [60, 60], [70, 70]])
    }
    age = tf.zeros([4])
    cache = negative_cache.NegativeCache(data, age)
    new_items = {
        '1': tf.convert_to_tensor([[0, 0], [1, 1], [3, 3], [8, 8], [9, 9]]),
        '2': tf.convert_to_tensor([[4, 4], [7, 7], [7, 7], [10, 10], [11, 11]])
    }
    is_in_cache_filter_fn = filter_fns.IsInCacheFilterFn(keys=('1',))
    mask = is_in_cache_filter_fn(cache, new_items)
    mask_expected = tf.convert_to_tensor([False, False, False, True, True])
    self.assertAllEqual(mask_expected, mask)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
