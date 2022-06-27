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

"""Tests for negative_cache.negative_cache."""

import tensorflow.compat.v2 as tf
from negative_cache import negative_cache


class NegativeCacheTest(tf.test.TestCase):

  def test_init_cache(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
        '3': tf.io.FixedLenFeature(shape=[3, 2], dtype=tf.float32)
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=6)
    cache = cache_manager.init_cache()
    self.assertEqual({'1', '2', '3'}, set(cache.data.keys()))
    self.assertAllEqual(tf.zeros([6, 2], dtype=tf.int32), cache.data['1'])
    self.assertAllEqual(tf.zeros([6, 3], dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(tf.zeros([6, 3, 2], dtype=tf.float32), cache.data['3'])
    self.assertAllEqual(tf.zeros([6], dtype=tf.int32), cache.age)

  def test_update_cache(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updates = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.int32), cache.age)
    updates = {
        '1': 2 * tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': 2.0 * tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [1, 1], [2, 2], [2, 2]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0],
                              [2.0, 2.0, 2.0]],
                             dtype=tf.float32), cache.data['2'])
    updates = {
        '1': 3 * tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': 3.0 * tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[3, 3], [3, 3], [2, 2], [2, 2]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0],
                              [2.0, 2.0, 2.0]],
                             dtype=tf.float32), cache.data['2'])

  def test_update_cache_with_non_multiple_cache_size(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=3)
    cache = cache_manager.init_cache()
    updates = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [1, 1], [0, 0]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([0, 0, 1], dtype=tf.int32), cache.age)
    updates = {
        '1': 2 * tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': 2.0 * tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[2, 2], [1, 1], [2, 2]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor(
            [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([0, 1, 0], dtype=tf.int32), cache.age)

  def test_update_caches_with_tf_function(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    init_cache_fn = tf.function(cache_manager.init_cache)
    cache = init_cache_fn()
    updates = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    update_cache_fn = tf.function(cache_manager.update_cache)
    cache = update_cache_fn(cache, new_items=updates)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             dtype=tf.float32), cache.data['2'])

  def test_raises_value_error_if_different_update_sizes(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    init_cache_fn = tf.function(cache_manager.init_cache)
    cache = init_cache_fn()
    updates = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': tf.ones(shape=[1, 3], dtype=tf.float32)
    }
    update_cache_fn = tf.function(cache_manager.update_cache)
    with self.assertRaises(ValueError):
      cache = update_cache_fn(cache, new_items=updates)

  def test_update_cache_with_existing_items(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updated_item_indices = tf.convert_to_tensor([1, 3], dtype=tf.int32)
    updated_item_data = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
        '2': tf.ones(shape=[2, 3], dtype=tf.float32)
    }
    cache = cache_manager.update_cache(
        cache,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0]],
                             dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([1, 0, 1, 0], dtype=tf.int32), cache.age)

  def test_partial_update_cache_with_existing_items(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updated_item_indices = tf.convert_to_tensor([1, 3], dtype=tf.int32)
    updated_item_data = {
        '1': tf.ones(shape=[2, 2], dtype=tf.int32),
    }
    cache = cache_manager.update_cache(
        cache,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(tf.zeros([4, 3], dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([1, 0, 1, 0], dtype=tf.int32), cache.age)

  def test_update_cache_with_new_items_and_existing_items(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=2)
    data = {
        '1': tf.convert_to_tensor([[0], [0], [3]], dtype=tf.int32),
        '2': tf.convert_to_tensor([[1], [2], [4]], dtype=tf.int32)
    }
    age = tf.convert_to_tensor([2, 1, 0])
    cache = negative_cache.NegativeCache(data=data, age=age)
    updated_item_indices = tf.convert_to_tensor([0], dtype=tf.int32)
    updated_item_data = {
        '1': tf.convert_to_tensor([[10]], dtype=tf.int32),
    }
    new_items = {
        '1': tf.convert_to_tensor([[11]], dtype=tf.int32),
        '2': tf.convert_to_tensor([[12]], dtype=tf.int32)
    }
    cache = cache_manager.update_cache(
        cache,
        new_items=new_items,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[10], [11], [3]], dtype=tf.int32),
        cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[1], [12], [4]], dtype=tf.int32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([0, 0, 1], dtype=tf.int32), cache.age)

  def test_raises_value_error_if_new_item_keys_not_equal_specs(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updates = {
        '1': tf.ones(shape=[2, 1], dtype=tf.int32),
    }
    with self.assertRaises(ValueError):
      cache = cache_manager.update_cache(cache, new_items=updates)
    updates = {
        '1': tf.ones(shape=[2, 1], dtype=tf.int32),
        '2': tf.ones(shape=[2, 1], dtype=tf.int32),
        '3': tf.ones(shape=[2, 1], dtype=tf.int32),
    }
    with self.assertRaises(ValueError):
      cache = cache_manager.update_cache(cache, new_items=updates)

  def test_raises_value_error_if_update_item_keys_not_in_specs(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[1], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updated_item_data = {
        '1': tf.ones(shape=[2, 1], dtype=tf.int32),
        '3': tf.ones(shape=[2, 1], dtype=tf.int32),
    }
    updated_item_indices = tf.convert_to_tensor([0])
    with self.assertRaises(ValueError):
      cache = cache_manager.update_cache(
          cache,
          updated_item_data=updated_item_data,
          updated_item_indices=updated_item_indices)

  def test_masked_update_cache_with_existing_items(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updated_item_indices = tf.convert_to_tensor([1, 3], dtype=tf.int32)
    updated_item_data = {
        '1':
            tf.convert_to_tensor([[1, 1], [2, 2]], dtype=tf.int32),
        '2':
            tf.convert_to_tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                                 dtype=tf.float32),
    }
    updated_item_mask = tf.convert_to_tensor([True, False])
    cache = cache_manager.update_cache(
        cache,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices,
        updated_item_mask=updated_item_mask)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[0, 0], [1, 1], [0, 0], [0, 0]],
                             dtype=tf.float32), cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([1, 0, 1, 1], dtype=tf.int32), cache.age)

  def test_masked_update_cache_with_existing_items_when_all_items_masked(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = negative_cache.NegativeCache(
        data={
            '1':
                tf.convert_to_tensor([[5, 5], [10, 10], [15, 15], [20, 20]],
                                     dtype=tf.int32)
        },
        age=tf.convert_to_tensor([2, 2, 2, 2], dtype=tf.int32))
    updated_item_indices = tf.convert_to_tensor([1, 3], dtype=tf.int32)
    updated_item_data = {
        '1': tf.convert_to_tensor([[1, 1], [2, 2]], dtype=tf.int32),
    }
    updated_item_mask = tf.convert_to_tensor([False, False])
    cache = cache_manager.update_cache(
        cache,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices,
        updated_item_mask=updated_item_mask)
    self.assertEqual({'1'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[5, 5], [10, 10], [15, 15], [20, 20]],
                             dtype=tf.float32), cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([3, 3, 3, 3], dtype=tf.int32), cache.age)

  def test_update_cache_without_lru(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(
        specs=specs, cache_size=4, use_lru=False)
    cache = negative_cache.NegativeCache(
        data={
            '1':
                tf.convert_to_tensor([[5, 5], [10, 10], [15, 15], [20, 20]],
                                     dtype=tf.int32)
        },
        age=tf.convert_to_tensor([1, 0, 1, 1], dtype=tf.int32))
    updated_item_indices = tf.convert_to_tensor([1, 3], dtype=tf.int32)
    updated_item_data = {
        '1': tf.convert_to_tensor([[1, 1], [2, 2]], dtype=tf.int32),
    }
    cache = cache_manager.update_cache(
        cache,
        updated_item_indices=updated_item_indices,
        updated_item_data=updated_item_data)
    cache_data_expected = [[5, 5], [1, 1], [15, 15], [2, 2]]
    cache_age_expected = [2, 1, 2, 2]
    self.assertAllEqual(cache_data_expected, cache.data['1'])
    self.assertAllEqual(cache_age_expected, cache.age)

  def test_masked_update_cache_with_existing_items_not_in_index_one(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        '2': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = cache_manager.init_cache()
    updated_item_indices = tf.convert_to_tensor([0, 3], dtype=tf.int32)
    updated_item_data = {
        '1':
            tf.convert_to_tensor([[1, 1], [2, 2]], dtype=tf.int32),
        '2':
            tf.convert_to_tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                                 dtype=tf.float32),
    }
    updated_item_mask = tf.convert_to_tensor([True, False])
    cache = cache_manager.update_cache(
        cache,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices,
        updated_item_mask=updated_item_mask)
    self.assertEqual({'1', '2'}, set(cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [0, 0], [0, 0], [0, 0]],
                             dtype=tf.float32), cache.data['1'])
    self.assertAllEqual(
        tf.convert_to_tensor([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             dtype=tf.float32), cache.data['2'])
    self.assertAllEqual(
        tf.convert_to_tensor([0, 1, 1, 1], dtype=tf.int32), cache.age)

  def test_new_items_with_mask(self):
    specs = {
        '1': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
    }
    cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
    cache = negative_cache.NegativeCache(
        data={
            '1':
                tf.convert_to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]],
                                     dtype=tf.int32)
        },
        age=tf.convert_to_tensor([0, 2, 1, 3], dtype=tf.int32))

    new_items = {
        '1': tf.convert_to_tensor([[5, 5], [6, 6], [7, 7]], dtype=tf.int32)
    }
    new_items_mask = tf.convert_to_tensor([True, False, True])
    cache = cache_manager.update_cache(
        cache, new_items=new_items, new_items_mask=new_items_mask)
    self.assertAllEqual(
        tf.convert_to_tensor([[1, 1], [7, 7], [3, 3], [5, 5]], dtype=tf.int32),
        cache.data['1'])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
