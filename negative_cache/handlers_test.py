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

"""Tests for negative_cache.handler."""

import tensorflow.compat.v2 as tf
from negative_cache import handlers
from negative_cache import losses
from negative_cache import negative_cache


class StubCacheLoss(object):

  def __init__(self, updated_item_data, updated_item_indices,
               updated_item_mask):
    self.updated_item_data = updated_item_data
    self.updated_item_indices = updated_item_indices
    self.updated_item_mask = updated_item_mask

  def __call__(self, doc_network, query_embeddings, pos_doc_embeddings, cache):
    return losses.CacheLossReturn(
        training_loss=0.0,
        interpretable_loss=0.0,
        updated_item_data=self.updated_item_data,
        updated_item_indices=self.updated_item_indices,
        updated_item_mask=self.updated_item_mask,
        staleness=0.0)


class HandlerTest(tf.test.TestCase):

  def test_initialize_cache(self):
    specs = {
        'data': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        'embedding': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32)
    }
    cache_manager = negative_cache.CacheManager(specs, cache_size=4)
    handler = handlers.CacheLossHandler(
        cache_manager,
        StubCacheLoss(None, None, None),
        embedding_key='embedding',
        data_keys=('data',))
    self.assertAllEqual({'data', 'embedding'}, set(handler.cache.data.keys()))
    self.assertAllEqual(
        tf.zeros(shape=[4, 2], dtype=tf.int32), handler.cache.data['data'])
    self.assertAllEqual(
        tf.zeros(shape=[4, 3], dtype=tf.float32),
        handler.cache.data['embedding'])
    self.assertAllEqual(tf.zeros(shape=[4], dtype=tf.int32), handler.cache.age)

  def test_check_cache_after_update(self):
    specs = {
        'data': tf.io.FixedLenFeature(shape=[2], dtype=tf.int32),
        'embedding': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32)
    }
    cache_manager = negative_cache.CacheManager(specs, cache_size=4)
    cache_loss = StubCacheLoss(
        updated_item_data={
            'cache': {
                'embedding': tf.convert_to_tensor([[1.0, 1.0, 1.0]])
            }
        },
        updated_item_indices={'cache': tf.convert_to_tensor([0])},
        updated_item_mask={'cache': tf.convert_to_tensor([True])})
    handler = handlers.CacheLossHandler(
        cache_manager,
        cache_loss,
        embedding_key='embedding',
        data_keys=('data',))
    loss_actual = handler.update_cache_and_compute_loss(
        item_network=None,
        query_embeddings=None,
        pos_item_embeddings=tf.convert_to_tensor([[2.0, 2.0, 2.0]]),
        features={'data': tf.convert_to_tensor([[2, 2]])})
    self.assertAllEqual({'data', 'embedding'}, set(handler.cache.data.keys()))
    self.assertAllEqual(
        tf.convert_to_tensor([[0, 0], [2, 2], [0, 0], [0, 0]]),
        handler.cache.data['data'])
    self.assertAllEqual(
        tf.convert_to_tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]]),
        handler.cache.data['embedding'])
    self.assertAllEqual(tf.convert_to_tensor([0, 0, 1, 1]), handler.cache.age)
    self.assertEqual(0.0, loss_actual)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
