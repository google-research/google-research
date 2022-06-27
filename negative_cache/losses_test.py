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

"""Tests for negative_cache.losses."""

import tensorflow.compat.v2 as tf
from negative_cache import losses
from negative_cache import negative_cache


class LossesTest(tf.test.TestCase):

  def test_cache_classification_loss_interpretable_loss(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0], [3.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0], [1.0]])

    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], reducer=None)
    interpretable_loss = loss_fn(doc_network, query_embedding,
                                 pos_doc_embedding, cache).interpretable_loss
    interpretable_loss_expected = [3.169846, 0.349012, 0.694385]
    self.assertAllClose(interpretable_loss_expected, interpretable_loss)

    loss_fn.reducer = tf.math.reduce_mean
    interpretable_loss = loss_fn(doc_network, query_embedding,
                                 pos_doc_embedding, cache).interpretable_loss
    interpretable_loss_expected = (3.169846 + 0.349012 + 0.694385) / 3.0
    self.assertAllClose(interpretable_loss_expected, interpretable_loss)

  def test_cache_classification_loss_training_loss(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0], [3.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0], [1.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], reducer=None)
    loss_fn._retrieval_fn = retrieval_fn
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    prob_pos = tf.convert_to_tensor([0.0420101, 0.705385, 0.499381])
    score_differences = tf.convert_to_tensor([1.0, -3.0, 0.0])
    training_loss_expected = (1.0 - prob_pos) * score_differences
    self.assertAllClose(training_loss_expected, training_loss)

    loss_fn.reducer = tf.math.reduce_mean
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    training_loss_expected = tf.reduce_mean(
        (1.0 - prob_pos) * score_differences)
    self.assertAllClose(training_loss_expected, training_loss)

  def test_cache_classification_loss_training_loss_with_score_transform(self):
    cached_embeddings = 2.0 * tf.convert_to_tensor([[1.0], [-1.0]])
    cached_data = 2.0 * tf.convert_to_tensor([[1.0], [-1.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = 2.0 * tf.convert_to_tensor([[-1.0], [1.0], [3.0]])
    pos_doc_embedding = 2.0 * tf.convert_to_tensor([[2.0], [2.0], [1.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    score_transform = lambda scores: 0.25 * scores
    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], score_transform=score_transform, reducer=None)
    loss_fn._retrieval_fn = retrieval_fn
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    prob_pos = tf.convert_to_tensor([0.0420101, 0.705385, 0.499381])
    score_differences = tf.convert_to_tensor([1.0, -3.0, 0.0])
    training_loss_expected = (1.0 - prob_pos) * score_differences
    self.assertAllClose(training_loss_expected, training_loss)

    loss_fn.reducer = tf.math.reduce_mean
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    training_loss_expected = tf.reduce_mean(
        (1.0 - prob_pos) * score_differences)
    self.assertAllClose(training_loss_expected, training_loss)

  def test_cache_classification_loss_training_loss_gradient(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    query_model = tf.Variable(1.0)
    doc_model = tf.Variable(1.0)
    doc_network = lambda data: data['data'] * doc_model

    query_embedding = tf.convert_to_tensor([[-1.0], [1.0], [3.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0], [1.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn = losses.CacheClassificationLoss('embeddings', ['data'])
    loss_fn._retrieval_fn = retrieval_fn

    with tf.GradientTape() as tape:
      query_embedding = query_model * query_embedding
      pos_doc_embedding = doc_model * pos_doc_embedding
      training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                              cache).training_loss
    gradient = tape.gradient(training_loss, [query_model, doc_model])
    gradient_expected = [0.024715006, 0.024715006]
    self.assertAllClose(gradient_expected, gradient)

  def test_cache_classification_loss_training_loss_with_multi_cache(self):
    cached_embeddings_1 = tf.convert_to_tensor([[1.0]])
    cached_embeddings_2 = tf.convert_to_tensor([[-1.0]])
    cached_data_1 = tf.convert_to_tensor([[1.0]])
    cached_data_2 = tf.convert_to_tensor([[-1.0]])
    cache = {
        'cache1':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings_1,
                    'data': cached_data_1
                },
                age=[0]),
        'cache2':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings_2,
                    'data': cached_data_2
                },
                age=[0]),
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0], [3.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0], [1.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], reducer=None)
    loss_fn._retrieval_fn = retrieval_fn
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    prob_pos = tf.convert_to_tensor([0.0420101, 0.705385, 0.499381])
    score_differences = tf.convert_to_tensor([1.0, -3.0, 0.0])
    training_loss_expected = (1.0 - prob_pos) * score_differences
    self.assertAllClose(training_loss_expected, training_loss)

    loss_fn.reducer = tf.math.reduce_mean
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    training_loss_expected = tf.reduce_mean(
        (1.0 - prob_pos) * score_differences)
    self.assertAllClose(training_loss_expected, training_loss)

  def test_cache_classification_loss_refreshed_embeddings(self):
    cached_embeddings_1 = tf.convert_to_tensor([[1.0], [2.0], [3.0]])
    cached_embeddings_2 = tf.convert_to_tensor([[-1.0], [-2.0]])
    cached_data_1 = tf.convert_to_tensor([[10.0], [20.0], [30.0]])
    cached_data_2 = tf.convert_to_tensor([[-10.0], [-20.0]])
    cache = {
        'cache1':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings_1,
                    'data': cached_data_1
                },
                age=[0, 0, 0]),
        'cache2':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings_2,
                    'data': cached_data_2
                },
                age=[0, 0]),
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[0.0], [0.0], [0.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0], [1.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [3]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn = losses.CacheClassificationLoss('embeddings', ['data'])
    loss_fn._retrieval_fn = retrieval_fn
    cache_loss_return = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                                cache)
    self.assertAllEqual(cache_loss_return.updated_item_mask['cache1'],
                        [True, True, False])
    self.assertAllEqual(cache_loss_return.updated_item_mask['cache2'],
                        [False, False, True])
    self.assertAllEqual(
        cache_loss_return.updated_item_data['cache1']['embeddings'][0:2],
        [[10.0], [20.0]])
    self.assertAllEqual(
        cache_loss_return.updated_item_data['cache2']['embeddings'][2], [-10.0])
    self.assertAllEqual(cache_loss_return.updated_item_indices['cache1'][0:2],
                        [0, 1])
    self.assertEqual(cache_loss_return.updated_item_indices['cache2'][2], 0)

  def test_cache_classification_loss_staleness(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0]])
    cached_data = tf.convert_to_tensor([[2.0], [-3.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[0.0], [0.0], [0.0]])
    pos_doc_embedding = tf.convert_to_tensor([[0.0], [0.0], [0.0]])
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [1], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn = losses.CacheClassificationLoss('embeddings', ['data'])
    loss_fn._retrieval_fn = retrieval_fn
    staleness = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                        cache).staleness
    staleness_expected = 0.31481481481
    self.assertAllClose(staleness_expected, staleness)

    loss_fn.reducer = None
    staleness = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                        cache).staleness
    staleness_expected = [1.0 / 4.0, 4.0 / 9.0, 1.0 / 4.0]
    self.assertAllClose(staleness_expected, staleness)

  def test_cache_classification_loss_interpretable_loss_with_top_k(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0]])

    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], reducer=None, top_k=2)
    interpretable_loss = loss_fn(doc_network, query_embedding,
                                 pos_doc_embedding, cache).interpretable_loss
    interpretable_loss_expected = [3.0949229, 1.407605964]
    self.assertAllClose(interpretable_loss_expected, interpretable_loss)

  def test_cache_classification_loss_training_loss_with_top_k(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0]])

    loss_fn = losses.CacheClassificationLoss(
        'embeddings', ['data'], reducer=None, top_k=2)
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn._retrieval_fn = retrieval_fn
    training_loss = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                            cache).training_loss
    prob_pos = tf.convert_to_tensor([0.045278503, 0.2447284712])
    score_differences = tf.convert_to_tensor([3.0, -1.0])
    training_loss_expected = (1.0 - prob_pos) * score_differences
    self.assertAllClose(training_loss_expected, training_loss)

  def test_cache_classification_loss_refreshed_embeddings_with_top_k(self):
    cached_embeddings = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cached_data = tf.convert_to_tensor([[1.0], [-1.0], [3.0], [2.0]])
    cache = {
        'cache':
            negative_cache.NegativeCache(
                data={
                    'embeddings': cached_embeddings,
                    'data': cached_data
                },
                age=[0, 0])
    }
    doc_network = lambda data: data['data']
    query_embedding = tf.convert_to_tensor([[-1.0], [1.0]])
    pos_doc_embedding = tf.convert_to_tensor([[2.0], [2.0]])

    loss_fn = losses.CacheClassificationLoss('embeddings', ['data'], top_k=2)
    # pylint: disable=g-long-lambda
    retrieval_fn = lambda scores: tf.convert_to_tensor([[0], [0]],
                                                       dtype=tf.int64)
    # pylint: enable=g-long-lambda
    loss_fn._retrieval_fn = retrieval_fn
    loss_fn_return = loss_fn(doc_network, query_embedding, pos_doc_embedding,
                             cache)
    updated_item_data = loss_fn_return.updated_item_data
    updated_item_indices = loss_fn_return.updated_item_indices
    updated_item_mask = loss_fn_return.updated_item_mask
    self.assertAllClose([[-1.0], [1.0]],
                        updated_item_data['cache']['embeddings'])
    self.assertAllEqual([1, 0], updated_item_indices['cache'])
    self.assertAllEqual([True, True], updated_item_mask['cache'])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
