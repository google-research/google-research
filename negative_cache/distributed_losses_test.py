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

"""Tests for distributed losses in negative_cache.losses."""
import tensorflow.compat.v2 as tf

from negative_cache import losses
from negative_cache import negative_cache
from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import


def make_distributed_tensor(strategy, tensors):
  stacked = tf.stack(tensors, axis=0)
  fn = tf.function(lambda t: t[xla.replica_id()])
  return strategy.run(fn, args=(stacked,))


class DistributedCacheClassificationLossTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    self.strategy = tf.distribute.TPUStrategy(resolver)

  def tpuAssertAllClose(self, a, b):
    return self.assertAllClose(a, b, rtol=1e-4, atol=1e-4)

  def testInterpretableLoss(self):
    cache_data_1 = {
        'data': tf.convert_to_tensor([[1.0], [2.0]]),
        'embedding': tf.convert_to_tensor([[1.0], [2.0]])
    }
    cache_data_2 = {
        'data': tf.convert_to_tensor([[-1.0], [-2.0]]),
        'embedding': tf.convert_to_tensor([[-1.0], [-2.0]])
    }
    cache_data_multi_replica = {}
    for key in cache_data_1:
      cache_data_multi_replica[key] = make_distributed_tensor(
          self.strategy, [cache_data_1[key], cache_data_2[key]])
    cache_age = tf.zeros([0], dtype=tf.int32)
    cache_age_multi_replica = make_distributed_tensor(self.strategy,
                                                      [cache_age, cache_age])
    cache = negative_cache.NegativeCache(
        data=cache_data_multi_replica, age=cache_age_multi_replica)
    query_embeddings_1 = tf.convert_to_tensor([[-1.0], [1.0]])
    query_embeddings_2 = tf.convert_to_tensor([[1.0], [1.0]])
    query_embeddings_multi_replica = make_distributed_tensor(
        self.strategy, [query_embeddings_1, query_embeddings_2])
    pos_doc_embeddings_1 = tf.convert_to_tensor([[2.0], [2.0]])
    pos_doc_embeddings_2 = tf.convert_to_tensor([[2.0], [2.0]])
    pos_doc_embeddings_multi_replica = make_distributed_tensor(
        self.strategy, [pos_doc_embeddings_1, pos_doc_embeddings_2])

    embedding_key = 'embedding'
    data_keys = ('data',)
    loss_obj = losses.DistributedCacheClassificationLoss(
        embedding_key=embedding_key, data_keys=data_keys)
    doc_network = lambda data: data['data']

    @tf.function
    def loss_fn_reduced(query_embedding, pos_doc_embedding, cache):
      return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

    output_reduced = self.strategy.run(
        loss_fn_reduced,
        args=(query_embeddings_multi_replica, pos_doc_embeddings_multi_replica,
              {
                  'cache': cache
              }))

    interpretable_loss_reduced = output_reduced.interpretable_loss.values
    interpretable_loss_reduced_expected = [(4.37452 + 0.890350) / 2.0, 0.890350]
    self.tpuAssertAllClose(interpretable_loss_reduced_expected,
                           interpretable_loss_reduced)

    loss_obj.reducer = None

    @tf.function
    def loss_fn_no_reduce(query_embedding, pos_doc_embedding, cache):
      return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

    output_no_reduce = self.strategy.run(
        loss_fn_no_reduce,
        args=(query_embeddings_multi_replica, pos_doc_embeddings_multi_replica,
              {
                  'cache': cache
              }))
    interpretable_loss_no_reduce = output_no_reduce.interpretable_loss.values
    interpretable_loss_no_reduce_expected = [(4.37452, 0.890350),
                                             (0.890350, 0.890350)]
    self.tpuAssertAllClose(interpretable_loss_no_reduce_expected,
                           interpretable_loss_no_reduce)

  def testTrainingLoss(self):
    cache_data_1 = {
        'data': tf.convert_to_tensor([[1.0], [2.0]]),
        'embedding': tf.convert_to_tensor([[1.0], [2.0]])
    }
    cache_data_2 = {
        'data': tf.convert_to_tensor([[-1.0], [-2.0]]),
        'embedding': tf.convert_to_tensor([[-1.0], [-2.0]])
    }
    cache_data_multi_replica = {}
    for key in cache_data_1:
      cache_data_multi_replica[key] = make_distributed_tensor(
          self.strategy, [cache_data_1[key], cache_data_2[key]])
    cache_age = tf.zeros([0], dtype=tf.int32)
    cache_age_multi_replica = make_distributed_tensor(self.strategy,
                                                      [cache_age, cache_age])
    cache = negative_cache.NegativeCache(
        data=cache_data_multi_replica, age=cache_age_multi_replica)
    query_embeddings_1 = tf.convert_to_tensor([[-1.0], [1.0]])
    query_embeddings_2 = tf.convert_to_tensor([[1.0], [1.0]])
    query_embeddings_multi_replica = make_distributed_tensor(
        self.strategy, [query_embeddings_1, query_embeddings_2])
    pos_doc_embeddings_1 = tf.convert_to_tensor([[2.0], [2.0]])
    pos_doc_embeddings_2 = tf.convert_to_tensor([[2.0], [2.0]])
    pos_doc_embeddings_multi_replica = make_distributed_tensor(
        self.strategy, [pos_doc_embeddings_1, pos_doc_embeddings_2])

    embedding_key = 'embedding'
    data_keys = ('data',)
    loss_obj = losses.DistributedCacheClassificationLoss(
        embedding_key=embedding_key, data_keys=data_keys)

    def mock_retrieval_fn(scores):
      if scores.shape[0] == 4:
        return tf.convert_to_tensor([[0], [1], [0], [1]], dtype=tf.int64)
      else:
        return tf.convert_to_tensor([[0], [1]], dtype=tf.int64)

    loss_obj._retrieval_fn = mock_retrieval_fn
    doc_network = lambda data: data['data']

    @tf.function
    def loss_fn_reduced(query_embedding, pos_doc_embedding, cache):
      return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

    output_reduced = self.strategy.run(
        loss_fn_reduced,
        args=(query_embeddings_multi_replica, pos_doc_embeddings_multi_replica,
              {
                  'cache': cache
              }))

    training_loss_reduced = output_reduced.training_loss.values
    training_loss_reduced_expected = [(0.9874058 + -2.35795) / 2.0,
                                      (-0.589488 + -2.35795) / 2.0]
    self.tpuAssertAllClose(training_loss_reduced_expected,
                           training_loss_reduced)

    loss_obj.reducer = None

    @tf.function
    def loss_fn_no_reduce(query_embedding, pos_doc_embedding, cache):
      return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

    output_no_reduce = self.strategy.run(
        loss_fn_no_reduce,
        args=(query_embeddings_multi_replica, pos_doc_embeddings_multi_replica,
              {
                  'cache': cache
              }))
    training_loss_no_reduce = output_no_reduce.training_loss.values
    training_loss_no_reduce_expected = [(0.9874058, -2.35795),
                                        (-0.589488, -2.35795)]
    self.tpuAssertAllClose(training_loss_no_reduce_expected,
                           training_loss_no_reduce)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
