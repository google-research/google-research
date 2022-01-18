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

import os
import pickle
import tempfile
import time
import numpy as np
from scann.scann_ops.py import scann_ops
import tensorflow as tf


class TestGraphMode(tf.test.TestCase):

  def serialization_tester(self, np_dataset, np_queries, searcher_lambda):
    with tempfile.TemporaryDirectory() as tmpdirname:
      g1 = tf.compat.v1.Graph()
      g2 = tf.compat.v1.Graph()
      with tf.compat.v1.Session(graph=g1) as sess:
        self.assertFalse(tf.executing_eagerly())

        ds = tf.convert_to_tensor(np_dataset, dtype=tf.float32)
        qs = tf.convert_to_tensor(np_queries, dtype=tf.float32)

        searcher = searcher_lambda(ds)
        idx, dis = searcher.search_batched(qs)
        module = searcher.serialize_to_module()

        t1 = time.time()
        np_dis, np_idx = sess.run([dis, idx])
        with open(os.path.join(tmpdirname, "dis.pkl"), "wb") as f:
          pickle.dump(np_dis, f)
        with open(os.path.join(tmpdirname, "idx.pkl"), "wb") as f:
          pickle.dump(np_idx, f)
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.saved_model.save(
            module,
            tmpdirname,
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))
        print("first run with on-the-fly training:", time.time() - t1)
      with tf.compat.v1.Session(graph=g2) as sess2:
        self.assertFalse(tf.executing_eagerly())

        ds = tf.convert_to_tensor(np_dataset, dtype=tf.float32)
        qs = tf.convert_to_tensor(np_queries, dtype=tf.float32)

        reloaded = tf.compat.v2.saved_model.load(export_dir=tmpdirname)
        searcher = scann_ops.searcher_from_module(reloaded)
        idx, dis = searcher.search_batched(qs)

        t1 = time.time()
        sess2.run(tf.compat.v1.global_variables_initializer())
        np_dis, np_idx = sess2.run([dis, idx])
        with open(os.path.join(tmpdirname, "dis.pkl"), "rb") as f:
          orig_dis = pickle.load(f)
        with open(os.path.join(tmpdirname, "idx.pkl"), "rb") as f:
          orig_idx = pickle.load(f)
        print("second run from serialization:", time.time() - t1)
        self.assertTrue(np.array_equal(orig_dis, np_dis))
        self.assertTrue(np.array_equal(orig_idx, np_idx))

  def test_ah_serialization(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)

    def searcher_maker(ds):
      return scann_ops.builder(ds, 15, "dot_product").score_ah(2).build()

    self.serialization_tester(np_dataset, np_queries, searcher_maker)

  def test_tree_ah_serialization(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)

    def searcher_maker(ds):
      return scann_ops.builder(ds, 15,
                               "dot_product").tree(100, 20).score_ah(2).build()

    self.serialization_tester(np_dataset, np_queries, searcher_maker)

  def test_tree_brute_force_serialization(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)

    def searcher_maker(ds):
      return scann_ops.builder(ds, 10, "dot_product").tree(
          100, 10).score_brute_force(True).build()

    self.serialization_tester(np_dataset, np_queries, searcher_maker)

  def test_brute_force_int8_serialization(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)

    def searcher_maker(ds):
      return scann_ops.builder(ds, 10,
                               "dot_product").score_brute_force(True).build()

    self.serialization_tester(np_dataset, np_queries, searcher_maker)

  def test_reordering_serialization(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)

    def searcher_maker(ds):
      return scann_ops.builder(ds, 10,
                               "dot_product").score_ah(2).reorder(40).build()

    self.serialization_tester(np_dataset, np_queries, searcher_maker)


class TestEagerMode(tf.test.TestCase):

  def numpy_brute_force_dp(self, dataset, queries, k):
    product = np.matmul(queries, np.transpose(dataset))
    indices = (-product).argsort()[:, :k]
    return np.sort(product)[:, :-k - 1:-1], indices

  def test_brute_force(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)
    dataset = tf.convert_to_tensor(np_dataset, dtype=tf.float32)
    queries = tf.convert_to_tensor(np_queries, dtype=tf.float32)
    k = 10

    searcher = scann_ops.builder(dataset, k,
                                 "dot_product").score_brute_force().build()

    indices, distances = searcher.search_batched(queries)
    np_distances, np_indices = self.numpy_brute_force_dp(
        np_dataset, np_queries, k)
    self.assertAllClose(distances, np_distances)
    self.assertAllEqual(indices, np_indices)

    indices2, distances2 = searcher.search_batched(
        queries, final_num_neighbors=20)
    np_distances2, np_indices2 = self.numpy_brute_force_dp(
        np_dataset, np_queries, 20)
    self.assertAllClose(distances2, np_distances2)
    self.assertAllEqual(indices2, np_indices2)

    indices3, distances3 = searcher.search(queries[0], final_num_neighbors=20)
    self.assertAllClose(distances3, np_distances2[0])
    self.assertAllEqual(indices3, np_indices2[0])

  def test_parallel(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(1000, 32)
    dataset = tf.convert_to_tensor(np_dataset, dtype=tf.float32)
    queries = tf.convert_to_tensor(np_queries, dtype=tf.float32)
    k = 10

    searcher = scann_ops.builder(dataset, k,
                                 "dot_product").tree(50, 6).score_ah(2).build()

    idx, dis = searcher.search_batched(queries)
    idx_parallel, dis_parallel = searcher.search_batched_parallel(queries)
    self.assertAllClose(dis, dis_parallel)
    self.assertAllEqual(idx, idx_parallel)

  def test_reordering_shapes(self):
    np_dataset = np.random.rand(10000, 32)
    np_queries = np.random.rand(100, 32)
    dataset = tf.convert_to_tensor(np_dataset, dtype=tf.float32)
    queries = tf.convert_to_tensor(np_queries, dtype=tf.float32)
    k = 5

    searcher = scann_ops.builder(dataset, k,
                                 "dot_product").score_ah(2).reorder(25).build()

    indices, distances = searcher.search_batched(queries)
    self.assertAllEqual((100, k), distances.shape)
    self.assertAllEqual((100, k), indices.shape)

    indices, distances = searcher.search_batched(queries, final_num_neighbors=8)
    self.assertAllEqual((100, 8), distances.shape)
    self.assertAllEqual((100, 8), indices.shape)

    indices, distances = searcher.search_batched(
        queries, pre_reorder_num_neighbors=8)
    self.assertAllEqual((100, k), distances.shape)
    self.assertAllEqual((100, k), indices.shape)

    indices, distances = searcher.search_batched(
        queries, final_num_neighbors=6, pre_reorder_num_neighbors=8)
    self.assertAllEqual((100, 6), distances.shape)
    self.assertAllEqual((100, 6), indices.shape)

    indices, distances = searcher.search(queries[0])
    self.assertAllEqual((k,), distances.shape)
    self.assertAllEqual((k,), indices.shape)

    indices, distances = searcher.search(queries[0], final_num_neighbors=8)
    self.assertAllEqual((8,), distances.shape)
    self.assertAllEqual((8,), indices.shape)

    indices, distances = searcher.search(
        queries[0], pre_reorder_num_neighbors=8)
    self.assertAllEqual((k,), distances.shape)
    self.assertAllEqual((k,), indices.shape)

    indices, distances = searcher.search(
        queries[0], final_num_neighbors=6, pre_reorder_num_neighbors=8)
    self.assertAllEqual((6,), distances.shape)
    self.assertAllEqual((6,), indices.shape)


if __name__ == "__main__":
  tf.test.main()
