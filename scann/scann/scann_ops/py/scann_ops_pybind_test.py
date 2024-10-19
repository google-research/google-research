# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
import tempfile

import numpy as np

from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind
from absl.testing import absltest
from absl.testing import parameterized


class ScannTest(parameterized.TestCase):

  def verify_serialization(self,
                           searcher,
                           n_dims,
                           n_queries,
                           relative_path=False):
    with tempfile.TemporaryDirectory() as tmpdir:
      searcher.serialize(tmpdir, relative_path=relative_path)
      queries = []
      indices = []
      distances = []
      for _ in range(n_queries):
        queries.append(np.random.rand(n_dims).astype(np.float32))
        idx_orig, dis_orig = searcher.search(queries[-1])
        indices.append(idx_orig)
        distances.append(dis_orig)

      s2 = scann_ops_pybind.load_searcher(tmpdir)
      for q, idx, dis in zip(queries, indices, distances):
        idx_new, dis_new = s2.search(q)
        np.testing.assert_array_equal(idx_new, idx)
        np.testing.assert_allclose(dis_new, dis, rtol=2e-7)

  def normalize(self, dataset):
    return dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

  def rand(self, *args, **kwargs):
    """Generate a normalized random dataset friendly to quantization."""
    return self.normalize(
        np.round(
            (np.random.rand(*args, **kwargs) - 0.5) * 256).astype(np.float32))

  def test_brute_force(self):

    def ground_truth(dataset, query, k):
      product = np.matmul(dataset, query)
      idx = np.argsort(product)[-k:]
      dis = np.sort(product)[-k:]
      return idx[::-1], dis[::-1]

    k = 10
    n_dims = 10
    n_points = 1000
    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    s = (
        scann_ops_pybind.builder(ds, k,
                                 "dot_product").score_brute_force().build())
    for _ in range(100):
      q = np.random.rand(n_dims).astype(np.float32)
      idx, dis = s.search(q)
      _, gt_dis = ground_truth(ds, q, 10)

      # The following leads to flakiness due to non-associativity of FP addition
      # np.testing.assert_array_equal(idx, gt_idx)
      # Instead, ensure distances are close to ground truth distances, and that
      # distances of indices returned are close to Numpy-computed distances for
      # those indices.
      selected_distances = np.matmul(ds[idx], q)
      np.testing.assert_allclose(dis, selected_distances, rtol=1e-6)
      np.testing.assert_allclose(dis, gt_dis, rtol=1e-6)

  def test_batching(self):
    k = 10
    n_dims = 10
    n_points = 1000
    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    s = (
        scann_ops_pybind.builder(ds, k,
                                 "dot_product").score_brute_force().build())

    qs = np.random.rand(n_points, n_dims).astype(np.float32)
    batch_idx, batch_dis = s.search_batched(qs)
    for i, q in enumerate(qs):
      _, dis = s.search(q)

      selected_distances = np.matmul(ds[batch_idx[i]], q)
      np.testing.assert_allclose(dis, selected_distances, rtol=1e-6)
      np.testing.assert_allclose(dis, batch_dis[i], rtol=1e-6)

  @parameterized.product(
      dist=["squared_l2", "dot_product"],
      quantize_tree=[True, False],
      reorder=[
          None,
          scann_builder.ReorderType.INT8,
          scann_builder.ReorderType.BFLOAT16,
          scann_builder.ReorderType.FLOAT32,
      ],
      soar=[(True, 2.0), (False, 2.0), (True, 1.2)],
  )
  def test_tree_ah(self, dist, quantize_tree, reorder, soar):
    if soar and dist != "dot_product":
      return
    avq = None
    # To avoid excessive numbers of combinations, we test AVQ if we're using
    # dot product distance (a prereq) and bfloat16, since bfloat16 behavior
    # should be independent of whether AVQ is enabled or not.
    if reorder == scann_builder.ReorderType.BFLOAT16 and dist == "dot_product":
      avq = 2.5

    n_dims = 50
    ds = np.random.rand(12345, n_dims).astype(np.float32)
    builder = (
        scann_ops_pybind.builder(ds, 10, dist).tree(
            300,
            30,
            min_partition_size=10,
            quantize_centroids=quantize_tree,
            avq=avq,
            soar_lambda=1.5 if soar[0] else None,
            overretrieve_factor=soar[1] if soar[0] else None,
        ).score_ah(2))
    if reorder:
      builder = builder.reorder(20, quantize=reorder)
    # Test absolute path serialization.
    self.verify_serialization(builder.build(), n_dims, 5)
    # Test relative path serialization.
    self.verify_serialization(builder.build(), n_dims, 5, relative_path=True)

  @parameterized.parameters(("squared_l2",), ("dot_product",))
  def test_pure_ah(self, dist):
    n_dims = 50
    ds = np.random.rand(12345, n_dims).astype(np.float32)
    s = scann_ops_pybind.builder(ds, 10, dist).score_ah(2).build()
    self.verify_serialization(s, n_dims, 5)
    self.verify_serialization(s, n_dims, 5, relative_path=True)

  @parameterized.product(
      dist=["squared_l2", "dot_product"],
      quantize=[True, False],
      soar=[True, False],
  )
  def test_tree_brute_force(self, dist, quantize, soar):
    if soar and dist != "dot_product":
      return
    n_dims = 100
    ds = np.random.rand(12345, n_dims).astype(np.float32)
    s = (
        scann_ops_pybind.builder(ds, 10, dist).tree(
            100, 10, soar_lambda=1.5
            if soar else None).score_brute_force(quantize).build())
    self.verify_serialization(s, n_dims, 5)
    self.verify_serialization(s, n_dims, 5, relative_path=True)

  def test_empty_partitions(self):
    n_dims = 100
    ds = np.random.rand(1234, n_dims).astype(np.float32)
    # with 1234 points and 200 partitions, k-means fails to work well and some
    # partitions are empty; make sure this serializes properly
    s = (
        scann_ops_pybind.builder(ds, 10, "dot_product").tree(
            200, 10, min_partition_size=5).score_ah(1).build())
    self.verify_serialization(s, n_dims, 500)
    self.verify_serialization(s, n_dims, 500, relative_path=True)

  @parameterized.product(
      dist=["squared_l2", "dot_product"],
      quant=[
          scann_builder.ReorderType.INT8,
          scann_builder.ReorderType.BFLOAT16,
      ],
  )
  def test_brute_force_quantized(self, dist, quant):
    n_dims = 100
    ds = np.random.rand(12345, n_dims).astype(np.float32)
    s = scann_ops_pybind.builder(ds, 10, dist).score_brute_force(quant).build()
    self.verify_serialization(s, n_dims, 5)
    self.verify_serialization(s, n_dims, 5, relative_path=True)

  def test_shapes(self):
    n_dims = 128
    k = 10
    ds = np.random.rand(1234, n_dims).astype(np.float32)
    # first look at AH searcher with reordering
    s = (
        scann_ops_pybind.builder(ds, k,
                                 "dot_product").score_ah(2).reorder(30).build())
    q = np.random.rand(n_dims).astype(np.float32)
    self.assertLen(s.search(q)[0], k)
    self.assertLen(s.search(q, final_num_neighbors=20)[0], 20)
    self.assertLen(s.search(q, pre_reorder_num_neighbors=50)[0], k)
    self.assertLen(
        s.search(q, final_num_neighbors=15, pre_reorder_num_neighbors=30)[0],
        15)

    batch_q = np.random.rand(18, n_dims).astype(np.float32)
    self.assertEqual(s.search_batched(batch_q)[0].shape, (18, k))
    self.assertEqual(
        s.search_batched(batch_q, final_num_neighbors=20)[0].shape, (18, 20))
    self.assertEqual(
        s.search_batched(batch_q, pre_reorder_num_neighbors=20)[0].shape,
        (18, k),
    )
    self.assertEqual(
        s.search_batched(
            batch_q, final_num_neighbors=20,
            pre_reorder_num_neighbors=40)[0].shape,
        (18, 20),
    )

    # now look at AH without reordering
    s2 = scann_ops_pybind.builder(ds, k, "dot_product").score_ah(2).build()
    self.assertLen(s2.search(q)[0], k)
    self.assertLen(s2.search(q, final_num_neighbors=20)[0], 20)

  def test_squared_l2(self):

    def ground_truth(dataset, query, k):
      squared_l2 = np.sum(np.square(dataset - query), axis=1)
      return np.argsort(squared_l2)[:k], np.sort(squared_l2)[:k]

    n_dims = 50
    k = 10
    ds = np.random.rand(2500, n_dims).astype(np.float32)
    qs = np.random.rand(500, n_dims).astype(np.float32)
    s = (
        scann_ops_pybind.builder(ds, k,
                                 "squared_l2").score_brute_force(False).build())
    idx, dis = s.search_batched(qs)

    for query, idx_row, dis_row in zip(qs, idx, dis):
      _, gt_dis = ground_truth(ds, query, k)
      np.testing.assert_allclose(dis_row, gt_dis, rtol=1e-5)
      selected_distances = np.sum(np.square(ds[idx_row] - query), axis=1)
      np.testing.assert_allclose(dis_row, selected_distances, rtol=1e-5)

  def test_parallel_batching(self):
    n_dims = 50
    k = 10
    ds = np.random.rand(12500, n_dims).astype(np.float32)
    qs = np.random.rand(2500, n_dims).astype(np.float32)
    s = (
        scann_ops_pybind.builder(ds, k,
                                 "squared_l2").tree(80, 10).score_ah(2).build())
    idx, dis = s.search_batched(qs)
    idx_parallel, dis_parallel = s.search_batched_parallel(qs)
    self.assertLess(np.mean(idx != idx_parallel), 1e-3)
    np.testing.assert_allclose(dis, dis_parallel, rtol=1e-5)

  # make sure spherical partitioning proto is valid and doesn't crash
  def test_spherical_kmeans(self):
    n_dims = 50
    k = 10
    ds = self.normalize(np.random.rand(12500, n_dims).astype(np.float32))
    s = (
        scann_ops_pybind.builder(ds, k, "squared_l2").tree(
            80, 10, spherical=True).score_ah(2).build())
    self.verify_serialization(s, n_dims, 20)

  def test_updates(self):
    k = 10
    n_dims = 10
    n_points = 1000
    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    docids = [str(i) for i in range(n_points)]
    s = (
        scann_ops_pybind.builder(
            ds, k, "dot_product").score_brute_force().build(docids=docids))

    # Test deletion.
    query = np.random.rand(1, n_dims).astype(np.float32)
    docids0, dist0 = s.search(query[0])
    s.delete(docids0[:5])
    docids1, dist1 = s.search(query[0])
    np.testing.assert_array_equal(docids0[5:], docids1[:5])
    np.testing.assert_array_almost_equal(dist0[5:], dist1[:5])
    with self.assertRaises(KeyError):
      s.delete(docids0[0])

    # Test insertion.
    s.upsert(
        [str(i + 1000) for i in range(5)],
        [ds[int(docids0[i])] for i in range(5)],
    )
    docids2, dist2 = s.search(query[0])
    np.testing.assert_array_equal(docids2[:5],
                                  [str(i + 1000) for i in range(5)])
    np.testing.assert_array_almost_equal(dist0, dist2)

    # Test update.
    s.upsert(docids0[5:], [np.zeros_like(query) for _ in range(5)])
    docids3, dist3 = s.search(query[0])
    self.assertFalse(np.equal(docids2[5:], docids3[5:]).any())
    self.assertFalse(np.equal(dist2[5:], dist3[5:]).any())

  def test_serialization_with_updates(self):
    k = 10
    n_dims = 10
    n_points = 10000
    query = np.random.rand(1, n_dims).astype(np.float32)
    # Test dynamic Tree-AH serialization.
    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    docids = [str(i) for i in range(n_points)]
    s1 = (
        scann_ops_pybind.builder(ds, k, "dot_product").tree(
            10, 1).score_ah(2).reorder(100).build(docids=docids))
    docids1, dist1 = s1.search(query[0])

    with tempfile.TemporaryDirectory() as tmpdir:
      # Remove everything.
      for i in range(n_points):
        s1.delete(str(i))
      s1.serialize(tmpdir)

      # Initialize from empty, serialized searcher.
      s2 = scann_ops_pybind.load_searcher(tmpdir)
      self.assertEmpty(s2.docids)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Add back everything, with offset of 1.
      for docid, vec in zip([str(i + 1) for i in range(n_points)], ds):
        s2.upsert(docid, vec)
      s2.serialize(tmpdir)

      s3 = scann_ops_pybind.load_searcher(tmpdir)
      self.assertNotIn("0", s3.docids)

      docids3, dist3 = s3.search(query[0])
      np.testing.assert_array_equal([int(i) for i in docids1],
                                    [(int(i) - 1) for i in docids3])
      np.testing.assert_array_almost_equal(dist1, dist3)

  def test_rebalance(self):
    k = 10
    n_dims = 10
    n_points = 10000

    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    docids = [str(i) for i in range(n_points)]
    s = (
        scann_ops_pybind.builder(ds[::2], k, "dot_product").tree(
            100, 20).score_ah(2).reorder(100).build(docids=docids[::2]))

    with tempfile.TemporaryDirectory() as tmpdir:
      s.serialize(tmpdir)
      p1 = open(tmpdir + "/serialized_partitioner.pb", "rb").read()

      for docid, vec in zip(docids[1::2], ds[1::2]):
        s.upsert(docid, vec)
      s.rebalance()
      s.serialize(tmpdir)
      p2 = open(tmpdir + "/serialized_partitioner.pb", "rb").read()

      # Make sure two partitioners are not identical.
      self.assertNotEqual(p1, p2)

  def test_autopilot(self):
    k = 10
    n_dims = 100
    n_points = 100000
    ds = np.random.rand(n_points, n_dims).astype(np.float32)
    docids = [str(i) for i in range(n_points)]
    s = (
        scann_ops_pybind.builder(
            ds[:1000, :], k,
            "dot_product").autopilot().build(docids=docids[:1000]))

    s.upsert(docids[1000:10000], ds[1000:10000, :])
    s.rebalance()
    config = s.config()
    # Start with bruteforce.
    self.assertRegex(config, "brute_force")

    # Phase change to TreeAh searcher.
    s.upsert(docids[10000:100000], ds[10000:100000, :])
    s.rebalance()
    config = s.config()
    self.assertRegex(config, "asymmetric_hash")
    self.assertRegex(config, "partitioning")

    # Make sure serialization works when autopilot is enabled.
    with tempfile.TemporaryDirectory() as tmpdir:
      s.serialize(tmpdir)
      s1 = scann_ops_pybind.load_searcher(tmpdir)
      self.assertEqual(s1.config(), config)

    # Make sure create_config() under autopilot also works.
    config2 = (
        scann_ops_pybind.builder(ds, k,
                                 "dot_product").autopilot().create_config())
    self.assertEqual(config2, config)

  def verify_accuracy(self, base, exp, gt, test, op, threshold=0.05):
    op(base)
    op(exp)
    op(gt)
    base_idx = np.array(base.search_batched(test)[0])
    exp_idx = np.array(exp.search_batched(test)[0])
    gt_idx = np.array(gt.search_batched(test)[0])
    base_accuracy = np.mean([
        len(np.intersect1d(base_idx[i], gt_idx[i])) / len(gt_idx[0])
        for i in range(gt_idx.shape[0])
    ])
    exp_accuracy = np.mean([
        len(np.intersect1d(exp_idx[i], gt_idx[i])) / len(gt_idx[0])
        for i in range(gt_idx.shape[0])
    ])
    base.rebalance()
    base_idx = np.array(base.search_batched(test)[0])
    after_accuracy = np.mean([
        len(np.intersect1d(base_idx[i], gt_idx[i])) / len(gt_idx[0])
        for i in range(gt_idx.shape[0])
    ])

    self.assertGreater(exp_accuracy, base_accuracy - threshold)
    self.assertGreater(exp_accuracy, after_accuracy - threshold)
    print(base_accuracy, exp_accuracy, after_accuracy)

  def test_online_training(self):
    dim = 20
    intrinsic_dim = 2
    n = 20000
    k = 2
    np.random.seed(42)
    proj = self.normalize(np.random.rand(intrinsic_dim, dim))
    train = self.normalize(
        (np.random.rand(n, intrinsic_dim) @ proj).astype(np.float32))
    test = self.normalize(
        (np.random.rand(1000, intrinsic_dim) @ proj).astype(np.float32))
    docids = [str(i) for i in range(n)]

    s0 = (
        scann_ops_pybind.builder(
            train[::2], k, "dot_product").tree(10, 4).score_ah(
                2, anisotropic_quantization_threshold=0.2).reorder(100).build(
                    docids=docids[::2]))
    s = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").tree(
            10, 4, incremental_threshold=0.2).score_ah(
                2, anisotropic_quantization_threshold=0.2).reorder(100).build(
                    docids=docids[::2]))
    bf = (
        scann_ops_pybind.builder(
            train[::2], k,
            "dot_product").score_brute_force().build(docids=docids[::2]))

    # Delete only.
    self.verify_accuracy(s0, s, bf, test, lambda x: x.delete(docids[::4]))
    # Update
    self.verify_accuracy(s0, s, bf, test,
                         lambda x: x.upsert(docids[::4], train[1::4]))
    # Add only
    self.verify_accuracy(s0, s, bf, test,
                         lambda x: x.upsert(docids[1::2], train[1::2]))

  def test_online_incremental(self):
    # Note for future tests:
    # 1. We need a relatively large dataset to invoke online incremental
    #    behavior. So feel free to ignore if the test is running slow.
    # 2. This test dataset is created by projecting a low-instrinsic dim
    #    dim dataset into high dim space, thus sensitive to random seed, n and
    #    dim. Feel free to increase threshold if accuracy check fails.
    dim = 128
    intrinsic_dim = 24
    n = 100000
    k = 10
    np.random.seed(42)
    proj = self.rand(intrinsic_dim, dim)
    train = self.normalize(
        (self.rand(n, intrinsic_dim) @ proj).astype(np.float32))
    test = self.normalize(
        (self.rand(1000, intrinsic_dim) @ proj).astype(np.float32))
    docids = [str(i) for i in range(n)]

    s0 = (
        scann_ops_pybind.builder(
            train[::2], k, "dot_product").autopilot().build(docids=docids[::2]))
    s = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL,
            quantize=scann_builder.ReorderType.BFLOAT16,
        ).build(docids=docids[::2]))
    bf = (
        scann_ops_pybind.builder(
            train[::2], k,
            "dot_product").score_brute_force().build(docids=docids[::2]))

    # Delete only.
    self.verify_accuracy(s0, s, bf, test, lambda x: x.delete(docids[::4]))
    # Update
    self.verify_accuracy(s0, s, bf, test,
                         lambda x: x.upsert(docids[::4], train[1::4]))
    # Add only
    self.verify_accuracy(s0, s, bf, test, lambda x: x.upsert(docids, train))
    with tempfile.TemporaryDirectory() as tmpdir:
      s.rebalance()
      s.serialize(tmpdir)
      self.assertTrue(
          os.path.isfile(os.path.join(tmpdir, "bfloat16_dataset.npy")))
      self.assertFalse(os.path.isfile(os.path.join(tmpdir, "dataset.npy")))
      s1 = scann_ops_pybind.load_searcher(tmpdir)
      s1.rebalance()
      self.verify_accuracy(s, s1, bf, test, lambda x: x)

  def test_online_incremental_batched(self):
    dim = 128
    intrinsic_dim = 24
    n = 100000
    k = 10
    np.random.seed(42)
    proj = self.rand(intrinsic_dim, dim)
    train = self.normalize(
        (self.rand(n, intrinsic_dim) @ proj).astype(np.float32))
    test = self.normalize(
        (self.rand(1000, intrinsic_dim) @ proj).astype(np.float32))
    docids = [str(i) for i in range(n)]

    s0 = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL).build(
                docids=docids[::2]))
    s = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL).build(
                docids=docids[::2]))
    bf = (
        scann_ops_pybind.builder(
            train, k, "dot_product").score_brute_force().build(docids=docids))

    # Add only
    s0.upsert(docids, train)
    s.upsert(docids, train, batch_size=256)
    # We want to make sure batched and non-batched upsertion has virtually
    # no quality difference.
    self.verify_accuracy(s0, s, bf, test, lambda x: None, threshold=0.01)

  def test_online_incremental_quantized(self):
    dim = 128
    n = 100000
    k = 10
    np.random.seed(42)
    train = self.rand(n, dim)
    test = self.rand(1000, dim)
    docids = [str(i) for i in range(n)]

    s0 = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL,
            quantize=scann_builder.ReorderType.FLOAT32,
        ).build(docids=docids[::2]))
    s1 = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL,
            quantize=scann_builder.ReorderType.BFLOAT16,
        ).build(docids=docids[::2]))
    s2 = (
        scann_ops_pybind.builder(train[::2], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL,
            quantize=scann_builder.ReorderType.INT8,
        ).build(docids=docids[::2]))
    bf = (
        scann_ops_pybind.builder(
            train, k, "dot_product").score_brute_force().build(docids=docids))

    # Add only
    s0.upsert(docids, train, batch_size=256)
    s1.upsert(docids, train, batch_size=256)
    s2.upsert(docids, train, batch_size=256)
    # We want to make sure quality difference between quantized searchers are
    # small.
    self.verify_accuracy(s0, s1, bf, test, lambda x: None)
    self.verify_accuracy(s0, s2, bf, test, lambda x: None)

  def test_online_incremental_phase_change(self):
    dim = 128
    n = 100000
    k = 10
    np.random.seed(42)
    train = self.rand(n, dim)
    test = self.rand(1000, dim)
    docids = [str(i) for i in range(n)]

    s0 = (
        scann_ops_pybind.builder(train[:0], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL).build(
                docids=docids[:0]))
    s = (
        scann_ops_pybind.builder(train[:0], k, "dot_product").autopilot(
            mode=scann_builder.IncrementalMode.ONLINE_INCREMENTAL).build(
                docids=docids[:0]))
    bf = (
        scann_ops_pybind.builder(
            train[:0], k,
            "dot_product").score_brute_force().build(docids=docids[:0]))

    bf.upsert(docids[:1000], train[:1000])
    self.verify_accuracy(s0, s, bf, test,
                         lambda x: x.upsert(docids[:1000], train[:1000]))
    # Start with bruteforce.
    self.assertRegex(s.config(), "brute_force")

    # Add only
    self.verify_accuracy(s0, s, bf, test,
                         lambda x: x.upsert(docids, train, batch_size=256))
    self.assertRegex(s.config(), "partitioning")

    # Coming back to bruteforce.
    self.verify_accuracy(s0, s, bf, test, lambda x: x.delete(docids[1000:]))
    self.assertRegex(s.config(), "brute_force")


if __name__ == "__main__":
  absltest.main()
