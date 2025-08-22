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

import os
import tempfile

import numpy as np

from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind
from absl.testing import absltest
from absl.testing import parameterized


class ScannTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Should suffice for most tests, but tests can use their own too.
    cls.default_dims = 20
    cls.ds = np.random.rand(1234, cls.default_dims).astype(np.float32)

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
        np.testing.assert_allclose(dis_new, dis, rtol=1e-6)

  def normalize(self, dataset):
    return dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

  def test_brute_force(self):

    def ground_truth(dataset, query, k):
      product = np.matmul(dataset, query)
      idx = np.argsort(product)[-k:]
      dis = np.sort(product)[-k:]
      return idx[::-1], dis[::-1]

    k = 10
    s = (
        scann_ops_pybind.builder(self.ds, k,
                                 "dot_product").score_brute_force().build())
    for _ in range(100):
      q = np.random.rand(self.default_dims).astype(np.float32)
      idx, dis = s.search(q)
      _, gt_dis = ground_truth(self.ds, q, k)

      # The following leads to flakiness due to non-associativity of FP addition
      # np.testing.assert_array_equal(idx, gt_idx)
      # Instead, ensure distances are close to ground truth distances, and that
      # distances of indices returned are close to Numpy-computed distances for
      # those indices.
      selected_distances = np.matmul(self.ds[idx], q)
      np.testing.assert_allclose(dis, selected_distances, rtol=1e-6)
      np.testing.assert_allclose(dis, gt_dis, rtol=1e-6)

  def test_batching(self):
    k = 10
    s = (
        scann_ops_pybind.builder(self.ds, k,
                                 "dot_product").score_brute_force().build())

    qs = np.random.rand(1000, self.default_dims).astype(np.float32)
    batch_idx, batch_dis = s.search_batched(qs)
    for i, q in enumerate(qs):
      _, dis = s.search(q)

      selected_distances = np.matmul(self.ds[batch_idx[i]], q)
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
      upper_tree=[True, False],
  )
  def test_tree_ah(self, dist, quantize_tree, reorder, soar, upper_tree):
    if soar and dist != "dot_product":
      return
    avq = None
    # To avoid excessive numbers of combinations, we test AVQ if we're using
    # dot product distance (a prereq) and bfloat16, since bfloat16 behavior
    # should be independent of whether AVQ is enabled or not.
    if reorder == scann_builder.ReorderType.BFLOAT16 and dist == "dot_product":
      avq = 2.5
    avq_or_nan = avq or float("nan")

    builder = (
        scann_ops_pybind.builder(self.ds, 10, dist).tree(
            50,
            3,
            min_partition_size=10,
            quantize_centroids=quantize_tree,
            avq=avq,
            soar_lambda=1.5 if soar[0] else None,
            overretrieve_factor=soar[1] if soar[0] else None,
        ).score_ah(2))
    if reorder:
      builder = builder.reorder(
          20, quantize=reorder, anisotropic_quantization_threshold=avq_or_nan)
    if upper_tree:
      builder = builder.upper_tree(
          num_leaves=8,
          num_leaves_to_search=5,
          avq=avq_or_nan,
          soar_lambda=1.5 if soar[0] else None,
          overretrieve_factor=soar[1] if soar[0] else None,
          scoring_mode=(reorder or scann_builder.ReorderType.INT8),
          anisotropic_quantization_threshold=avq_or_nan,
      )
    # Test absolute path serialization.
    self.verify_serialization(builder.build(), self.default_dims, 5)
    # Test relative path serialization.
    self.verify_serialization(
        builder.build(), self.default_dims, 5, relative_path=True)

  @parameterized.parameters(("squared_l2",), ("dot_product",))
  def test_pure_ah(self, dist):
    s = scann_ops_pybind.builder(self.ds, 10, dist).score_ah(2).build()
    self.verify_serialization(s, self.default_dims, 5)
    self.verify_serialization(s, self.default_dims, 5, relative_path=True)

  @parameterized.product(
      dist=["squared_l2", "dot_product"],
      quantize=[True, False],
      soar=[True, False],
  )
  def test_tree_brute_force(self, dist, quantize, soar):
    if soar and dist != "dot_product":
      return
    s = (
        scann_ops_pybind.builder(self.ds, 10, dist).tree(
            30, 3, soar_lambda=1.5
            if soar else None).score_brute_force(quantize).build())
    self.verify_serialization(s, self.default_dims, 5)
    self.verify_serialization(s, self.default_dims, 5, relative_path=True)

  def test_empty_partitions(self):
    # self.ds has 1234 points, so with 200 partitions, k-means fails to work
    # well and some partitions are empty; make sure this serializes properly.
    s = (
        scann_ops_pybind.builder(self.ds, 10, "dot_product").tree(
            200, 10, min_partition_size=5).score_ah(1).build())
    self.verify_serialization(s, self.default_dims, 500)
    self.verify_serialization(s, self.default_dims, 500, relative_path=True)

  @parameterized.product(
      dist=["squared_l2", "dot_product"],
      quant=[
          scann_builder.ReorderType.INT8,
          scann_builder.ReorderType.BFLOAT16,
      ],
  )
  def test_brute_force_quantized(self, dist, quant):
    s = (
        scann_ops_pybind.builder(self.ds, 10,
                                 dist).score_brute_force(quant).build())
    self.verify_serialization(s, self.default_dims, 5)
    self.verify_serialization(s, self.default_dims, 5, relative_path=True)

  def test_shapes(self):
    k = 10
    # first look at AH searcher with reordering
    s = (
        scann_ops_pybind.builder(self.ds, k,
                                 "dot_product").score_ah(2).reorder(30).build())
    q = np.random.rand(self.default_dims).astype(np.float32)
    self.assertLen(s.search(q)[0], k)
    self.assertLen(s.search(q, final_num_neighbors=20)[0], 20)
    self.assertLen(s.search(q, pre_reorder_num_neighbors=50)[0], k)
    self.assertLen(
        s.search(q, final_num_neighbors=15, pre_reorder_num_neighbors=30)[0],
        15)

    batch_q = np.random.rand(18, self.default_dims).astype(np.float32)
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
    s2 = scann_ops_pybind.builder(self.ds, k, "dot_product").score_ah(2).build()
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
    k = 10
    s = (
        scann_ops_pybind.builder(self.normalize(self.ds), k, "squared_l2").tree(
            30, 3, spherical=True).score_ah(2).build())
    self.verify_serialization(s, self.default_dims, 20)

  def test_truncation(self):
    reduction_dim = 12
    s = (
        scann_ops_pybind.builder(self.ds, 10, "dot_product").truncate(
            reduction_dim=reduction_dim).tree(30, 3).score_ah(2).build())
    self.assertRegex(s.config(), "TRUNCATE")
    with tempfile.TemporaryDirectory() as tmpdir:
      s.serialize(tmpdir, relative_path=True)
      hashed = np.load(os.path.join(tmpdir, "hashed_dataset.npy"))
      self.assertEqual(hashed.shape[1], reduction_dim // 2)
    self.verify_serialization(s, self.default_dims, 10)

  def test_pca(self):
    k = 10
    s = (
        scann_ops_pybind.builder(self.ds, k, "dot_product").pca(
            pca_significance_threshold=0.8,
            pca_truncation_threshold=1.0).tree(80, 10).score_ah(2).build())
    self.assertRegex(s.config(), "PCA")
    with tempfile.TemporaryDirectory() as tmpdir:
      s.serialize(tmpdir, relative_path=True)
      hashed = np.load(os.path.join(tmpdir, "hashed_dataset.npy"))
      self.assertLessEqual(hashed.shape[1], self.default_dims / 2)
    self.verify_serialization(s, self.default_dims, 10)


if __name__ == "__main__":
  absltest.main()
