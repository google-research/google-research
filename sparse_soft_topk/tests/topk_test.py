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

"""Tests for isotonic_pav."""

from absl.testing import absltest
from absl.testing import parameterized

import itertools
import numpy as np
import jax
import jax.numpy as jnp

from sparse_soft_topk import hard_topk_mag
from sparse_soft_topk import hard_topk_mask
from sparse_soft_topk import sparse_soft_topk_mask_pav
from sparse_soft_topk import sparse_soft_topk_mag_pav
from sparse_soft_topk import sparse_soft_topk_mask_dykstra
from sparse_soft_topk import sparse_soft_topk_mag_dykstra

PS = (4 / 3, 2.0, 3 / 2, 2, 3.0)


class SparseSoftTopkTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(PS))
  def test_sparsity_topk_mask_pav(self, p, l=1e-2):
    """Checks the sparsity of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_pav = sparse_soft_topk_mask_pav(y, k, l=l, p=p)
    self.assertSequenceAlmostEqual(
        y.argsort()[::-1][:k].sort(), out_top_k_pav.argsort()[::-1][:k].sort()
    )
    self.assertAlmostEqual(len(jnp.where(out_top_k_pav > 0)[0]), k)

  @parameterized.parameters(itertools.product(PS))
  def test_multi_dimentional_input_pav_mask(
      self, p, l=1e-1, n_features=5, n_batches=3
  ):
    """Checks the vmaping."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n_batches, n_features))
    k = 2
    z = sparse_soft_topk_mask_pav(y, k, l=l, p=p)
    for i in range(n_batches):
      self.assertSequenceAlmostEqual(
          z[i], sparse_soft_topk_mask_pav(y[i], k, l=l, p=p), places=6
      )

  @parameterized.parameters(itertools.product(PS))
  def test_multi_dimentional_input_pav_mag(
      self, p, l=1e-1, n_features=5, n_batches=3
  ):
    """Checks the vmaping."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n_batches, n_features))
    k = 2
    z = sparse_soft_topk_mag_pav(y, k, l=l, p=p)
    for i in range(n_batches):
      self.assertSequenceAlmostEqual(
          z[i], sparse_soft_topk_mag_pav(y[i], k, l=l, p=p), places=6
      )

  def test_sparsity_topk_mask_dykstra(self, l=1e-2):
    """Checks the sparsity of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_dykstra = sparse_soft_topk_mask_dykstra(y, k, l=l)
    self.assertAlmostEqual(
        y.argsort()[::-1][:k].sort().sum(),
        out_top_k_dykstra.argsort()[::-1][:k].sort().sum(),
    )
    self.assertAlmostEqual(len(jnp.where(out_top_k_dykstra > 0)[0]), k)

  @parameterized.parameters(itertools.product(PS))
  def test_sparsity_topk_mag_pav(self, p, l=1e-3):
    """Checks the sparsity of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_pav = sparse_soft_topk_mag_pav(y, k, l=l, p=p)
    self.assertAlmostEqual(
        len(jnp.where(jnp.absolute(out_top_k_pav) > 0)[0]), k
    )

  def test_sparsity_topk_mag_dykstra(self, l=1e-3):
    """Checks the sparsity of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_dykstra = sparse_soft_topk_mag_dykstra(y, k, l=l)
    self.assertAlmostEqual(
        len(jnp.where(jnp.absolute(out_top_k_dykstra) > 0)[0]), k
    )

  @parameterized.parameters(itertools.product(PS))
  def test_magnitude_topk_mag_pav(self, p, l=1e-3):
    """Checks the magnitude of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_pav = sparse_soft_topk_mag_pav(y, k, l=l, p=p)
    true_top_k_mag = jnp.where(
        jnp.absolute(y) > jnp.absolute(y).sort()[::-1][k], y, 0
    )
    self.assertAlmostEqual(
        ((out_top_k_pav - true_top_k_mag) ** 2).mean(), 0.0, places=3
    )

  def test_magnitude_topk_mag(self, l=1e-3):
    """Checks the magnitude of the topk."""
    n = int(1e3)
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 10
    out_top_k_dykstra = sparse_soft_topk_mag_dykstra(y, k, l=l)
    true_top_k_mag = jnp.where(
        jnp.absolute(y) > jnp.absolute(y).sort()[::-1][k], y, 0
    )
    self.assertAlmostEqual(
        ((out_top_k_dykstra - true_top_k_mag) ** 2).mean(), 0.0, places=3
    )


class HardTopkTest(parameterized.TestCase):

  def test_hard_topk_mask_and_mag(self):
    """Checks the sparsity of the topk."""
    n = 20
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n))
    k = 3
    out_top_k_mask = hard_topk_mask(y, k)
    self.assertSequenceAlmostEqual(
        y.argsort()[::-1][:k].sort(), out_top_k_mask.argsort()[::-1][:k].sort()
    )
    self.assertAlmostEqual(len(jnp.where(out_top_k_mask > 0)[0]), k)
    out_top_k_mag = hard_topk_mag(y, k)
    self.assertSequenceAlmostEqual(
        y * hard_topk_mask(jnp.absolute(y), k), out_top_k_mag
    )


if __name__ == "__main__":
  absltest.main()
