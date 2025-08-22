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

"""Tests for isotonic_dykstra."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from sparse_soft_topk._src import isotonic_pav
from sparse_soft_topk._src import isotonic_dykstra

LS = (1e-3, 1e-2, 1e-1, 1.0, 1e1)


class IsotonicDystraMaskTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(LS))
  def test_compare_with_pav(self, l, p=2.0):
    """Comparing the output with projected gradient descent."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.rand(10))
    w = jnp.array(seed.rand(10))
    bisect_max_iter = 100
    out_dykstra = isotonic_dykstra.isotonic_dykstra_mask(y - w * l)
    out_pav = isotonic_pav.isotonic_mask_pav(
        y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
    self.assertSequenceAlmostEqual(out_dykstra, out_pav, places=4)

  def test_jvp(self, eps=1e-3):
    """Compare the jvp with finite differences."""
    seed = np.random.RandomState(0)
    n = 5
    y = jnp.array(seed.randn(n))
    v = jnp.array(seed.randn(n))
    jvp = (
        jax.jacobian(isotonic_dykstra.isotonic_dykstra_mask)(
            y
        )
        @ v
    )
    approx = (
        isotonic_dykstra.isotonic_dykstra_mask(
            y + eps * v
        )
        - isotonic_dykstra.isotonic_dykstra_mask(
            y
        )
    ) / eps
    self.assertSequenceAlmostEqual(jvp, approx, places=3)


class IsotonicDystraMagTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(LS))
  def test_compare_with_pav(self, l, p=2.0):
    """Comparing the output with projected gradient descent."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.rand(10))
    w = jnp.array(seed.rand(10))
    bisect_max_iter = 100
    out_dykstra = isotonic_dykstra.isotonic_dykstra_mag(y / (1 + w * l), w, l)
    out_pav = isotonic_pav.isotonic_mag_pav(
        y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
    self.assertSequenceAlmostEqual(out_dykstra, out_pav, places=4)


  @parameterized.parameters(itertools.product(LS))
  def test_jvp(self, l, eps=1e-3):
    """Compare the jvp with finite differences."""
    seed = np.random.RandomState(0)
    n = 10
    y = jnp.array(seed.randn(n))
    w = jnp.array(seed.rand(n))
    v = jnp.array(seed.randn(n))
    jvp = (
        jax.jacobian(isotonic_dykstra.isotonic_dykstra_mag)(
            y, w, l=l
        )
        @ v
    )
    approx = (
        isotonic_dykstra.isotonic_dykstra_mag(
            y + eps * v, w, l=l
        )
        - isotonic_dykstra.isotonic_dykstra_mag(
            y, w, l=l
        )
    ) / eps
    self.assertSequenceAlmostEqual(jvp, approx, places=3)

if __name__ == "__main__":
  absltest.main()
