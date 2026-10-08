# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Unit tests for dp_clip."""

from absl.testing import absltest
import numpy as np
import scipy.sparse as sp

from dp_clip import dp_clip_lib
from dp_clip import experiment_lib


class DpClipLibTest(absltest.TestCase):

  def test_get_vector_mse(self):
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 4])
    mse = experiment_lib.get_vector_mse(v1, v2)
    self.assertAlmostEqual(mse, 1 / 3)

  def test_compute_optimal_clipping_bound(self):
    # Simple dataset
    data = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    bound = dp_clip_lib.compute_optimal_clipping_bound(
        data, rho=1.0, num_trials=1
    )
    self.assertGreaterEqual(bound[0], 1.0)

  def test_unbounded_quantile_no_shift(self):
    # With high rho (negligible noise), median of all 10.0s should be ~10.0.
    data = np.full(1000, 10.0)
    estimates = dp_clip_lib.unbounded_quantile(
        data, beta=1.001, quantile=0.5, rho=1e6, num_trials=5
    )
    for est in estimates:
      self.assertAlmostEqual(est, 10.0, delta=0.05)

  def test_optimize_rc_norm_at_one(self):
    # Vectors with norm exactly 1.0 suffer zero clipping bias at C=1.0 and
    # should not cause optimize_rc to overshoot 1.0 when rho is high.
    data = sp.csr_matrix(np.eye(100))
    bounds = dp_clip_lib.optimize_rc(
        data,
        beta=1.001,
        rho_sum=0.1,
        rho_clip=1e6,
        num_trials=5,
    )
    for b in bounds:
      self.assertAlmostEqual(b, 1.0, delta=1e-5)

  def test_optimize_rc_nonnegative_data_toggle(self):
    data = sp.csr_matrix(np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]))
    np.random.seed(42)
    bounds_nonneg = dp_clip_lib.optimize_rc(
        data,
        beta=1.001,
        rho_sum=0.1,
        rho_clip=0.1,
        num_trials=5,
        nonnegative_data=True,
    )
    np.random.seed(42)
    bounds_general = dp_clip_lib.optimize_rc(
        data,
        beta=1.001,
        rho_sum=0.1,
        rho_clip=0.4,
        num_trials=5,
        nonnegative_data=False,
    )
    np.testing.assert_allclose(bounds_nonneg, bounds_general)

  def test_precompute_rc(self):
    # Two aligned vectors of norms 1.5 and 3.0 along e_1, with beta=2.0:
    # bins = [floor(log2(1.5)), floor(log2(3.0))] = [0, 1], max_bin = 1.
    # rc_vals for k = 0, 1: k=0 -> 2, k=1 -> 1.
    data = sp.csr_matrix(np.array([[1.5, 0.0], [3.0, 0.0]]))
    rc_vals = dp_clip_lib.precompute_rc(data, beta=2.0)
    np.testing.assert_allclose(rc_vals, np.array([2.0, 1.0]))

  def test_hly_uq(self):
    norms = np.arange(1.0, 1001.0)
    bounds = dp_clip_lib.hly_uq(
        norms,
        d=50,
        beta=1.001,
        rho_sum=1.0,
        rho_clip=1e6,
        num_trials=5,
    )
    # k* = sqrt(2 * 50 / 1.0) = 10 -> 990th norm (~990.0)
    for b in bounds:
      self.assertAlmostEqual(b, 990.0, delta=5.0)


if __name__ == "__main__":
  absltest.main()

