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

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

# Avoid an XLA:CPU thread-pool starvation deadlock on 2-core Forge runners
# (NPROC=2): ThunkExecutor can schedule up to kMaxWorkers-1 = 3 tasks onto the
# 2-thread XLAEigen pool while jaxlib's ParallelBatchMap (lapack_kernels.cc)
# schedules sub-batches onto the same XLAEigen pool and blocks on an
# absl::BlockingCounter when num_threads > 1.
os.environ.setdefault("PJRT_NPROC", "1")

import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from f_divergence_tests.divergence import kernel as kernel_module  # pylint: disable=g-import-not-at-top
from f_divergence_tests.divergence import likelihood_ratio  # pylint: disable=g-import-not-at-top
from f_divergence_tests.hypothesis_test import fuse  # pylint: disable=g-import-not-at-top

_RUN_LARGE_MONTE_CARLO = flags.DEFINE_bool(
    "run_large_monte_carlo",
    False,
    "Whether to run the large Monte-Carlo Type-I error and power tests.",
)


class _NanOnTargetLambdaDivergence(likelihood_ratio.DrMMDSymmetric):
  """Test-only divergence that returns NaN when `lmbda == nan_lmbda`."""

  def __init__(self, lmbda, nan_lmbda = 0.01):
    super().__init__(lmbda=lmbda)
    self._nan_lmbda = nan_lmbda

  def directional(self, k_xx, k_yy, k_xy):
    val = super().directional(k_xx, k_yy, k_xy)
    return jnp.where(jnp.isclose(self.lmbda, self._nan_lmbda), jnp.nan, val)


class _NanQueryRatioOnTargetLambdaDivergence(likelihood_ratio.DrMMDSymmetric):
  """Test-only divergence whose query ratio has a NaN when `lmbda == nan_lmbda`."""

  symmetric = False

  def __init__(self, lmbda, nan_lmbda = 0.01):
    super().__init__(lmbda=lmbda)
    self._nan_lmbda = nan_lmbda

  def likelihood_ratio_yx_on_z(self, k_zx, k_zy, k_xx, cho_factor_k_xx, k_xy):
    ratios = super().likelihood_ratio_yx_on_z(
        k_zx, k_zy, k_xx, cho_factor_k_xx, k_xy
    )
    nan_first = ratios.at[0].set(jnp.nan)
    return jnp.where(
        jnp.isclose(self.lmbda, self._nan_lmbda), nan_first, ratios
    )


class _AlwaysNanDivergence(likelihood_ratio.DrMMDSymmetric):
  """Test-only divergence that always returns NaN."""

  symmetric = False

  def directional(self, k_xx, k_yy, k_xy):
    del k_xx, k_yy, k_xy
    return jnp.array(jnp.nan, dtype=jnp.float32)


def _spearman_rho(x, y):
  """Computes Spearman rank correlation between 1D arrays without ties."""
  rx = np.argsort(np.argsort(x)).astype(np.float64)
  ry = np.argsort(np.argsort(y)).astype(np.float64)
  return float(np.corrcoef(rx, ry)[0, 1])


class _VectorizedDummyDivergence(kernel_module.KernelDivergence):
  """Test-only divergence exercising the `has_vectorized_kernel=True` path."""

  def __init__(self):
    super().__init__(has_vectorized_kernel=True)

  def __call__(self, kernel_matrix, num_permutations, key):
    del key
    diag_sum = jnp.trace(kernel_matrix)
    mean_val = jnp.mean(kernel_matrix) + diag_sum
    return jnp.full((num_permutations + 1,), mean_val, dtype=jnp.float32)


class FuseTest(parameterized.TestCase):

  def _make_standard_fuse_test(
      self,
      *,
      significance = 0.1,
      num_permutations = 40,
      min_memory_permutations = True,
  ):
    return fuse.FuseTest(
        divergence_class=likelihood_ratio.DrMMDSymmetric,
        kernels=[
            kernel_module.Kernel.GAUSSIAN,
            kernel_module.Kernel.LAPLACE,
        ],
        significance=significance,
        number_bandwidths=3,
        num_permutations=num_permutations,
        min_memory_kernel=False,
        min_memory_permutations=min_memory_permutations,
        batch_size_permutations=20,
        divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
    )

  def test_h0_type1_error_and_deterministic_rejection_pattern(self):
    # Evaluated across 20 fixed seeds (seeds 0..19, n=60, d=2, B=40, alpha=0.1)
    # with DrMMDSymmetric:
    # Observed p-values for seeds 0..9:
    #   [0.6098, 0.2439, 0.3902, 0.2683, 0.3902,
    #    0.6829, 0.2195, 0.9024, 0.3902, 0.1707]
    # First 10 seeds (0..9) have 0 rejections (<= 2 headroom); across all 20
    # seeds (0..19) there are <= 4 rejections at alpha=0.1.
    test = self._make_standard_fuse_test(significance=0.1, num_permutations=40)
    num_permutations = 40
    min_p_val = 1.0 / (num_permutations + 1)

    rejections_20 = []
    p_vals_20 = []
    for seed in range(20):
      rng = np.random.default_rng(1000 + seed)
      x = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
      y = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
      res = test.run(x, y, key_seed=seed)
      p_val = float(res.p_val)
      rejected = bool(res.result_test)
      p_vals_20.append(p_val)
      rejections_20.append(rejected)
      self.assertGreaterEqual(p_val, min_p_val - 1e-6)
      self.assertLessEqual(p_val, 1.0 + 1e-6)

    rejections_10 = rejections_20[:10]
    self.assertLessEqual(
        sum(rejections_10),
        2,
        msg=f"rejections_10={rejections_10}, p_vals={p_vals_20[:10]}",
    )
    self.assertLessEqual(
        sum(rejections_20),
        4,
        msg=f"rejections_20={rejections_20}, p_vals={p_vals_20}",
    )
    # Assert exact deterministic rejection pattern for the 10 fixed seeds.
    expected_pattern_10 = [False] * 10
    self.assertEqual(
        rejections_10,
        expected_pattern_10,
        msg=f"rejections_10={rejections_10}, p_vals={p_vals_20[:10]}",
    )

  def test_h1_mean_shift_rejects_all_fixed_seeds(self):
    # Evaluated across 20 fixed seeds (seeds 0..19, n=60, d=2, shift=1.0, B=40)
    # with DrMMDSymmetric: all 20 seeds reject at significance=0.05 with
    # minimum possible permutation p_val == 1/(B+1) ~= 0.0244.
    test = self._make_standard_fuse_test(significance=0.05, num_permutations=40)
    for seed in range(3):
      rng = np.random.default_rng(2000 + seed)
      x = jnp.asarray(rng.normal(loc=0.0, size=(60, 2)), dtype=jnp.float32)
      y = jnp.asarray(rng.normal(loc=1.0, size=(60, 2)), dtype=jnp.float32)
      res = test.run(x, y, key_seed=seed)
      self.assertTrue(
          bool(res.result_test),
          msg=f"seed={seed}, p_val={float(res.p_val)}",
      )
      self.assertLessEqual(float(res.p_val), 0.05)

  def test_masking_drops_nan_lambda_rows_and_preserves_finite_p_value(self):
    test = fuse.FuseTest(
        divergence_class=_NanOnTargetLambdaDivergence,
        divergence_params={"nan_lmbda": 0.01},
        kernels=[
            kernel_module.Kernel.GAUSSIAN,
            kernel_module.Kernel.LAPLACE,
        ],
        significance=0.05,
        number_bandwidths=2,
        num_permutations=20,
        min_memory_kernel=False,
        min_memory_permutations=True,
        batch_size_permutations=10,
        divergence_sweep_hyperparams={"lmbda": [0.01, 0.1, 1.0]},
    )
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(size=(50, 2)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(loc=1.0, size=(50, 2)), dtype=jnp.float32)

    mock_get_kernel_matrix = self.enter_context(
        mock.patch.object(
            kernel_module,
            "get_kernel_matrix",
            autospec=True,
            spec_set=True,
            side_effect=kernel_module.get_kernel_matrix,
        )
    )
    res = test.run(x, y, key_seed=7)

    rows_per_lambda = 2 * 2  # 2 kernels * 2 bandwidths
    self.assertEqual(mock_get_kernel_matrix.call_count, rows_per_lambda)
    self.assertEqual(int(res.additional_info["num_hyperparams"]), 12)
    self.assertEqual(
        int(res.additional_info["num_valid_hyperparams"]),
        12 - rows_per_lambda,
    )
    for norm_key in ("l1", "l2"):
      bw_arr = np.asarray(res.additional_info["bw"][norm_key])
      self.assertEqual(bw_arr.shape, (2,))
      self.assertNotAlmostEqual(float(bw_arr[0]), float(bw_arr[1]))
    self.assertTrue(np.isfinite(float(res.p_val)))
    self.assertTrue(np.all(np.isfinite(np.asarray(res.all_statistics))))
    self.assertTrue(bool(res.result_test))

  def test_all_rows_invalid_returns_non_rejection_and_p_val_one(self):
    test = fuse.FuseTest(
        divergence_class=_AlwaysNanDivergence,
        kernels=[kernel_module.Kernel.GAUSSIAN],
        significance=0.05,
        number_bandwidths=2,
        num_permutations=10,
        min_memory_kernel=False,
        min_memory_permutations=False,
        divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
    )
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(30, 2)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(30, 2)), dtype=jnp.float32)

    mock_get_distances = self.enter_context(
        mock.patch.object(
            kernel_module,
            "get_distances",
            autospec=True,
            spec_set=True,
            side_effect=kernel_module.get_distances,
        )
    )
    res = test.run(x, y, key_seed=0)

    self.assertEqual(mock_get_distances.call_count, 1)
    self.assertEqual(int(res.additional_info["num_valid_hyperparams"]), 0)
    self.assertEqual(int(res.additional_info["num_hyperparams"]), 4)
    self.assertEqual(float(res.p_val), 1.0)
    self.assertFalse(bool(res.result_test))

  def test_no_tracer_leak_and_consistent_bw_shapes_across_runs(self):
    test = fuse.FuseTest(
        divergence_class=likelihood_ratio.DrMMDSymmetric,
        kernels=[
            kernel_module.Kernel.GAUSSIAN,
            kernel_module.Kernel.LAPLACE,
        ],
        significance=0.1,
        number_bandwidths=2,
        num_permutations=20,
        min_memory_kernel=False,
        min_memory_permutations=False,
        divergence_sweep_hyperparams={"lmbda": [0.1]},
    )
    rng = np.random.default_rng(123)
    x1 = jnp.asarray(rng.normal(size=(40, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.normal(size=(40, 2)), dtype=jnp.float32)
    x2 = jnp.asarray(rng.normal(size=(50, 2)), dtype=jnp.float32)
    y2 = jnp.asarray(rng.normal(size=(50, 2)), dtype=jnp.float32)

    with jax.checking_leaks():
      res1 = test.run(x1, y1, key_seed=0)
      res2 = test.run(x2, y2, key_seed=1)

    self.assertFalse(hasattr(test, "bandwidths"))
    self.assertCountEqual(res1.additional_info["bw"].keys(), ["l1", "l2"])
    self.assertCountEqual(res2.additional_info["bw"].keys(), ["l1", "l2"])
    for norm_key in ("l1", "l2"):
      self.assertEqual(res1.additional_info["bw"][norm_key].shape, (2,))
      self.assertEqual(res2.additional_info["bw"][norm_key].shape, (2,))
      self.assertTrue(
          np.all(np.isfinite(np.asarray(res1.additional_info["bw"][norm_key])))
      )
      self.assertTrue(
          np.all(np.isfinite(np.asarray(res2.additional_info["bw"][norm_key])))
      )

  def test_fixed_bandwidths_and_vectorized_kernel_branch(self):
    test = fuse.FuseTest(
        divergence_class=_VectorizedDummyDivergence,
        kernels=kernel_module.Kernel.GAUSSIAN,
        significance=0.05,
        number_bandwidths=None,
        bandwidths=[0.5, 1.5],
        num_permutations=10,
        min_memory_kernel=True,
        min_memory_permutations=False,
        divergence_sweep_hyperparams=None,
    )
    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.normal(size=(20, 2)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(20, 2)), dtype=jnp.float32)

    with jax.checking_leaks():
      res = test.run(x, y, key_seed=0)

    self.assertEqual(int(res.additional_info["num_hyperparams"]), 2)
    self.assertEqual(int(res.additional_info["num_valid_hyperparams"]), 2)
    np.testing.assert_allclose(
        res.additional_info["bw"]["l2"],
        np.array([0.5, 1.5], dtype=np.float32),
    )

    n_samples = x.shape[0]
    pooled = jnp.concatenate((x, y)).astype(jnp.float32)
    pairwise = kernel_module.get_distances(
        pooled, pooled, norm=kernel_module.Norm.L2.value, min_memory=True
    ).astype(jnp.float32)
    expected_kernel_norms = jnp.stack([
        jnp.sqrt(
            jnp.sum(
                kernel_module.get_kernel_matrix(
                    pairwise,
                    kernel=kernel_module.Kernel.GAUSSIAN,
                    bandwidth=bw,
                ).astype(jnp.float32)
                ** 2
            )
            / (n_samples * (n_samples - 1))
        )
        for bw in (0.5, 1.5)
    ])
    lambda_reg = jnp.sqrt(float(n_samples * (n_samples - 1.0)))
    stats_matrix = res.additional_info["statistics_matrix"]
    expected_fused = (
        jax.scipy.special.logsumexp(
            lambda_reg * stats_matrix / expected_kernel_norms[:, None],
            axis=0,
            b=0.5,
        )
        / lambda_reg
    )
    np.testing.assert_allclose(
        np.asarray(res.all_statistics),
        np.asarray(expected_fused),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_validation_errors(self):
    with self.assertRaisesRegex(ValueError, "Only sweeping over one"):
      fuse.FuseTest(
          divergence_class=likelihood_ratio.SymmetricHockeyStick,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=2,
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=False,
          divergence_sweep_hyperparams={"lmbda": [0.1], "order": [1.0]},
      )

    with self.assertRaisesRegex(
        ValueError, "Only specify number_bandwiths or bandwidths"
    ):
      fuse.FuseTest(
          divergence_class=likelihood_ratio.SymmetricHockeyStick,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=2,
          bandwidths=[1.0, 2.0],
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=False,
      )

    with self.assertRaisesRegex(
        ValueError, "Only specify number_bandwiths or bandwidths"
    ):
      fuse.FuseTest(
          divergence_class=likelihood_ratio.SymmetricHockeyStick,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=None,
          bandwidths=None,
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=False,
      )

    test = self._make_standard_fuse_test(num_permutations=10)
    with self.assertRaisesRegex(ValueError, "2D arrays"):
      test.run(
          jnp.zeros((20,), dtype=jnp.float32),
          jnp.zeros((20, 2), dtype=jnp.float32),
          key_seed=0,
      )
    with self.assertRaisesRegex(ValueError, "same number of samples"):
      test.run(
          jnp.zeros((20, 2), dtype=jnp.float32),
          jnp.zeros((25, 2), dtype=jnp.float32),
          key_seed=0,
      )
    with self.assertRaisesRegex(ValueError, "at least 2 samples"):
      test.run(
          jnp.zeros((1, 2), dtype=jnp.float32),
          jnp.zeros((1, 2), dtype=jnp.float32),
          key_seed=0,
      )
    with self.assertRaisesRegex(ValueError, "same feature dimension"):
      test.run(
          jnp.zeros((20, 2), dtype=jnp.float32),
          jnp.zeros((20, 3), dtype=jnp.float32),
          key_seed=0,
      )

  def test_witness_scores_1d_gaussian_shift_and_orientation(self):
    # For numerator ~ N(1, 1) and denominator ~ N(0, 1), the true density ratio
    # dP_num/dP_den(z) = exp(z - 0.5) is strictly increasing in z.
    # Evaluated across 20 fixed seeds (seeds 0..19, n=200, q=64 on [-1, 2]):
    # forward Spearman rho > 0.9, and swapping numerator/denominator reverses
    # the ranking with Spearman rho < -0.9 across all 20 seeds.
    test = fuse.FuseTest(
        divergence_class=likelihood_ratio.SymmetricHockeyStick,
        kernels=[
            kernel_module.Kernel.GAUSSIAN,
            kernel_module.Kernel.LAPLACE,
        ],
        significance=0.05,
        number_bandwidths=None,
        bandwidths=[0.5, 1.0, 2.0],
        num_permutations=20,
        min_memory_kernel=False,
        min_memory_permutations=True,
        divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
    )
    z = np.linspace(-1.0, 2.0, 64, dtype=np.float32)[:, None]
    z_jax = jnp.asarray(z)
    true_ratio = np.exp(z[:, 0] - 0.5)

    forward_rhos = []
    reverse_rhos = []
    for seed in range(20):
      rng = np.random.default_rng(3000 + seed)
      numerator = jnp.asarray(
          rng.normal(loc=1.0, scale=1.0, size=(200, 1)), dtype=jnp.float32
      )
      denominator = jnp.asarray(
          rng.normal(loc=0.0, scale=1.0, size=(200, 1)), dtype=jnp.float32
      )

      scores_fwd, info_fwd = test.compute_fused_witness_scores(
          numerator_samples=numerator,
          denominator_samples=denominator,
          query_samples=z_jax,
      )
      scores_rev, info_rev = test.compute_fused_witness_scores(
          numerator_samples=denominator,
          denominator_samples=numerator,
          query_samples=z_jax,
      )

      rho_fwd = _spearman_rho(np.asarray(scores_fwd), true_ratio)
      rho_rev = _spearman_rho(np.asarray(scores_rev), true_ratio)
      forward_rhos.append(rho_fwd)
      reverse_rhos.append(rho_rev)

      self.assertEqual(scores_fwd.shape, (64,))
      self.assertEqual(int(info_fwd["num_valid"]), test.number_statistics)
      self.assertEqual(int(info_rev["num_valid"]), test.number_statistics)
      weights_fwd = np.asarray(info_fwd["weights"])
      self.assertEqual(weights_fwd.shape, (test.number_statistics,))
      self.assertTrue(np.all(weights_fwd >= 0.0))
      np.testing.assert_allclose(np.sum(weights_fwd), 1.0, atol=1e-5)

    for seed, (rho_fwd, rho_rev) in enumerate(zip(forward_rhos, reverse_rhos)):
      self.assertGreater(
          rho_fwd,
          0.9,
          msg=f"seed={seed}, rho_fwd={rho_fwd}, min_fwd={min(forward_rhos)}",
      )
      self.assertLess(
          rho_rev,
          -0.9,
          msg=f"seed={seed}, rho_rev={rho_rev}, max_rev={max(reverse_rhos)}",
      )

  def test_witness_scores_identical_distributions_median_near_one(self):
    # Evaluated across 20 fixed seeds (seeds 0..19, n=100, d=2, q=80):
    # When numerator and denominator follow the same distribution N(0, I_2),
    # the true density ratio is identically 1.0 and the median fused witness
    # score lies well inside [0.7, 1.3] for all 20 seeds.
    test = self._make_standard_fuse_test(num_permutations=20)
    medians = []
    for seed in range(20):
      rng = np.random.default_rng(4000 + seed)
      numerator = jnp.asarray(rng.normal(size=(100, 2)), dtype=jnp.float32)
      denominator = jnp.asarray(rng.normal(size=(100, 2)), dtype=jnp.float32)
      queries = jnp.asarray(rng.normal(size=(80, 2)), dtype=jnp.float32)
      scores, info = test.compute_fused_witness_scores(
          numerator_samples=numerator,
          denominator_samples=denominator,
          query_samples=queries,
      )
      median_score = float(np.median(np.asarray(scores)))
      medians.append(median_score)
      self.assertEqual(int(info["num_valid"]), test.number_statistics)

    for seed, median_score in enumerate(medians):
      self.assertBetween(
          median_score,
          0.7,
          1.3,
          msg=(
              f"seed={seed}, median={median_score}, min={min(medians)},"
              f" max={max(medians)}"
          ),
      )

  def test_witness_scores_masking_and_all_invalid_rows(self):
    rng = np.random.default_rng(77)
    numerator = jnp.asarray(
        rng.normal(loc=1.0, size=(30, 2)), dtype=jnp.float32
    )
    denominator = jnp.asarray(
        rng.normal(loc=0.0, size=(30, 2)), dtype=jnp.float32
    )
    queries = jnp.asarray(rng.normal(size=(15, 2)), dtype=jnp.float32)

    with self.subTest("partial_masking_from_nonfinite_stat"):
      test_stat_nan = fuse.FuseTest(
          divergence_class=_NanOnTargetLambdaDivergence,
          divergence_params={"nan_lmbda": 0.01},
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=2,
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=True,
          divergence_sweep_hyperparams={"lmbda": [0.01, 0.1, 1.0]},
      )
      scores, info = test_stat_nan.compute_fused_witness_scores(
          numerator, denominator, queries
      )
      weights = np.asarray(info["weights"])
      self.assertEqual(int(info["num_valid"]), 4)
      self.assertEqual(weights.shape, (6,))
      self.assertTrue(np.all(weights >= 0.0))
      np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-5)
      # For each bandwidth, the first lambda (0.01) is masked (`weights[0::3]`).
      np.testing.assert_array_equal(weights[0::3], 0.0)
      self.assertGreater(
          float(np.sum(weights[1::3]) + np.sum(weights[2::3])), 0.99
      )
      self.assertTrue(np.all(np.isfinite(np.asarray(scores))))

    with self.subTest("partial_masking_from_nonfinite_query_ratio"):
      test_ratio_nan = fuse.FuseTest(
          divergence_class=_NanQueryRatioOnTargetLambdaDivergence,
          divergence_params={"nan_lmbda": 0.01},
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=1,
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=True,
          divergence_sweep_hyperparams={"lmbda": [0.01, 0.1]},
      )
      scores, info = test_ratio_nan.compute_fused_witness_scores(
          numerator, denominator, queries
      )
      weights = np.asarray(info["weights"])
      self.assertEqual(int(info["num_valid"]), 1)
      np.testing.assert_array_equal(weights[:1], 0.0)
      np.testing.assert_allclose(weights[1:], 1.0, atol=1e-5)
      self.assertTrue(np.all(np.isfinite(np.asarray(scores))))

    with self.subTest("all_rows_invalid"):
      test_all_nan = fuse.FuseTest(
          divergence_class=_AlwaysNanDivergence,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=1,
          num_permutations=10,
          min_memory_kernel=False,
          min_memory_permutations=False,
          divergence_sweep_hyperparams={"lmbda": [0.1]},
      )
      scores, info = test_all_nan.compute_fused_witness_scores(
          numerator, denominator, queries
      )
      self.assertEqual(int(info["num_valid"]), 0)
      self.assertEqual(scores.shape, (15,))
      self.assertTrue(np.all(np.isnan(np.asarray(scores))))
      np.testing.assert_array_equal(
          np.asarray(info["weights"]), np.zeros(1, dtype=np.float32)
      )

  def test_witness_scores_query_batch_size_invariance_and_no_tracer_leak(self):
    test = fuse.FuseTest(
        divergence_class=likelihood_ratio.SymmetricKLDivergence,
        kernels=[kernel_module.Kernel.GAUSSIAN],
        significance=0.05,
        number_bandwidths=None,
        bandwidths=[1.0],
        num_permutations=10,
        min_memory_kernel=True,
        min_memory_permutations=True,
        divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
    )
    rng = np.random.default_rng(2026)
    numerator = jnp.asarray(
        rng.normal(loc=0.5, size=(30, 3)), dtype=jnp.float32
    )
    denominator = jnp.asarray(
        rng.normal(loc=-0.5, size=(30, 3)), dtype=jnp.float32
    )
    # Prime number of queries q=23 to exercise non-multiple batch sizes.
    queries = jnp.asarray(rng.normal(size=(23, 3)), dtype=jnp.float32)

    with jax.checking_leaks():
      scores_full, info_full = test.compute_fused_witness_scores(
          numerator, denominator, queries, query_batch_size=4096
      )
      scores_non_mult, info_non_mult = test.compute_fused_witness_scores(
          numerator, denominator, queries, query_batch_size=7
      )
      scores_one, info_one = test.compute_fused_witness_scores(
          numerator, denominator, queries, query_batch_size=1
      )

    for scores_b, info_b in (
        (scores_non_mult, info_non_mult),
        (scores_one, info_one),
    ):
      np.testing.assert_allclose(
          np.asarray(scores_b), np.asarray(scores_full), rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          np.asarray(info_b["weights"]),
          np.asarray(info_full["weights"]),
          rtol=1e-5,
          atol=1e-6,
      )
      self.assertEqual(int(info_b["num_valid"]), int(info_full["num_valid"]))

  def test_witness_scores_validation_errors(self):
    test = self._make_standard_fuse_test(num_permutations=10)
    num_ok = jnp.zeros((20, 2), dtype=jnp.float32)
    den_ok = jnp.zeros((20, 2), dtype=jnp.float32)
    q_ok = jnp.zeros((10, 2), dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, "2D arrays"):
      test.compute_fused_witness_scores(
          jnp.zeros((20,), dtype=jnp.float32), den_ok, q_ok
      )
    with self.assertRaisesRegex(ValueError, "2D array"):
      test.compute_fused_witness_scores(
          num_ok, den_ok, jnp.zeros((10,), dtype=jnp.float32)
      )
    with self.assertRaisesRegex(ValueError, "same number of samples"):
      test.compute_fused_witness_scores(
          num_ok, jnp.zeros((25, 2), dtype=jnp.float32), q_ok
      )
    with self.assertRaisesRegex(ValueError, "at least 2 samples"):
      test.compute_fused_witness_scores(
          jnp.zeros((1, 2), dtype=jnp.float32),
          jnp.zeros((1, 2), dtype=jnp.float32),
          q_ok,
      )
    with self.assertRaisesRegex(ValueError, "same feature dimension"):
      test.compute_fused_witness_scores(
          num_ok, jnp.zeros((20, 3), dtype=jnp.float32), q_ok
      )
    with self.assertRaisesRegex(ValueError, "same feature dimension"):
      test.compute_fused_witness_scores(
          num_ok, den_ok, jnp.zeros((10, 3), dtype=jnp.float32)
      )
    with self.assertRaisesRegex(ValueError, "at least one sample"):
      test.compute_fused_witness_scores(
          num_ok, den_ok, jnp.zeros((0, 2), dtype=jnp.float32)
      )
    with self.assertRaisesRegex(
        ValueError, "query_batch_size must be positive"
    ):
      test.compute_fused_witness_scores(
          num_ok, den_ok, q_ok, query_batch_size=0
      )

  def test_large_monte_carlo_type1_error(self):
    if not _RUN_LARGE_MONTE_CARLO.value:
      self.skipTest("Large Monte-Carlo test; pass --run_large_monte_carlo.")
    alpha = 0.05
    num_reps = 200
    test = self._make_standard_fuse_test(
        significance=alpha, num_permutations=100
    )
    rejections = 0
    for rep in range(num_reps):
      rng = np.random.default_rng(5000 + rep)
      x = jnp.asarray(rng.normal(size=(80, 2)), dtype=jnp.float32)
      y = jnp.asarray(rng.normal(size=(80, 2)), dtype=jnp.float32)
      res = test.run(x, y, key_seed=rep)
      rejections += int(bool(res.result_test))
    rejection_rate = rejections / num_reps
    se = np.sqrt(alpha * (1.0 - alpha) / num_reps)
    self.assertLessEqual(rejection_rate, alpha + 3.0 * se)


if __name__ == "__main__":
  absltest.main()
