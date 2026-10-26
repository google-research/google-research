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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from scipy import stats  # pylint: disable=g-import-not-at-top

from f_divergence_tests.divergence import kernel  # pylint: disable=g-import-not-at-top
from f_divergence_tests.divergence import likelihood_ratio  # pylint: disable=g-import-not-at-top

os.environ.setdefault("PJRT_NPROC", "1")


def _gaussian_kernels(
    x, y, bandwidth = 1.0
):
  """Computes Gaussian kernel matrices `(k_xx, k_yy, k_xy)` in float32."""
  x_jax = jnp.asarray(x, dtype=jnp.float32)
  y_jax = jnp.asarray(y, dtype=jnp.float32)
  d_xx = kernel.get_distances(x_jax, x_jax, norm="l2", min_memory=False)
  d_yy = kernel.get_distances(y_jax, y_jax, norm="l2", min_memory=False)
  d_xy = kernel.get_distances(x_jax, y_jax, norm="l2", min_memory=False)
  k_xx = kernel.get_kernel_matrix(d_xx, kernel.Kernel.GAUSSIAN, bandwidth)
  k_yy = kernel.get_kernel_matrix(d_yy, kernel.Kernel.GAUSSIAN, bandwidth)
  k_xy = kernel.get_kernel_matrix(d_xy, kernel.Kernel.GAUSSIAN, bandwidth)
  return k_xx, k_yy, k_xy


def _ref_likelihood_ratio_inv(
    k_zx,
    k_zy,
    k_xx,
    k_xy,
    lmbda,
):
  """Reference likelihood ratio using explicit matrix inverse `jnp.linalg.inv`."""
  m = k_xx.shape[0]
  inv_k_xx = jnp.linalg.inv(k_xx + m * lmbda * jnp.eye(m, dtype=k_xx.dtype))
  first_term = k_zy.mean(axis=1) - k_zx.mean(axis=1)
  second_term = -(k_zx @ inv_k_xx @ k_xy).mean(axis=1)
  third_term = k_zx @ inv_k_xx @ k_xx.mean(axis=1)
  chi_square_vals = (
      (first_term + second_term + third_term) / lmbda * 2.0 * (1.0 + lmbda)
  )
  return chi_square_vals + 1.0


def _ref_f_divergence_directional_inv(
    k_xx,
    k_yy,
    k_xy,
    lmbda,
    g,
    f_star,
):
  """Reference directional f-divergence using `jnp.linalg.inv`."""
  r_x = _ref_likelihood_ratio_inv(k_xx, k_xy, k_xx, k_xy, lmbda)
  r_y = _ref_likelihood_ratio_inv(k_xy.T, k_yy, k_xx, k_xy, lmbda)
  return jnp.mean(g(r_y)) - jnp.mean(f_star(g(r_x)))


def _ref_drmmd_directional_inv(
    k_xx,
    k_yy,
    k_xy,
    lmbda,
):
  """Reference directional DrMMD using `jnp.linalg.inv`."""
  m = k_xx.shape[0]
  inv_k_xx = jnp.linalg.inv(k_xx + m * lmbda * jnp.eye(m, dtype=k_xx.dtype))
  part1 = k_yy.mean() + k_xx.mean() - 2.0 * k_xy.mean()
  part2 = -(k_xy.T @ inv_k_xx @ k_xy).mean()
  part3 = (k_xx.T @ inv_k_xx @ k_xy).mean() * 2.0
  part4 = -(k_xx.T @ inv_k_xx @ k_xx).mean()
  return (part1 + part2 + part3 + part4) / lmbda * (1.0 + lmbda)


class LikelihoodRatioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="hockey_stick",
          estimator_cls=likelihood_ratio.SymmetricHockeyStick,
          is_drmmd=False,
      ),
      dict(
          testcase_name="kl_divergence",
          estimator_cls=likelihood_ratio.SymmetricKLDivergence,
          is_drmmd=False,
      ),
      dict(
          testcase_name="drmmd",
          estimator_cls=likelihood_ratio.DrMMDSymmetric,
          is_drmmd=True,
      ),
  )
  def test_cholesky_matches_matrix_inverse_reference(
      self, estimator_cls, is_drmmd
  ):
    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(50, 3)).astype(np.float32)
    y = rng.normal(loc=0.3, scale=1.0, size=(50, 3)).astype(np.float32)
    k_xx, k_yy, k_xy = _gaussian_kernels(x, y, bandwidth=1.5)
    lmbda = 0.5
    estimator = estimator_cls(lmbda=lmbda)

    actual_dir = estimator.directional(k_xx, k_yy, k_xy)
    actual_rev = estimator.directional(k_yy, k_xx, k_xy.T)
    actual_call = estimator(k_xx, k_yy, k_xy)

    if is_drmmd:
      expected_dir = _ref_drmmd_directional_inv(k_xx, k_yy, k_xy, lmbda)
      expected_rev = _ref_drmmd_directional_inv(k_yy, k_xx, k_xy.T, lmbda)
    else:
      expected_dir = _ref_f_divergence_directional_inv(
          k_xx,
          k_yy,
          k_xy,
          lmbda,
          estimator.ratio_postprocess_fn,
          estimator.conjugate_fn,
      )
      expected_rev = _ref_f_divergence_directional_inv(
          k_yy,
          k_xx,
          k_xy.T,
          lmbda,
          estimator.ratio_postprocess_fn,
          estimator.conjugate_fn,
      )
    expected_call = jnp.maximum(expected_dir, expected_rev)

    with self.subTest("directional_xy"):
      self.assertTrue(bool(jnp.isfinite(actual_dir)))
      np.testing.assert_allclose(actual_dir, expected_dir, rtol=1e-4, atol=1e-5)
    with self.subTest("directional_yx"):
      self.assertTrue(bool(jnp.isfinite(actual_rev)))
      np.testing.assert_allclose(actual_rev, expected_rev, rtol=1e-4, atol=1e-5)
    with self.subTest("call_symmetric"):
      self.assertTrue(bool(jnp.isfinite(actual_call)))
      np.testing.assert_allclose(
          actual_call, expected_call, rtol=1e-4, atol=1e-5
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="hockey_stick",
          dir_cls=likelihood_ratio.HockeyStickKernelFDivergence,
          sym_cls=likelihood_ratio.SymmetricHockeyStick,
      ),
      dict(
          testcase_name="kl",
          dir_cls=likelihood_ratio.KullbackLeiblerFDivergence,
          sym_cls=likelihood_ratio.SymmetricKLDivergence,
      ),
      dict(
          testcase_name="total_variation",
          dir_cls=likelihood_ratio.TotalVariationFDivergence,
          sym_cls=likelihood_ratio.SymmetricTotalVariation,
      ),
      dict(
          testcase_name="pearson_chi_squared",
          dir_cls=likelihood_ratio.PearsonChiSquaredFDivergence,
          sym_cls=likelihood_ratio.SymmetricPearsonChiSquared,
      ),
  )
  def test_symmetric_vs_directional_behavior(self, dir_cls, sym_cls):
    rng = np.random.default_rng(7)
    x = rng.normal(loc=0.0, scale=0.5, size=(40, 2)).astype(np.float32)
    y = rng.normal(loc=1.2, scale=1.5, size=(40, 2)).astype(np.float32)
    k_xx, k_yy, k_xy = _gaussian_kernels(x, y, bandwidth=1.0)
    dir_est = dir_cls(lmbda=0.1)
    sym_est = sym_cls(lmbda=0.1)

    d_xy = dir_est.directional(k_xx, k_yy, k_xy)
    d_yx = dir_est.directional(k_yy, k_xx, k_xy.T)

    with self.subTest("non_symmetric_flag_and_call"):
      self.assertFalse(dir_est.symmetric)
      np.testing.assert_allclose(dir_est(k_xx, k_yy, k_xy), d_xy, rtol=1e-6)
    with self.subTest("symmetric_flag_name_and_call"):
      self.assertTrue(sym_est.symmetric)
      self.assertEqual(sym_cls.__name__, f"Symmetric{dir_cls.__name__}")
      np.testing.assert_allclose(
          sym_est(k_xx, k_yy, k_xy), jnp.maximum(d_xy, d_yx), rtol=1e-6
      )

  def test_drmmd_symmetric_flag_and_directional(self):
    rng = np.random.default_rng(11)
    x = rng.normal(loc=0.0, scale=0.5, size=(30, 2)).astype(np.float32)
    y = rng.normal(loc=1.0, scale=1.5, size=(30, 2)).astype(np.float32)
    k_xx, k_yy, k_xy = _gaussian_kernels(x, y, bandwidth=1.0)
    est = likelihood_ratio.DrMMDSymmetric(lmbda=0.1)

    d_xy = est.directional(k_xx, k_yy, k_xy)
    d_yx = est.directional(k_yy, k_xx, k_xy.T)
    sym_val = est(k_xx, k_yy, k_xy)

    self.assertTrue(est.symmetric)
    self.assertIsInstance(est, likelihood_ratio.LikelihoodRatioEstimator)
    np.testing.assert_allclose(sym_val, jnp.maximum(d_xy, d_yx), rtol=1e-6)

    # Also verify non-symmetric branch of DrMMDSymmetric.__call__.
    est.symmetric = False
    np.testing.assert_allclose(est(k_xx, k_yy, k_xy), d_xy, rtol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name="hockey_stick",
          estimator_cls=likelihood_ratio.SymmetricHockeyStick,
      ),
      dict(
          testcase_name="kl_divergence",
          estimator_cls=likelihood_ratio.SymmetricKLDivergence,
      ),
      dict(
          testcase_name="drmmd",
          estimator_cls=likelihood_ratio.DrMMDSymmetric,
      ),
  )
  def test_identical_samples_and_mean_shift_divergences(self, estimator_cls):
    # Tested across 20 fixed seeds (seeds 0..19, n=60, d=2, bw=0.5, lmbda=1.0):
    # - identical samples: 0.0 for all three estimators (|div| < 1e-6).
    # - 2-sigma mean shift: all 20 seeds finite and > 0.1 for HockeyStick, KL,
    #   and DrMMD.
    estimator = estimator_cls(lmbda=1.0)
    identical_vals = []
    shifted_vals = []
    for seed in range(20):
      rng = np.random.default_rng(seed)
      x = rng.normal(loc=0.0, scale=1.0, size=(60, 2)).astype(np.float32)
      y_shifted = rng.normal(loc=2.0, scale=1.0, size=(60, 2)).astype(
          np.float32
      )
      k_xx, _, _ = _gaussian_kernels(x, x, bandwidth=0.5)
      k_xx_s, k_yy_s, k_xy_s = _gaussian_kernels(x, y_shifted, bandwidth=0.5)
      identical_vals.append(float(estimator(k_xx, k_xx, k_xx)))
      shifted_vals.append(float(estimator(k_xx_s, k_yy_s, k_xy_s)))

    for div_identical in identical_vals:
      self.assertAlmostEqual(div_identical, 0.0, delta=0.05)
    for seed, div_shifted in enumerate(shifted_vals):
      self.assertTrue(
          np.isfinite(div_shifted),
          msg=f"seed={seed}, vals={shifted_vals}",
      )
      self.assertGreater(
          div_shifted,
          0.1,
          msg=f"seed={seed}, min={min(shifted_vals)}, max={max(shifted_vals)}",
      )

  def test_likelihood_ratio_ranks_1d_gaussian_shift(self):
    # For x ~ N(0, 1) and y ~ N(1, 1), true dP_y/dP_x(z) = exp(z - 0.5).
    # Tested across 20 fixed seeds (seeds 0..19, n=200, lmbda=0.1, bw=1.0):
    # Spearman rho > 0.98 for all 20 seeds (asserted > 0.9 for headroom).
    z = np.linspace(-1.0, 2.0, 50, dtype=np.float32)[:, None]
    z_jax = jnp.asarray(z)
    true_ratio = np.exp(z[:, 0] - 0.5)
    bw = 1.0
    estimator = likelihood_ratio.KullbackLeiblerFDivergence(lmbda=0.1)

    for seed in range(20):
      rng = np.random.default_rng(seed)
      x = rng.normal(loc=0.0, scale=1.0, size=(200, 1)).astype(np.float32)
      y = rng.normal(loc=1.0, scale=1.0, size=(200, 1)).astype(np.float32)
      x_jax = jnp.asarray(x)
      y_jax = jnp.asarray(y)
      k_xx = kernel.get_kernel_matrix(
          kernel.get_distances(x_jax, x_jax, norm="l2", min_memory=False),
          kernel.Kernel.GAUSSIAN,
          bw,
      )
      k_xy = kernel.get_kernel_matrix(
          kernel.get_distances(x_jax, y_jax, norm="l2", min_memory=False),
          kernel.Kernel.GAUSSIAN,
          bw,
      )
      k_zx = kernel.get_kernel_matrix(
          kernel.get_distances(z_jax, x_jax, norm="l2", min_memory=False),
          kernel.Kernel.GAUSSIAN,
          bw,
      )
      k_zy = kernel.get_kernel_matrix(
          kernel.get_distances(z_jax, y_jax, norm="l2", min_memory=False),
          kernel.Kernel.GAUSSIAN,
          bw,
      )

      cho_x = likelihood_ratio.regularized_cho_factor(k_xx, estimator.lmbda)
      estimated_ratio = np.asarray(
          estimator.likelihood_ratio_yx_on_z(k_zx, k_zy, k_xx, cho_x, k_xy)
      )
      rho = stats.spearmanr(estimated_ratio, true_ratio).statistic
      self.assertGreater(rho, 0.9, msg=f"seed={seed}, rho={rho}")

  def test_single_query_returns_1d_shape_one(self):
    rng = np.random.default_rng(99)
    x = rng.normal(size=(10, 2)).astype(np.float32)
    y = rng.normal(size=(10, 2)).astype(np.float32)
    k_xx, _, k_xy = _gaussian_kernels(x, y, bandwidth=1.0)
    k_zx = k_xx[:1, :]
    k_zy = k_xy[:1, :]

    estimator = likelihood_ratio.KullbackLeiblerFDivergence(lmbda=0.1)
    cho_x = likelihood_ratio.regularized_cho_factor(k_xx, estimator.lmbda)
    ratio = estimator.likelihood_ratio_yx_on_z(k_zx, k_zy, k_xx, cho_x, k_xy)

    self.assertEqual(ratio.shape, (1,))

  def test_hockey_stick_postprocess_fn_returns_float_dtype(self):
    estimator = likelihood_ratio.HockeyStickKernelFDivergence(
        lmbda=0.1, order=1.0
    )
    ratios = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float32)

    transformed = estimator.ratio_postprocess_fn(ratios)

    self.assertEqual(transformed.dtype, jnp.float32)
    np.testing.assert_array_equal(
        transformed, np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )


if __name__ == "__main__":
  absltest.main()
