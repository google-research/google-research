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

from absl.testing import absltest
import numpy as np

from dp_l2 import l2


class L2Test(absltest.TestCase):

  def test_get_radius_upper_bound(self):
    # gammaincc(5, 5.745 / 1.23) = 0.5 to tolerance 1e-4.
    d = 5
    sigma = 1.23
    beta = 0.5
    self.assertAlmostEqual(
        l2.get_radius_upper_bound(d, sigma, beta),
        5.745,
        delta=1e-3
    )

  def test_get_cap_fraction_from_h_zero(self):
    d = 10
    h = 0
    r = 5
    self.assertAlmostEqual(
        l2.get_cap_fraction_from_h(d, h, r),
        0,
        delta=1e-3
    )

  def test_get_cap_fraction_from_h_half(self):
    d = 10
    h = 5
    r = 5
    self.assertAlmostEqual(
        l2.get_cap_fraction_from_h(d, h, r),
        0.5,
        delta=1e-3
    )

  def test_get_cap_fraction_from_h_whole(self):
    d = 10
    h = 10
    r = 5
    self.assertAlmostEqual(
        l2.get_cap_fraction_from_h(d, h, r),
        1,
        delta=1e-3
    )

  def test_get_cap_fraction_from_h_raises_error_big_h(self):
    d = 10
    h = 11
    r = 5
    with self.assertRaises(RuntimeError):
      l2.get_cap_fraction_from_h(d, h, r)

  def test_get_cap_fraction_from_h_raises_error_negative_h(self):
    d = 10
    h = -1
    r = 5
    with self.assertRaises(RuntimeError):
      l2.get_cap_fraction_from_h(d, h, r)

  def test_get_cap_height(self):
    # For eps = 1, r = 5, and sigma = 0.5, the cap height is
    # 0.5 * (6 ^ 2 - 5.5 ^ 2) = 2.875.
    eps = 1
    r = 5
    sigma = 0.5
    self.assertAlmostEqual(
        l2.get_cap_height(eps, r, sigma),
        2.875,
        delta=1e-3
    )

  def test_get_cap_height_too_small(self):
    # For eps = 1, r = 5, and sigma = 2, the cap height is
    # 0.5 * (6 ^ 2 - 7 ^ 2) = -6.5, which should be clipped to 0.
    eps = 1
    r = 5
    sigma = 2
    self.assertAlmostEqual(
        l2.get_cap_height(eps, r, sigma),
        0,
        delta=1e-3
    )

  def test_get_cap_height_too_big(self):
    # For eps = 1, r = 0.1, and sigma = 0, the cap height is
    # 0.5 * (1.1 ^ 2 - 0.1 ^ 2) = 0.6, which should be clipped to 2 * r = 0.2.
    eps = 1
    r = 0.1
    sigma = 0
    self.assertAlmostEqual(
        l2.get_cap_height(eps, r, sigma),
        0.2,
        delta=1e-3
    )

  def test_get_cap_fraction_for_loss_zero(self):
    # If sigma > 1/eps, then the mechanism is eps-DP, and the cap fraction is
    # 0.
    d = 10
    eps = 1
    r = 1.23
    sigma = 1.23
    self.assertAlmostEqual(
        l2.get_cap_fraction_for_loss(d, eps, r, sigma),
        0,
        delta=1e-3
    )

  def test_get_cap_fraction_for_loss_one(self):
    # If r <= (1 - eps * sigma) / 2, then the cap fraction is 1.
    # See Corollary 3.7 in the paper for details.
    d = 10
    eps = 1
    sigma = 0.1
    r = 0.490
    self.assertAlmostEqual(
        l2.get_cap_fraction_for_loss(d, eps, r, sigma),
        1,
        delta=1e-3
    )

  def test_compute_shell_mass_half(self):
    # gammaincc(5, 5.745 / 1.23) = 0.5 to tolerance 1e-4.
    d = 5
    sigma = 1.23
    self.assertAlmostEqual(
        l2.compute_shell_mass(d, sigma, 0, 5.745),
        0.5,
        delta=1e-3
    )

  def test_get_plrv_1_upper_bound_large_sigma(self):
    # If sigma >= 1/eps, then the mechanism is eps-DP, and the probability of
    # L_1 >= eps is 0.
    d = 10
    sigma = 1.23
    num_rs = 10
    last_r = 1.23
    eps = 1
    self.assertAlmostEqual(
        l2.get_plrv_1_upper_bound(d, sigma, num_rs, last_r, eps),
        0,
        delta=1e-3
    )

  def test_get_plrv_1_upper_bound_small_sigma_d_1(self):
    # If sigma < 1 / eps and d=1, then the mechanism is Laplace, and the
    # probability of L_1 >= eps is 1 - 0.5 * e^(0.5 * (eps - 1 / sigma)).
    # Here, that is 1 - 0.5 * e^(0.5 * (1 - 2)) ~= 0.6967.
    d = 1
    sigma = 0.5
    num_rs = 10
    last_r = 1.23
    eps = 1
    self.assertAlmostEqual(
        l2.get_plrv_1_upper_bound(d, sigma, num_rs, last_r, eps),
        0.6967,
        delta=1e-3
    )

  def test_get_plrv_1_upper_bound_small_sigma_large_d(self):
    # See Lemma 3.13 in the paper for details.
    # first_r = (1 - (eps * sigma) ^ 2) / (2 * (1 + eps * sigma))
    # = (3 / 4) / 3 = 0.25, and the upper bound is
    # gamma(3, 0.5) / Gamma(3)
    #   + [gamma(3, 2) - gamma(3, 0.5)] * F_{0.25, h(0.25)} / Gamma(3)
    #   + Gamma(3, 2) * F_{1, h(1)}) / Gamma(3)
    # ~= 0.0143877 + 0.3089359 * 1 + 0.6766764 * 0.4375
    # (using Python's gammaincc for Gamma(d, x) / Gamma(d) and gammainc for
    # gamma(d, x) / Gamma(d), and get_cap_fraction_from_loss for F_{x, h(x)})
    # = 0.61937.
    d = 3
    sigma = 0.5
    num_rs = 2
    last_r = 1
    eps = 1
    self.assertAlmostEqual(
        l2.get_plrv_1_upper_bound(d, sigma, num_rs, last_r, eps),
        0.61937,
        delta=1e-3
    )

  def test_get_1_centered_cap_fraction_for_loss_zero(self):
    # If e1_r < (1 + eps * sigma) / 2, then the cap fraction is 0.
    d = 10
    eps = 1
    sigma = 0.1
    e1_r = 0.499
    self.assertAlmostEqual(
        l2.get_1_centered_cap_fraction_for_loss(d, eps, sigma, e1_r),
        0,
    )

  def test_get_1_centered_cap_fraction_for_loss_one(self):
    # The maximum height of the radius-e1_r spherical cap in V is
    # e1_r(1 - eps * sigma) - (1 - eps^2 * sigma^2) / 2 (see the proof of
    # Lemma 3.19 for details).
    # Here, that is 2(1 - 0.1) - (1 - 0.01) / 2 = 1.8 - 0.495 = 1.305, and
    # get_cap_fraction_from_h(10, 1.305, 2) ~= 0.147523.
    d = 10
    eps = 1
    sigma = 0.1
    e1_r = 2
    self.assertAlmostEqual(
        l2.get_1_centered_cap_fraction_for_loss(d, eps, sigma, e1_r),
        0.147523,
        delta=1e-3
    )

  def test_get_plrv_2_lower_bound_large_sigma(self):
    d = 10
    eps = 1
    sigma = 1.23
    e1_rs = np.linspace(1, 1.23, 2)
    self.assertAlmostEqual(
        l2.get_plrv_2_lower_bound(d, eps, sigma, e1_rs),
        0,
        delta=1e-3
    )

  def test_get_plrv_2_lower_bound_small_sigma_d_1(self):
    # If d=1, then this should return
    # 0.5 * e^(0.5 * (-eps - 1 / sigma)) = 0.5 * e^(-3/2)~= 0.111565.
    d = 1
    eps = 1
    sigma = 0.5
    e1_rs = np.linspace(1, 1.23, 2)
    self.assertAlmostEqual(
        l2.get_plrv_2_lower_bound(d, eps, sigma, e1_rs),
        0.111,
        delta=1e-3
    )

  def test_get_plrv_2_lower_bound_small_sigma_large_d(self):
    # See Lemma 3.21 in the paper for details.
    # The lower bound is (gamma(3, 6) - gamma(3, 4)) * F_{2, h(2)} / 2
    # + Gamma(3, 6) * F_{3, h(3)} / 2
    # where h(2) = get_cap_fraction_from_h(3, 2(1 - 0.5) - (1 - 0.5^2) / 2, 2)
    # = get_cap_fraction_from_h(3, 0.625, 2) = 0.15625
    # and h(3) = get_cap_fraction_from_h(3, 3(1 - 0.5) - (1 - 0.5^2) / 2, 3)
    # = get_cap_fraction_from_h(3, 1.125, 3) = 0.10417.
    # Thus, the expected answer is ~= 0.176134 * 0.15625 + 0.061969 * 0.1875
    # ~= 0.03914.
    d = 3
    eps = 1
    sigma = 0.5
    e1_rs = np.linspace(2, 3, 2)
    self.assertAlmostEqual(
        l2.get_plrv_2_lower_bound(d, eps, sigma, e1_rs),
        0.0391,
        delta=1e-3
    )

  def test_get_plrv_difference_large_sigma(self):
    # If sigma >= 1 / eps, then the mechanism is eps-DP, and the PLRV
    # difference is 0.
    d = 10
    eps = 1
    delta = 0.01
    num_rs = 10
    num_e1_rs = 10
    sigma = 1.23
    self.assertAlmostEqual(
        l2.get_plrv_difference(d, eps, delta, num_rs, num_e1_rs, sigma),
        0,
        delta=1e-3
    )

  def test_get_plrv_difference_small_sigma_d_1(self):
    # See Lemma 3.3 and Lemma 3.15 in the paper for details.
    # The PLRV difference is
    # [1 - 0.5 * e^(0.5 * (eps - 1 / sigma))]
    # - e^(eps) * [0.5 * e^(0.5 * (-eps - 1 / sigma))]
    # = [1 - 0.5 * e^(-0.5)] - e * [0.5 * e^(-1.5)] ~= 0.393469
    d = 1
    eps = 1
    delta = 0.01
    num_rs = 10
    num_e1_rs = 10
    sigma = 0.5
    self.assertAlmostEqual(
        l2.get_plrv_difference(d, eps, delta, num_rs, num_e1_rs, sigma),
        0.39346,
        delta=1e-3
    )

  def test_get_l2_sigma_valid_sigma(self):
    d = 10
    eps = 1
    delta = 0.01
    num_rs = 10
    num_e1_rs = 10
    sigma = l2.get_l2_sigma(d, eps, delta, num_rs, num_e1_rs)
    # We verify that the estimated sigma yields (eps, delta)-DP.
    self.assertLess(
        l2.get_plrv_difference(d, eps, delta, num_rs, num_e1_rs, sigma),
        delta
    )

  def test_get_l2_sigma_minimal_sigma(self):
    d = 10
    eps = 1
    delta = 0.01
    num_rs = 10
    num_e1_rs = 10
    sigma = l2.get_l2_sigma(d, eps, delta, num_rs, num_e1_rs)
    # We verify that a smaller sigma does not yield (eps, delta)-DP.
    self.assertLess(
        delta,
        l2.get_plrv_difference(
            d, eps, delta, num_rs, num_e1_rs, sigma - 0.001
        )
    )

  def test_sample_l2_ball_returns_valid_samples(self):
    d = 10
    num_samples = 1000
    samples = l2.sample_l2_ball(d, num_samples)
    self.assertEqual(samples.shape, (num_samples, d))
    self.assertLess(np.amax(np.linalg.norm(samples, axis=1)), 1)


if __name__ == '__main__':
  absltest.main()
