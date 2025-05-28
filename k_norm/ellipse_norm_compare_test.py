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

"""Tests for ellipse_norm_compare."""

from absl.testing import absltest

from k_norm import ellipse_norm_compare


class EllipseNormCompareTest(absltest.TestCase):

  def test_compare_count_ball_ellipse_norms(self):
    d = 4
    k = 2
    # By Theorem 8.9 in the paper, the ellipse has squared axis lengths
    # a_1^2 = sqrt(lambda * ||v_1(k)||_2^2), and
    # a_2^2 = sqrt(lambda * ||v_2(k)||_2^2 / (d-1))
    # where lambda = (||v_1(k)||_2 + ||v_2(k)||_2 * sqrt(d-1))^2. Using
    # ||v_1(k)||_2 = k / sqrt(d), and
    # ||v_2(k)||_2 = sqrt(k(d-k) / d)
    # and substituting in d = 4 and k = 2 gives
    # ||v_1(k)||_2 = 1, and
    # ||v_2(k)||_2 = 1.
    # Thus
    # lambda = (1 + sqrt(3))^2
    # a_1^2 = sqrt((1 + sqrt(3))^2 * 1) = 1 + sqrt(3)
    # a_2^2 = sqrt((1 + sqrt(3))^2 * 1 / 3) = (1 + sqrt(3)) / sqrt(3)
    # and by Lemma 4.7, the expected squared l_2 norm is
    # (a_1^2 + (d-1) * a_2^2) / (d + 2)
    # = (1 + sqrt(3) + sqrt(3) * (1 + sqrt(3))) / 6.
    sqrt_3 = 3 ** (1/2)
    expected_ellipse_norm = (1 + sqrt_3 + sqrt_3 * (1 + sqrt_3)) / 6
    # The minimum l_2 ball containing the count ball has radius sqrt(k), so by
    # Lemma 4.7, its expected squared l_2 norm is
    # d * k / (d + 2) = 4 / 3.
    expected_ball_norm = 4 / 3
    expected_squared_l2_norm_ratio = expected_ellipse_norm / expected_ball_norm
    squared_l2_norm_ratio = ellipse_norm_compare.compare_count_norms(d, k)
    self.assertAlmostEqual(expected_squared_l2_norm_ratio,
                           squared_l2_norm_ratio)

  def test_compare_vote_ball_ellipse_norms(self):
    d = 4
    # By Theorem 9.10 in the paper, the ellipse has squared axis lengths
    # a_1^2 = sqrt(lambda * ||w_1||_2^2), and
    # a_2^2 = sqrt(lambda * ||w_2||_2^2 / (d-1))
    # where lambda = (||w_1||_2 + ||w_2||_2 * sqrt(d-1))^2. Using
    # ||w_1||_2 = (d-1) * sqrt(d) / 2, and
    # ||w_2||_2 = sqrt(d * (d^2 - 1) / 12)
    # and substituting in d = 4 gives
    # ||w_1||_2 = 3, and
    # ||w_2||_2 = sqrt(5).
    # Thus
    # lambda = (3 + sqrt(15)) ^ 2
    # a_1^2 = sqrt((3 + sqrt(15)) ^ 2 * 9) = 3 * (3 + sqrt(15))
    # a_2^2 = sqrt((3 + sqrt(15)) ^ 2 * 5 / 3) = (3 + sqrt(15)) * sqrt(5 / 3)
    # and by Lemma 4.7, the expected squared l_2 norm is
    # (a_1^2 + (d-1) * a_2^2) / (d+2)
    # = (3 * (3 + sqrt(15)) + 3 * (3 + sqrt(15)) * sqrt(5/3)) / 6
    # = ((3 + sqrt(15)) * (1 + sqrt(5/3))) / 2.
    sqrt_15 = 15 ** (1/2)
    sqrt_53 = (5 / 3) ** (1/2)
    expected_ellipse_norm = ((3 + sqrt_15) * (1 + sqrt_53)) / 2
    # The minimum l_2 ball containing the count ball has radius
    # sqrt(sum_i=0^{d-1} i^2 = sqrt(1 + 4 + 9) = sqrt(14), so by
    # Lemma 4.7, its expected squared l_2 norm is
    # d * 14 / (d + 2) = 28 / 3.
    expected_ball_norm = 28 / 3
    expected_squared_l2_norm_ratio = expected_ellipse_norm / expected_ball_norm
    squared_l2_norm_ratio = ellipse_norm_compare.compare_vote_norms(d)
    self.assertAlmostEqual(expected_squared_l2_norm_ratio,
                           squared_l2_norm_ratio, places=5)


if __name__ == '__main__':
  absltest.main()
