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

from absl.testing import absltest
import numpy as np
from scipy import special

from dp_ripple import ripple_count
from dp_ripple import ripple_sum


comb = special.comb


# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name
class RippleCountTest(absltest.TestCase):
  def test_compute_K_n(self,):
    # Test for checking that K_n is computed correctly for various examples.
    max_d = 10
    max_k = 10
    max_n = 10
    C_per_n = np.zeros((max_d + 1, max_k * max_n + 1, max_n + 1))
    for n in range(max_n):
      C_n = ripple_sum.compute_ordered_restricted_partitions_table(
          max_d, max_k * n, n - 1
      )
      C_per_n[: C_n.shape[0], : C_n.shape[1], n] = C_n

    d = 10
    k = 6
    n = 0
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = 1
    assert expected_K_n == computed_K_n

    d = 10
    k = 6
    n = 1
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = 0
    for i in range(1, k + 1):
      expected_K_n += comb(d, i) * 2
    assert expected_K_n == computed_K_n

    d = 1
    k = 1
    n = 9
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = 2
    assert expected_K_n == computed_K_n

    d = 5
    k = 1
    n = 3
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = (
        5 * 2 + 20 * 4 + 10 * 8
    )  # (3,0,0,0,0) x 5, (1,2,0,0,0) x 20, (1,1,1,0,0) x 10
    print("expected_K_n = " + str(expected_K_n))
    print("computed_K_n = " + str(computed_K_n))
    assert expected_K_n == computed_K_n

    d = 5
    k = 2
    n = 2
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    # (1,-1,0,0,0) x 20, (1,1,-1,0,0) x 30, (1,-1,-1,0,0) x 30, (1,1,-1,-1,0) x 30
    # The following counts should be doubled for the negative orthant
    # (2,2,0,0,0) x 10 , (2,1,1,0,0) x 30, (2,1,0,0,0) x 20, (2,0,0,0,0) x 5,
    # (1,1,1,1,0) x 5, (1,1,1,0,0) x 10

    expected_K_n = 2 * (10 + 30 + 20 + 5 + 5 + 10) + 20 + 30 + 30 + 30
    assert expected_K_n == computed_K_n

    d = 5
    k = 2
    n = 3
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    # (2,-1,0,0,0) x 20, (-2,1,0,0,0) x 20, (2,-1,-1,0,0) x 30, (-2,1,1,0,0) x 30,
    # (2,1,-1,0,0) x 60, (-2,-1,1,0,0) x 60,
    # (1,-2,-2,0,0) x 30, (-1,2,2,0,0) x 30, (2,2,-1,-1,0) x 30, (-2,-2,1,1,0) x 30,
    # (1,2,-1,-1,0) x 60, (-1,-2,1,1,0) x 60
    # (1,2,1,-1,0) x 60, (-1,-2,-1,1,0) x 60, (1,1,1,-1,0) x 20, (-1,-1,-1,1,0) x 20
    # (1,1,1,1,-1) x 5, (-1,-1,-1,-1,1) x 5, (1,2,1,-1,-1) x 30, (-1,-2,-1,1,1) x 30
    # (1,1,1,-1,-1) x 10, (-1,-1,-1,1,1) x 10,
    # These counts should be doubled for the negative orthant
    # (3,3,0,0,0) x 10, (3,2,1,0,0) x 60, (3,1,1,1,0) x 20, (3,2,0,0,0) x 20,
    # (3,1,1,0,0) x 30, (3,1,0,0,0) x 20, (3,0,0,0,0) x 5, (2,2,2,0,0) x 10,
    # (2,2,1,1,0) x 30, (2,1,1,1,1) x 5, (2,2,1,0,0) x 30, (2,1,1,1,0) x 20,
    # (1,1,1,1,1) x 1

    expected_K_n = 2 * (
        10 + 60 + 20 + 20 + 30 + 20 + 5 + 10 + 30 + 5 + 30 + 20 + 1
    ) + (20 + 20 + 30 + 30 + 60 + 60 + 30 + 30 + 30 + 30 + 60 + 60 + 60 + 60 + 20 + 20 + 5 + 5 + 30 + 30 + 10 + 10)
    assert expected_K_n == computed_K_n

    d = 2
    k = 1
    n = 0
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = 1
    assert expected_K_n == computed_K_n

    d = 2
    k = 1
    n = 1
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = 4  # (1,0) x 2, (-1,0) x 2
    assert expected_K_n == computed_K_n

    d = 2
    k = 1
    n = 2
    K_n_p_m_weights = ripple_count._compute_K_n_p_m_weights(d, n, k, C_per_n)
    computed_K_n = np.sum(K_n_p_m_weights)
    expected_K_n = (
        8  # (2,0) x 2, (1,1) x 1, (1,-1) x 2, (-1,-1) x 1, (-2,0) x 2
    )
    assert expected_K_n == computed_K_n


def test_w_n_appoximately_sum_to_1(self,):
  # Test for checking that \sum_{n=0}^{max_n}w_n approximately sums to 1.
  max_d = 4
  max_k = 4
  max_n = 200
  eps = 0.1
  C_per_n = np.zeros((max_d + 1, max_k * max_n + 1, max_n + 1))
  for n in range(max_n):
    C_n = ripple_sum.compute_ordered_restricted_partitions_table(
        max_d, max_k * n, n - 1
    )
    C_per_n[: C_n.shape[0], : C_n.shape[1], n] = C_n
  for d in range(1, max_d):
    for k in range(0, d + 1):
      normalizing_constant = (
          ripple_count._compute_normalizing_constant_for_count(d, k, eps)
      )
      first_max_n_ws = []
      for n in range(max_n):
        w_n = ripple_count._compute_w_n_for_count(
            d, k, n, eps, normalizing_constant, C_per_n
        )
        first_max_n_ws.append(w_n)
      np.testing.assert_allclose(sum(first_max_n_ws), 1, atol=1e-3)
      print("(d, k) = " + str((d, k)))
      print(sum(first_max_n_ws))
  return


if __name__ == "__main__":
  absltest.main()
