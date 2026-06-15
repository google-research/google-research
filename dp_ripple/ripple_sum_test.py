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

from dp_ripple import ripple_sum


comb = special.comb


# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name
class RippleSumTest(absltest.TestCase):

  def test_compute_ordered_restricted_partitions_table_against_convolution(
      self,
  ):
    """Test compute_ordered_restricted_partitions_table against convolution.

    Alternatively, we use the generating function $g(x) = (x + x^{2} + ...
    + x^{n-1})^{d} = (\frac{x^{n+1}-1}{x-1})^{d}$. Define $g(x)[s]$ is
    the coefficent of $x^{s}$. Note that $g(x)[s] = C(d,s,n-1)$.
    If we represent $(1 + x + x^{2} + ... + x^{n-1})$ as an array
    $J = [0,1,...,1] \in \mathbb{R}^{n}$ then the coefficients of $g(x)$ are
    given by the $d$-fold convolution $J * J * ... * J$.
    """

    d = 7
    m = 5
    J = [0] + ([1] * m)
    d_fold_convolution = J
    for _ in range(d - 1):
      d_fold_convolution = np.convolve(d_fold_convolution, J)
    coefficients_up_to_d_times_m = (
        ripple_sum.compute_ordered_restricted_partitions_table(d, d * m, m)
    )
    expected_result = np.zeros_like(d_fold_convolution)
    assert len(d_fold_convolution) == len(coefficients_up_to_d_times_m[d, :])
    np.testing.assert_allclose(
        (d_fold_convolution - coefficients_up_to_d_times_m[d, :]),
        expected_result,
    )


def test_sample_ordered_restricted_partition(
    self,
):
  """Test that each partition of 5 into parts of size <= 3 occur equally often.

  Let X ~ Binomial(n, p). Then for any 0 < \delta < 1, we have that
  P[X > (1 + \delta)np] < \exp(-\delta^{2}np/3) and
  P[X < (1 - \delta)np] < \exp(-\delta^{2}np/2). Here n = 6000, p = 1/6.
  Let's set \delta = 0.126 so that the test succeeds with probability 0.99.
  See theorem 6.2.1 in the following reference for details.
  courses.cs.washington.edu/courses/cse312/20su/files/student_drive/6.2.pdf
  """
  d = 3
  s = 5
  m = 3
  C = ripple_sum.compute_ordered_restricted_partitions_table(d, s, m)
  num_decompositions = 6
  expected_num_samples_per_decomposition = 1000
  freq_counts = {}
  for _ in range(num_decompositions * expected_num_samples_per_decomposition):
    sample = ripple_sum._sample_ordered_restricted_partition(d, s, m, C)
    if str(sample) in freq_counts:
      freq_counts[str(sample)] += 1
    else:
      freq_counts[str(sample)] = 1

  for _, v in freq_counts.items():
    np.testing.assert_allclose(
        v,
        expected_num_samples_per_decomposition,
        atol=0.126 * expected_num_samples_per_decomposition,
    )


def test_sample_X_J_point(
    self,
):
  # Test for checking that frequencies split by chosen_index are correct
  # where chosen_index is the number of coordinates that are equal to n.
  d = 10
  k = 5
  n = 7
  s = 8
  J = list(range(8))
  num_samples = 10000
  C = ripple_sum.compute_ordered_restricted_partitions_table(d, n * k, n - 1)
  freq_per_chosen_index_sum = {}
  for i in range(1, k + 1):
    freq_per_chosen_index_sum[i] = {}
  for _ in range(num_samples):
    sample = ripple_sum._sample_X_J_point(d, s, J, k, n, C)
    chosen_index = np.size(np.where(np.abs(sample) == n))
    t = np.sum(np.abs(sample)) - n * chosen_index
    if t not in freq_per_chosen_index_sum[chosen_index]:
      freq_per_chosen_index_sum[chosen_index][t] = 1
    else:
      freq_per_chosen_index_sum[chosen_index][t] += 1

  X_J_i_weights = [[]] + [
      ripple_sum._compute_X_J_i_weights(s, k, n, i, C) for i in range(1, k + 1)
  ]
  unnormalized_index_probabilities = [
      comb(s, i) * np.sum(X_J_i_weights[i]) for i in range(1, k + 1)
  ]
  index_probabilities = unnormalized_index_probabilities / np.sum(
      unnormalized_index_probabilities
  )
  unnormalized_chosen_index_freqs = np.zeros(k + 1)
  for i in range(1, k + 1):
    for t in freq_per_chosen_index_sum[i]:
      unnormalized_chosen_index_freqs[i] += freq_per_chosen_index_sum[i][t]
  normalized_chosen_index_freqs = unnormalized_chosen_index_freqs / np.sum(
      unnormalized_chosen_index_freqs
  )
  np.testing.assert_allclose(
      normalized_chosen_index_freqs[1:], index_probabilities, atol=1e-02
  )


def test_sample_X_J_complement_point(
    self,
):
  # Test for checking that frequencies split by s are weighted properly.
  d = 10
  k = 5
  n = 7
  s = 8
  J = list(range(8))
  num_samples = 10000
  C = ripple_sum.compute_ordered_restricted_partitions_table(d, n * k, n - 1)
  freq_per_sum = {}
  for _ in range(num_samples):
    sample = ripple_sum._sample_X_J_complement_point(d, s, J, k, n, C)
    t = int(np.sum(np.abs(sample)))
    if t not in freq_per_sum:
      freq_per_sum[t] = 1
    else:
      freq_per_sum[t] += 1
  weights = C[s][max(0, n * k - k + 1) : n * k + 1]
  normalized_weights = weights / np.sum(weights)
  unnormalized_sum_freqs = np.zeros(n * k + 1)
  for t in freq_per_sum:
    unnormalized_sum_freqs[t] += freq_per_sum[t]
  normalized_sum_freqs = unnormalized_sum_freqs / np.sum(unnormalized_sum_freqs)
  np.testing.assert_allclose(
      normalized_sum_freqs[max(0, n * k - k + 1) :],
      normalized_weights,
      atol=1e-02,
  )


def test_compute_L_n(
    self,
):
  # Test for checking that L_n is computed correctly for various examples.
  max_d = 10
  max_k = 10
  max_n = 10
  C_per_n = {}
  for n in range(max_n):
    C_per_n[n] = ripple_sum.compute_ordered_restricted_partitions_table(
        max_d, max_k * n, n - 1
    )

  d = 10
  k = 6
  n = 0
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 1
  assert expected_L_n == computed_L_n

  d = 10
  k = 6
  n = 1
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 0
  for i in range(1, k + 1):
    expected_L_n += comb(d, i) * (2**i)
  assert expected_L_n == computed_L_n

  d = 1
  k = 1
  n = 9
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 2
  assert expected_L_n == computed_L_n

  d = 5
  k = 1
  n = 3
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = (
      5 * 2 + 20 * 4 + 10 * 8
  )  # (3,0,0,0,0) x 5, (1,2,0,0,0) x 20, (1,1,1,0,0) x 10
  assert expected_L_n == computed_L_n

  d = 5
  k = 2
  n = 2
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  # (2,2,0,0,0) x 10, (2,1,1,0,0) x 30, (2,1,0,0,0) x 20, (2,0,0,0,0) x 5,
  # (1,1,1,1,0) x 5, (1,1,1,0,0) x 10
  expected_L_n = 5 * 2 + 30 * 4 + 40 * 8 + 5 * 16
  assert expected_L_n == computed_L_n

  d = 5
  k = 2
  n = 3
  # computed_L_n = np.sum(ripple_sum.compute_X_Xc_sizes(d, k, n, C_per_n[n]))
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  # (3,3,0,0,0) x 10, (3,2,1,0,0) x 60, (3,1,1,1,0) x 20, (3,2,0,0,0) x 20,
  # (3,1,1,0,0) x 30, (3,1,0,0,0) x 20, (3,0,0,0,0) x 5, (2,2,2,0,0) x 10,
  # (2,2,1,1,0) x 30, (2,1,1,1,1) x 5, (2,2,1,0,0) x 30, (2,1,1,1,0) x 20,
  # (1,1,1,1,1) x 1
  expected_L_n = 5 * 2 + 50 * 4 + 130 * 8 + 70 * 16 + 6 * 32
  assert expected_L_n == computed_L_n

  d = 2
  k = 1
  n = 0
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 1
  assert expected_L_n == computed_L_n

  d = 2
  k = 1
  n = 1
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 2 * 2  # (1,0) x 2
  assert expected_L_n == computed_L_n

  d = 2
  k = 1
  n = 2
  L_n_s_weights = ripple_sum._compute_L_n_s_weights(d, k, n, C_per_n[n])
  computed_L_n = np.sum(L_n_s_weights)
  expected_L_n = 2 * 2 + 1 * 4  # (2,0) x 2, (1,1) x 1
  assert expected_L_n == computed_L_n


def test_w_n_appoximately_sum_to_1(
    self,
):
  # Test for checking that \sum_{n=0}^{max_n}w_n approximately sums to 1.
  max_n = 100
  eps = 1
  for d in range(1, 10):
    for k in range(0, d + 1):
      normalizing_constant = ripple_sum._compute_normalizing_constant_for_sum(
          d, k, eps
      )
      first_max_n_ws = []
      for n in range(max_n):
        C_table = ripple_sum.compute_ordered_restricted_partitions_table(
            d, n * k, n - 1 if n >= 1 else -1
        )
        # Calculate the weight of this entire layer n
        L_n_weight = np.sum(ripple_sum._compute_L_n_s_weights(d, k, n, C_table))
        w_n = L_n_weight * np.exp(-n * eps) / normalizing_constant
        first_max_n_ws.append(w_n)
      np.testing.assert_allclose(sum(first_max_n_ws), 1, atol=1e-4)
  return


if __name__ == "__main__":
  absltest.main()
