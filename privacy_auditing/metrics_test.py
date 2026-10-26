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

"""Tests for the privacy metrics module."""

from concurrent import futures

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from privacy_auditing import metrics


def _normal_scores(n_in, n_out):
  rng = np.random.default_rng(seed=0xBAD5EED)
  return rng.normal(1, 1, n_in), rng.normal(0, 1, n_out)


def _binomial_scores(n_in, n_out):
  # Binomial scores to get ties.
  rng = np.random.default_rng(seed=0xBAD5EED)
  return rng.binomial(100, 0.6, n_in), rng.binomial(100, 0.4, n_out)


class MetricsTest(parameterized.TestCase):

  @parameterized.product(
      mu=(0.7, 1.0, 1.2),
      out_samples_ratio=(0.5, 1.0),
      reverse=(False, True),
      shrinkage=(0.0, 0.5),
  )
  def test_gdp_upper_bound_tight(
      self, mu, out_samples_ratio, reverse, shrinkage
  ):
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_samples = 100_000
    out_samples = int(in_samples * out_samples_ratio)
    in_scores = rng.normal(mu, 1, in_samples)
    out_scores = rng.normal(0, 1, out_samples)
    if reverse:
      in_scores, out_scores = out_scores, in_scores
    mu_low, mu_hat, mu_high = metrics.audit_gdp(
        in_scores,
        out_scores,
        min_count=(in_samples + out_samples) // 100,
        shrinkage=shrinkage,
        seed=0xBAD5EED,
        central_with_interval=True,
    )
    self.assertBetween(mu_hat, mu_low, mu_high)
    # Lower bound can be higher than true because this is a max statistic.
    self.assertBetween(mu_low, 0.95 * mu, 1.05 * mu)
    self.assertBetween(mu_high, mu, 1.1 * mu)

  @parameterized.parameters(True, False)
  def test_gdp_upper_bound_inf_no_shrinkage(self, use_shrinkage):
    in_scores = np.ones(100)
    out_scores = np.zeros(100)
    mu_ub = metrics.audit_gdp(
        in_scores,
        out_scores,
        min_count=50,
        shrinkage=0.5 if use_shrinkage else 0.0,
        seed=0xBAD5EED,
    )
    if use_shrinkage:
      self.assertLess(mu_ub, np.inf)
    else:
      self.assertEqual(mu_ub, np.inf)

  @parameterized.parameters(
      (0, -np.inf),
      (0.1, -np.inf),
      (0.25, 0.0),
      (0.3, 0.2),
      (0.5, 1.0),
      (0.5000001, np.inf),
      (0.625, np.inf),
      (0.75, np.inf),
      (0.9, np.inf),
      (1.0, np.inf),
  )
  def test_safe_quantile(self, q, expected):
    values = np.array([-np.inf, 0, 1, np.inf, np.inf])
    self.assertAlmostEqual(metrics._safe_quantile(values, q), expected)

  def test_safe_quantile_vectorized(self):
    values = np.array([-np.inf, 0, 1, np.inf, np.inf])
    q = np.linspace(0, 1, 9)
    result = metrics._safe_quantile_vectorized(values, q)
    expected = [-np.inf, -np.inf, 0.0, 0.5, 1.0, np.inf, np.inf, np.inf, np.inf]
    np.testing.assert_allclose(result, expected)

  @parameterized.product(
      reverse=(False, True),
      score_gen=(_normal_scores, _binomial_scores),
      out_samples_ratio=(0.5, 1.0, 1.5),
      shrinkage=(0.0, 0.5),
  )
  def test_efficient_jackknife(
      self, reverse, score_gen, out_samples_ratio, shrinkage
  ):
    n_in = 1000
    n_out = int(n_in * out_samples_ratio)
    in_scores, out_scores = score_gen(n_in, n_out)
    in_scores.sort()
    out_scores.sort()
    min_count = 200

    def jackknife_task(i):
      in_sample = in_scores if reverse else np.delete(in_scores, i)
      out_sample = np.delete(out_scores, i) if reverse else out_scores
      all_mus, _ = metrics._compute_mus(
          in_sample, out_sample, min_count, shrinkage
      )
      return np.max(all_mus)

    with futures.ThreadPoolExecutor() as executor:
      n = n_out if reverse else n_in
      jackknife_results = list(executor.map(jackknife_task, range(n)))

    target_scores = out_scores if reverse else in_scores
    other_scores = in_scores if reverse else out_scores
    jackknife_max_mu = metrics._max_mu_jackknife_one_side(
        target_scores, other_scores, min_count, shrinkage
    )
    np.testing.assert_allclose(jackknife_max_mu, jackknife_results)


if __name__ == '__main__':
  absltest.main()
