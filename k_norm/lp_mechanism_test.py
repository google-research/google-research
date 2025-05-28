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

"""Tests for lp_mechanism."""

from absl.testing import absltest
import numpy as np

from k_norm import lp_mechanism


class LpMechanismTest(absltest.TestCase):

  def test_random_signs_uniform_distribution(self):
    d = 10
    ones = np.ones(d)
    num_positives = np.zeros(d)
    i = 0
    num_trials = 10000
    overall_failure_probability = 0.001
    while i < num_trials:
      num_positives += lp_mechanism.random_signs(ones) > 0
      i += 1
    # This confidence interval comes from Hoeffding's inequality, see for
    # example https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds,
    # and solve for k and apply a union bound over d.
    confidence_interval_width = np.sqrt(
        num_trials * np.log(2 / (overall_failure_probability / (2 * d)))
    )
    np.testing.assert_array_less(
        num_positives,
        np.ones(d) * ((num_trials / 2) + confidence_interval_width),
    )
    np.testing.assert_array_less(
        np.ones(d) * ((num_trials / 2) - confidence_interval_width),
        num_positives,
    )

  def test_sample_lp_ball_norm(self):
    d = 10
    num_samples = 1000
    ps = [1, 2, 3, 5, 10, 100, np.inf]
    for p in ps:
      samples = [lp_mechanism.sample_lp_ball(d, p) for _ in range(num_samples)]
      num_big_norm = np.sum(np.linalg.norm(samples, ord=p, axis=1) > 1)
      self.assertEqual(num_big_norm, 0)

if __name__ == '__main__':
  absltest.main()
