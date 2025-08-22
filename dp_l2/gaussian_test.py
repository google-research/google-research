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
from dp_l2 import gaussian


class GaussianTest(absltest.TestCase):

  def test_gaussian_cdf_check(self):
    # Let Phi be the standard Gaussian CDF. Then
    # Phi(0) = 0.5 and Phi(-1) = 0.1586552539 is accurate to 1e-10.
    # If we set l2_sensitivity / (2 * sigma) = 1/2 and
    # eps * sigma / l2_sensitivity = 1/2, then
    # gaussian_cdf_check returns should return Phi(0) - e^(eps) * Phi(-1).
    # Thus gaussian_cdf_check(eps = 1/2, sigma = 1, l2_sensitivity = 1)
    # should return Phi(0) - e^(1/2) * Phi(-1) ~ 0.2384.
    eps = 0.5
    sigma = 1
    l2_sensitivity = 1
    self.assertAlmostEqual(
        gaussian.gaussian_cdf_check(eps, l2_sensitivity, sigma),
        0.2384,
        delta=1e-4,
    )

  def test_get_gaussian_sigma(self):
    # This value is taken from the Google C++ library
    # (https://github.com/google/differential-privacy/blob/main/cc/algorithms/numerical-mechanisms_test.cc#L1119).
    eps = np.log(3)
    l2_sensitivity = 1
    delta = 1e-5
    self.assertAlmostEqual(
        gaussian.get_gaussian_sigma(eps, delta, l2_sensitivity),
        3.425,
        delta=1e-3,
    )

if __name__ == '__main__':
  absltest.main()
