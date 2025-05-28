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

from dp_l2 import utils


class UtilsTest(absltest.TestCase):

  def test_linear_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -0.5), 0.501, delta=2e-3
    )

  def test_quadratic_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -(x**2), -2.0), 1.415, delta=2e-3
    )

  def test_exponential_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -(2**x), -5.0), 2.322, delta=2e-3
    )

  def test_zero_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, 0.0), 0.001, delta=2e-3
    )

  def test_negative_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -1.0), 1.001, delta=2e-3
    )

  def test_high_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -100.0), 100.001, delta=2e-3
    )

  def test_small_tolerance(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -0.5, tolerance=1e-6), 0.5, delta=2e-6
    )

  def test_large_tolerance(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -0.5, tolerance=1e-1), 0.5, delta=2e-1
    )

if __name__ == '__main__':
  absltest.main()
