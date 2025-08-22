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

"""Tests for utils."""

import math

from absl.testing import absltest as test
from abstract_nas.evolution import utils


class UtilsTest(test.TestCase):

  def test_pareto_curve1(self):
    points = [(3, 1), (2, 2), (1, 3), (3, 3)]
    curve = utils.ParetoCurve(points, first_high=False, second_high=False)
    self.assertSameElements(curve.standardize_obs(points[0:3]),
                            curve.curve)

  def test_pareto_curve2(self):
    points = [(3, 1), (2, 2), (1, 3), (3, 3)]
    curve = utils.ParetoCurve(points, first_high=True, second_high=True)
    self.assertSameElements(curve.standardize_obs(points[3:4]),
                            curve.curve)

  def test_pareto_weights1(self):
    points = [(3, 1), (2, 2), (1, 3), (3, 3)]
    curve = utils.ParetoCurve(points, first_high=False, second_high=False)
    points = [(3, 1), (2, 2), (1, 3), (3, 3), (1, 1)]
    weights = curve.get_weights(points, normalize=False)
    self.assertAlmostEqual(0, weights[0])
    self.assertAlmostEqual(0, weights[1])
    self.assertAlmostEqual(0, weights[2])
    self.assertAlmostEqual(math.sqrt(2), weights[3])
    self.assertAlmostEqual(-math.sqrt(2), weights[4])

  def test_pareto_weights2(self):
    points = [(3, 1), (2, 2), (1, 3), (3, 3)]
    curve = utils.ParetoCurve(points, first_high=True, second_high=True)
    points = [(3, 1), (2, 2), (1, 3), (3, 3), (4, 4)]
    weights = curve.get_weights(points, normalize=False)
    self.assertAlmostEqual(0, weights[0])
    self.assertAlmostEqual(1, weights[1])
    self.assertAlmostEqual(0, weights[2])
    self.assertAlmostEqual(0, weights[3])
    self.assertAlmostEqual(-math.sqrt(2), weights[4])

if __name__ == '__main__':
  test.main()
