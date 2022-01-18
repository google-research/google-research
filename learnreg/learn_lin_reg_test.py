# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for learn_lin_reg.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized
import numpy as np
import six

from learnreg import learn_lin_reg


DataPoint = learn_lin_reg.DataPoint


class TestCase(parameterized.TestCase):

  @parameterized.parameters(
      ([2, 1, 3], 1),
      ([3, 2, 1], 2),
  )
  def test_argmin(self, value_list, expected_index):
    actual_index = learn_lin_reg.argmin(value_list)
    self.assertEqual(expected_index, actual_index)

  @parameterized.named_parameters(
      # Minimize a, subject to a >= 5.
      ('case1', {'a': 1}, [], [({'a': -1}, -5)], {'a': 5}),
      # Minimize 2*a-b, subject to 3*a=1, b <= 2.
      ('case2',
       {'a': 2, 'b': -1},
       [({'a': 1}, 1./3)],
       [({'b': 1}, 2)],
       {'a': 1./3, 'b': 2}),
  )
  def test_solve_lp(self, cost, equality_contraints, upper_bounds,
                    expected_variables):
    actual_variables = learn_lin_reg.solve_lp(cost, equality_contraints,
                                              upper_bounds)
    self.assertSetEqual(set(six.iterkeys(expected_variables)),
                        set(six.iterkeys(actual_variables)))
    for name, value in six.iteritems(expected_variables):
      self.assertAlmostEqual(value, actual_variables[name])

  @parameterized.named_parameters(
      ('case1', [DataPoint(1., 0., [1.])], [1.]),
      ('case2', [DataPoint(1.1, 0., [1., 1.]), DataPoint(2, 0., [0, 2])],
       [.1, 1]),
      ('case3', [DataPoint(1., .9, [1.]), DataPoint(.9, .8, [1.])],
       [.1]),
  )
  def test_learn_linear_regularizer(self, data_points,
                                    expected_normalized_coefficients):
    alpha, coefficients = learn_lin_reg.learn_linear_regularizer(data_points)
    normalized_coefficients = coefficients / alpha
    np.testing.assert_almost_equal(expected_normalized_coefficients,
                                   normalized_coefficients)


if __name__ == '__main__':
  unittest.main()
