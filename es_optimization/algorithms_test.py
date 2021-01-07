# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for optimization algorithms."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from es_optimization import algorithms

perturbation_array = np.array([[0, 1], [2, -1], [4, 2],
                               [-2, -2], [0, 3], [0, -3], [0, 4], [0, -4],
                               [-1, 5], [1, -5], [2, 6], [8, -6]])
function_value_array = np.array(
    [-1, 1, 10, -10, -2, 2, -0.5, 0.5, 4, -4, -8, 8])


class ESOptimizationAlgorithmsTest(parameterized.TestCase):

  @parameterized.parameters(
      (perturbation_array, function_value_array, 'antithetic', 3,
       np.array([[4, 2], [2, 6], [-1, 5], [-2, -2], [8, -6], [1, -5]
                ]), np.array([10, -8, 4, -10, 8, -4])),
      (perturbation_array, function_value_array, 'forward_fd', 5,
       np.array([[4, 2], [8, -6], [-1, 5], [0, -3], [2, -1]
                ]), np.array([10, 8, 4, 2, 1])))
  def test_filtering(self, perturbations, function_values, est_type,
                     num_top_directions, expected_ps, expected_fs):
    top_ps, top_fs = algorithms.filter_top_directions(perturbations,
                                                      function_values, est_type,
                                                      num_top_directions)
    np.testing.assert_array_equal(expected_ps, top_ps)
    np.testing.assert_array_equal(expected_fs, top_fs)

  @parameterized.parameters(
      (perturbation_array, function_value_array, 'antithetic', 3,
       np.array([100, -16])), (perturbation_array, function_value_array,
                               'forward_fd', 5, np.array([76, -9])),
      (perturbation_array, function_value_array, 'antithetic', 0,
       np.array([102, -34])), (perturbation_array, function_value_array,
                               'forward_fd', 0, np.array([74, -34])))
  def test_mc_gradient(self, perturbations, function_values, est_type,
                       num_top_directions, expected_gradient):
    precision_parameter = 0.1
    step_size = 0.01
    current_value = 2
    blackbox_object = algorithms.MCOptimizer(precision_parameter, est_type,
                                             False, 'no_method', None,
                                             step_size, num_top_directions)
    current_input = np.zeros(2)
    step = blackbox_object.run_step(perturbations, function_values,
                                    current_input, current_value)
    gradient = step * (precision_parameter**2) / step_size
    if num_top_directions == 0:
      gradient *= len(perturbations)
    else:
      gradient *= num_top_directions

    np.testing.assert_array_almost_equal(expected_gradient, gradient)


if __name__ == '__main__':
  absltest.main()
