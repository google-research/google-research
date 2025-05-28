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

"""Tests for metrics.

This test is taken from
https://github.com/google-research/google-research/blob/master/dp_regression/experiment_test.py.
The only modification is to specify it to evaluating a single model at a time,
consistent with our code.
"""


from absl.testing import absltest
import numpy as np

from private_kendall import metrics


class MetricsTest(absltest.TestCase):

  def test_r_squared(self):
    labels = np.asarray([1, 2, 3])
    predictions_1 = np.asarray([3, -1, 2])
    r_squared_1 = metrics.r_squared_from_predictions(predictions_1, labels)
    # residual_sum_squares:
    # (3 - 1)^2 + (-1 - 2)^2 + (2 - 3)^2 = 14
    # total sum squares:
    # (1 - 2)^2 + (2 - 2)^2 + (3 - 2)^2 = 2
    # R^2:
    # 1 - (14 / 2) = -6
    # 1 - (1 / 2) = 0.5
    np.testing.assert_array_almost_equal(r_squared_1, -6)
    predictions_2 = np.asarray([0, 2, 3])
    r_squared_2 = metrics.r_squared_from_predictions(predictions_2, labels)
    # residual_sum_squares:
    # (0-1)^2 + (2-2)^2 + (3-3)^2 = 1
    # R^2:
    # 1 - (1 / 2) = 0.5
    np.testing.assert_array_almost_equal(r_squared_2, 0.5)

if __name__ == '__main__':
  absltest.main()
