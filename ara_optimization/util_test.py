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
from absl.testing import parameterized
import numpy as np
import pandas as pd
from ara_optimization import util


class UtilTest(parameterized.TestCase):

  def test_randomized_round(self):
    np.random.seed(1234)
    series = pd.Series([0.2]*1000)
    rounded_series = [int(x) for x in util.randomized_round(series)]
    self.assertBetween(rounded_series, [0]*1000, [1]*1000)
    self.assertEqual(sum(rounded_series), 219)

  @parameterized.named_parameters(
      ("at endpoints", pd.Series([0, 10]), 10, 5, pd.Series([0, 10])),
      ("beyond endpoints", pd.Series([-1, 11]), 10, 5, pd.Series([0, 10])),
      ("scaling down", pd.Series([3, 6, 9]), 12, 8, pd.Series([3, 6, 9])),
      ("scaling up", pd.Series([2, 4, 6]), 8, 12, pd.Series([2, 4, 6]))
  )
  def test_randomized_snap_row(self, value, max_value, cap, rounded_value):
    snap_row = list(util.randomized_snap_row(value, max_value, cap))
    self.assertAlmostEqual(snap_row, list(rounded_value))

  @parameterized.named_parameters(
      ("equal split", [1, 1], 10, [5, 5]),
      ("unequal split", [1, 4], 10, [2, 8]),
      ("rounds exactly", [1, 3], 5, [1, 4]),
      ("rounds up first", [1, 1, 1], 4, [2, 1, 1]),
      ("rounds up second", [3, 4, 3], 11, [3, 5, 3]),
      ("rounds down first", [2, 2, 2], 5, [1, 2, 2]),
      ("rounds down second", [3, 4, 3], 9, [3, 3, 3]),
      ("avoid zero value", [1, 1000], 5, [1, 4])
  )
  def test_snap_contribution_bounds(self, bounds, total, rounded_bounds):
    snap_bounds = util.snap_contribution_bounds(bounds, total)
    self.assertEqual(snap_bounds, rounded_bounds)

  @parameterized.named_parameters(
      ("small test", pd.DataFrame({"A": [1, 2, 3]}), ["A"], 5, [10, 5]),
      ("medium test", pd.DataFrame({"A": [1, 2, 3, 4, 5],
                                    "B": [2, 2, 2, 2, 2],
                                    "C": [10, 10, 10, 2, 2]}), ["A", "B", "C"],
       5, [15, 10, 50, 5]),
      ("different scale factor", pd.DataFrame({"A": [3]}), ["A"], 12, [36, 12])
  )
  def test_rmsre_tau_error_metrics(self, df, value_columns, mult_factor, taus):
    error_metrics = util.rmsre_tau_error_metrics(df, value_columns, mult_factor)
    self.assertEqual([metr.tau for metr in error_metrics], taus)

if __name__ == "__main__":
  absltest.main()
