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
import pandas as pd
from ara_optimization import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("zero bias", pd.Series(0), pd.Series(4), pd.Series(0), 2/5),
      ("nonzero bias", pd.Series(2), pd.Series(5), pd.Series(0), 3/5),
      ("large true value", pd.Series(0), pd.Series(4), pd.Series(10), 2/10)
  )
  def test_RMSRE_Tau_error(self, bias, variance, true_value, error):
    metric = metrics.RMSRETauMetric(5)
    self.assertAlmostEqual(metric.error(bias, variance, true_value)[0], error)

  @parameterized.named_parameters(
      ("zero bias", pd.Series(0), pd.Series(4), pd.Series(0), 2),
      ("nonzero bias", pd.Series(2), pd.Series(5), pd.Series(0), 3),
      ("true value ignored", pd.Series(2), pd.Series(5), pd.Series(10), 3)
  )
  def test_RMSE_error(self, bias, variance, true_value, error):
    metric = metrics.RMSEMetric()
    self.assertAlmostEqual(metric.error(bias, variance, true_value)[0], error)

  @parameterized.named_parameters(
      ("length 1", [1], 1),
      ("length 3", [1, 2, 2], 3**0.5),
      ("all equal", [5, 5, 5, 5, 5], 5),
      ("pandas series", pd.Series([1, 2, 3]), (14/3)**0.5)
  )
  def test_L2_avg_error(self, errors, avg_error):
    metric = metrics.RMSEMetric()
    self.assertAlmostEqual(metric.avg_error(errors), avg_error)

if __name__ == "__main__":
  absltest.main()
