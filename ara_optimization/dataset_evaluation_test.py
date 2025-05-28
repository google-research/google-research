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
from scipy import stats
from ara_optimization import dataset_evaluation
from ara_optimization import metrics


class DatasetEvaluationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("small_example", pd.DataFrame({"A": [1, 2],
                                      "B": [3, 4],
                                      "C": [1, 1],
                                      "slices": [0, 0]}),
       ["slices"], ["A", "B"], "C", [metrics.RMSEMetric()]*3, 1, [1, 3], [5, 5],
       True, False, 20*(10/9)**0.5
      ),
      ("exact_objective", pd.DataFrame({"A": [1, 2],
                                        "B": [3, 4],
                                        "C": [1, 1],
                                        "slices": [0, 0]}),
       ["slices"], ["A", "B"], "C", [metrics.RMSEMetric()]*3, 1, [1, 3], [5, 5],
       False, False, (1/2**16)*(stats.dlaplace(1/2**16).var() *
                                (20**2 * 10/9 + 3) / 3)**0.5
      ),
      ("separate_count_estimation", pd.DataFrame({"A": [1, 2],
                                                  "B": [3, 4],
                                                  "C": [1, 1],
                                                  "slices": [0, 0]}),
       ["slices"], ["A", "B"], "C", [metrics.RMSEMetric()]*3, 1, [1, 3, 4],
       [5, 5, 1], True, True, (24054)**0.5 * 2/9
      )
  )
  def test_evaluate_objective(self, df, slice_columns, value_columns,
                              count_column, error_metric, total_privacy_budget,
                              contribution_budgets, clipping_thresholds,
                              approximate_objective, separate_count_estimation,
                              overall_error):
    dataset = dataset_evaluation.ARADataset(df, slice_columns, value_columns,
                                            count_column)
    self.assertAlmostEqual(overall_error, dataset.evaluate_objective(
        error_metric, total_privacy_budget, contribution_budgets,
        clipping_thresholds, approximate_objective, separate_count_estimation
    ))


if __name__ == "__main__":
  absltest.main()
