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

"""Tests for optimize_ara."""

import math
from absl.testing import absltest
import pandas as pd
from scipy import stats
from ara_optimization import dataset_evaluation
from ara_optimization import metrics
from ara_optimization import optimize_ara


class OptimizeAraTest(absltest.TestCase):

  def test_initial_value(self):
    dataset = dataset_evaluation.ARADataset(
        pd.DataFrame({"A": [1]*99+[5]+[15]}), [], ["A"], "")
    initial_value = [1, math.log2(5)]
    objective = optimize_ara.ARAObjective(dataset, [], 1.0)
    for a, b in zip(objective.initial_value(), initial_value):
      self.assertAlmostEqual(a, b)

  def test_optimize_ara(self):
    dataset = dataset_evaluation.ARADataset(
        pd.DataFrame({"A": [1], "C": [1], "slice": [1]}), ["slice"], ["A"], "C")
    privacy_budget = 1.0
    answer = optimize_ara.ConfigurationError(
        [1], [1], (stats.dlaplace(1/2**16).var() * 3 / 2**33)**0.5)
    optimization_output = optimize_ara.optimize_ara(dataset, dataset,
                                                    [metrics.RMSEMetric()] * 2,
                                                    privacy_budget)
    self.assertAlmostEqual(optimization_output.contribution_budgets[0],
                           answer.contribution_budgets[0])
    self.assertAlmostEqual(optimization_output.clipping_thresholds[0],
                           answer.clipping_thresholds[0])
    self.assertAlmostEqual(optimization_output.error, answer.error)

  def test_baseline(self):
    dataset = dataset_evaluation.ARADataset(
        pd.DataFrame({"A": [1], "C": [1], "slice": [1]}), ["slice"], ["A"], "C")
    privacy_budget = 1.0
    quantile = 0.99
    answer = optimize_ara.ConfigurationError(
        [0.5], [1], (stats.dlaplace(1/2**16).var() / 2**30)**0.5)
    baseline_output = optimize_ara.baseline(dataset, dataset,
                                            [metrics.RMSEMetric()] * 2,
                                            privacy_budget, quantile)
    self.assertAlmostEqual(baseline_output.contribution_budgets[0],
                           answer.contribution_budgets[0])
    self.assertAlmostEqual(baseline_output.clipping_thresholds[0],
                           answer.clipping_thresholds[0])
    self.assertAlmostEqual(baseline_output.error, answer.error)

if __name__ == "__main__":
  absltest.main()
