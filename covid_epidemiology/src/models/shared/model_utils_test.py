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

import unittest

import numpy as np
import pandas as pd

from covid_epidemiology import model
from covid_epidemiology.src.models.shared import model_utils


class ModelUtilsTest(unittest.TestCase):

  def test_update_metric_to_predictions(self):
    metric_to_predictions = {
        "Infected": [
            model.Prediction(
                time_horizon=1, predicted_value=20, ground_truth=12)
        ]
    }
    gt_data = [7, 2, 5, 7]

    got = model_utils.update_metric_to_predictions(
        metric="recovered_cases",
        values=np.array([[12], [23]], np.float32),
        metric_to_predictions=metric_to_predictions,
        train_end_of_window=2,
        gt_data=gt_data)

    wanted = {
        "Infected": [
            model.Prediction(
                time_horizon=1, predicted_value=20, ground_truth=12)
        ],
        "recovered_cases": [
            model.Prediction(
                time_horizon=1, predicted_value=12, ground_truth=5),
            model.Prediction(
                time_horizon=2, predicted_value=23, ground_truth=7)
        ],
    }
    self.assertEqual(wanted, got)

  def test_update_metric_to_predictions_with_quantiles(self):
    metric_to_predictions = {}
    gt_data = [7, 2, 5, 7]

    got = model_utils.update_metric_to_predictions(
        metric="recovered_cases",
        values=np.array([[12, 14], [23, 25]], np.float32),
        metric_to_predictions=metric_to_predictions,
        train_end_of_window=2,
        gt_data=gt_data,
        quantiles=[0.1, 0.9],
        metric_string_format="{metric}_{quantile}_quantile")

    wanted = {
        "recovered_cases_0.1_quantile": [
            model.Prediction(
                time_horizon=1, predicted_value=12, ground_truth=5),
            model.Prediction(
                time_horizon=2, predicted_value=23, ground_truth=7)
        ],
        "recovered_cases_0.9_quantile": [
            model.Prediction(
                time_horizon=1, predicted_value=14, ground_truth=5),
            model.Prediction(
                time_horizon=2, predicted_value=25, ground_truth=7)
        ],
    }
    self.assertEqual(wanted, got)

  def test_update_metric_to_predictions_offset(self):
    metric_to_predictions = {
        "Infected": [
            model.Prediction(
                time_horizon=1, predicted_value=20, ground_truth=12)
        ]
    }
    gt_data = [7, 2, 5, 7]

    got = model_utils.update_metric_to_predictions(
        metric="recovered_cases",
        values=np.array([[12], [23]], np.float32),
        metric_to_predictions=metric_to_predictions,
        train_end_of_window=2,
        gt_data=gt_data,
        time_horizon_offset=2)

    wanted = {
        "Infected": [
            model.Prediction(
                time_horizon=1, predicted_value=20, ground_truth=12)
        ],
        "recovered_cases": [
            model.Prediction(
                time_horizon=-1, predicted_value=12, ground_truth=5),
            model.Prediction(
                time_horizon=0, predicted_value=23, ground_truth=7)
        ],
    }
    self.assertEqual(wanted, got)

  def test_populate_gt_list(self):
    gt_list = np.zeros((2, 4))
    gt_indicator = np.ones((2, 4))
    location_to_gt = {
        "IR": pd.Series([12, 4, 5]),
        "US": pd.Series([3, 4, 5, 6, 7])
    }
    gt_list, gt_indicator = model_utils.populate_gt_list(
        0, location_to_gt, "IR", 4, gt_list, gt_indicator)
    gt_list, gt_indicator = model_utils.populate_gt_list(
        1, location_to_gt, "US", 4, gt_list, gt_indicator)
    wanted = np.array([[12, 4, 5, 0], [3, 4, 5, 6]], np.float32)
    np.testing.assert_equal(gt_list, wanted)
    np.testing.assert_equal(gt_indicator, gt_indicator)


if __name__ == "__main__":
  unittest.main()
