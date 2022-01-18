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

from unittest import mock

import numpy as np
import pandas as pd
from pandas_gbq import gbq
import tensorflow as tf

from covid_epidemiology.src.models.shared import feature_utils
from covid_epidemiology.src.models.shared import model_spec


class FeatureUtilsTest(tf.test.TestCase):

  def test_filter_data_based_on_location(self):
    ts_data = pd.DataFrame([{
        "feature_name": "death",
        "dt": "2020/01/01",
        "country_code": "IR"
    }, {
        "feature_name": "death",
        "dt": "2020/01/01",
        "country_code": "US"
    }, {
        "feature_name": "death",
        "dt": "2020/01/01",
        "country_code": "CH"
    }])
    static_data = pd.DataFrame([{
        "population": 70,
        "country_code": "IR"
    }, {
        "population": 50,
        "country_code": "US"
    }, {
        "population": 10,
        "country_code": "CH"
    }])
    got_static_data, got_ts_data = feature_utils.filter_data_based_on_location(
        static_data=static_data, ts_data=ts_data, locations=["IR", "US"])
    wanted_ts_data = pd.DataFrame([{
        "feature_name": "death",
        "dt": "2020/01/01",
        "country_code": "IR"
    }, {
        "feature_name": "death",
        "dt": "2020/01/01",
        "country_code": "US"
    }])
    wanted_static_data = pd.DataFrame([{
        "population": 70,
        "country_code": "IR"
    }, {
        "population": 50,
        "country_code": "US"
    }])

    pd.testing.assert_frame_equal(got_ts_data, wanted_ts_data)
    pd.testing.assert_frame_equal(got_static_data, wanted_static_data)

  def test_static_feature_map_for_locations(self):
    static_data = {
        "population": {
            "US": 100,
            "CH": 200,
            "IT": 30,
        },
        "land_area": {
            "US": 100,
            "CH": 150,
        }
    }
    static_feature_specs = [
        model_spec.FeatureSpec(name="population", initializer=None),
        model_spec.FeatureSpec(name="land_area", initializer=None),
    ]
    locations = ["US", "CH"]
    expected = np.array([[100, 100], [200, 150]])
    actual = feature_utils.static_feature_to_dense(static_data,
                                                   static_feature_specs,
                                                   locations)
    self.assertAllEqual(actual, expected)

  def test_covariates_as_tensors_for_location(self):
    ts_data = {
        "temperature": {
            "US": [70.5, 73.0],
            "CH": [72.5, 75.3],
        },
        "mobility": {
            "US": [98.4, 70.1],
            "CH": [73.5, 65.3],
            "IT": [83.5, 65.0],
        }
    }
    covariate_feature_specs = [
        model_spec.FeatureSpec(name="temperature", initializer=None),
        model_spec.FeatureSpec(name="mobility", initializer=None),
    ]
    expected = [
        np.array([[70.5, 98.4], [72.5, 73.5]]),
        np.array([[73.0, 70.1], [75.3, 65.3]])
    ]
    actual = feature_utils.covariate_features_to_dense(ts_data,
                                                       covariate_feature_specs,
                                                       ["US", "CH"], 2)
    self.assertAllClose(actual, expected)

  def test_covariates_as_tensors_for_location_filters_nones(self):
    ts_data = {
        "temperature": {
            "US": [70.5, 73.0],
            "CH": [72.5, 75.3],
        },
        "mobility": {
            "US": [98.4, 70.1],
            "CH": [73.5, 65.3],
            "IT": [83.5, 65.0],
        },
        "humidity": {
            "US": [34.3, 38.2],
            "CH": [44.2, 42.4],
            "IT": None,
        }
    }
    covariate_feature_specs = [
        model_spec.FeatureSpec(name="temperature", initializer=None),
        model_spec.FeatureSpec(name="mobility", initializer=None),
        model_spec.FeatureSpec(name="humidity", initializer=None),
    ]
    expected = [
        np.array([[70.5, 98.4, 34.3], [72.5, 73.5, 44.2]]),
        np.array([[73.0, 70.1, 38.2], [75.3, 65.3, 42.4]])
    ]
    actual = feature_utils.covariate_features_to_dense(ts_data,
                                                       covariate_feature_specs,
                                                       ["US", "CH"], 2)
    self.assertAllClose(actual, expected)

  def test_get_categorical_features_mask_for_ts(self):
    covariate_feature_specs = [
        model_spec.FeatureSpec(name="temperature", initializer=None),
        model_spec.FeatureSpec(name="mobility", initializer=None),
        model_spec.FeatureSpec(name="chc_npi_School", initializer=None),
    ]
    categorical_features = ["chc_npi_School"]
    expected = tf.constant(np.array([[0, 0, 1], [0, 0, 1]]))
    actual = feature_utils.get_categorical_features_mask(
        covariate_feature_specs, categorical_features, 2, is_static=False)
    self.assertAllClose(actual, expected)

  def test_periodic_forecast_works_for_weekly_case(self):
    example_periods = np.array([
        [2, 3, 4, 5, 6, 7, 1],
        [0, 0, 0, 0, 1, 1, 0],
    ])
    expected_output = np.array([
        [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    ])
    output_array = np.empty((2, 14))
    feature_utils.periodic_forecast(example_periods, output_array, period=7)
    np.testing.assert_equal(output_array, expected_output)

  def test_periodic_forecast_works_shorter_than_period(self):
    num_locations = 5
    num_days = 4
    start_day = 2
    short_example = np.arange(start_day, start_day + num_days)
    most_recent_days = np.tile(short_example, (num_locations, 1))
    most_recent_days[-1, :] += 1
    output_array = np.empty((num_locations, 5))
    expected_output = np.tile([4, 5, 2, 3, 4], (num_locations, 1))
    expected_output[-1, :] += 1
    feature_utils.periodic_forecast(most_recent_days, output_array, period=6)
    np.testing.assert_equal(output_array, expected_output)


class TestForecastFeatures(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.covariates = np.tile(np.arange(28).reshape((-1, 1, 1)), (1, 3, 4))
    cls.feature_specs = {
        "none":
            model_spec.FeatureSpec(
                "none", forecast_method=model_spec.ForecastMethod.NONE),
        "constant":
            model_spec.FeatureSpec(
                "constant", forecast_method=model_spec.ForecastMethod.CONSTANT),
        "week":
            model_spec.FeatureSpec(
                "week",
                forecast_method=model_spec.ForecastMethod.PERIODIC_WEEKLY),
        "xg":
            model_spec.FeatureSpec(
                "xg", forecast_method=model_spec.ForecastMethod.XGBOOST),
    }
    _, n_locations, n_features = cls.covariates.shape

    cls.expected_output = np.zeros((10, n_locations, n_features))
    cls.expected_output[:, :, 1] = 27
    cls.expected_output[:7, :, 2] = np.arange(21, 28).reshape(-1, 1)
    cls.expected_output[7:, :, 2] = np.arange(21, 24).reshape(-1, 1)
    cls.expected_output[:, :, 3] = 1.0

  def test_forecast_features_same_length(self):
    output_time_points = 5
    forecasts = feature_utils.forecast_features(
        self.covariates,
        self.feature_specs,
        num_forecast_steps=output_time_points,
    )
    _, n_locations, n_features = self.covariates.shape
    self.assertTupleEqual(forecasts.shape,
                          (output_time_points, n_locations, n_features))
    np.testing.assert_allclose(forecasts,
                               self.expected_output[:output_time_points, :, :])

  def test_forecast_features_forecast_shorter(self):
    output_time_points = 3
    _, n_locations, n_features = self.covariates.shape
    expected_output = np.zeros((output_time_points, n_locations, n_features))
    expected_output[:, :, 1] = 27
    expected_output[:, :, 2] = np.arange(21, 24).reshape(-1, 1)
    expected_output[:, :, 3] = 1.0
    forecasts = feature_utils.forecast_features(
        self.covariates,
        self.feature_specs,
        num_forecast_steps=output_time_points,
    )
    _, n_locations, n_features = self.covariates.shape
    self.assertTupleEqual(forecasts.shape,
                          (output_time_points, n_locations, n_features))
    np.testing.assert_allclose(forecasts,
                               self.expected_output[:output_time_points, :, :])

  def test_forecast_features_num_threads(self):
    output_time_points = 5
    forecasts = feature_utils.forecast_features(
        self.covariates,
        self.feature_specs,
        num_forecast_steps=output_time_points,
        num_threads=1,
    )
    forecasts_parallel = feature_utils.forecast_features(
        self.covariates,
        self.feature_specs,
        num_forecast_steps=output_time_points,
        num_threads=2,
    )
    _, n_locations, n_features = self.covariates.shape
    self.assertTupleEqual(forecasts.shape,
                          (output_time_points, n_locations, n_features))
    np.testing.assert_allclose(forecasts,
                               self.expected_output[:output_time_points, :, :])
    np.testing.assert_allclose(forecasts, forecasts_parallel)

  def test_read_from_a_project_raises_if_no_projects_specified(self):
    with self.assertRaises(ValueError):
      feature_utils.read_from_a_project("SELECT * from FAKE", [])

  @mock.patch.object(
      pd, "read_gbq", side_effect=gbq.NotFoundException, autospec=pd.read_gbq)
  def test_read_from_a_project_raises_if_not_found_in_any_projects(
      self, read_mock):
    with self.assertRaises(gbq.NotFoundException):
      feature_utils.read_from_a_project("SELECT * from FAKE",
                                        ["test_project", "test_project_2"])

    read_mock.assert_has_calls([
        mock.call("SELECT * from FAKE", project_id="test_project"),
        mock.call("SELECT * from FAKE", project_id="test_project_2")
    ])

  @mock.patch.object(
      pd,
      "read_gbq",
      side_effect=[gbq.NotFoundException,
                   pd.DataFrame([1])],
      autospec=pd.read_gbq)
  def test_read_from_a_project_can_handle_missing_project(self, read_mock):
    output_df = feature_utils.read_from_a_project(
        "SELECT * from FAKE", ["test_project", "test_project_2"])

    read_mock.assert_has_calls([
        mock.call("SELECT * from FAKE", project_id="test_project"),
        mock.call("SELECT * from FAKE", project_id="test_project_2")
    ])
    pd.testing.assert_frame_equal(output_df, pd.DataFrame([1]))


if __name__ == "__main__":
  tf.test.main()
