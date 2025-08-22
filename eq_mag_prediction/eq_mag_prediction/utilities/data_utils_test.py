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

"""Tests for utilities.data_utils."""

import numpy as np
import pandas as pd

from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import time_conversions


class ComputationUtilityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Longitude = 180', 180.0, 180.0),
      ('Longitude = -180', -180.0, -180.0),
      ('Longitude = 181', 181.0, -179.0),
  )
  def test_longitude(self, lng_input, expected):
    self.assertEqual(data_utils.normalize_longitude(lng_input), expected)

  @parameterized.named_parameters(
      ('Lat, Lng = 90.0, 10.0', 90.0, 10.0, (90.0, 10.0)),
      ('Lat, Lng = -90.0, -10.0', -90.0, -10.0, (-90.0, -10.0)),
      ('Lat, Lng = -90.0, 270.0', -90.0, 270.0, (-90.0, -90.0)),
  )
  def test_latitude_longitude_normalization(self, lat, lng, expected):
    self.assertEqual(data_utils.normalize_lat_lng(lat, lng), expected)

  @parameterized.named_parameters(
      ('Lat, Lng = -91.0, 10.0', -91.0, 10.0),
      ('Lat, Lng = 91.0, -10.0', 91.0, -10.0),
      ('Lat, Lng = 0.0, 361', 0, 361),
      ('Lat, Lng = 0.0, -181', 0, -181),
  )
  def test_latitude_longitude_out_of_range(self, lat, lng):
    self.assertRaises(ValueError, data_utils.normalize_lat_lng, lat, lng)

  def test_concatenate_polyline(self):
    pline = [np.arange(2 * k).reshape(2, k) for k in [3, 10, 38]]
    result = data_utils.concatenate_polyline(pline)
    self.assertEqual(
        result.shape, (2, sum(line.shape[1] for line in pline) + len(pline) - 1)
    )

  def test_inside_circle(self):
    epsilon = 1e-3  # Safety margin for several usegaes
    x_center = 3
    y_center = 20
    # Define a theoretical and actual radii to consider for numerical errors
    circle_radius = 2 + epsilon
    # create points inside and outside of given circle to examine upon
    examine_radii = [1, 2, 3]
    examine_angles = np.linspace(0, 2 * np.pi - epsilon, 10)
    for r in examine_radii:
      for t in examine_angles:
        x_point = r * np.cos(t) + x_center
        y_point = r * np.sin(t) + y_center
        point = np.array([(x_point, y_point)])  # Create the point to examine
        is_point_inside_circle = data_utils.inside_circle(
            point, (x_center, y_center), circle_radius
        )
        should_be_inside = True if (r <= circle_radius) else False
        self.assertEqual(is_point_inside_circle, should_be_inside)

  def test_separate_repeating_times_in_catalog(self):
    rnd_seed = np.random.RandomState(seed=1905)
    dataframe_len = 100
    # define time vector with repetitions:
    time_vec = np.linspace(0, 300, 100)
    dt = time_vec[1] - time_vec[0]  # expected dt fot later verification
    repeating_value = time_vec[25]
    n_repeating = 5
    time_vec[25 : 25 + n_repeating] = (
        repeating_value  # create repeats in time vector
    )
    rnd_seed.shuffle(time_vec)
    repeating_elements = np.where(time_vec == repeating_value)[0]

    # create a simulating catalog:
    simulating_df = pd.DataFrame({
        'a': rnd_seed.uniform(size=(dataframe_len)),
        'b': rnd_seed.uniform(size=(dataframe_len)),
        'time': time_vec,
        'c': rnd_seed.uniform(size=(dataframe_len)),
        'd': rnd_seed.uniform(size=(dataframe_len)),
    })

    orders_of_magnitude = 4
    df_no_repating_time = data_utils.separate_repeating_times_in_catalog(
        simulating_df, orders_of_magnitude=orders_of_magnitude
    )

    # verify columns other than 'time' haven't changed
    self.assertTrue(
        (
            simulating_df[['a', 'b', 'c', 'd']].values
            == df_no_repating_time[['a', 'b', 'c', 'd']].values
        ).all()
    )

    # verify all time values are now unique:
    self.assertEqual(
        np.unique(df_no_repating_time.time.values).size, dataframe_len
    )

    # verify shifted values are in expected margin:
    new_time_values = df_no_repating_time.time.values[repeating_elements]
    time_margin = dt / (10**orders_of_magnitude)
    min_val = repeating_value - time_margin
    self.assertEqual(min_val, new_time_values.min())
    max_val = repeating_value + time_margin
    self.assertEqual(max_val, new_time_values.max())

    # verify non repeating items remain untouched
    remaining_original = np.delete(
        simulating_df.time.values, repeating_elements
    )
    remaining_fixed = np.delete(
        df_no_repating_time.time.values, repeating_elements
    )
    self.assertTrue((remaining_original == remaining_fixed).all())

  def test_smear_binned_magnitudes(self):
    # JMA catalog is known to have discretization of 0.1:
    jma_catalog = data_utils.jma_dataframe()
    jma_smear_magnitude = data_utils.smear_binned_magnitudes(jma_catalog)

    min_diff = np.diff(np.unique(jma_catalog.magnitude)).min()
    new_min_diff = np.diff(np.unique(jma_smear_magnitude.magnitude)).min()
    self.assertLess(new_min_diff, min_diff)

    # Histograms should be equal after re-binning the data:
    bins = np.arange(
        jma_catalog.magnitude.min() - 3 * min_diff / 2,
        jma_catalog.magnitude.max() + 3 * min_diff / 2,
        min_diff,
    )
    old_hist, _ = np.histogram(jma_catalog.magnitude.values, bins)
    new_hist, _ = np.histogram(jma_smear_magnitude.magnitude.values, bins)
    np.testing.assert_equal(old_hist, new_hist)


class DataRetrievalUtilityTest(parameterized.TestCase):

  def test_jma_dataframe(self):
    data_frame = data_utils.jma_dataframe()
    self.assertEqual(
        time_conversions.time_to_datetime_japan(data_frame.time.min()).year,
        1922,
    )
    self.assertEqual(
        time_conversions.time_to_datetime_japan(data_frame.time.max()).year,
        2020,
    )

    # As an example, we check that the details of the Tohoku earthquake are ok.
    tohoku_earthquake = data_frame.loc[data_frame.magnitude.idxmax()]
    date_time = time_conversions.time_to_datetime_utc(tohoku_earthquake.time)
    self.assertEqual(date_time.year, 2011)
    self.assertEqual(date_time.month, 3)
    self.assertEqual(date_time.day, 11)
    self.assertEqual(date_time.hour, 5)
    self.assertAlmostEqual(tohoku_earthquake.time_std, 2.6)
    self.assertAlmostEqual(tohoku_earthquake.latitude, 38.1035)
    self.assertAlmostEqual(tohoku_earthquake.latitude_std, 0.09333333333)
    self.assertAlmostEqual(tohoku_earthquake.longitude, 142.861)
    self.assertAlmostEqual(tohoku_earthquake.longitude_std, 0.145)
    self.assertAlmostEqual(tohoku_earthquake.depth, 23.74)
    self.assertAlmostEqual(tohoku_earthquake.magnitude, 9.0)


if __name__ == '__main__':
  absltest.main()
