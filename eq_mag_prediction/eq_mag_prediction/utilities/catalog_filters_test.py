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

"""Tests for utilities.catalog_filters."""

import functools

import numpy as np
import pandas as pd

from absl.testing import absltest
from eq_mag_prediction.utilities import catalog_filters
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import test_utils as global_test_utils
from eq_mag_prediction.utilities import time_conversions


class CatalogFiltersTest(absltest.TestCase):

  def test_limit_catalog_to_square(self):
    catalog = global_test_utils.sample_catalog().copy()
    center = geometry.Point(143, 39)
    side = 1.5
    limited = catalog_filters.limit_catalog_to_square(catalog, center, 2 * side)
    complement = catalog[(
        (catalog.longitude >= center.lng + side)
        | (catalog.longitude < center.lng - side)
        | (catalog.latitude >= center.lat + side)
        | (catalog.latitude < center.lat - side)
    )]

    self.assertNotEmpty(limited)
    self.assertTrue((limited.longitude >= center.lng - side).all())
    self.assertTrue((limited.longitude < center.lng + side).all())
    self.assertTrue((limited.latitude >= center.lat - side).all())
    self.assertTrue((limited.latitude < center.lat + side).all())
    self.assertLen(catalog, len(limited) + len(complement))

  def test_is_in_rectangle(self):
    catalog = global_test_utils.sample_catalog().copy()
    longitude_range = (142.5, 143.5)
    latitude_range = (38.5, 39.5)
    limited = catalog[
        catalog_filters.is_in_rectangle(
            catalog, longitude_range, latitude_range
        )
    ]

    self.assertNotEmpty(limited)
    self.assertTrue((limited.longitude >= longitude_range[0]).all())
    self.assertTrue((limited.longitude < longitude_range[1]).all())
    self.assertTrue((limited.latitude >= latitude_range[0]).all())
    self.assertTrue((limited.latitude < latitude_range[1]).all())

  def _create_test_catalog(self, coordinate_list):
    longitude, latitude = list(zip(*coordinate_list))
    magnitude = np.arange(len(coordinate_list))
    time = np.arange(len(coordinate_list))
    return pd.DataFrame(
        data={
            'longitude': longitude,
            'latitude': latitude,
            'magnitude': magnitude,
            'time': time,
        },
        columns=['longitude', 'latitude', 'magnitude', 'time'],
    )

  def test_is_in_polygon(self):
    external_coordinates = (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    )
    epsilon = 1e-2
    test_coordinates = (
        (0, 1 - epsilon),
        (1 - epsilon, 1 - epsilon),
        (1 - epsilon, 0),
        (1 - epsilon, -(1 - epsilon)),
        (0, -(1 - epsilon)),
        (-(1 - epsilon), -(1 - epsilon)),
        (-(1 - epsilon), 0),
        (-(1 - epsilon), 1 - epsilon),
    )
    polygon_longitude, polygon_latitude = list(zip(*external_coordinates))
    test_catalog = self._create_test_catalog(test_coordinates)
    # Test for a square
    in_polygon_logical = catalog_filters.is_in_polygon(
        test_catalog, polygon_longitude, polygon_latitude
    )
    expected_in = np.full_like(polygon_longitude, True)
    np.testing.assert_equal(in_polygon_logical, expected_in)
    # Test for a rhombus
    in_polygon_logical = catalog_filters.is_in_polygon(
        test_catalog, polygon_longitude[::2], polygon_latitude[::2]
    )
    expected_in = np.array([True, False]*4)
    np.testing.assert_equal(in_polygon_logical, expected_in)

  def test_criterion(self):
    lng_range = (141, 143.5)
    lat_range = (36.5, 38)
    max_depth = 40
    min_magnitude = 3.5
    start_timestamp = time_conversions.datetime_japan_to_time(1985)
    catalog = global_test_utils.sample_catalog().copy()
    criterion = functools.partial(
        catalog_filters.earthquake_criterion,
        longitude_range=lng_range,
        latitude_range=lat_range,
        start_timestamp=start_timestamp,
        max_depth=max_depth,
        min_magnitude=min_magnitude,
    )
    subcatalog = catalog[criterion(catalog)]

    self.assertGreater(len(subcatalog), 100)
    self.assertTrue(criterion(subcatalog).all())

    self.assertTrue((subcatalog.magnitude >= min_magnitude).all())
    self.assertTrue((subcatalog.depth <= max_depth).all())
    self.assertTrue((subcatalog.time >= start_timestamp).all())
    self.assertTrue((subcatalog.longitude >= lng_range[0]).all())
    self.assertTrue((subcatalog.longitude < lng_range[1]).all())
    self.assertTrue((subcatalog.latitude >= lat_range[0]).all())
    self.assertTrue((subcatalog.latitude < lat_range[1]).all())

  def test_criterion_with_end_time(self):
    catalog = global_test_utils.sample_catalog().copy()
    end_timestamp = time_conversions.datetime_japan_to_time(2000)
    criterion_w_endtime = functools.partial(
        catalog_filters.earthquake_criterion, end_timestamp=end_timestamp
    )
    subcatalog_w_endtime = catalog[criterion_w_endtime(catalog)]

    self.assertTrue((subcatalog_w_endtime.time <= end_timestamp).all())

  def test_return_entire_catalog_criterion(self):
    catalog = global_test_utils.sample_catalog().copy()
    filteres_catalog = catalog[
        catalog_filters.return_entire_catalog_criterion(catalog)
    ]
    pd.testing.assert_frame_equal(catalog, filteres_catalog)


if __name__ == '__main__':
  absltest.main()
