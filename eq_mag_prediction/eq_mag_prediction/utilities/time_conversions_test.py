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

"""Tests for utilities.time_conversions."""

import pytz
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import time_conversions


class TimeConversionsTest(parameterized.TestCase):

  def test_time_to_bytes_utc(self):
    self.assertEqual(
        time_conversions.time_to_bytes_utc(1351090801), b'2012-10-24T15:00:01'
    )
    self.assertEqual(
        time_conversions.time_to_bytes_utc(1351090801.042),
        b'2012-10-24T15:00:01.042000',
    )


  def test_time_to_datetime(self):
    date_time = time_conversions.time_to_datetime(
        1351090801, pytz.timezone('UTC')
    )
    self.assertEqual(date_time.year, 2012)
    self.assertEqual(date_time.month, 10)
    self.assertEqual(date_time.day, 24)
    self.assertEqual(date_time.hour, 15)
    self.assertEqual(date_time.minute, 0)
    self.assertEqual(date_time.second, 1)

  def test_datetime_to_time(self):
    self.assertEqual(
        time_conversions.datetime_to_time(
            pytz.timezone('UTC'),
            year=2012,
            month=10,
            day=24,
            hour=15,
            minute=0,
            second=1,
        ),
        1351090801,
    )

  @parameterized.parameters(1, 54321, 123456789, 11223344556)
  def test_time_to_datetime_and_back(self, timestamp):
    conversion_functions_pair = (
        (
            time_conversions.time_to_datetime_japan,
            time_conversions.datetime_japan_to_time,
        ),
        (
            time_conversions.time_to_datetime_pst,
            time_conversions.datetime_pst_to_time,
        ),
    )
    for (
        time_to_datetime_zone,
        datetime_zone_to_time,
    ) in conversion_functions_pair:
      date_time = time_to_datetime_zone(timestamp)
      self.assertEqual(
          timestamp,
          datetime_zone_to_time(
              date_time.year,
              date_time.month,
              date_time.day,
              date_time.hour,
              date_time.minute,
              date_time.second,
          ),
      )


if __name__ == '__main__':
  absltest.main()
