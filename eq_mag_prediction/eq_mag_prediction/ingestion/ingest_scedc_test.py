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

"""Tests for ingestion.ingest_scedc."""

from absl.testing import absltest
from eq_mag_prediction.ingestion import ingest_scedc


class IngestScedcTest(absltest.TestCase):

  def test_parse_line(self):
    line = (
        '1111/11/11 11:11:11.11 eq  l 1.11 h    11.111 -222.222  -0.1 C  '
        '1111111    1   11'
    )
    parsed = ingest_scedc._parse_line(line)
    expected = {
        'time': -27080311729,
        'event_type': 'eq',
        'geographical_type': 'l',
        'magnitude': 1.11,
        'magnitude_type': 'h',
        'latitude': 11.111,
        'longitude': -222.222,
        'depth': -0.1,
        'location_quality': 'C',
        'event_id': 1111111,
        'picked_phases': 1,
        'seismograms': 11,
    }
    self.assertDictEqual(parsed, expected)

  def test_parse_line_error(self):
    line = (
        '1111/11/11 11:87:11.11 eq  l 1.11 h    11.111 -222.222  -0.1 C '
        ' 1111111    1   11'
    )
    with self.assertRaisesRegex(ValueError, 'does not match format'):
      ingest_scedc._parse_line(line)


if __name__ == '__main__':
  absltest.main()
