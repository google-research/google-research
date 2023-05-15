# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Tests for DePlot metrics."""

from absl.testing import absltest

from deplot import metrics


_TARGET = """title | my table
year | argentina | brazil
1999 | 200 | 158
"""

_TRANSPOSED_TARGET = """title | my table
country | 1999
argentina | 200
brazil | 158
"""

_PREDICTION = """title | my table
year | argentina | brazil
1999 | 202 | 0
"""


class MetricsTest(absltest.TestCase):

  def test_get_table_datapoints(self):
    self.assertDictEqual(metrics._get_table_datapoints(_TARGET), {
        "title": "my table",
        "1999 argentina": "200",
        "1999 brazil": "158",
    })

  def test_get_datapoint_metric(self):
    self.assertEqual(metrics._get_datapoint_metric(
        ("1999 argentina", "200"), ("1999 argentina", "202")), 0.99)

  def test_table_datapoints_precision_recall(self):
    result = metrics.table_datapoints_precision_recall(
        [[_TARGET]], [_PREDICTION])
    transposed_result = metrics.table_datapoints_precision_recall(
        [[_TRANSPOSED_TARGET]], [_PREDICTION])
    self.assertDictEqual(result, {
        # 1 comes from a matched title, 99 from the argentina entry.
        "table_datapoints_precision": 100 * 1.99 / 3,
        "table_datapoints_recall": 100 * 1.99 / 3,
        "table_datapoints_f1": 100 * 1.99 / 3,
    })
    self.assertDictEqual(transposed_result, result)

  def test_table_numbers_match(self):
    self.assertEqual(
        metrics._table_numbers_match("0.5 | foo | 0.6", "0.5 | foo | 0.6"),
        1)

    # This is 1 - relative_distance(0.4, 0.5) / 2. Note that the 0.6 is ignored.
    self.assertEqual(
        metrics._table_numbers_match("baz | foo | 0.4", "0.5 | foo | 0.6"),
        0.875)


if __name__ == "__main__":
  absltest.main()
