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

_MISALIGNED_PREDICTION = """title | my table
argentina | brazil | time
200 | 158 | 1999
"""

_TARGET_SCATTER = """x | y
10 | 20
30 | 40
50 | 60
"""

_PREDICTION_SCATTER = """y | x
40 | 30
"""


class MetricsTest(absltest.TestCase):

  def test_get_table_datapoints(self):
    table = metrics._parse_table(_TARGET)
    self.assertDictEqual(metrics._get_table_datapoints(table), {
        "title": "my table",
        "1999 argentina": "200",
        "1999 brazil": "158",
    })

  def test_align_table(self):
    table, score = metrics._parse_table(_MISALIGNED_PREDICTION).aligned(
        ("year", "argentina", "brazil"))
    self.assertTupleEqual(table.headers, ("time", "argentina", "brazil"))
    self.assertEqual(score, 1)

  def test_transposed_table(self):
    transposed_table = metrics._parse_table(_TRANSPOSED_TARGET, transposed=True)
    table = metrics._parse_table(_TARGET)
    self.assertTupleEqual(transposed_table.rows, table.rows)

  def test_row_datapoints_precision_recall(self):
    result = metrics.row_datapoints_precision_recall(
        [[_TARGET]], [_PREDICTION])
    self.assertDictEqual(result, {
        # 1 comes from a matched title, 99 from the argentina entry.
        # Result comes from transposing both tables and considering columns.
        "row_datapoints_precision": 100 * 1.99 / 3,
        "row_datapoints_recall": 100 * 1.99 / 3,
        "row_datapoints_f1": 100 * 1.99 / 3,
    })

  def test_row_datapoints_scatter(self):
    result = metrics.row_datapoints_precision_recall(
        [[_TARGET_SCATTER]], [_PREDICTION_SCATTER])
    self.assertAlmostEqual(result["row_datapoints_precision"], 100)
    self.assertAlmostEqual(result["row_datapoints_recall"], 100.0 / 3)
    self.assertAlmostEqual(result["row_datapoints_f1"], 100 / 2.0)

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
