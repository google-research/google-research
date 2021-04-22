# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for aqt.tfevent_utils.

Note on test coverage / mutants: get_parsed_tfevents() and get_tfevent_paths()
are currently not covered by unit tests as setting up tfevent test files is
non-trivial. Their functionality has been tested on TFEvent files generated
by experiments, and these functions are unlikely to fail silently.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

from aqt.utils import tfevent_utils


class TfEventUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_duplicates',
          events=tfevent_utils.EventSeries(
              name='dummy',
              steps=onp.array([0, 1]),
              values=onp.array([0, 1]),
              wall_times=onp.array([0, 1]),
          ),
          exp=tfevent_utils.EventSeries(
              name='dummy',
              steps=onp.array([0, 1]),
              values=onp.array([0, 1]),
              wall_times=None,
          )),
      dict(
          testcase_name='one_duplicate',
          events=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0, 0]),
              values=onp.array([0, 1]),
              wall_times=onp.array([0, 1]),
          ),
          exp=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0]),
              values=onp.array([1]),
              wall_times=None,
          )),
      dict(
          testcase_name='one_duplicate_unsorted',
          events=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0, 0]),
              values=onp.array([1, 0]),
              wall_times=onp.array([1, 0]),
          ),
          exp=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0]),
              values=onp.array([1]),
              wall_times=None,
          )),
      dict(
          testcase_name='multiple_duplicates',
          events=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0, 0, 1, 2, 2]),
              values=onp.array([0, 1, 2, 4, 5]),
              wall_times=onp.array([0, 1, 2, 3, 4]),
          ),
          exp=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0, 1, 2]),
              values=onp.array([1, 2, 5]),
              wall_times=None,
          )),
      dict(
          testcase_name='multiple_duplicates_unsorted',
          events=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([2, 1, 0, 0, 2]),
              values=onp.array([4, 2, 1, 0, 5]),
              wall_times=onp.array([3, 2, 1, 0, 4]),
          ),
          exp=tfevent_utils.EventSeries(
              name='test_events',
              steps=onp.array([0, 1, 2]),
              values=onp.array([1, 2, 5]),
              wall_times=None,
          )),
  )
  def test_sort_and_deduplicate_entries(self, events, exp):
    res = tfevent_utils._sort_and_deduplicate_entries(events)
    self.assertEqual(res.name, exp.name)
    onp.testing.assert_array_equal(res.steps, exp.steps)
    onp.testing.assert_array_equal(res.values, exp.values)
    onp.testing.assert_array_equal(res.wall_times, exp.wall_times)

  def test_sort_and_deduplicate_entries_should_raise_error_when_wall_times_unsorted(
      self):
    events = tfevent_utils.EventSeries(
        name='test_events',
        steps=onp.array([0, 1]),
        values=onp.array([0, 1]),
        wall_times=onp.array([1, 0]))
    with self.assertRaises(ValueError):
      tfevent_utils._sort_and_deduplicate_entries(events)


if __name__ == '__main__':
  absltest.main()
