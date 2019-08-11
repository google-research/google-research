# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for metrics.py libraries.

There are two kinds of tests being run in this file. The first one is comparison
of metrics calculated for oracle predictions. The second is comparison of
metrics calculated for a known prediction with the known ground truth values.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import unittest
from schema_guided_dst import metrics

ACCURACY_METRICS = [
    metrics.AVERAGE_GOAL_ACCURACY,
    metrics.AVERAGE_CAT_ACCURACY,
    metrics.AVERAGE_NONCAT_ACCURACY,
    metrics.JOINT_GOAL_ACCURACY,
    metrics.JOINT_CAT_ACCURACY,
    metrics.JOINT_NONCAT_ACCURACY,
]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class MetricsTest(unittest.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    data_file = os.path.join(THIS_DIR, "test_data", "metrics_test_refdata.json")
    with open(data_file) as f:
      test_data = json.load(f)
    self.frame_ref = test_data["frame_ref"]
    self.frame_hyp = test_data["frame_hyp"]
    self.known_metrics = test_data["metrics_hyp"]
    self.utterance = test_data["utterance"]

    schema_file = os.path.join(THIS_DIR, "test_data",
                               "metrics_test_refschema.json")
    with open(schema_file) as f:
      self.schema = json.load(f)

  def _assert_dicts_almost_equal(self, ref_dict, other_dict):
    self.assertCountEqual(ref_dict.keys(), other_dict.keys())
    for metric in ref_dict.keys():
      self.assertAlmostEqual(ref_dict[metric], other_dict[metric])

  def test_active_intent_accuracy(self):
    # (1) Test on oracle frame.
    intent_acc_oracle = metrics.get_active_intent_accuracy(
        self.frame_ref, self.frame_ref)
    self.assertAlmostEqual(1.0, intent_acc_oracle)

    # (2) Test on a previously known frame.
    intent_acc_hyp = metrics.get_active_intent_accuracy(self.frame_ref,
                                                        self.frame_hyp)
    self.assertAlmostEqual(self.known_metrics[metrics.ACTIVE_INTENT_ACCURACY],
                           intent_acc_hyp)

  def test_slot_tagging_f1(self):
    # (1) Test on oracle frame.
    slot_tagging_f1_oracle = metrics.get_slot_tagging_f1(
        self.frame_ref, self.frame_ref, self.utterance, self.schema)
    # Ground truth values for oracle prediction are all 1.0.
    self._assert_dicts_almost_equal(
        {k: 1.0 for k in self.known_metrics[metrics.SLOT_TAGGING_F1]},
        slot_tagging_f1_oracle._asdict())

    # (2) Test on a previously known frame.
    slot_tagging_f1_hyp = metrics.get_slot_tagging_f1(self.frame_ref,
                                                      self.frame_hyp,
                                                      self.utterance,
                                                      self.schema)
    self._assert_dicts_almost_equal(self.known_metrics[metrics.SLOT_TAGGING_F1],
                                    slot_tagging_f1_hyp._asdict())

  def test_requested_slots_f1(self):
    # (1) Test on oracle frame.
    requestable_slots_f1_oracle = metrics.get_requested_slots_f1(
        self.frame_ref, self.frame_ref)
    # Ground truth values for oracle prediction are all 1.0.
    self._assert_dicts_almost_equal(
        {k: 1.0 for k in self.known_metrics[metrics.REQUESTED_SLOTS_F1]},
        requestable_slots_f1_oracle._asdict())

    # (2) Test on a previously known frame.
    requested_slots_f1_hyp = metrics.get_requested_slots_f1(
        self.frame_ref, self.frame_hyp)
    self._assert_dicts_almost_equal(
        self.known_metrics[metrics.REQUESTED_SLOTS_F1],
        requested_slots_f1_hyp._asdict())

  def test_average_and_joint_goal_accuracy(self):
    # (1) Test on oracle frame.
    goal_accuracy_oracle = metrics.get_average_and_joint_goal_accuracy(
        self.frame_ref, self.frame_ref, self.schema)
    # Ground truth values for oracle prediction are all 1.0.
    self._assert_dicts_almost_equal({k: 1.0 for k in ACCURACY_METRICS},
                                    goal_accuracy_oracle)

    # (2) Test on a previously known frame.
    goal_accuracy_hyp = metrics.get_average_and_joint_goal_accuracy(
        self.frame_ref, self.frame_hyp, self.schema)
    self._assert_dicts_almost_equal(
        {k: self.known_metrics[k] for k in ACCURACY_METRICS}, goal_accuracy_hyp)


if __name__ == "__main__":
  unittest.main()
