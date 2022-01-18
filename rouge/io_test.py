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

"""Tests for rouge input/output library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import absltest
from rouge import io
from rouge import rouge_scorer
from rouge import scoring
from rouge import test_util


class IoTest(absltest.TestCase):

  def testProducesValidOutput(self):
    with tempfile.NamedTemporaryFile() as output_file:
      output_filename = output_file.name
      scorer = rouge_scorer.RougeScorer(["rouge1"], False)
      io.compute_scores_and_write_to_csv(test_util.TARGETS_FILE,
                                         test_util.PREDICTIONS_FILE,
                                         output_filename, scorer,
                                         scoring.BootstrapAggregator())
      with open(output_filename) as f:
        csv_lines = f.readlines()
      output_types = tuple((line.split(",")[0] for line in csv_lines))
      self.assertEqual(output_types[0], "score_type")
      self.assertSameElements(output_types[1:],
                              ["rouge1-P", "rouge1-R", "rouge1-F"])

  def testUnAggregated(self):
    with tempfile.NamedTemporaryFile() as output_file:
      output_filename = output_file.name
      scorer = rouge_scorer.RougeScorer(["rouge1"], False)
      io.compute_scores_and_write_to_csv(test_util.TARGETS_FILE,
                                         test_util.PREDICTIONS_FILE,
                                         output_filename, scorer, None)
      with open(output_filename) as f:
        csv_lines = f.readlines()
      ids = tuple((line.split(",")[0] for line in csv_lines))
      self.assertEqual(ids[0], "id")
      self.assertLen(csv_lines, 3)

  def testDelimitedFile(self):
    with tempfile.NamedTemporaryFile() as output_file:
      output_filename = output_file.name
      scorer = rouge_scorer.RougeScorer(["rouge1"], False)
      io.compute_scores_and_write_to_csv(
          test_util.DELIMITED_FILE,
          test_util.DELIMITED_FILE,
          output_filename,
          scorer,
          None,
          delimiter=":")
      with open(output_filename) as f:
        csv_lines = f.readlines()
      ids = tuple((line.split(",")[0] for line in csv_lines))
      self.assertEqual(ids[0], "id")
      self.assertLen(csv_lines, 5)

  def testAssertsOnInvalidInputFiles(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"], False)
    with self.assertRaises(ValueError):
      io.compute_scores_and_write_to_csv("invalid*", "invalid*", "invalid",
                                         scorer, scoring.BootstrapAggregator())

  def testAssertsOnInvalidRougeTypes(self):
    scorer = rouge_scorer.RougeScorer(["rougex"], False)
    with self.assertRaises(ValueError):
      io.compute_scores_and_write_to_csv(test_util.TARGETS_FILE,
                                         test_util.PREDICTIONS_FILE, "", scorer,
                                         scoring.BootstrapAggregator())


if __name__ == "__main__":
  absltest.main()
