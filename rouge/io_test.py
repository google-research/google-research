# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
from pathlib import Path

from absl.testing import absltest
from rouge_score import io
from rouge_score import rouge_scorer
from rouge_score import scoring
from rouge_score import test_util


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

  def testGlobPairsFilesByMatchingWildcardIdentifiers(self):
    # Lexical sort produces a different order for 1, 10, ..., 2 if only
    # one side zero-pads ids. The files must be matched by actual id.
    with tempfile.TemporaryDirectory() as directory:
      for number in range(1, 13):
        line = "unique%d token%d only%d\n" % (number, number, number)
        (Path(directory) / ("doc%d.targets" % number)).write_text(line)
        (Path(directory) / ("doc%d.decodes" % number)).write_text(line)
      out = Path(directory) / "scores.csv"
      io.compute_scores_and_write_to_csv(
          str(Path(directory) / "*.targets"),
          str(Path(directory) / "*.decodes"),
          str(out),
          rouge_scorer.RougeScorer(["rouge1"], False), None)
      rows = out.read_text().splitlines()
      self.assertLen(rows, 13)
      self.assertTrue(all(float(row.split(",")[-1]) == 1.0
                          for row in rows[1:]))

  def testRejectsSameCountButMismatchedWildcardIdentifiers(self):
    with tempfile.TemporaryDirectory() as directory:
      for number in range(1, 13):
        line = "unique%d token%d only%d\n" % (number, number, number)
        (Path(directory) / ("ref_%d.txt" % number)).write_text(line)
        (Path(directory) / ("out_%02d.txt" % number)).write_text(line)
      out = Path(directory) / "scores.csv"
      with self.assertRaisesRegex(ValueError, "matching wildcard"):
        io.compute_scores_and_write_to_csv(
            str(Path(directory) / "ref_*.txt"),
            str(Path(directory) / "out_*.txt"),
            str(out),
            rouge_scorer.RougeScorer(["rouge1"], False), None)
      self.assertFalse(out.exists())

  def testRejectsDifferentIdsWithEqualFileCounts(self):
    with tempfile.TemporaryDirectory() as directory:
      for target, prediction in zip(("a", "b"), ("a", "c")):
        (Path(directory) / (target + ".targets")).write_text("same\n")
        (Path(directory) / (prediction + ".decodes")).write_text("same\n")
      with self.assertRaisesRegex(ValueError, "matching wildcard"):
        io.compute_scores_and_write_to_csv(
            str(Path(directory) / "*.targets"),
            str(Path(directory) / "*.decodes"),
            str(Path(directory) / "scores.csv"),
            rouge_scorer.RougeScorer(["rouge1"], False), None)

  def testWildcardKeysAlsoSupportQuestionMarksAndClasses(self):
    with tempfile.TemporaryDirectory() as directory:
      for number in (1, 2):
        line = "word%d match%d\n" % (number, number)
        (Path(directory) / ("ref_%d.targets" % number)).write_text(line)
        (Path(directory) / ("out_%d.decodes" % number)).write_text(line)
      output_file = Path(directory) / "scored.csv"
      io.compute_scores_and_write_to_csv(
          str(Path(directory) / "ref_[12].targets"),
          str(Path(directory) / "out_?.decodes"),
          str(output_file),
          rouge_scorer.RougeScorer(["rouge1"], False), None)
      scores = output_file.read_text().splitlines()[1:]
      self.assertLen(scores, 2)
      self.assertTrue(all(float(row.split(",")[-1]) == 1.0
                          for row in scores))

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
