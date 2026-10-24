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

import os
import shutil
import tempfile

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

  def _write_corpus(self, directory, filenames_prefix, filenames_suffix,
                    ids, zero_pad):
    """Writes one file per id, each holding text unique to that id."""
    os.makedirs(directory)
    for example_id in ids:
      stem = "%0*d" % (2, example_id) if zero_pad else str(example_id)
      with open(os.path.join(directory, filenames_prefix + stem +
                             filenames_suffix), "w") as f:
        f.write("unique%d token%d only%d\n" % (example_id, example_id,
                                                example_id))

  def _score_corpus(self, target_dir, prediction_dir, target_glob,
                    prediction_glob):
    """Scores a corpus and returns the per-row rouge1 F-measures."""
    output_filename = os.path.join(target_dir, "scores.csv")
    scorer = rouge_scorer.RougeScorer(["rouge1"], False)
    io.compute_scores_and_write_to_csv(
        os.path.join(target_dir, target_glob),
        os.path.join(prediction_dir, prediction_glob), output_filename, scorer,
        None)
    with open(output_filename) as f:
      rows = f.read().strip().split("\n")[1:]
    return [float(row.split(",")[3]) for row in rows]

  def testPairsFilesByExampleIdDespiteSortOrderDifference(self):
    """Files are paired by id, not by position in a sorted listing.

    With ids 1..12 on one side and 01..12 on the other, lexical sort orders the
    two sides differently. Pairing by sorted position scored 11 of 12 examples
    against the wrong target, which drove the mean rouge1-F from 1.0 to 0.083.
    """
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory)
    target_dir = os.path.join(directory, "targets")
    prediction_dir = os.path.join(directory, "predictions")

    ids = range(1, 13)
    self._write_corpus(target_dir, "ref_", ".txt", ids, zero_pad=False)
    self._write_corpus(prediction_dir, "out_", ".txt", ids, zero_pad=True)

    fmeasures = self._score_corpus(target_dir, prediction_dir, "ref_*.txt",
                                   "out_*.txt")

    self.assertLen(fmeasures, 12)
    for fmeasure in fmeasures:
      self.assertAlmostEqual(fmeasure, 1.0, places=6)

  def testPairsTargetsAndDecodesSuffixes(self):
    """The documented *.targets / *.decodes pairing still works."""
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory)
    target_dir = os.path.join(directory, "targets")
    prediction_dir = os.path.join(directory, "predictions")

    ids = range(1, 13)
    self._write_corpus(target_dir, "doc", ".targets", ids, zero_pad=False)
    self._write_corpus(prediction_dir, "doc", ".decodes", ids, zero_pad=False)

    fmeasures = self._score_corpus(target_dir, prediction_dir, "*.targets",
                                   "*.decodes")

    self.assertLen(fmeasures, 12)
    for fmeasure in fmeasures:
      self.assertAlmostEqual(fmeasure, 1.0, places=6)

  def testRaisesWhenExampleIdsDoNotMatch(self):
    """Equal file counts with different ids is an error, not a silent score."""
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory)
    target_dir = os.path.join(directory, "targets")
    prediction_dir = os.path.join(directory, "predictions")

    self._write_corpus(target_dir, "ref_", ".txt", range(1, 13),
                       zero_pad=False)
    self._write_corpus(prediction_dir, "other_", ".txt", range(1, 13),
                       zero_pad=False)

    scorer = rouge_scorer.RougeScorer(["rouge1"], False)
    with self.assertRaises(ValueError):
      io.compute_scores_and_write_to_csv(
          os.path.join(target_dir, "*.txt"),
          os.path.join(prediction_dir, "*.txt"),
          os.path.join(directory, "scores.csv"), scorer, None)

  def testPairKeyNormalizesZeroPadding(self):
    """Ids compare equal with and without zero padding, and are order stable."""
    self.assertEqual(io._pair_key("ref_*.txt", "ref_07.txt"),
                     io._pair_key("out_*.txt", "out_7.txt"))

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
