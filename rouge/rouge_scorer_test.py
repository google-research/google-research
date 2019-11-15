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

# Lint as: python2, python3
"""Tests for rouge scorer.

Tests for both correctness and for consistency with the official ROUGE-1.5.5
implementation.

"Ground truth" scores are taken from manual runs of ROUGE-1.5.5.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
from rouge import rouge_scorer
from rouge import test_util


class RougeScorerTest(parameterized.TestCase):

  def setUp(self):
    super(RougeScorerTest, self).setUp()
    with open(test_util.TARGETS_FILE) as f:
      self.targets = f.readlines()
    with open(test_util.PREDICTIONS_FILE) as f:
      self.predictions = f.readlines()

  @parameterized.parameters(["rougen", "rouge0", "rouge10"])
  def testInvalidRougeTypes(self, rouge_type):
    with self.assertRaises(ValueError):
      scorer = rouge_scorer.RougeScorer([rouge_type])
      scorer.score("testing one two", "testing")

  @parameterized.parameters(["rouge1", "rouge9", "rougeL", "rougeLsum"])
  def testValidRougeTypes(self, rouge_type):
    scorer = rouge_scorer.RougeScorer([rouge_type])
    result = scorer.score("testing one two", "testing")
    self.assertSameElements(list(result.keys()), [rouge_type])

  def testRouge1(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    result = scorer.score("testing one two", "testing")
    self.assertAlmostEqual(1, result["rouge1"].precision)
    self.assertAlmostEqual(1 / 3, result["rouge1"].recall)
    self.assertAlmostEqual(1 / 2, result["rouge1"].fmeasure)

  @parameterized.parameters(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  def testRougeEmpty(self, rouge_type):
    scorer = rouge_scorer.RougeScorer([rouge_type])
    result = scorer.score("testing one two", "")
    self.assertAlmostEqual(0, result[rouge_type].precision)
    self.assertAlmostEqual(0, result[rouge_type].recall)
    self.assertAlmostEqual(0, result[rouge_type].fmeasure)

  def testRouge2(self):
    scorer = rouge_scorer.RougeScorer(["rouge2"])
    result = scorer.score("testing one two", "testing one")
    self.assertAlmostEqual(1, result["rouge2"].precision)
    self.assertAlmostEqual(1 / 2, result["rouge2"].recall)
    self.assertAlmostEqual(2 / 3, result["rouge2"].fmeasure)

  def testRougeLConsecutive(self):
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    result = scorer.score("testing one two", "testing one")
    self.assertAlmostEqual(1, result["rougeL"].precision)
    self.assertAlmostEqual(2 / 3, result["rougeL"].recall)
    self.assertAlmostEqual(4 / 5, result["rougeL"].fmeasure)

  def testRougeLNonConsecutive(self):
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    result = scorer.score("testing one two", "testing two")
    self.assertAlmostEqual(1, result["rougeL"].precision)
    self.assertAlmostEqual(2 / 3, result["rougeL"].recall)
    self.assertAlmostEqual(4 / 5, result["rougeL"].fmeasure)

  def testMultipleRougeTypes(self):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    result = scorer.score("testing one two", "testing one")
    self.assertSameElements(list(result.keys()), ["rouge1", "rougeL"])
    self.assertAlmostEqual(1, result["rouge1"].precision)
    self.assertAlmostEqual(2 / 3, result["rouge1"].recall)
    self.assertAlmostEqual(4 / 5, result["rouge1"].fmeasure)
    self.assertAlmostEqual(1, result["rougeL"].precision)
    self.assertAlmostEqual(2 / 3, result["rougeL"].recall)
    self.assertAlmostEqual(4 / 5, result["rougeL"].fmeasure)

  def testRouge1AgainstRouge155(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.40741, result["rouge1"].recall, 5)
    self.assertAlmostEqual(0.68750, result["rouge1"].precision, 5)
    self.assertAlmostEqual(0.51163, result["rouge1"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.40476, result["rouge1"].recall, 5)
    self.assertAlmostEqual(0.65385, result["rouge1"].precision, 5)
    self.assertAlmostEqual(0.50000, result["rouge1"].fmeasure, 5)

  def testRouge1AgainstRouge155WithStemming(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.40741, result["rouge1"].recall, 5)
    self.assertAlmostEqual(0.68750, result["rouge1"].precision, 5)
    self.assertAlmostEqual(0.51163, result["rouge1"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.42857, result["rouge1"].recall, 5)
    self.assertAlmostEqual(0.69231, result["rouge1"].precision, 5)
    self.assertAlmostEqual(0.52941, result["rouge1"].fmeasure, 5)

  def testRouge2AgainstRouge155(self):
    scorer = rouge_scorer.RougeScorer(["rouge2"])
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.30769, result["rouge2"].recall, 5)
    self.assertAlmostEqual(0.53333, result["rouge2"].precision, 5)
    self.assertAlmostEqual(0.39024, result["rouge2"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.29268, result["rouge2"].recall, 5)
    self.assertAlmostEqual(0.48000, result["rouge2"].precision, 5)
    self.assertAlmostEqual(0.36364, result["rouge2"].fmeasure, 5)

  def testRouge2AgainstRouge155WithStemming(self):
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.30769, result["rouge2"].recall, 5)
    self.assertAlmostEqual(0.53333, result["rouge2"].precision, 5)
    self.assertAlmostEqual(0.39024, result["rouge2"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.29268, result["rouge2"].recall, 5)
    self.assertAlmostEqual(0.48000, result["rouge2"].precision, 5)
    self.assertAlmostEqual(0.36364, result["rouge2"].fmeasure, 5)

  def testRougeLAgainstRouge155(self):
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.40741, result["rougeL"].recall, 5)
    self.assertAlmostEqual(0.68750, result["rougeL"].precision, 5)
    self.assertAlmostEqual(0.51163, result["rougeL"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.40476, result["rougeL"].recall, 5)
    self.assertAlmostEqual(0.65385, result["rougeL"].precision, 5)
    self.assertAlmostEqual(0.50000, result["rougeL"].fmeasure, 5)

  def testRougeLSumAgainstRouge155WithStemming(self):
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    target = test_util.get_text(
        os.path.join(test_util.PYROUGE_DIR, "target_multi.0.txt"))
    prediction = test_util.get_text(
        os.path.join(test_util.PYROUGE_DIR, "prediction_multi.0.txt"))
    result = scorer.score(target, prediction)

    self.assertAlmostEqual(0.36538, result["rougeLsum"].recall, places=5)
    self.assertAlmostEqual(0.66667, result["rougeLsum"].precision, places=5)
    self.assertAlmostEqual(0.47205, result["rougeLsum"].fmeasure, places=5)

  def testLcsTable(self):
    ref = [1, 2, 3, 4, 5]
    c1 = [2, 5, 3, 4]
    t = rouge_scorer._lcs_table(ref, c1)
    self.assertEqual(3, t[len(ref)][len(c1)])
    def _read_lcs(t, ref, can):
      return rouge_scorer._backtrack_norec(t, ref, can)
    # Indices
    self.assertEqual([1, 2, 3],
                     _read_lcs(t, ref, c1))
    # Values
    self.assertEqual([2, 3, 4],
                     [ref[i] for i in _read_lcs(t, ref, c1)])

    # No common subsequence.
    c2 = [8, 9]
    t = rouge_scorer._lcs_table(ref, c2)
    self.assertEqual(0, t[len(ref)][len(c2)])
    self.assertEqual([],
                     _read_lcs(t, ref, c2))

  def testUnionLcs(self):
    # Example in Section 3.2 of https://www.aclweb.org/anthology/W04-1013,
    # except using indices into ref.

    # First test helper.
    lcs1 = [0, 1]  # lcs [1, 2]
    lcs2 = [0, 2, 4]
    self.assertEqual([0, 1, 2, 4], rouge_scorer._find_union([lcs1, lcs2]))
    self.assertEqual([0, 1, 2, 4], rouge_scorer._find_union([lcs2, lcs1]))

    ref = [1, 2, 3, 4, 5]
    c1 = [1, 2, 6, 7, 8]  # lcs = [1, 2]
    c2 = [1, 3, 8, 9, 5]  # lcs = [1, 3, 5]
    self.assertEqual([1, 2, 3, 5],
                     rouge_scorer._union_lcs(ref, [c1, c2]))
    self.assertEqual([1, 2, 3, 5],
                     rouge_scorer._union_lcs(ref, [c1, c2]))

  def testSummaryLevelLcs(self):
    refs = [
        [1, 2, 3, 4, 5]
    ]
    cans = [
        [1, 2, 6, 7, 8],  # lcs = [1, 2]
        [1, 3, 8, 9, 5]   # lcs = [1, 3, 5]
    ]
    score = rouge_scorer._summary_level_lcs(refs, cans)
    self.assertEqual(0.8, score.recall)   # 4 / 5
    self.assertEqual(0.4, score.precision)   # 4 / 10
    # 0.4*0.8 / (0.4 + 0.8)
    self.assertAlmostEqual(0.5333, score.fmeasure, places=3)

    # Tokenizer may drop all tokens, resulting in empty candidate list.
    score = rouge_scorer._summary_level_lcs([["reference"]], [[]])
    self.assertEqual(0.0, score.recall)

  def testRougeLsum(self):
    scorer = rouge_scorer.RougeScorer(["rougeLsum"])
    result = scorer.score("w1 w2 w3 w4 w5", "w1 w2 w6 w7 w8\nw1 w3 w8 w9 w5")
    self.assertAlmostEqual(0.8, result["rougeLsum"].recall)
    self.assertAlmostEqual(0.4, result["rougeLsum"].precision)
    self.assertAlmostEqual(0.5333, result["rougeLsum"].fmeasure, places=3)

    # Empty case
    result = scorer.score("w1 w2 w3 w4 w5", "")
    self.assertAlmostEqual(0.0, result["rougeLsum"].fmeasure, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].recall, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].precision, places=3)

    result = scorer.score("", "w1")
    self.assertAlmostEqual(0.0, result["rougeLsum"].fmeasure, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].recall, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].precision, places=3)

    # Case in which summary is all non-word characters.
    result = scorer.score("w1 w2 w3 w4 w5", "/")
    self.assertAlmostEqual(0.0, result["rougeLsum"].fmeasure, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].recall, places=3)
    self.assertAlmostEqual(0.0, result["rougeLsum"].precision, places=3)

  def testRougeLsumLarge(self):
    with open(test_util.LARGE_PREDICTIONS_FILE) as f:
      prediction = f.read()
    with open(test_util.LARGE_TARGETS_FILE) as f:
      target = f.read()
    scorer = rouge_scorer.RougeScorer(["rougeLsum"])
    result = scorer.score(target, prediction)
    self.assertAlmostEqual(0.533, result["rougeLsum"].fmeasure, places=3)


if __name__ == "__main__":
  absltest.main()
