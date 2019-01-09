# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for rouge scorer.

Tests for both correctness and for consistency with the official ROUGE-1.5.5
implementation.

"Ground truth" scores are taken from manual runs of ROUGE-1.5.5.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

  @parameterized.parameters(["rouge1", "rouge9", "rougeL"])
  def testValidRougeTypes(self, rouge_type):
    scorer = rouge_scorer.RougeScorer([rouge_type])
    result = scorer.score("testing one two", "testing")
    self.assertSameElements(result.keys(), [rouge_type])

  def testRouge1(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    result = scorer.score("testing one two", "testing")
    self.assertAlmostEqual(1, result["rouge1"].precision)
    self.assertAlmostEqual(1 / 3, result["rouge1"].recall)
    self.assertAlmostEqual(1 / 2, result["rouge1"].fmeasure)

  def testRouge1Empty(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    result = scorer.score("testing one two", "")
    self.assertAlmostEqual(0, result["rouge1"].precision)
    self.assertAlmostEqual(0, result["rouge1"].recall)
    self.assertAlmostEqual(0, result["rouge1"].fmeasure)

  def testRouge2(self):
    scorer = rouge_scorer.RougeScorer(["rouge2"])
    result = scorer.score("testing one two", "testing one")
    self.assertAlmostEqual(1, result["rouge2"].precision)
    self.assertAlmostEqual(1 / 2, result["rouge2"].recall)
    self.assertAlmostEqual(2 / 3, result["rouge2"].fmeasure)

  def testRouge2Empty(self):
    scorer = rouge_scorer.RougeScorer(["rouge2"])
    result = scorer.score("testing one two", "")
    self.assertAlmostEqual(0, result["rouge2"].precision)
    self.assertAlmostEqual(0, result["rouge2"].recall)
    self.assertAlmostEqual(0, result["rouge2"].fmeasure)

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

  def testRougeLEmpty(self):
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    result = scorer.score("testing one two", "")
    self.assertAlmostEqual(0, result["rougeL"].precision)
    self.assertAlmostEqual(0, result["rougeL"].recall)
    self.assertAlmostEqual(0, result["rougeL"].fmeasure)

  def testMultipleRougeTypes(self):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    result = scorer.score("testing one two", "testing one")
    self.assertSameElements(result.keys(), ["rouge1", "rougeL"])
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

  def testRougeLAgainstRouge155WithStemming(self):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(self.targets[0], self.predictions[0])
    self.assertAlmostEqual(0.40741, result["rougeL"].recall, 5)
    self.assertAlmostEqual(0.68750, result["rougeL"].precision, 5)
    self.assertAlmostEqual(0.51163, result["rougeL"].fmeasure, 5)
    result = scorer.score(self.targets[1], self.predictions[1])
    self.assertAlmostEqual(0.42857, result["rougeL"].recall, 5)
    self.assertAlmostEqual(0.69231, result["rougeL"].precision, 5)
    self.assertAlmostEqual(0.52941, result["rougeL"].fmeasure, 5)


if __name__ == "__main__":
  absltest.main()
