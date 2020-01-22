# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for rouge scoring and aggregation.

Checks for both correctness, and for consistency with values from the perl ROUGE
implementation which this package replicates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
import numpy as np
from six.moves import range
from six.moves import zip
from rouge import rouge_scorer
from rouge import scoring
from rouge import test_util

# Delta for matching against ground truth rouge values. Must be relatively
# high compared to the individual rouge tests since bootstrap sampling
# introduces randomness.
_DELTA = 0.002

# Use a fixed random seed, or tests may fail with nonzero probability.
_RANDOM_SEED = 123


class BootstrapAggregatorTest(absltest.TestCase):

  def setUp(self):
    super(BootstrapAggregatorTest, self).setUp()
    np.random.seed(_RANDOM_SEED)
    with open(test_util.LARGE_TARGETS_FILE) as f:
      self.targets = f.readlines()
    with open(test_util.LARGE_PREDICTIONS_FILE) as f:
      self.predictions = f.readlines()

  def assertSimilarAggregates(self, precision, recall, fmeasure, aggregate,
                              delta=_DELTA):
    """Helper method for asserting matching aggregate scores.

    Args:
      precision: Tuple of (low, mid, high) precision scores.
      recall: Tuple of (low, mid, high) recall scores.
      fmeasure: Tuple of (low, mid, high) fmeasure scores.
      aggregate: An AggregateScore object.
      delta: Tolerance delta for matching values.
    """

    self.assertAlmostEqual(precision[0], aggregate.low.precision, delta=delta)
    self.assertAlmostEqual(precision[1], aggregate.mid.precision, delta=delta)
    self.assertAlmostEqual(precision[2], aggregate.high.precision, delta=delta)
    self.assertAlmostEqual(recall[0], aggregate.low.recall, delta=delta)
    self.assertAlmostEqual(recall[1], aggregate.mid.recall, delta=delta)
    self.assertAlmostEqual(recall[2], aggregate.high.recall, delta=delta)
    self.assertAlmostEqual(fmeasure[0], aggregate.low.fmeasure, delta=delta)
    self.assertAlmostEqual(fmeasure[1], aggregate.mid.fmeasure, delta=delta)
    self.assertAlmostEqual(fmeasure[2], aggregate.high.fmeasure, delta=delta)

  def testConsistentPercentiles(self):
    aggregator = scoring.BootstrapAggregator(confidence_interval=0.9)
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=1, recall=1 / 3, fmeasure=1 / 2)
    })
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=0, recall=0, fmeasure=0)
    })
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=1, recall=1, fmeasure=1)
    })
    result = aggregator.aggregate()

    self.assertSimilarAggregates((1 / 3, 2 / 3, 3 / 3),
                                 (1 / 9, 4 / 9, 7 / 9),
                                 (1 / 6, 3 / 6, 5 / 6),
                                 result["rouge1"], delta=1e-8)

  def testLargeConfidence(self):
    aggregator = scoring.BootstrapAggregator(confidence_interval=0.0)
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=1, recall=1 / 3, fmeasure=1 / 2)
    })
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=0, recall=0, fmeasure=0)
    })
    aggregator.add_scores({
        "rouge1": scoring.Score(precision=1, recall=1, fmeasure=1)
    })
    result = aggregator.aggregate()

    self.assertSimilarAggregates((2 / 3, 2 / 3, 2 / 3),
                                 (4 / 9, 4 / 9, 4 / 9),
                                 (3 / 6, 3 / 6, 3 / 6),
                                 result["rouge1"], delta=1e-8)

  def testMultipleRougeTypes(self):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    aggregator = scoring.BootstrapAggregator()
    for target, prediction in zip(self.targets[:5], self.predictions[:5]):
      aggregator.add_scores(scorer.score(target, prediction))
    result = aggregator.aggregate()

    self.assertSameElements(list(result.keys()), ["rouge1", "rougeL"])

  def testConfidenceIntervalsAgainstRouge155(self):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    aggregator = scoring.BootstrapAggregator()
    for target, prediction in zip(self.targets, self.predictions):
      aggregator.add_scores(scorer.score(target, prediction))
    result = aggregator.aggregate()

    self.assertSimilarAggregates((0.48695, 0.49879, 0.51131),
                                 (0.31106, 0.31950, 0.32849),
                                 (0.37614, 0.38554, 0.39581),
                                 result["rouge1"])

  def testConfidenceIntervalsAgainstRouge155WithStemming(self):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for target, prediction in zip(self.targets, self.predictions):
      aggregator.add_scores(scorer.score(target, prediction))
    result = aggregator.aggregate()

    self.assertSimilarAggregates((0.51027, 0.52434, 0.53788),
                                 (0.32563, 0.33580, 0.34548),
                                 (0.39380, 0.40524, 0.41661),
                                 result["rouge1"])
    self.assertSimilarAggregates((0.50759, 0.52104, 0.53382),  # P
                                 (0.32418, 0.33377, 0.34362), # R
                                 (0.39157, 0.40275, 0.41383), # F
                                 result["rougeL"])

  def testConfidenceIntervalsAgainstRouge155WithStemmingMultiLine(self):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    t_files = [os.path.join(test_util.PYROUGE_DIR, 'target_multi.%d.txt' % i) for i in range(0, 250)]
    p_files = [os.path.join(test_util.PYROUGE_DIR, 'prediction_multi.%d.txt' % i) for i in range(0, 250)]

    targets = [test_util.get_text(x) for x in t_files]
    predictions = [test_util.get_text(x) for x in p_files]
    assert len(targets) == len(predictions)
    assert len(targets) == 250
    for target, prediction in zip(targets, predictions):
      aggregator.add_scores(scorer.score(target, prediction))
    result = aggregator.aggregate()

    # DIR = testdata/pyrouge_evaluate_plain_text_files
    #  pyrouge_evaluate_plain_text_files -s $DIR -sfp "prediction_multi.(.*).txt"
    #    -m $DIR -mfp target_multi.#ID#.txt
    self.assertSimilarAggregates((0.58963, 0.59877, 0.60822),    # P
                                 (0.37327, 0.38091, 0.38914),    # R
                                 (0.45607, 0.46411, 0.47244),    # F
                                 result["rouge1"])
    self.assertSimilarAggregates((0.35429, 0.36516, 0.37665),    # P
                                 (0.22341, 0.23109, 0.23916),    # R
                                 (0.27312, 0.28209, 0.29133),    # F
                                 result["rouge2"])
    self.assertSimilarAggregates((0.58604, 0.59491, 0.60444),    # P
                                 (0.37084, 0.37846, 0.38671),    # R
                                 (0.45305, 0.46113, 0.46946),    # F
                                 result["rougeLsum"])


if __name__ == "__main__":
  absltest.main()
