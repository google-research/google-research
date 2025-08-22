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

"""Tests for scorer."""

from absl.testing import absltest
import numpy as np

from smart_eval import matching_functions as mf
from smart_eval import scorer


class ScorerTest(absltest.TestCase):

  def test_smart_score_precomputation(self):
    matching_fn = mf.ChrfMatchingFunction()
    smart_scorer = scorer.SmartScorer(
        smart_types=['smart1', 'smart2', 'smartL'], matching_fn=matching_fn)

    ref = ['This is the first sentence.', 'This is the second.']
    can = ['This is the beginning of the text.', 'This is the last.']
    score = smart_scorer.smart_score(ref, can)

    pairwise_scores = []
    for ref_sent in ref:
      for can_sent in can:
        pairwise_scores.append(matching_fn(ref_sent, can_sent))
    score_matrix = np.reshape(pairwise_scores, newshape=(2, 2))
    score_precomp = smart_scorer.smart_score_precomputed(
        ref_score_matrix=score_matrix)

    self.assertDictEqual(score, score_precomp)

  def test_smart_score(self):
    smart_scorer = scorer.SmartScorer(
        smart_types=['smart1', 'smart2', 'smartL'])
    # Reference and candidate not used when score_matrix is set.
    score_matrix = [[0.81, 0.06, 0.43, 0.64], [0.00, 0.32, 0.27, 0.57],
                    [0.49, 0.29, 0.24, 0.56]]
    score = smart_scorer.smart_score_precomputed(score_matrix)

    # Test Sent-R1.
    r1_rec = (0.81 + 0.57 + 0.56) / 3
    r1_prec = (0.81 + 0.32 + 0.43 + 0.64) / 4
    self.assertAlmostEqual(r1_rec, score['smart1']['recall'])
    self.assertAlmostEqual(r1_prec, score['smart1']['precision'])

    # Test Sent-R2.
    # bigram score matrix would be
    # 0.405 0.030 0.215 0.320 0.000
    # 0.000 0.565 0.165 0.500 0.320
    # 0.245 0.145 0.280 0.415 0.285
    # 0.000 0.245 0.145 0.120 0.280
    r2_rec = (0.405 + 0.565 + 0.415 + 0.280) / 4
    r2_prec = (0.405 + 0.565 + 0.280 + 0.5 + 0.320) / 5
    self.assertAlmostEqual(r2_rec, score['smart2']['recall'])
    self.assertAlmostEqual(r2_prec, score['smart2']['precision'])

    # Test Sent-RL
    rl_rec = (0.81 + 0.57 + 0.56) / 3
    rl_prec = (0.81 + 0.32 + 0.27 + 0.57) / 4
    self.assertAlmostEqual(rl_rec, score['smartL']['recall'])
    self.assertAlmostEqual(rl_prec, score['smartL']['precision'])

  def test_smart_score_with_source(self):
    smart_scorer = scorer.SmartScorer(
        smart_types=['smart1', 'smart2', 'smartL'])
    # Reference and candidate not used when score_matrix is set.
    src_score_matrix = [[0.81, 0.06, 0.43, 0.64], [0.00, 0.32, 0.27, 0.57],
                        [0.49, 0.29, 0.24, 0.56]]
    ref_score_matrix = [[0.81, 0.06, 0.43], [0.00, 0.32, 0.27]]
    score = smart_scorer.smart_score_precomputed(ref_score_matrix,
                                                 src_score_matrix)

    # Test Sent-R1.
    src_r1_rec = (0.81 + 0.57 + 0.56) / 3
    src_r1_prec = (0.81 + 0.32 + 0.43 + 0.64) / 4
    ref_r1_rec = (0.81 + 0.32) / 2
    ref_r1_prec = (0.31 + 0.32 + 0.43) / 3
    r1_rec = max(src_r1_rec, ref_r1_rec)
    r1_prec = max(src_r1_prec, ref_r1_prec)
    self.assertAlmostEqual(r1_rec, score['smart1']['recall'])
    self.assertAlmostEqual(r1_prec, score['smart1']['precision'])

    # Test Sent-R2.
    # src bigram score matrix would be
    # 0.405 0.030 0.215 0.320 0.000
    # 0.000 0.565 0.165 0.500 0.320
    # 0.245 0.145 0.280 0.415 0.285
    # 0.000 0.245 0.145 0.120 0.280
    # ref bigram score matrix would be
    # 0.405 0.030 0.215 0.000
    # 0.000 0.565 0.165 0.215
    # 0.000 0.000 0.160 0.135
    src_r2_rec = (0.405 + 0.565 + 0.415 + 0.280) / 4
    src_r2_prec = (0.405 + 0.565 + 0.280 + 0.5 + 0.320) / 5
    ref_r2_rec = (0.405 + 0.565 + 0.160) / 3
    ref_r2_prec = (0.405 + 0.565 + 0.215 + 0.215) / 4
    r2_rec = max(src_r2_rec, ref_r2_rec)
    r2_prec = max(src_r2_prec, ref_r2_prec)
    self.assertAlmostEqual(r2_rec, score['smart2']['recall'])
    self.assertAlmostEqual(r2_prec, score['smart2']['precision'])

    # Test Sent-RL
    src_rl_rec = (0.81 + 0.57 + 0.56) / 3
    src_rl_prec = (0.81 + 0.32 + 0.27 + 0.57) / 4
    ref_rl_rec = (0.81 + 0.32) / 2
    ref_rl_prec = (0.81 + 0.32 + 0.27) / 3
    rl_rec = max(src_rl_rec, ref_rl_rec)
    rl_prec = max(src_rl_prec, ref_rl_prec)
    self.assertAlmostEqual(rl_rec, score['smartL']['recall'])
    self.assertAlmostEqual(rl_prec, score['smartL']['precision'])

  def test_zero_f1(self):
    smart_scorer = scorer.SmartScorer(
        smart_types=['smart1', 'smart2', 'smartL'])
    score_matrix = [[0 for _ in range(3)] for _ in range(4)]
    score = smart_scorer.smart_score_precomputed(score_matrix)

    self.assertAlmostEqual(0, score['smart1']['fmeasure'])
    self.assertAlmostEqual(0, score['smart2']['fmeasure'])
    self.assertAlmostEqual(0, score['smartL']['fmeasure'])

  def test_assymetric_matching(self):
    # Example assymetric matching fn
    def assym_matching_fn(alist, blist):
      return [min(len(a), len(b)) / len(a) for a, b in zip(alist, blist)]

    can = [
        'abcde',
        'ab'
    ]
    ref = [
        'ab',
        'a'
    ]
    # Resulting score_matrix should be
    # 1.0 1.0
    # 1.0 1.0
    # Resulting rev_score_matrix should be
    # 0.4 0.2
    # 1.0 0.5

    smart_scorer = scorer.SmartScorer(
        matching_fn=assym_matching_fn, is_symmetric_matching=False)
    score = smart_scorer.smart_score(ref, can)

    # SMART-1
    r1_rec = (1.0 + 1.0) / 2
    r1_prec = (0.4 + 1.0) / 2

    # SMART-2
    # bigram_score_matrix would be
    # 0.5 0.5 0.0
    # 0.5 1.0 0.5
    # 0.0 0.5 0.5
    # rev_bigram_score_matrix would be
    # 0.2 0.1  0.0
    # 0.5 0.45 0.1
    # 0.0 0.5  0.25
    r2_rec = (0.5 + 1.0 + 0.5) / 3
    r2_prec = (0.2 + 0.5 + 0.5) / 3

    # SMART-L
    rl_rec = (1.0 + 1.0) / 2
    rl_prec = (0.4 + 1.0) / 2

    self.assertAlmostEqual(r1_rec, score['smart1']['recall'])
    self.assertAlmostEqual(r1_prec, score['smart1']['precision'])
    self.assertAlmostEqual(r2_rec, score['smart2']['recall'])
    self.assertAlmostEqual(r2_prec, score['smart2']['precision'])
    self.assertAlmostEqual(rl_rec, score['smartL']['recall'])
    self.assertAlmostEqual(rl_prec, score['smartL']['precision'])


if __name__ == '__main__':
  absltest.main()
