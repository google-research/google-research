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

"""Tests for summeval_experiments."""

from absl.testing import absltest

from smart_eval import matching_functions as mf
from smart_eval import summeval_utils as utils


class SummevalUtilsTest(absltest.TestCase):

  def test_precalculated_smart_score(self):
    src = ['This is a source text.', 'It is quite short.']
    # There are multiple references in SummEval.
    refs = [['This is the first sentence.', 'This is the second.'],
            ['This is another sentence.', 'This is another one.']]
    can = ['This is the beginning of the text.', 'This is the last.']

    # Create dummy list of SummEval examples.
    examples = [
        utils.SummevalExample(
            instance_key='',
            system_key='',
            source=src,
            references=refs,
            candidate=can,
            scores=dict())
    ]

    # Create pairwise scores.
    # Pair candidate with source.
    src_can_pairs = []
    for src_sent in src:
      for can_sent in can:
        src_can_pairs.append((src_sent, can_sent))
    src_pairwise_scores = [
        mf.chrf_matcher(src_sent, can_sent)
        for src_sent, can_sent in src_can_pairs
    ]

    # Pair candidate with reference.
    ref_pairwise_scores = []
    for ref in refs:
      ref_can_pairs = []
      for ref_sent in ref:
        for can_sent in can:
          ref_can_pairs.append((ref_sent, can_sent))
      ref_pairwise_scores += [
          mf.chrf_matcher(ref_sent, can_sent)
          for ref_sent, can_sent in ref_can_pairs
      ]

    # Get SMART without precalculation.
    utils.calculate_smart_score(examples, 'chrf', matcher=mf.chrf_matcher)
    print('got SMART without precalculcation')

    # Get SMART with precalculation.
    pairwise_scores = {'src': src_pairwise_scores, 'ref': ref_pairwise_scores}
    utils.calculate_smart_score(
        examples, 'chrf_precalc', pairwise_scores=pairwise_scores)
    print('got SMART with precalculcation')

    for example in examples:
      self.assertAlmostEqual(
          example.scores['src_smart1_fmeasure_chrf'],
          example.scores['src_smart1_fmeasure_chrf_precalc'])
      self.assertAlmostEqual(
          example.scores['src_smart2_fmeasure_chrf'],
          example.scores['src_smart2_fmeasure_chrf_precalc'])
      self.assertAlmostEqual(
          example.scores['src_smartL_fmeasure_chrf'],
          example.scores['src_smartL_fmeasure_chrf_precalc'])


if __name__ == '__main__':
  absltest.main()
