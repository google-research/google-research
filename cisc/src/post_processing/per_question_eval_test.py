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

from absl.testing import absltest
import pandas as pd
from cisc.src.post_processing import aggregators
from cisc.src.post_processing import per_question_eval


class PerQuestionEvalTest(absltest.TestCase):

  def test_score(self):
    df = pd.DataFrame({
        'question_id': ['q1', 'q1', 'q1'],
        'verbal_confidence': [1, 1, 1],
        'binary_confidence': [1, 1, 1],
        'answer': ['ans1', 'ans2', 'ans2'],
        'golden_label': ['ans2', 'ans2', 'ans2'],
        'logit_confidence': [None, None, None],
        'response_probability': [None, None, None],
        'prompt': [None, None, None],
        'response': [None, None, None],
    })
    df = per_question_eval.group_by_question_id(df)
    scores = per_question_eval.score(
        df,
        eval_func_configs=[
            aggregators.AggregatorConfig(
                aggregator_type=aggregators.AggregatorType.SC,
                norm_type=aggregators.NormalizationType.NONE,
            )
        ],
        traces_lens=[1, 2, 3],
        num_bootstrap=1,
    )
    self.assertLen(scores, 1)
    self.assertIn('SC_NONE', scores)
    self.assertLen(scores['SC_NONE'], 3)
    # We can only be sure that the last score is 1, because it used all 3 three
    # traces (no real randomization).
    self.assertEqual(scores['SC_NONE'][-1], 1)

  def test_wqd_eval_one_sample(self):
    df = pd.DataFrame({
        'verbal_confidence': [[2, 1]],
        'binary_confidence': [[1, 1]],
        'answer': [['ans1', 'ans2']],
        'golden_label': ['ans1'],
        'logit_confidence': [[1, 2]],
        'response_probability': [[None, None]],
    })
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='verbal_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=1,
            num_higher_better=1,
            num_confidence_ties=0,
        ),
    )
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='logit_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=1,
            num_higher_better=0,
            num_confidence_ties=0,
        ),
    )

  def test_wqd_eval_one_sample_reverese(self):
    # In this case there is nothing really to compare, so we return None.
    df = pd.DataFrame({
        'verbal_confidence': [[1, 2]],
        'binary_confidence': [[1, 1]],
        'answer': [['ans1', 'ans2']],
        'golden_label': ['ans1'],
        'logit_confidence': [[2, 1]],
        'response_probability': [[None, None]],
    })
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='verbal_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=1,
            num_higher_better=0,
            num_confidence_ties=0,
        ),
    )
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='logit_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=1,
            num_higher_better=1,
            num_confidence_ties=0,
        ),
    )

  def test_wqd_eval_one_sample_two_samples_half_correct(self):
    # In this case there is nothing really to compare, so we return None.
    df = pd.DataFrame({
        'verbal_confidence': [[1, 2], [2, 1]],
        'binary_confidence': [[1, 1], [1, 1]],
        'answer': [['ans1', 'ans2'], ['ans1', 'ans2']],
        'golden_label': ['ans1', 'ans1'],
        'logit_confidence': [[1, 2], [2, 1]],
        'response_probability': [[None, None], [None, None]],
    })
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='verbal_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=2,
            num_higher_better=1,
            num_confidence_ties=0,
        ),
    )
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='logit_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=2,
            num_higher_better=1,
            num_confidence_ties=0,
        ),
    )

  def test_wqd_eval_all_correct(self):
    # In this case there is nothing really to compare, so we return None.
    df = pd.DataFrame({
        'verbal_confidence': [[1, 1]],
        'binary_confidence': [[1, 1]],
        'answer': [['ans2', 'ans2']],
        'golden_label': ['ans2'],
        'logit_confidence': [[1, 1]],
        'response_probability': [[None, None]],
    })
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='verbal_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=0,
            num_higher_better=0,
            num_confidence_ties=0,
        ),
    )

  def test_wqd_eval_all_ties(self):
    # In this case there is nothing really to compare, so we return None.
    df = pd.DataFrame({
        'verbal_confidence': [[1, 1.000000000001]],
        'binary_confidence': [[1, 1]],
        'answer': [['ans1', 'ans2']],
        'golden_label': ['ans2'],
        'logit_confidence': [[1, 1]],
        'response_probability': [[None, None]],
    })
    self.assertEqual(
        per_question_eval.wqd_eval(
            df, confidence_col_name='verbal_confidence', num_bootstrap=1
        ),
        per_question_eval.WQDEvalStats(
            num_pairs=1,
            num_higher_better=0,
            num_confidence_ties=1,
        ),
    )


if __name__ == '__main__':
  absltest.main()
