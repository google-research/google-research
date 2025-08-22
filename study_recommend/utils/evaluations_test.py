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

import unittest

import numpy as np
import pandas as pd
from study_recommend import types
from study_recommend.utils import evaluation


def test_data():
  titles_read = {
      1: ['A', 'B', 'A'],
      2: ['B', 'B', 'C'],
      3: ['A', 'B', 'C', 'C', 'A', 'D'],
  }
  recommendations = {
      1: [['C', 'D'], ['C', 'D'], ['A', 'B']],
      2: [['B', 'D'], ['A', 'B'], ['C', 'D']],
      3: [['G', 'K'], ['C', 'K'], ['A', 'B'], ['D', 'K']],
  }

  reference_counts = {
      'all': {1: 3, 2: 3, 3: 4},
      'non_continuation': {1: 3, 2: 2, 3: 3},
      'non_history': {1: 2, 2: 2, 3: 2},
  }
  reference_hits = {
      'all': {1: 1, 2: 3, 3: 3},
      'non_continuation': {1: 1, 2: 2, 3: 2},
      'non_history': {1: 0, 2: 2, 3: 1},
  }
  return titles_read, recommendations, reference_counts, reference_hits


class EvaluationsTest(unittest.TestCase):

  def test_student_breakdown(self):
    titles_read, recommendations, reference_counts, reference_hits = test_data()
    counts, hits = evaluation.compute_per_student_hits(
        titles_read,
        recommendations,
        n=2,
        compute_non_continuation=True,
        compute_non_history=True,
    )

    self.assertEqual(counts, reference_counts)
    self.assertEqual(hits, reference_hits)

  def test_aggregate(self):
    _, _, reference_counts, reference_hits = test_data()
    score = evaluation.aggregate(reference_counts['all'], reference_hits['all'])
    self.assertAlmostEqual(score, 0.6944444444444443)

  def test_utility_fn_final_score(self):
    titles_read, recommendations, _, _ = test_data()
    score = evaluation.hits_at_n(titles_read, recommendations, n=2)
    self.assertAlmostEqual(score, 0.6944444444444443)

  def test_aggregate_function_no_recommendations(self):
    score = evaluation.aggregate({1: 0, 2: 0}, {})
    self.assertIsNone(score)

  def test_interaction_df_to_title_array(self):
    dataframe = pd.DataFrame(
        data=[[1, 'A1'], [1, 'A2'], [2, 'B1'], [2, 'B2'], [3, 'C1']],
        columns=[
            types.StudentActivityFields.STUDENT_ID,
            types.StudentActivityFields.BOOK_ID,
        ],
    ).groupby(types.StudentActivityFields.STUDENT_ID)
    titles_array = evaluation.interaction_df_to_title_array(dataframe)
    titles_array = {key: value.tolist() for key, value in titles_array.items()}
    self.assertEqual(
        titles_array, {1: ['A1', 'A2'], 2: ['B1', 'B2'], 3: ['C1']}
    )

  def test_get_hits_at_n_dataframe(self):
    titles_read, recommendations, _, _ = test_data()
    dataframe = evaluation.compute_per_student_hits_at_n_dataframe(
        titles_read,
        recommendations,
        n_values=[1, 2],
        compute_non_continuation=True,
        compute_non_history=True,
    )

    reference_hits = np.array([
        0.3333333333333333,
        0.6666666666666666,
        0.75,
        0.3333333333333333,
        1.0,
        0.6666666666666666,
        0.0,
        1.0,
        0.5,
        0.3333333333333333,
        1.0,
        0.75,
        0.3333333333333333,
        1.0,
        0.6666666666666666,
        0.0,
        1.0,
        0.5,
    ])
    reference_n_recommendations = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    reference_eval_type = [
        'all',
        'all',
        'all',
        'non_continuation',
        'non_continuation',
        'non_continuation',
        'non_history',
        'non_history',
        'non_history',
        'all',
        'all',
        'all',
        'non_continuation',
        'non_continuation',
        'non_continuation',
        'non_history',
        'non_history',
        'non_history',
    ]
    reference_student_id = [
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
    ]

    self.assertTrue(
        np.allclose(
            reference_hits, dataframe[types.ResultsRecordFields.HITS_AT_N]
        )
    )
    self.assertEqual(
        reference_n_recommendations,
        dataframe[types.ResultsRecordFields.N_RECOMMENDATIONS].tolist(),
    )
    self.assertEqual(
        reference_eval_type,
        dataframe[types.ResultsRecordFields.EVAL_TYPE].tolist(),
    )
    self.assertEqual(
        reference_student_id,
        dataframe[types.ResultsRecordFields.STUDENT_ID].tolist(),
    )


if __name__ == '__main__':
  unittest.main()
