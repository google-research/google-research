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

"""Tests for score_calculation."""

import pandas as pd


from psyborgs import score_calculation
from psyborgs import survey_bench_lib




def _load_test_admin_session():
  test_admin_session_filepath = 'datasets/test_admin_session.json'

  return survey_bench_lib.load_admin_session(test_admin_session_filepath)


class ScoreCalculationTest(googletest.TestCase):

  def test_reshape_response_choice_probability_scores(self):
    """Test for reshape_response_choice_probability_scores()."""
    # define the input DataFrame
    input_df = pd.DataFrame({
        'prompt_text': ['Prompt 1', 'Prompt 1', 'Prompt 2', 'Prompt 2'],
        'continuation_text': [
            'Continuation 1',
            'Continuation 2',
            'Continuation 1',
            'Continuation 2',
        ],
        'score': [0.1, 0.2, 0.3, 0.4],
        'item_preamble_id': [
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
        ],
        'item_id': ['itemid1', 'itemid1', 'itemid2', 'itemid2'],
        'item_postamble_id': [
            'plk-bfi-0',
            'plk-bfi-0',
            'plk-bfi-0',
            'plk-bfi-0',
        ],
        'response_scale_id': ['likert2', 'likert2', 'likert2', 'likert2'],
        'response_value': [1, 5, 1, 5],
        'response_choice': ['choice1', 'choice2', 'choice1', 'choice2'],
        'response_choice_postamble_id': ['none', 'none', 'none', 'none'],
        'model_id': ['model1', 'model1', 'model1', 'model1'],
    })

    # define the expected output DataFrame
    expected_output_df = pd.DataFrame({
        'item_preamble_id': ['a2-r1-g2-cs6'],
        'item_postamble_id': ['plk-bfi-0'],
        'response_scale_id': ['likert2'],
        'response_choice_postamble_id': ['none'],
        'model_id': ['model1'],
        'itemid1_1': [0.1],
        'itemid1_5': [0.2],
        'itemid2_1': [0.3],
        'itemid2_5': [0.4],
    })

    pd.testing.assert_frame_equal(
        score_calculation.reshape_response_choice_probability_scores(input_df),
        expected_output_df,
        check_dtype=False,
    )

  def test_score_session(self):
    """Test for score_session()."""
    # define the test Admin Session
    test_admin_session = _load_test_admin_session()

    # define the input DataFrame
    input_df = pd.DataFrame({
        'prompt_text': ['Prompt 1', 'Prompt 1', 'Prompt 2', 'Prompt 2'],
        'continuation_text': [
            'Continuation 1',
            'Continuation 2',
            'Continuation 1',
            'Continuation 2',
        ],
        'score': [0.1, 0.2, 0.3, 0.4],
        'item_preamble_id': [
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
            'a2-r1-g2-cs6',
        ],
        'item_id': ['itemid1', 'itemid1', 'itemid2', 'itemid2'],
        'item_postamble_id': [
            'plk-bfi-0',
            'plk-bfi-0',
            'plk-bfi-0',
            'plk-bfi-0',
        ],
        'response_scale_id': ['likert2', 'likert2', 'likert2', 'likert2'],
        'response_value': [1, 5, 1, 5],
        'response_choice': ['choice1', 'choice2', 'choice1', 'choice2'],
        'response_choice_postamble_id': ['none', 'none', 'none', 'none'],
        'model_id': ['model1', 'model1', 'model1', 'model1'],
    })

    # define the expected output DataFrame
    expected_output_df = pd.DataFrame({
        'item_preamble_id': ['a2-r1-g2-cs6'],
        'item_postamble_id': ['plk-bfi-0'],
        'response_scale_id': ['likert2'],
        'response_choice_postamble_id': ['none'],
        'model_id': ['model1'],
        'itemid1': [5],
        'itemid2': [5],
        'itemid1_1': [0.1],
        'itemid1_5': [0.2],
        'itemid2_1': [0.3],
        'itemid2_5': [0.4],
        'TM': [1.5],
    })

    pd.testing.assert_frame_equal(
        score_calculation.score_session(test_admin_session, input_df),
        expected_output_df,
        check_dtype=False,
    )


