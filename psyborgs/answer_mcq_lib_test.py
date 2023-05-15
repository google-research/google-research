# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Tests for social_science_mcq_lib_integration."""

from unittest import mock

from absl.testing import absltest

from psyborgs import answer_mcq_lib


class MockMCQLibTest(absltest.TestCase):

  def test_score_item_choice(self):
    mock_score_with_llm = mock.MagicMock()
    mock_score_with_llm.return_value = [42]

    self.assertEqual(
        answer_mcq_lib.score_item_choice(
            scoring_function=mock_score_with_llm,
            item_text=(
                'Regarding the statement, "I am the life of the party", I'
                ' tend to '
            ),
            response_text='strongly disagree',
        ),
        42,
    )

  def test_score_item(self):
    mock_score_with_llm = mock.MagicMock()
    mock_score_with_llm.return_value = [42]

    likert5_scale = answer_mcq_lib.ResponseScale(
        description='5-point Likert scale',
        _choices=answer_mcq_lib.LIKERT5_CHOICES
        )

    self.assertEqual(
        answer_mcq_lib.score_item(
            scoring_function=mock_score_with_llm,
            item_text=(
                'Regarding the statement, "I am the life of the party", I'
                ' tend to '
            ),
            response_scale=likert5_scale,
        ),
        {
            (1, 'strongly disagree'): 42,
            (2, 'disagree'): 42,
            (3, 'neither agree nor disagree'): 42,
            (4, 'agree'): 42,
            (5, 'strongly agree'): 42
            })

  def test_sort_choice_scores(self):
    score_dict = {
        (1, 'strongly disagree'): -9.356916,
        (2, 'disagree'): -5.9719963,
        (3, 'neither agree nor disagree'): -12.580297,
        (4, 'agree'): -5.7852287,
        (5, 'strongly agree'): -10.555626
        }

    sorted_list = [
        ((4, 'agree'), -5.7852287),
        ((2, 'disagree'), -5.9719963),
        ((1, 'strongly disagree'), -9.356916),
        ((5, 'strongly agree'), -10.555626),
        ((3, 'neither agree nor disagree'), -12.580297)
        ]

    self.assertEqual(
        answer_mcq_lib.sort_choice_scores(score_dict),
        sorted_list)


if __name__ == '__main__':
  absltest.main()
