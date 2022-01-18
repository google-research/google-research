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

# Lint as: python3
""""Tests for T5 CBQA metrics."""

from absl.testing import absltest
from t5.evaluation import test_utils

from t5_closed_book_qa.t5_cbqa import metrics


class MetricsTest(test_utils.BaseMetricsTest):

  def test_natural_questions_metrics(self):
    targets = [
        [('yes',), ('no',), ('yes',), ('maybe',)],
        [('Ashley', 'Mary-Kate'), ('Ashley and Mary-Kate',)],
        [('Colin', 'Adam')],
        [('yes',), ('yes',), ('yes',)],
        [('no', 'not really'), ('no',), ()],
        [('no', 'not really'), ('no',), ()],
    ]
    predictions = [
        [('yes',)],  # correct
        [('Mary-Kate', 'Ashley')],  # correct
        [('Colin', 'Adam')],  # correct, but not golden
        [('no',)],  # incorrect
        [('no', 'Not  really',)],  # needs normalization
        [()],  # incorrect
    ]

    self.assertDictClose(
        metrics.natural_questions(targets, predictions),
        {
            'recall': 3/5*100,
            'golden_answers': 5,
        })

    self.assertDictClose(
        metrics.natural_questions(
            targets, predictions,
            non_null_threshold=1),
        {
            'recall': 4/6*100,
            'golden_answers': 6,
        })

    self.assertDictClose(
        metrics.natural_questions(
            targets, predictions,
            non_null_threshold=1,
            normalize_answers=False),
        {
            'recall': 3/6*100,
            'golden_answers': 6,
        })


if __name__ == '__main__':
  absltest.main()

