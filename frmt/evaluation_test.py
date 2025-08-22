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

"""Tests for evaluation."""

from collections.abc import Sequence
from unittest import mock

from absl.testing import absltest
import bleurt.score as bleurt_lib
from frmt import evaluation


def _get_pt_predictions():
  return [
      evaluation.TranslationPair(
          source="This is an example.", translation="Isso é um exemplo."
      ),
      evaluation.TranslationPair(
          source="I like eating pineapple.",
          translation="Eu gosto de comer abacaxi.",
      ),
      evaluation.TranslationPair(
          source="I am talking to Maria.",
          translation="Estou falando com Maria.",
      ),
  ]


def _get_pt_references():
  return [
      evaluation.TranslationPair(
          source="This is an example.", translation="Isto é um exemplo."
      ),
      evaluation.TranslationPair(
          source="I like eating pineapple.",
          translation="Eu gosto de comer ananás.",
      ),
      evaluation.TranslationPair(
          source="I am talking to Maria.",
          translation="Estou a falar com Maria.",
      ),
  ]


def get_mock_bleurt_scorer(return_values):
  mock_bleurt_scorer = mock.MagicMock()
  mock_bleurt_scorer.score.side_effect = return_values
  return mock_bleurt_scorer


class EvaluationTest(absltest.TestCase):

  def test_validate_predictions_and_references(self):
    with self.subTest("Valid predictions and references"):
      evaluation.MetricType._validate_predictions_and_references(
          predictions=_get_pt_predictions(), references=_get_pt_references()
      )

    with self.subTest("No predictions or references"):
      with self.assertRaises(ValueError):
        evaluation.MetricType._validate_predictions_and_references(
            predictions=[], references=_get_pt_references()
        )
      with self.assertRaises(ValueError):
        evaluation.MetricType._validate_predictions_and_references(
            predictions=_get_pt_predictions(), references=[]
        )

    with self.subTest("Misaligned predictions and references"):
      with self.assertRaises(ValueError):
        evaluation.MetricType._validate_predictions_and_references(
            predictions=_get_pt_predictions()[::-1],
            references=_get_pt_references(),
        )

  def test_bleu(self):
    expected_bleu = 0.48565133395124815
    actual_bleu = evaluation.MetricType._compute_bleu(
        predictions=_get_pt_predictions(),
        references=_get_pt_references(),
        language="pt",
    )
    self.assertEqual(actual_bleu, expected_bleu)

  def test_chrf(self):
    expected_chrf = 0.672376902523307
    actual_chrf = evaluation.MetricType._compute_chrf(
        predictions=_get_pt_predictions(), references=_get_pt_references()
    )
    self.assertEqual(actual_chrf, expected_chrf)

  @mock.patch.object(bleurt_lib, "LengthBatchingBleurtScorer", autospec=True)
  def test_bleurt(self, mock_bleurt_scorer_cls):
    mock_bleurt_scorer = mock_bleurt_scorer_cls()
    mock_bleurt_scorer.score.return_value = [0.1, 0.2, 0.5]
    expected_bleurt = 0.26666666666666666  # 0.8 / 3
    actual_bleurt = evaluation.MetricType._compute_bleurt(
        predictions=_get_pt_predictions(),
        references=_get_pt_references(),
        bleurt_scorer=mock_bleurt_scorer,
    )
    self.assertEqual(actual_bleurt, expected_bleurt)

  @mock.patch.object(evaluation, "BleurtScorer", autospec=True)
  def test_evaluate(self, mock_bleurt_scorer_cls):
    mock_bleurt_scorer_cls().score.return_value = [0.1, 0.2, 0.5]

    with self.subTest("Test valid eval metrics."):
      expected_metrics = evaluation.Metrics(
          bleu=0.48565133395124815,
          chrf=0.672376902523307,
          bleurt=0.26666666666666666,
      )
      actual_metrics = evaluation.evaluate(
          predictions=_get_pt_predictions(),
          references=_get_pt_references(),
          eval_metrics=[
              evaluation.MetricType.BLEU,
              evaluation.MetricType.CHRF,
              evaluation.MetricType.BLEURT,
          ],
          language="pt",
          bleurt_scorer_cache=evaluation.BleurtScorerCache(""),
      )
      self.assertDictEqual(actual_metrics.as_dict(), expected_metrics.as_dict())

    with self.subTest("Test invalid eval metrics."):
      with self.assertRaises(ValueError):
        evaluation.evaluate(
            predictions=_get_pt_predictions(),
            references=_get_pt_references(),
            eval_metrics=[
                evaluation.MetricType.BLEU,
                evaluation.MetricType.UNDEFINED,  # Invalid metric.
                evaluation.MetricType.BLEURT,
            ],
            language="pt",
            bleurt_scorer_cache=evaluation.BleurtScorerCache(""),
        )


if __name__ == "__main__":
  absltest.main()
