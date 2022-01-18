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

"""Tests for eval_utils."""

from absl.testing import absltest
import numpy as np

from etcmodel.models.openkp import eval_utils
from etcmodel.models.openkp import generate_examples_lib as lib

EXAMPLE_JSON = r"""
{
  "url": "http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html",
  "text": "Star Trek Discovery Season 1 Director",
  "KeyPhrases": [
    [
      "Star",
      "Trek"
    ],
    [
      "Discovery",
      "Season"
    ]
  ]
}
"""

EXAMPLE_VDOM = r"""
"VDOM": "[{\"Id\":0,\"text\":\"Star Trek Discovery Season 1\",\"feature\":[44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0,44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0],\"start_idx\":0,\"end_idx\":5},{\"Id\":0,\"text\":\"Director\",\"feature\":[208.0,49.0,138.0,15.0,0.0,0.0,0.0,0.0,12.0,1.0,198.0,564.0,138.0,15.0,1.0,0.0,0.0,0.0,12.0,1.0],\"start_idx\":5,\"end_idx\":6}]"
"""


class GenerateExamplesLibTest(absltest.TestCase):

  def test_text_example_from_json(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)

    expected = eval_utils.OpenKpTextExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        words=['Star', 'Trek', 'Discovery', 'Season', '1', 'Director'],
        key_phrases=set(['discovery season', 'star trek']))

    self.assertEqual(expected, example)

  def test_json_from_text_example(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    json_str = example.to_json_string()
    example2 = eval_utils.OpenKpTextExample.from_json(json_str)
    self.assertEqual(example, example2)

  def test_text_example_from_openkp_example(self):
    expected = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    openkp_example = lib.OpenKpExample.from_json(EXAMPLE_JSON.strip()[:-1] +
                                                 ',' + EXAMPLE_VDOM + '}')
    converted_example = eval_utils.OpenKpTextExample.from_openkp_example(
        openkp_example)
    self.assertEqual(expected, converted_example)

  def test_get_key_phrase_predictions(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    pr1 = eval_utils.KpPositionPrediction(start_idx=4, phrase_len=2, logit=0.1)
    pr3 = eval_utils.KpPositionPrediction(start_idx=0, phrase_len=2, logit=0.3)
    pr2 = eval_utils.KpPositionPrediction(start_idx=2, phrase_len=2, logit=0.2)
    predictions = example.get_key_phrase_predictions([pr1, pr2, pr3],
                                                     max_predictions=2)
    self.assertEqual(predictions, ['star trek', 'discovery season'])

  def test_get_key_phrase_predictions_skip_invalid_indices(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    pr1 = eval_utils.KpPositionPrediction(start_idx=4, phrase_len=2, logit=0.1)
    # start_idx=20 is longer than the document.
    pr3 = eval_utils.KpPositionPrediction(start_idx=20, phrase_len=2, logit=0.3)
    pr2 = eval_utils.KpPositionPrediction(start_idx=2, phrase_len=2, logit=0.2)
    predictions = example.get_key_phrase_predictions([pr1, pr2, pr3],
                                                     max_predictions=2)
    self.assertEqual(predictions, ['discovery season', '1 director'])

  def test_get_score_full(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    candidates = ['Star Trek', 'Something']
    p, r, f1 = example.get_score_full(candidates, max_depth=3)
    self.assertAlmostEqual(p, [1.0, 0.5, 1 / 3])
    self.assertAlmostEqual(r, [0.5, 0.5, 0.5])
    self.assertAlmostEqual(f1, [2 / 3, 0.5, 0.4])

  def test_score_examples(self):
    example = eval_utils.OpenKpTextExample.from_json(EXAMPLE_JSON)
    pr1 = eval_utils.KpPositionPrediction(start_idx=4, phrase_len=2, logit=0.1)
    pr2 = eval_utils.KpPositionPrediction(start_idx=0, phrase_len=2, logit=0.3)
    summary = eval_utils.score_examples([example, example], [[pr1], [pr2]])
    expected = [0.5, 1 / 6, 1 / 10, 0.25, 0.25, 0.25, 1 / 3, 0.2, 1 / 7]
    for i in range(9):
      self.assertAlmostEqual(summary[i], expected[i])

  def test_logits_to_predictions(self):
    logits = np.array([[0.1, 0.9, 0.5, -0.3], [0.8, 0.3, 0.4, -0.5]])
    predictions = eval_utils.logits_to_predictions(logits, max_predictions=3)
    expected1 = eval_utils.KpPositionPrediction(
        start_idx=1, phrase_len=1, logit=0.9)
    expected2 = eval_utils.KpPositionPrediction(
        start_idx=0, phrase_len=2, logit=0.8)
    expected3 = eval_utils.KpPositionPrediction(
        start_idx=2, phrase_len=1, logit=0.5)
    predictions.sort(key=lambda prediction: prediction.logit, reverse=True)
    self.assertEqual(predictions[0], expected1)
    self.assertEqual(predictions[1], expected2)
    self.assertEqual(predictions[2], expected3)


if __name__ == '__main__':
  absltest.main()
