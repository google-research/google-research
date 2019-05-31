# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for score_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from wiki_split_bleu_eval import score_lib


class ScoreLibTest(absltest.TestCase):

  def setUp(self):
    super(ScoreLibTest, self).setUp()
    # Predictions.
    self.pred_data = [
        "aSrc ppnt jBL9 rSubRHTb . <::::> 9IU2 eH1V J6k17QPB jBL9 WIdmKeId.",
        "BLOI jBL9 zNWW Qn6p TJXXiRjg vpbG eH1V TTFk ."
    ]

    # Gold references.
    self.gold_data = [
        "\t".join([
            "aSrc ppnt jBL9 U6TlcN . <::::> 9IU2 eH1V J6k17QPB 3zIH lBCy .",
            "aSrc ppnt jBL9 U6TlcN poyY eH1V J6k17QPB WIdmKeId ."
        ]), "\t".join(["aSrc TTFk jBL9 77SJ poyY akLg jBL9 Qn6p TJXXiRjg ."])
    ]

  def test_non_parallel_data(self):
    """Validates detection of non-parallel predictions and gold items."""
    self.gold_data = self.gold_data[:1]
    gold = score_lib.ReadParcels(self.gold_data)
    pred = score_lib.ReadParcels(self.pred_data, reduce_to_single_analysis=True)
    with self.assertRaises(AssertionError) as cm:
      _ = score_lib.PerformEval(gold=gold, pred=pred)
      self.assertTrue(cm.exception.message.startswith("Got unequal"))

  def test_validate_scoring(self):
    """Simple test to validate scoring."""

    gold = score_lib.ReadParcels(self.gold_data)
    pred = score_lib.ReadParcels(self.pred_data, reduce_to_single_analysis=True)

    results = score_lib.PerformEval(gold=gold, pred=pred)
    print(results)

    # Compare against previously captured results.
    results = {k: round(v, 2) for k, v in results.items()}
    self.assertDictEqual(
        {
            "bleu.corpus.decomp": 24.20,
            "bleu.macro_avg_sent.decomp": 19.00,
            "counts.gold_inputs": 2,
            "counts.pred_inputs": 2,
            "counts.predictions": 2,
            "counts.references": 3,
            "lengths.simple_per_complex": 1.5,
            "lengths.tokens_per_simple": 7.0,
            "lengths.tokens_per_simple_micro": 6.33,
            "ref_lengths.simple_per_complex": 1.25,
            "ref_lengths.tokens_per_simple": 8.62,
            "refs_per_input.avg": 1.5,
            "refs_per_input.max": 2,
            "refs_per_input.min": 1,
            "uniq_refs_per_input.avg": 1.5,
            "uniq_refs_per_input.max": 2,
            "uniq_refs_per_input.min": 1
        }, results)


if __name__ == "__main__":
  absltest.main()
