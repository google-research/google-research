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

"""Tests for NarrativeQA evaluation."""

from absl.testing import absltest

from readtwice.models.narrative_qa import evaluation


class EvaluationTest(absltest.TestCase):

  def assert_near(self, a, b, message):
    self.assertLess(abs(a - b), 1e-2, '%f vs %f: %s' % (a, b, message))

  def test_evaluate_narrative_qa_simple(self):
    groud_thruth = {
        1: [
            'jim was arrested for knocking dave out using a jock bone from a '
            'mule .', 'for hitting dave with a hock bone'
        ],
    }
    predictions = {1: 'he was arrested for stealing a mule bone .'}
    metrics = evaluation.evaluate_narrative_qa(groud_thruth, predictions)
    self.assert_near(metrics['Bleu_1'], 100.0 * 0.7499999999062501, 'Bleu_1')
    self.assert_near(metrics['Bleu_2'], 100.0 * 0.5669467094379106, 'Bleu_2')
    self.assert_near(metrics['Bleu_3'], 100.0 * 0.37697372050997574, 'Bleu_3')
    self.assert_near(metrics['Bleu_4'], 5.7212484236409495e-03, 'Bleu_4')
    self.assert_near(metrics['ROUGE_L'], 49.19354838709677, 'ROUGE_L')
    self.assert_near(metrics['CIDEr'], 100.0 * 0.0, 'CIDEr')

if __name__ == '__main__':
  absltest.main()
