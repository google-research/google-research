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

"""Evaluation script for NarrativeQA dataset."""

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge


def evaluate_narrative_qa(ground_truth, predicted_answers):
  """Evaluation NarrativeQA predictions."""
  scorers = [(Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
             (Rouge(), 'ROUGE_L'), (Cider(), 'CIDEr')]

  def preprocess(text):
    return text.lower().rstrip(' .').strip()

  common_keys = [k for k in predicted_answers if k in ground_truth]
  refs = {k: [preprocess(s) for s in ground_truth[k]] for k in common_keys}
  hyps = {k: [preprocess(predicted_answers[k])] for k in common_keys}

  ret_scores = dict(common=len(common_keys))
  for scorer, method in scorers:
    score, scores = scorer.compute_score(refs, hyps)
    if isinstance(method, list):
      for sc, _, m in zip(score, scores, method):
        # print('%s: %0.6f' % (m, sc))
        ret_scores[m] = sc * 100
    else:
      # print('%s: %0.6f' % (method, score))
      ret_scores[method] = score * 100
    if isinstance(scorer, Meteor):
      scorer.close()
  del scorers
  return ret_scores
