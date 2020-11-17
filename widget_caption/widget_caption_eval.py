# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Functions for evaluations."""
import collections
from typing import Dict, List

from nltk.translate import bleu_score

from rouge import rouge_scorer


def coco_evaluate(references, hypothesis):
  """Evaluate the hypothesis against the references for COCO scores.

  Args:
    references: A list of reference string.
    hypothesis: A hypothesis string.

  Returns:
    A dict containing COCO scores.
  """
  # Tokenize both hypothesis and references.
  hyp_words = hypothesis.split(' ')
  ref_words_list = [ref.split(' ') for ref in references]

  rouge_scores = _rouge_scores(references, hypothesis)
  scores = {
      'BLEU-1': _bleu_n(1, ref_words_list, hyp_words),
      'BLEU-2': _bleu_n(2, ref_words_list, hyp_words),
      'BLEU-3': _bleu_n(3, ref_words_list, hyp_words),
      'BLEU-4': _bleu_n(4, ref_words_list, hyp_words),
      'ROUGE-1-f1-mean': rouge_scores['rouge1_mean'],
      'ROUGE-1-f1-min': rouge_scores['rouge1_min'],
      'ROUGE-1-f1-max': rouge_scores['rouge1_max'],
      'ROUGE-2-f1-mean': rouge_scores['rouge2_mean'],
      'ROUGE-2-f1-min': rouge_scores['rouge2_min'],
      'ROUGE-2-f1-max': rouge_scores['rouge2_max'],
      'ROUGE-L-f1-mean': rouge_scores['rougeLsum_mean'],
      'ROUGE-L-f1-min': rouge_scores['rougeLsum_min'],
      'ROUGE-L-f1-max': rouge_scores['rougeLsum_max'],
  }

  return scores


def _bleu_n(n, ref_words_list,
            hyp_words):
  """Computes BLEU score up to n-gram."""
  # Average weights across different n-grams.
  weights = [1.0 / n] * n
  return bleu_score.sentence_bleu(
      ref_words_list,
      hyp_words,
      weights=weights,
      smoothing_function=bleu_score.SmoothingFunction().method1)


def _rouge_scores(references, hypothesis):
  """Computes min, max and mean Rouge scores over all the referneces."""

  score_keys = ['rouge1', 'rouge2', 'rougeLsum']
  scorer = rouge_scorer.RougeScorer(score_keys)
  scores = collections.defaultdict(list)
  for ref in references:
    score = scorer.score(target=ref, prediction=hypothesis)
    # Collect all the scores' F1 measure.
    for key in score_keys:
      scores[key].append(score[key].fmeasure)

  # Compute mean, max and min values over all the references.
  results = {}
  for key, values in scores.items():
    results[key + '_max'] = max(values)
    results[key + '_min'] = min(values)
    results[key + '_mean'] = sum(values) / len(values)
  return results
