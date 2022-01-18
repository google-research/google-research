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

"""Ground-truth evaluation related library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
from absl import logging
import six
from six.moves import range
from rouge import rouge_scorer
from rouge import scoring

TRUNC_LEN = 20

GroundTruthExample = collections.namedtuple('GroundTruthExample',
                                            ['index', 'story', 'storyid',
                                             'summary_list'])


class Metrics(object):
  """Container for ground-truth metrics."""

  ROUGE1_F = 'rouge-1'
  ROUGE1_R = 'rouge-1r'
  ROUGE1_P = 'rouge-1p'

  ROUGE2_F = 'rouge-2'
  ROUGE2_R = 'rouge-2r'
  ROUGE2_P = 'rouge-2p'

  ROUGEL_F = 'rouge-L'
  ROUGEL_R = 'rouge-Lr'
  ROUGEL_P = 'rouge-Lp'

  ROUGE1_R_TRUNC = 'rouge-1r-trunc20'
  ROUGE2_R_TRUNC = 'rouge-2r-trunc20'
  ROUGEL_R_TRUNC = 'rouge-Lr-trunc20'
  ROUGE1_F_TRUNC = 'rouge-1-trunc20'
  ROUGE2_F_TRUNC = 'rouge-2-trunc20'
  ROUGEL_F_TRUNC = 'rouge-L-trunc20'
  PERIODS = 'nperiods'
  WORDS = 'nwords'
  ALL_METRICS = [
      ROUGE1_F, ROUGE1_R, ROUGE1_P, ROUGE2_F, ROUGE2_R, ROUGE2_P, ROUGEL_F,
      ROUGEL_R, ROUGEL_P, ROUGE1_R_TRUNC, ROUGE2_R_TRUNC, ROUGEL_R_TRUNC,
      ROUGE1_F_TRUNC, ROUGE2_F_TRUNC, ROUGEL_F_TRUNC, PERIODS, WORDS
  ]

  def __init__(self):
    # Dict of metric-name (str) to float
    self.metrics = {}

  def add_metric(self, name, value):
    self.metrics[name] = float(value)

  def __str__(self):
    """CSV representation of metrics."""
    columns = list(self.metrics.keys())
    columns.sort()
    out = '%s\n' % ','.join(columns)
    values = [str(self.metrics[c]) for c in columns]
    out += '%s\n' % ','.join(values)
    return out


def get_summaries(s):
  return s.context.feature['summaries'].bytes_list.value


def get_first_sentence(s):
  return tuple(
      [v for v in
       s.feature_lists.feature_list['sentences'].feature[0].int64_list.value])


def get_summary_first_sentence(s):
  for i in range(len(s)):
    if s[i] in ['.', '?', '!']:
      return s[:(i + 1)]
  return s  # if no .?! (not a complete sent), then return the original s


def get_summary_truncated(s, l):
  words = s.split()
  return ' '.join(words[:min(l, len(words))])


def get_summary_n_periods(s):
  n_period = 0
  for i in range(len(s)):
    if s[i] in ['.', '?', '!']:
      n_period += 1
  return n_period


def count_words(s):
  # remove all punctuations in the text
  s_clean = re.sub(r'[^\w\s]', ' ', six.ensure_str(s, 'utf-8'))
  return len(s_clean.split())


class P2sEval(object):
  """Computes metrics against collected ground-truth."""

  def __init__(self, seq_ex_list):
    self.sent2summ = {}  # int64 array to list of summaries
    self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                                 use_stemmer=True)
    for s in seq_ex_list:
      first = get_first_sentence(s)
      if first in self.sent2summ:
        logging.fatal('duplicate first sentence: %s', str(first))
      self.sent2summ[first] = get_summaries(s)

  def compute_metrics(self,
                      model_sent2summ):
    """Compute metrics.

    Args:
      model_sent2summ: dict of int list to list of summaries

    Returns:
      Metrics object.
    """
    # TODO(peterjliu): Check that we have the right number of examples.
    if len(list(model_sent2summ.keys())) != len(list(self.sent2summ.keys())):
      logging.info('number of keys mismatch: %d %d',
                   len(list(model_sent2summ.keys())),
                   len(list(self.sent2summ.keys())))

    targets = []
    predictions = []
    agg = scoring.BootstrapAggregator()
    t_agg = scoring.BootstrapAggregator()
    for first_sent, model_summary in six.iteritems(model_sent2summ):
      first_model_summary = get_summary_first_sentence(model_summary)
      trunc_model_summary = get_summary_truncated(first_model_summary,
                                                  TRUNC_LEN)
      try:
        for s in self.sent2summ[first_sent]:
          agg.add_scores(self.rouge_scorer.score(s,  # reference first
                                                 model_summary))
          t_agg.add_scores(self.rouge_scorer.score(s,  # reference first
                                                   trunc_model_summary))
          targets.append(s)
          predictions.append(model_summary)
      except KeyError:
        logging.error('key not found %s', first_sent)
        raise Exception('key not found %s. %s' %
                        (str(first_sent), str(list(self.sent2summ.keys()))))

    rouge_scores = agg.aggregate()
    trunc_rouge_scores = t_agg.aggregate()

    m = Metrics()
    m.add_metric(Metrics.ROUGE1_F, rouge_scores['rouge1'].mid.fmeasure)
    m.add_metric(Metrics.ROUGE1_R, rouge_scores['rouge1'].mid.recall)
    m.add_metric(Metrics.ROUGE1_P, rouge_scores['rouge1'].mid.precision)

    m.add_metric(Metrics.ROUGE2_F, rouge_scores['rouge2'].mid.fmeasure)
    m.add_metric(Metrics.ROUGE2_R, rouge_scores['rouge2'].mid.recall)
    m.add_metric(Metrics.ROUGE2_P, rouge_scores['rouge2'].mid.precision)

    m.add_metric(Metrics.ROUGEL_F, rouge_scores['rougeL'].mid.fmeasure)
    m.add_metric(Metrics.ROUGEL_R, rouge_scores['rougeL'].mid.recall)
    m.add_metric(Metrics.ROUGEL_P, rouge_scores['rougeL'].mid.precision)

    # Truncated rouge
    m.add_metric(Metrics.ROUGE1_R_TRUNC,
                 trunc_rouge_scores['rouge1'].mid.recall)
    m.add_metric(Metrics.ROUGE2_R_TRUNC,
                 trunc_rouge_scores['rouge2'].mid.recall)
    m.add_metric(Metrics.ROUGEL_R_TRUNC,
                 trunc_rouge_scores['rougeL'].mid.recall)
    m.add_metric(Metrics.ROUGE1_F_TRUNC,
                 trunc_rouge_scores['rouge1'].mid.fmeasure)
    m.add_metric(Metrics.ROUGE2_F_TRUNC,
                 trunc_rouge_scores['rouge2'].mid.fmeasure)
    m.add_metric(Metrics.ROUGEL_F_TRUNC,
                 trunc_rouge_scores['rougeL'].mid.fmeasure)

    m.add_metric(
        Metrics.PERIODS,
        sum([get_summary_n_periods(s) for s in predictions]) / len(predictions))
    m.add_metric(Metrics.WORDS,
                 sum([count_words(s) for s in predictions]) / len(predictions))
    return m
