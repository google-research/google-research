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

r"""Human and extractive baseline evaluation.

human_and_extractive \
  --data_dir=$ROCSTORIES_DATA \
  --eval_subset=test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf

from rouge import rouge_scorer
from rouge import scoring
from summae import p2s_eval
from summae import util

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '.', 'Data directory.')
flags.DEFINE_string('eval_subset', 'test',
                    'which subset (valid/test) to eval/decode.')
flags.DEFINE_string('output_dir', '/tmp/12342',
                    'local directory to save extractive oracle')
flags.DEFINE_string('vocab_file', '',
                    'Subword vocab file.')  # for detok first sentence

my_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                           use_stemmer=True)


def get_extracts(s):
  # get 5 sentences as the extractive baselines
  sents = s.feature_lists.feature_list['untokenized_sentences'].feature
  assert len(sents) == 5
  return tuple([sents[i].bytes_list.value[0] for i in range(5)])


def human_ave(summ_list):
  """Average pairwise rouge between two human summaries."""
  agg = scoring.BootstrapAggregator()
  for s1_id, s1 in enumerate(summ_list):
    for s2_id, s2 in enumerate(summ_list):
      if s1_id >= s2_id:  # only compute for s1_id < s2_id
        continue
      s2_trunc = p2s_eval.get_summary_truncated(
          p2s_eval.get_summary_first_sentence(s2), p2s_eval.TRUNC_LEN)
      s1_s2_trunc_score = my_rouge_scorer.score(s1, s2_trunc)
      agg.add_scores(s1_s2_trunc_score)
  agg_ave = agg.aggregate()
  score_ave = {
      rouge_type: agg_ave[rouge_type].mid for rouge_type in agg_ave  # mid=mean
  }
  nwords_ave = np.mean([p2s_eval.count_words(s) for s in summ_list])
  return (score_ave, nwords_ave)


def human_max(summ_list):
  """Maximum pairwise rouge between any two human summaries."""
  score_max = None
  rouge_1r_trunc_max = 0
  for s1_id, s1 in enumerate(summ_list):
    for s2_id, s2 in enumerate(summ_list):
      if s1_id >= s2_id:
        continue
      s2_trunc = p2s_eval.get_summary_truncated(
          p2s_eval.get_summary_first_sentence(s2), p2s_eval.TRUNC_LEN)
      s1_s2_trunc_score = my_rouge_scorer.score(s1, s2_trunc)
    if s1_s2_trunc_score['rouge1'].recall >= rouge_1r_trunc_max:
      score_max = s1_s2_trunc_score
      rouge_1r_trunc_max = s1_s2_trunc_score['rouge1'].recall
  nwords_max = np.max([p2s_eval.count_words(s) for s in summ_list])
  return (score_max, nwords_max)


def extract_ave(e, summ_list):
  """Average rouge between ith sentence and human summaries."""
  agg = scoring.BootstrapAggregator()
  e_trunc = p2s_eval.get_summary_truncated(
      p2s_eval.get_summary_first_sentence(e),
      p2s_eval.TRUNC_LEN)  # get_summary_first_sentence may not be necessary
  for s in summ_list:
    s_e_trunc_score = my_rouge_scorer.score(s, e_trunc)
    agg.add_scores(s_e_trunc_score)
  agg_ave = agg.aggregate()
  score_ave = {
      rouge_type: agg_ave[rouge_type].mid for rouge_type in agg_ave  # mid=mean
  }
  nwords_e = p2s_eval.count_words(e)
  return (score_ave, nwords_e)


def extract_oracle(extract_list, summ_list):
  """Choose sentence with maximum average rouge."""
  # Choose sentence with maximum average rouge.
  score_accum = []
  for e in extract_list:
    e_trunc = p2s_eval.get_summary_truncated(
        p2s_eval.get_summary_first_sentence(e),
        p2s_eval.TRUNC_LEN)  # get_summary_first_sentence may not be necessary

    accum_rouge_1r_trunc = 0
    for s in summ_list:
      s_e_trunc_score = my_rouge_scorer.score(s, e_trunc)
      # for computing accumulative rouge
      accum_rouge_1r_trunc += s_e_trunc_score['rouge1'].recall
    score_accum.append(accum_rouge_1r_trunc)
  e_id_o = np.argmax(score_accum)
  e_o = extract_list[e_id_o]

  # Compute average rouge for the oracle sentence
  agg = scoring.BootstrapAggregator()
  e_o_trunc = p2s_eval.get_summary_truncated(
      p2s_eval.get_summary_first_sentence(e_o),
      p2s_eval.TRUNC_LEN)  # get_summary_first_sentence may not be necessary
  for s in summ_list:
    e_o_trunc_score = my_rouge_scorer.score(s, e_o_trunc)
    agg.add_scores(e_o_trunc_score)
  agg_o = agg.aggregate()
  score_o = {
      rouge_type: agg_o[rouge_type].mid for rouge_type in agg_o  # mid=mean
  }
  nwords_o = p2s_eval.count_words(e_o)
  return (score_o, nwords_o, e_o)


def print_agg_score(label, agg, nwords):
  print(
      '%s: \n\t rouge-1r-trunc20=%.3f \t rouge-Lr-trunc20=%.3f \t nwords=%.1f' %
      (label, agg.aggregate()['rouge1'].mid.recall,
       agg.aggregate()['rougeL'].mid.recall, np.mean(nwords)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.mkdir(FLAGS.output_dir)

  data_file = os.path.join(
      FLAGS.data_dir,
      'rocstories_gt.' + six.ensure_str(FLAGS.eval_subset) + '.tfrecord')
  seq_ex_list = util.get_seq_exs(data_file)
  print('Input data %s' % data_file)

  # Human summary baselines.
  # We have 3 human summaries for each example, and
  # 2 human performance variants:
  #   1. 'a': average pairwise rouge between two summaries
  #   2. 'm': maximum pairwise rouge between any two summaries
  agg_human = {}
  nwords_human = {}
  for h in ['a', 'm']:
    agg_human[h] = scoring.BootstrapAggregator()
    nwords_human[h] = []

  # Extractive baselines
  #   1. '1','2','3','4','5': rouge between ith sentence and human summary
  #   2. 'o': for each example, choose sentence with maximum average rouge
  agg_extract = {}
  nwords_extract = {}
  for e in [str(x) for x in list(range(5))] + ['o']:
    agg_extract[e] = scoring.BootstrapAggregator()
    nwords_extract[e] = []

  # human performance
  sent2oracle = {}
  for ex in seq_ex_list:
    summ_list = p2s_eval.get_summaries(ex)
    summ_list = [x.decode('utf-8') for x in summ_list]

    # human eval
    score, nwords = human_ave(summ_list)
    agg_human['a'].add_scores(score)
    nwords_human['a'].append(nwords)

    score, nwords = human_max(summ_list)
    agg_human['m'].add_scores(score)
    nwords_human['m'].append(nwords)

    # extractive eval
    extract_list = get_extracts(ex)
    extract_list = [x.decode('utf-8') for x in extract_list]
    for e_id, e in enumerate(extract_list):
      score, nwords = extract_ave(e, summ_list)
      agg_extract[str(e_id)].add_scores(score)
      nwords_extract[str(e_id)].append(nwords)

    score, nwords, e_o = extract_oracle(extract_list, summ_list)
    agg_extract['o'].add_scores(score)
    nwords_extract['o'].append(nwords)

    # save story and oracle sentence for future use
    first = p2s_eval.get_first_sentence(ex)
    if first in sent2oracle:
      logging.fatal('duplicate first sentence: %s', str(first))
    sent2oracle[first] = (' '.join(extract_list), e_o)  # (story, oracle)

  # write each example and the corresponding oracle to disk
  tk, _ = util.get_tokenizer_with_special(FLAGS.vocab_file, [])

  def detok(s):
    return tk.decode(util.strip_after_eos(s))

  keys_sorted = sorted(sent2oracle.keys(), key=detok)

  out_file = os.path.join(
      FLAGS.output_dir, 'rocstories_gt.' + six.ensure_str(FLAGS.eval_subset) +
      '.firstsent2oracle.txt')
  with tf.gfile.Open(out_file, 'w') as f:
    for k in keys_sorted:
      f.write('%s\n' % (sent2oracle[k][1]))

  # print out rouge scores for human performance
  print_agg_score('human average', agg_human['a'], nwords_human['a'])
  print_agg_score('human max', agg_human['m'], nwords_human['m'])
  for e_id in range(5):
    print_agg_score('extractive baseline{}'.format(e_id),
                    agg_extract[str(e_id)], nwords_extract[str(e_id)])
  print_agg_score('extractive oracle', agg_extract['o'], nwords_extract['o'])


if __name__ == '__main__':
  app.run(main)
