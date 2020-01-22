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

r"""Evaluation program for split+rephrase sentence decomposition.

Used for calculating metrics reported in the paper:
Learning to Split and Rephrase From Wikipedia Edit History, (Botha et al., 2018)

Metrics included:
  - Multi-reference corpus-level BLEU
  - Macro-averaged sentence-level BLEU
  - Length-ratios as reported in the original WebSplit paper:
      output_sentences per input_sentence, tokens per output_sentence

Scores are written as tab-separated rows to standard output.

Usage:
$ cd google_research
$ python -m wiki_split_bleu_eval.score_main \
    --gold="/path/to/references.tsv" \
    --pred="/path/to/predictions.txt" \

The files should be parallel line for line.

- predictions: system output for decomposing a sentence into one or more simpler
    sentences. (The code refers to each such simpler sentence as a parcel.)
    Format:
        parcel_1 SEP parcel_2 ...
    For example, a decomposition of "I think , therefore I am ." into two
    sentences (parcels) should be represented as:
        I think . <::::> Therefore I am .

- gold: ground truth decomposition(s) for the corresponding line.
    Format:
        decomposition_1 <TAB> decomposition_2 [<TAB> decomposition_3 ...]
      where each decomposition has the format
         parcel_1 SEP parcel_2 ...
    Example of two alternative reference decompositions:
         I think . <::::> Therefore I am . <TAB> I think . <::::> Thus I am .

The --parcel_sep flag controls the <::::>-separator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags
from absl import logging

from wiki_split_bleu_eval import score_lib

flags.DEFINE_string(
    'gold', None,
    'Gold (ground-truth) decompositions. Single line per instance: '
    'references are split by tabs and sentence boundaries by --parcel_sep.')
flags.DEFINE_string(
    'pred', None,
    'Predicted decompositions. Single line per instance, containing one '
    'decomposition with sentence boundaries split by --parcel_sep.')
flags.DEFINE_string('parcel_sep', '<::::>',
                    'Separator between parcels, for parallel-mode')
flags.DEFINE_string('output_sep', '\t', 'Delimiter for results output')
flags.DEFINE_bool('debug', False, 'output debug info')

FLAGS = flags.FLAGS


def main(unused_argv):
  logging.info('Scoring file "%s"', FLAGS.pred)

  with open(FLAGS.gold, 'r') as gold_fd:
    gold = score_lib.ReadParcels(gold_fd, parcel_sep=FLAGS.parcel_sep)

  with open(FLAGS.pred, 'r') as pred_fd:
    pred = score_lib.ReadParcels(
        pred_fd, parcel_sep=FLAGS.parcel_sep, reduce_to_single_analysis=True)

  results = {}
  results = score_lib.PerformEval(gold=gold, pred=pred, debug=FLAGS.debug)

  results['_gold_file'] = FLAGS.gold
  results['_pred_file'] = FLAGS.pred

  # Output scoring results in TSV format.
  def as_tsv(results):
    """Helper to format dict as TSV."""
    results_list = [
        '{}{}{}'.format(k, FLAGS.output_sep, v)
        for k, v in sorted(results.items(), key=lambda x: x[0])
    ]
    return '\n'.join(results_list)

  tsv_results = as_tsv(results)
  print(tsv_results)


if __name__ == '__main__':
  assert sys.version_info[0] >= 3, 'This code targets Python 3.'
  app.run(main)
