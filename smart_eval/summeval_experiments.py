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

r"""Replicate SummEval experiments.

To run:
python summeval_experiments.py --bartscore_file=${BARTSCORE_PATH} \
--summeval_file=${SUMMEVAL_PATH} -- output_file=${OUTPUT_PATH}
"""

import csv
import json

from absl import app
from absl import flags
from smart_eval import matching_functions as mf
from smart_eval.google import summeval_utils as utils

flags.DEFINE_string('bartscore_file', None,
                    'The json file with data from BARTScore.')
flags.DEFINE_string('summeval_file', None,
                    'The jsonl file with data from SummEval.')
flags.DEFINE_string('output_file', None,
                    'The output file path.')

FLAGS = flags.FLAGS


def main(argv):
  del argv

  summeval_pairs = utils.get_data(FLAGS.bartscore_file, FLAGS.summeval_file)

  # Calculate SMART with non-model-based matchers.
  # Uncomment other matchers below if necessary.
  matchers = {
      'chrf': mf.chrf_matcher,
      # 'meteor': mf.meteor_matcher,
      # 'bleu': mf.bleu_matcher,
      # 'rouge1': mf.rouge_1_matcher,
      # 'rouge2': mf.rouge_2_matcher,
      # 'rougeL': mf.rouge_l_matcher
  }
  for matcher_name, matcher in matchers.items():
    utils.calculate_smart_score(summeval_pairs, matcher_name, matcher=matcher)

  # Calculate SMART with pre-calculated scores from model-based matchers.
  # Uncomment other matchers below if necessary.
  matchers = {
      'bleurt': [
          'SMART/summeval/precomputed_scores/src_can_pairs.bleurt.csv',
          'SMART/summeval/precomputed_scores/ref_can_pairs.bleurt.csv',
      ],
      # 'anli': [
      #     'SMART/summeval/precomputed_scores/src_can_pairs.anli.csv',
      #     'SMART/summeval/precomputed_scores/ref_can_pairs.anli.csv',
      # ],
      # 'bertscore': [
      #     'SMART/summeval/precomputed_scores/src_can_pairs.bertscore.csv',
      #     'SMART/summeval/precomputed_scores/ref_can_pairs.bertscore.csv',
      # ],
  }
  for matcher_name, matcher_files in matchers.items():
    # Source-specific pairwise scores.
    with utils.open_file(matcher_files[0], 'r') as f:
      reader = csv.reader(f)
      rows = list(reader)
    src_pairwise_scores = [float(row[-1]) for row in rows]

    # Reference-specific pairwise scores.
    with utils.open_file(matcher_files[1], 'r') as f:
      reader = csv.reader(f)
      rows = list(reader)
    if matcher_name == 'anli':
      # Only ANLI files have headers.
      rows = rows[1:]
    ref_pairwise_scores = [float(row[-1]) for row in rows]

    pairwise_scores = {'src': src_pairwise_scores, 'ref': ref_pairwise_scores}
    utils.calculate_smart_score(
        summeval_pairs, matcher_name, pairwise_scores=pairwise_scores)

  # Calculate correlation.
  score_names = """max_smartL_fmeasure_bleurt
  max_smart1_fmeasure_bleurt
  max_smart2_fmeasure_bleurt
  max_smartL_fmeasure_chrf
  max_smart1_fmeasure_chrf
  max_smart2_fmeasure_chrf
  bart_score_cnn_src_hypo
  bart_score_src_hypo
  bert_score_f
  chrf
  mover_score
  prism_src_hypo
  rouge1_f
  rouge2_f
  rougel_f
  sentence_movers_glove_sms""".split('\n')
  score_names = [score_name.strip() for score_name in score_names]
  comparisons = ['coherence', 'consistency', 'fluency', 'relevance']
  correlation_dict = utils.get_correlation(
      summeval_pairs, score_names=score_names, comparisons=comparisons)

  with utils.open_file(FLAGS.output_file, 'w') as f:
    json.dump(correlation_dict, f)


if __name__ == '__main__':
  app.run(main)
