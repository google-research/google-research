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

"""Computes simple extractive baselines.

Baselines:
  1) Choose nth sentence as summary.

Metric:
  1) ROUGE-1
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import csv
import re

from absl import app
from absl import flags
from absl import logging

from rouge import rouge_scorer
from rouge import scoring

FLAGS = flags.FLAGS

flags.DEFINE_string('mturk_csv', '',
                    'MTurk csv.')

_STORY = 'Input.story'
_SUMM = 'Answer.summary'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with open(FLAGS.mturk_csv) as f:
    ignored = 0
    total = 0
    reader = csv.DictReader(f)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    aggregators = {}
    for i in range(5):
      aggregators[str(i)] = scoring.BootstrapAggregator()
    for row in reader:
      # TODO(peterjliu): Average across replicas for same example?
      total += 1
      sentences = re.split('[.!]', row[_STORY])
      sentences.pop()
      if len(sentences) != 5:
        # TODO(peterjliu): Just read sentences from raw csv file.
        logging.error('ignored %s %s', sentences, row[_STORY])
        ignored += 1
        continue
      summary = row[_SUMM]
      for i in range(5):
        aggregators[str(i)].add_scores(scorer.score(summary, sentences[i]))
    for i in range(5):
      print('ROUGE-1 for sentence-%d' % i)
      print(aggregators[str(i)].aggregate())
    logging.info('total %d, ignored %d', total, ignored)


if __name__ == '__main__':
  app.run(main)
