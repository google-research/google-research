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

r"""Creates new train/dev/test splits for SentEval data by bootstrap resampling.

Bootstrap resampling is a standard method to create a new sample by randomly
drawing with replacement from the original data. For more details please refer
to: https://en.wikipedia.org/wiki/Resampling_(statistics).

This script generates the following directory structure under the
`base_out_dir`. This follows the original SentEval directory structure and thus
allows running the original scripts with no change.

base_out_dir/
  probing/TASK_NAME.txt  # Data for this task.
  probing/TASK_NAME.txt-settings.json  # Some information on the generated data.

TASK_NAME.txt contains all examples with their respective set labels attached.
This file follows the original SentEval data format.

Example call:
python -m \
  talk_about_random_splits.probing.split_with_cross_validation \
  --senteval_path="/tmp/senteval/task_data/probing" \
  --base_out_dir="YOUR_PATH_HERE" --alsologtostderr
"""
import csv
import json
import os
import random

from absl import app
from absl import flags
from absl import logging


from talk_about_random_splits.probing import probing_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('senteval_path', None,
                    'Path to the original SentEval data in tsv format.')
flags.DEFINE_string(
    'base_out_dir', None,
    'Base working dir in which to create subdirs for this script\'s results.')
flags.DEFINE_string(
    'split_name', 'bootstrap_resampling',
    'Determines the base name of result sub-directories in `base_out_dir`.')
flags.DEFINE_integer('num_trials', 5, 'Number of trials to generate data for.')
flags.DEFINE_integer('dev_set_size', 10000,
                     'Number of requested examples in dev sets.')
flags.DEFINE_integer('test_set_size', 10000,
                     'Number of requested examples in test sets.')
flags.DEFINE_list('tasks', [
    'word_content.txt', 'sentence_length.txt', 'bigram_shift.txt',
    'tree_depth.txt', 'top_constituents.txt', 'past_present.txt',
    'subj_number.txt', 'obj_number.txt', 'odd_man_out.txt',
    'coordination_inversion.txt'
], 'Tasks for which to generate new data splits.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for task_name in FLAGS.tasks:
    logging.info('Starting task: %s', task_name)
    df = probing_utils.read_senteval_data(FLAGS.senteval_path, task_name)

    experiment_base_dir = os.path.join(
        FLAGS.base_out_dir,
        '{}{}'.format(FLAGS.num_trials, FLAGS.split_name) + '-{}')

    for trial_id in range(FLAGS.num_trials):
      split_dir = experiment_base_dir.format(trial_id)
      probing_dir = os.path.join(split_dir, 'probing')
      settings_path = os.path.join(probing_dir,
                                   '{}-settings.json'.format(task_name))
      data_out_path = os.path.join(probing_dir, '{}'.format(task_name))

      logging.info('Starting run: %s.', task_name)

      total_count = len(df)
      all_indexes = set(range(total_count))

      test_indexes = random.sample(all_indexes, FLAGS.test_set_size)
      train_dev_indexes = list(all_indexes - set(test_indexes))
      train_indexes = [
          random.choice(train_dev_indexes)
          for _ in range(total_count - FLAGS.dev_set_size - FLAGS.test_set_size)
      ]
      dev_indexes = [
          random.choice(train_dev_indexes) for _ in range(FLAGS.dev_set_size)
      ]

      logging.info('Writing output to file: %s.', data_out_path)

      # Set new labels.
      df.loc[df.index[train_indexes], 'set'] = 'tr'
      df.loc[df.index[dev_indexes], 'set'] = 'va'
      df.loc[df.index[test_indexes], 'set'] = 'te'

      os.make_dirs(probing_dir)

      with open(settings_path, 'w') as settings_file:
        settings = {
            'task_name': task_name,
            'trial_id': trial_id,
            'train_size': len(train_indexes),
            'dev_size': len(dev_indexes),
            'test_size': len(test_indexes),
            'train_size_unique': len(set(train_indexes)),
            'dev_size_unique': len(set(dev_indexes)),
            'test_size_unique': len(set(test_indexes)),
        }
        logging.info('Settings:\n%r', settings)
        json.dump(settings, settings_file, indent=2)

      with open(data_out_path, 'w') as data_file:
        # Don't add quoting to retain the original format unaltered.
        df[['set', 'target', 'text']].to_csv(
            data_file,
            sep='\t',
            header=False,
            index=False,
            quoting=csv.QUOTE_NONE,
            doublequote=False)


if __name__ == '__main__':
  flags.mark_flags_as_required(['senteval_path', 'base_out_dir'])
  app.run(main)
