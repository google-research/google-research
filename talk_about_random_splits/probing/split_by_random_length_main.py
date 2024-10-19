# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

r"""Creates new train/dev/test split for SentEval data based on text length.

This script randomly samples text lengths until the number of examples having
these lengths is within an accepted window. Since this is a random process,
the script can generate multiple such samples.

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
  talk_about_random_splits.probing.split_by_random_length_main \
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
import numpy as np


from talk_about_random_splits.probing import probing_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('senteval_path', None,
                    'Path to the original SentEval data in tsv format.')
flags.DEFINE_string(
    'base_out_dir', None,
    'Base working dir in which to create subdirs for this script\'s results.')
flags.DEFINE_string(
    'split_name', 'length_split_random',
    'Determines the base name of result sub-directories in `base_out_dir`.')
flags.DEFINE_integer(
    'trial_count', 5,
    'Splitting by random length is a random process. Here, specify the number '
    'of trials you want to generate data for.')
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
    logging.info('Starting task: %s.', task_name)
    df = probing_utils.read_senteval_data(FLAGS.senteval_path, task_name)
    df['text_len'] = df['text'].str.split().apply(len)
    experiment_base_dir = os.path.join(FLAGS.base_out_dir,
                                       FLAGS.split_name) + '-{}'

    for trial_id in range(FLAGS.trial_count):
      split_dir = experiment_base_dir.format(trial_id)
      probing_dir = os.path.join(split_dir, 'probing')
      settings_path = os.path.join(probing_dir,
                                   '{}-settings.json'.format(task_name))
      data_out_path = os.path.join(probing_dir, '{}'.format(task_name))
      logging.info('Starting run: %d.', trial_id)

      test_lengths, test_mask = probing_utils.split_by_random_length(
          df, FLAGS.test_set_size)
      train_dev_indexes = np.nonzero(~test_mask)[0]
      dev_indexes = random.sample(train_dev_indexes.tolist(),
                                  FLAGS.dev_set_size)
      train_indexes = list(set(train_dev_indexes) - set(dev_indexes))

      logging.info('Writing output to file: %s.', data_out_path)

      # Set new labels.
      df.loc[test_mask, 'set'] = 'te'
      df.loc[df.index[train_indexes], 'set'] = 'tr'
      df.loc[df.index[dev_indexes], 'set'] = 'va'

      os.make_dirs(probing_dir)

      with open(settings_path, 'w') as settings_file:
        settings = {
            'task_name':
                task_name,
            'trial_id':
                trial_id,
            'all_lengths':
                sorted(set(df['text_len'])),
            'test_lengths':
                sorted(test_lengths),
            'train/dev_lengths':
                sorted(set(df['text_len']) - set(test_lengths)),
            'train_size':
                len(train_indexes),
            'dev_size':
                len(dev_indexes),
            'test_size':
                int(test_mask.sum()),
            'test_mask':
                test_mask.values.tolist(),
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
