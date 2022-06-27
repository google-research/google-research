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

r"""Splits SentEval data into k stratified folds.

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

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from sklearn import model_selection


from talk_about_random_splits.probing import probing_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('senteval_path', None,
                    'Path to the original SentEval data in tsv format.')
flags.DEFINE_string(
    'base_out_dir', None,
    'Base working dir in which to create subdirs for this script\'s results.')
flags.DEFINE_string(
    'split_name', 'fold_xval',
    'Determines the base name of result sub-directories in `base_out_dir`.')
flags.DEFINE_integer('num_folds', 10,
                     'Number of folds into which to split the data.')
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

    experiment_base_dir = os.path.join(
        FLAGS.base_out_dir,
        '{}{}'.format(FLAGS.num_folds, FLAGS.split_name) + '-{}')
    skf = model_selection.StratifiedKFold(n_splits=FLAGS.num_folds)

    for current_fold_id, (train_indexes, test_indexes) in enumerate(
        skf.split(df['text'], df['target'])):
      split_dir = experiment_base_dir.format(current_fold_id)
      probing_dir = os.path.join(split_dir, 'probing')
      settings_path = os.path.join(probing_dir,
                                   '{}-settings.json'.format(task_name))
      data_out_path = os.path.join(probing_dir, '{}'.format(task_name))
      logging.info('Starting run: %d.', current_fold_id)

      # Use the same data for train and dev, because the probing code does some
      # hyperparameter search on dev. We don't wanna tune on the test portion.
      train_set = df.iloc[train_indexes].copy()
      train_set.loc[:, 'set'] = 'tr'
      dev_set = df.iloc[train_indexes].copy()
      dev_set.loc[:, 'set'] = 'va'
      test_set = df.iloc[test_indexes].copy()
      test_set.loc[:, 'set'] = 'te'
      new_data = pd.concat([train_set, dev_set, test_set], ignore_index=True)

      logging.info('Writing output to file: %s.', data_out_path)
      os.make_dirs(probing_dir)

      with open(settings_path, 'w') as settings_file:
        settings = {
            'task_name': task_name,
            'fold_id': current_fold_id,
            'train_size': len(train_indexes),
            'dev_size': len(train_indexes),
            'test_size': len(test_indexes),
        }
        logging.info('Settings:\n%r', settings)
        json.dump(settings, settings_file, indent=2)

      with open(data_out_path, 'w') as data_file:
        # Don't add quoting to retain the original format unaltered.
        new_data[['set', 'target', 'text']].to_csv(
            data_file,
            sep='\t',
            header=False,
            index=False,
            quoting=csv.QUOTE_NONE,
            doublequote=False)


if __name__ == '__main__':
  flags.mark_flags_as_required(['senteval_path', 'base_out_dir'])
  app.run(main)
