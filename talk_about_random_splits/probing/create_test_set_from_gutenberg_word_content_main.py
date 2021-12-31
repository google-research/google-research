# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
r"""Creates test sets for the SentEval word_content task from Gutenberg data.

In the word_content task, all sentences contain exactly 1 out of a set of
mid-frequency words. This script creates new test sets conforming to the same
setup.

Please note that the number of target words in the new test set might be smaller
than the original 1000, because we might not have enough sentences for all of
the target words. This is not a problem, because it resembles exactly possible
test time scenarios.

The script assumes that the Gutenberg data was preprocessed with
fl_preprocess_gutenberg_main.

Train and dev sets are kept as is, meaning they are the original SentEval
train/dev data. This way the existing architecture can be trained as usual but
tested on new Gutenberg data.

This script doesn't do any case normalization, neither on the target words nor
on sentence duplicate detection.

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
  talk_about_random_splits.probing.\
create_test_set_from_gutenberg_word_content_main \
  --senteval_path="/tmp/senteval/task_data/probing" \
  --base_out_dir="YOUR_PATH_HERE" \
  --gutenberg_path="YOUR_PREPROCESSED_DATA_PATH_HERE" --alsologtostderr \
"""
import csv
import json
import os
import random

from absl import app
from absl import flags
from absl import logging
import pandas as pd


from talk_about_random_splits.probing import probing_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'gutenberg_path', None,
    'Path to the preprocessed Gutenberg data. Data must be preprocessed with '
    'fl_preprocess_gutenberg_main.')
flags.DEFINE_string('senteval_path', None,
                    'Path to the original SentEval data in tsv format.')
flags.DEFINE_string(
    'base_out_dir', None,
    'Base working dir in which to create subdirs for this script\'s results.')
flags.DEFINE_string(
    'split_name', 'gutenberg',
    'Determines the base name of result sub-directories in `base_out_dir`.')
flags.DEFINE_integer('num_trials', 5, 'Number of trials to generate data for.')
flags.DEFINE_integer('test_set_size', 10000,
                     'Number of requested examples in test sets.')

_TASK_NAME = 'word_content.txt'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  word_content_data = probing_utils.read_senteval_data(FLAGS.senteval_path,
                                                       _TASK_NAME)
  target_words = set(word_content_data['target'])
  senteval_sentences = set(word_content_data['text'])
  target_word_to_sentences = probing_utils.get_target_word_to_sentence_mapping(
      target_words, senteval_sentences,
      probing_utils.get_strings_from_sharded_recordio(FLAGS.gutenberg_path))
  class_count = len(target_words)
  items_per_class = FLAGS.test_set_size // class_count
  target_word_counts = {k: len(v) for k, v in target_word_to_sentences.items()}
  valid_target_words = {
      k for k, v in target_word_counts.items() if v >= items_per_class
  }
  logging.info('Number of target words for which we have enough data: %d.',
               len(valid_target_words))
  assert valid_target_words

  train_set = word_content_data.loc[word_content_data['set'] == 'tr', :]
  dev_set = word_content_data.loc[word_content_data['set'] == 'va', :]

  experiment_base_dir = os.path.join(FLAGS.base_out_dir,
                                     f'{FLAGS.split_name}-{{}}')

  for trial_id in range(FLAGS.num_trials):
    split_dir = experiment_base_dir.format(trial_id)
    probing_dir = os.path.join(split_dir, 'probing')
    settings_path = os.path.join(probing_dir, f'{_TASK_NAME}-settings.json')
    data_out_path = os.path.join(probing_dir, _TASK_NAME)

    logging.info('Starting run: %d.', trial_id)

    data_sample = []

    for valid_target_word in valid_target_words:
      sentences = target_word_to_sentences[valid_target_word]
      current_sample = random.sample(sentences, items_per_class)
      data_sample.extend(
          ('te', valid_target_word, sample) for sample in current_sample)

    test_set = pd.DataFrame(data_sample, columns=train_set.columns)
    new_data = pd.concat([train_set, dev_set, test_set], ignore_index=True)

    logging.info('Writing output to file: %s.', data_out_path)
    os.make_dirs(probing_dir)

    with open(settings_path, 'w') as settings_file:
      settings = {
          'task_name': _TASK_NAME,
          'trial_id': trial_id,
          'train_size': len(train_set),
          'dev_size': len(dev_set),
          'test_size': len(test_set),
          'valid_target_words_size': len(valid_target_words),
          'valid_target_words': sorted(valid_target_words),
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
  flags.mark_flags_as_required(
      ['gutenberg_path', 'senteval_path', 'base_out_dir'])
  app.run(main)
