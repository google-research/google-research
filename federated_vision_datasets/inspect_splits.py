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

"""A simple tool for inspecting the federated visual classification data splits.

Usage:
python inspect_splits.py --dataset=cifar --train_file=train.csv
--test_file=test.csv

At least one of the flags in `--train_file` and `--test_file` shall be provided.
"""

import collections
import csv
from os import path

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset', 'landmarks', ['landmarks', 'inat', 'cifar'],
                  'The split from which dataset to parse.')

flags.DEFINE_string('train_file', None, 'Path to the training split csv file.')

flags.DEFINE_string('test_file', None, 'Path to the test split csv file.')


def get_parser(dataset_name):
  """Returns a csv line parser function for the given dataset."""

  def inat_parser(line, is_train=True):
    if is_train:
      user_id, image_id, class_id, _ = line
      return user_id, image_id, class_id
    else:
      image_id, class_id, _ = line
      return image_id, class_id

  def landmarks_parser(line, is_train=True):
    if is_train:
      user_id, image_id, class_id = line
      return user_id, image_id, class_id
    else:
      image_id, class_id = line
      return image_id, class_id

  parsers = {
      'inat': inat_parser,
      'landmarks': landmarks_parser,
      'cifar': landmarks_parser  # landmarks and cifar uses the same parser.
  }

  return parsers[dataset_name]


def inspect_train_file(train_file, parser):
  """Inspects the federated train split."""
  print('Train file: %s' % train_file)
  if not path.exists(train_file):
    print('Error: file does not exist.')
    return

  user_image_counter = collections.Counter()
  unique_images = set()
  unique_classes = set()

  with open(train_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    for line in reader:
      user_id, image_id, class_id = parser(line, is_train=True)

      user_image_counter[user_id] += 1
      unique_images.add(image_id)
      unique_classes.add(class_id)

    print('  {:,} users.'.format(len(user_image_counter)))
    print('  {:,} images.'.format(len(unique_images)))
    print('  {:,} classes.'.format(len(unique_classes)))


def inspect_test_file(test_file, parser):
  """Inspects the test split."""
  print('Test file: %s' % test_file)
  if not path.exists(test_file):
    print('Error: file does not exist.')
    return

  unique_images = set()
  unique_classes = set()

  with open(test_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    for line in reader:
      image_id, class_id = parser(line, is_train=False)

      unique_images.add(image_id)
      unique_classes.add(class_id)

    print('  {:,} images.'.format(len(unique_images)))
    print('  {:,} classes.'.format(len(unique_classes)))


def main(_):
  parser = get_parser(FLAGS.dataset)

  if FLAGS.train_file is None and FLAGS.test_file is None:
    print('Must provide at least one of these flags: \n'
          '  --train_file=/path/to/train_file.csv \n'
          '  --test_file=/path/to/test_file.csv')
    return

  if FLAGS.train_file is not None:
    inspect_train_file(FLAGS.train_file, parser)

  if FLAGS.test_file is not None:
    inspect_test_file(FLAGS.test_file, parser)


if __name__ == '__main__':
  app.run(main)
