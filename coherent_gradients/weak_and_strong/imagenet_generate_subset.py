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

# Lint as: python3

"""Generates a new subset of Imagenet for custom training.

This script generates a new subset of Imagenet, creating
hard links to the existing (full) Imagenet dataset.
It can be used to generate "easy"/"hard" Imagenet (e.g. subset of Imagenet
consisting only of examples that are easy/hard to learn.
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import torchvision


def link_files(location, file_list):
  for path in file_list:
    directory = os.path.basename(os.path.dirname(path))
    file_name = os.path.basename(path)
    new_directory_path = os.path.join(location, directory)
    os.makedirs(new_directory_path, exist_ok=True)
    link_path = os.path.join(new_directory_path, file_name)
    os.link(path, link_path)


def generate_new_imagenet(args, indices):
  """Generates new Imagenet subset using given training indices."""
  if len(indices) < args.train_size + args.test_size:
    raise ValueError(('Insufficient number of examples generated. '
                      'You may want to rerun training or let it train longer'))
  print('Separating indices into training and val...')
  perm = np.random.permutation(len(indices))
  original_imagenet = torchvision.datasets.ImageFolder(
      os.path.join(args.original_imagenet, 'train'))
  train_files = []
  test_files = []
  for i in perm[:args.train_size]:
    train_files.append(original_imagenet.imgs[indices[i]][0])
  for i in perm[args.train_size:args.train_size + args.test_size]:
    test_files.append(original_imagenet.imgs[indices[i]][0])

  print('Size of the training set: {}'.format(len(train_files)))
  print('Size of the test set: {}'.format(len(test_files)))

  assert not set(train_files).intersection(set(test_files))

  print('Generating train...')
  link_files(os.path.join(args.new_imagenet, 'train'), train_files)
  print('Generating test...')
  link_files(os.path.join(args.new_imagenet, 'val'), test_files)


def main():
  parser = argparse.ArgumentParser(description='Generating subset of Imagenet')
  parser.add_argument(
      'original_imagenet', help='Location of the original Imagenet')
  parser.add_argument(
      'new_imagenet', help='Location where new Imagenet should be created')
  parser.add_argument(
      'indices_file',
      help=('Location of the list of indices that '
            'should show up in the new Imagenet'))
  parser.add_argument(
      '--train-size',
      default=500000,
      type=int,
      help='Size of the new training set')
  parser.add_argument(
      '--test-size', default=100000, type=int, help='Size of the new test set')
  args = parser.parse_args()

  indices_df = pd.read_csv(args.indices_file)
  indices = list(set(indices_df['index'].values.tolist()))
  generate_new_imagenet(args, indices)


if __name__ == '__main__':
  main()
