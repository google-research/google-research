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

# Lint as: python3
"""preprocessing data helper functions."""


import os
import pickle
import numpy as np
import tensorflow.compat.v2 as tf


def process_dataset(to_skip_dict, random=False):
  """Splits (user, item) dataset to train ,valid and test.

  Args:
    to_skip_dict: Dict of sorted examples.
    random: Bool whether to extract valid, test by random. If false, valid, test
    are the last two items per user.

  Returns:
    examples: Dictionary mapping splits 'train','valid','test' to Numpy array
      containing corresponding CF pairs, and 'to_skip' to a Dictionary
      containing filters for each user.
  """
  examples = {}
  examples['to_skip'] = to_skip_dict
  examples['train'] = []
  examples['valid'] = []
  examples['test'] = []
  for uid in examples['to_skip']:
    if random:
      np.random.shuffle(examples['to_skip'][uid])
    examples['test'].append([uid, examples['to_skip'][uid][-1]])
    examples['valid'].append([
        uid, examples['to_skip'][uid][-2]
    ])
    for iid in examples['to_skip'][uid][0:-2]:
      examples['train'].append([uid, iid])
  for split in ['train', 'valid', 'test']:
    examples[split] = np.array(examples[split]).astype('int64')
  return examples


def save_as_pickle(dataset_path, examples_dict):
  """Saves data to train, valid, test and to_skip pickle files.

  Args:
    dataset_path: String path to dataset directory.
    examples_dict: Dictionary mapping splits 'train','valid','test'
      to Numpy array containing corresponding CF pairs, and 'to_skip' to
      a Dictionary containing filters for each user .
  """
  for dataset_split in ['train', 'valid', 'test', 'to_skip']:
    save_path = os.path.join(dataset_path, dataset_split + '.pickle')
    with tf.gfile.Open(save_path, 'wb') as save_file:
      pickle.dump(examples_dict[dataset_split], save_file)
