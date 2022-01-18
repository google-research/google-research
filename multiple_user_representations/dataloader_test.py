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

"""Tests for dataloader."""

import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from multiple_user_representations import dataloader

FLAGS = flags.FLAGS
_TESTDATA_DIR = 'third_party/google_research/google_research/multiple_user_representations/testdata'


class DataloaderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DataloaderTest, self).setUp()
    self._synthetic_data_path = os.path.join(_TESTDATA_DIR,
                                             'test_synthetic_data')
    self._amazon_data_path = os.path.join(_TESTDATA_DIR,
                                          'test_amazon_category_data')

  @parameterized.parameters(('step', False), ('user', False), ('step', True),
                            ('user', True))
  def test_prep_candidate_sampling_probability(self, split_type,
                                               use_validation):

    prep_candidates = dataloader._prep_candidate_sampling_probability_for_training

    items = np.array([1, 1, 0, 4, 4, 4, 6, 5, 1, 6])
    user_item_seqs = np.random.randint(20, size=(len(items), 3))
    user_item_seqs = np.concatenate(
        [user_item_seqs,
         np.expand_dims(items, 1),
         np.expand_dims(items, 1),
         np.expand_dims(items, 1)],
        axis=1)
    candidate_sampling_prob, head_items, _ = prep_candidates(
        user_item_seqs, split_type, use_validation)
    self.assertSequenceAlmostEqual(
        candidate_sampling_prob,
        [0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.2, 0.1, 0.3, 0.2])
    self.assertSequenceEqual(head_items, [4])

  @parameterized.parameters(('step', False), ('user', False), ('step', True),
                            ('user', True))
  def test_load_dataset_for_synthetic_dataset(self, split_type, use_validation):
    data_config = {
        'dataset_name': 'conditional_synthetic',
        'dataset_path': self._synthetic_data_path
    }

    data = dataloader.load_dataset(
        **data_config, split_type=split_type, use_validation=use_validation)
    self.assertIn('train_dataset', data)
    self.assertIn('test_dataset', data)
    self.assertIn('item_dataset', data)
    self.assertIn('num_items', data)

    if use_validation:
      self.assertIn('valid_dataset', data)
      valid_dataset = list(data['valid_dataset'].as_numpy_iterator())

    train_dataset = list(data['train_dataset'].as_numpy_iterator())
    test_dataset = list(data['test_dataset'].as_numpy_iterator())

    self.assertEqual(data['num_items'], 15)
    self.assertIn('candidate_sampling_probability', train_dataset[0])
    self.assertIn('is_head_item', train_dataset[0])
    self.assertLen(test_dataset[0]['user_item_sequence'], 19)

    if split_type == 'step':
      self.assertLen(train_dataset, 100)
      self.assertLen(test_dataset, 100)
      if use_validation:
        self.assertLen(valid_dataset, 100)
        self.assertLen(valid_dataset[0]['user_item_sequence'], 18)
        self.assertLen(train_dataset[0]['user_item_sequence'], 17)
      else:
        self.assertLen(train_dataset[0]['user_item_sequence'], 18)

    elif split_type == 'user':
      if use_validation:
        self.assertLen(train_dataset, 80)
        self.assertLen(valid_dataset, 10)
        self.assertLen(test_dataset, 10)
        self.assertLen(train_dataset[0]['user_item_sequence'], 19)
        self.assertLen(valid_dataset[0]['user_item_sequence'], 19)
      else:
        self.assertLen(train_dataset, 90)
        self.assertLen(test_dataset, 10)
        self.assertLen(train_dataset[0]['user_item_sequence'], 19)

  @parameterized.parameters(('step'), ('user'))
  def test_load_dataset_for_amazon_data(self, split_type):
    data_config = {
        'dataset_name': 'amazon_review_category',
        'dataset_path': self._amazon_data_path
    }

    data = dataloader.load_dataset(**data_config, split_type=split_type)
    self.assertIn('train_dataset', data)
    self.assertIn('test_dataset', data)
    self.assertIn('item_dataset', data)
    self.assertIn('num_items', data)

    train_dataset = list(data['train_dataset'].as_numpy_iterator())
    test_dataset = list(data['test_dataset'].as_numpy_iterator())

    self.assertEqual(data['num_items'], 10)  # 1 extra for padding
    self.assertLen(test_dataset[0]['user_item_sequence'], 29)
    self.assertIn('candidate_sampling_probability', train_dataset[0])

    if split_type == 'step':
      self.assertLen(train_dataset, 4)
      self.assertLen(test_dataset, 4)
      self.assertLen(train_dataset[0]['user_item_sequence'], 28)
    elif split_type == 'user':
      self.assertLen(train_dataset, 3)
      self.assertLen(test_dataset, 1)
      self.assertLen(train_dataset[0]['user_item_sequence'], 29)

  def test_random_dataset_value_error(self):

    data_config = {
        'dataset_name': 'random_dataset_name',
        'dataset_path': self._synthetic_data_path
    }

    with self.assertRaises(ValueError):
      dataloader.load_dataset(**data_config)

  def test_load_preprocessed_real_data(self):
    """Tests test_load_preprocessed_real_data using the dataset in testdata dir.

    See user_item_mapped.txt for the test data being used.
    """

    dataset = dataloader.load_preprocessed_real_data(
        self._amazon_data_path, max_seq_size=5, stride=5)

    self.assertIn('user_item_sequences', dataset)
    self.assertIn('num_items', dataset)
    self.assertLen(dataset['user_item_sequences'], 5)
    self.assertLen(dataset['user_negative_items'], 5)
    self.assertEqual(dataset['num_items'], 9)
    self.assertEqual(dataset['max_seq_size'], 5)

    # user 1
    self.assertSequenceEqual(dataset['user_item_sequences'][0], [0, 0, 2, 3, 1])
    # user 2
    self.assertSequenceEqual(dataset['user_item_sequences'][1], [8, 6, 1, 2, 4])
    # user 3 (split into 2)
    self.assertSequenceEqual(dataset['user_item_sequences'][2], [4, 1, 2, 3, 4])
    self.assertSequenceEqual(dataset['user_item_sequences'][3], [3, 4, 5, 6, 7])
    # user 4
    self.assertSequenceEqual(dataset['user_item_sequences'][4], [0, 0, 0, 6, 9])


if __name__ == '__main__':
  tf.test.main()
