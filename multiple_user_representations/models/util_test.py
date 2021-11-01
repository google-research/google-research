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

"""Tests for util."""

import os

from absl import flags
import tensorflow as tf

from multiple_user_representations import dataloader
from multiple_user_representations.models import util

FLAGS = flags.FLAGS
_TESTDATA_DIR = 'third_party/google_research/google_research/multiple_user_representations/testdata'


class UtilTest(tf.test.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()

    dataset_path = os.path.join(_TESTDATA_DIR,
                                'test_synthetic_data')
    data_config = {
        'dataset_name': 'conditional_synthetic',
        'dataset_path': dataset_path
    }

    data = dataloader.load_dataset(**data_config)
    self.train_dataset = data['train_dataset']
    self.item_count_weights = data['item_count_probs']

  def test_update_train_dataset_with_sample_weights(self):

    updated_train_dataset = util.update_train_dataset_with_sample_weights(
        self.train_dataset, self.item_count_weights)

    expected_train_dataset = list(
        self.train_dataset.batch(100).as_numpy_iterator())[0]
    updated_train_dataset = list(
        updated_train_dataset.batch(100).as_numpy_iterator())[0]

    # Check if all dataset features are present in updated dataset.
    for key in expected_train_dataset:
      if key != util.SAMPLE_WEIGHT:
        self.assertAllEqual(expected_train_dataset[key],
                            updated_train_dataset[key])

    # Check is sample_weight key is present in updated dataset.
    self.assertIn(util.SAMPLE_WEIGHT, updated_train_dataset)

    # Check if sample_weights are uniform.
    # Initial weights in self.item_count_weights lead to uniform sample weights.
    sample_weights = updated_train_dataset[util.SAMPLE_WEIGHT]
    number_sequences = len(sample_weights)
    uniform_weights = [1.0 / number_sequences] * number_sequences
    self.assertAllCloseAccordingToType(sample_weights, uniform_weights)


if __name__ == '__main__':
  tf.test.main()
