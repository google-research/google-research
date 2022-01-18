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

"""Tests for retrieval."""

import os

from absl import flags
import tensorflow as tf

from multiple_user_representations import dataloader
from multiple_user_representations.models import density_smoothed_retrieval
from multiple_user_representations.models import parametric_attention
from multiple_user_representations.models import retrieval
from multiple_user_representations.models import task

FLAGS = flags.FLAGS
_TESTDATA_DIR = 'third_party/google_research/google_research/multiple_user_representations/testdata'


class RetrievalTest(tf.test.TestCase):

  def setUp(self):
    super(RetrievalTest, self).setUp()
    dataset_path = os.path.join(_TESTDATA_DIR,
                                'test_synthetic_data')
    data_config = {
        'dataset_name':
            'conditional_synthetic',
        'dataset_path': dataset_path
    }

    data = dataloader.load_dataset(
        **data_config, split_type='user', use_validation=True)
    self.train_dataset = data['train_dataset']
    self.val_dataset = data['valid_dataset']
    self.max_seq_size = data['max_seq_size']
    self.num_items = data['num_items']
    self.item_dataset = data['item_dataset']
    self.item_count_weights = data['item_count_probs']

  def test_retrieval_model(self):
    user_model = parametric_attention.SimpleParametricAttention(
        output_dimension=2,
        input_embedding_dimension=2,
        vocab_size=self.num_items,
        num_representations=3,
        max_sequence_size=self.max_seq_size)
    item_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=2)])
    model = retrieval.RetrievalModel(user_model, item_model,
                                     task.MultiShotRetrievalTask(),
                                     self.num_items)
    self.assertIsInstance(model, tf.keras.Model)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    training_history = model.fit(self.train_dataset.batch(2), epochs=2)
    self.assertLen(training_history.history['loss'], 2)

  def test_density_weighted_retrieval_model(self):

    user_model = parametric_attention.SimpleParametricAttention(
        output_dimension=2,
        input_embedding_dimension=2,
        vocab_size=self.num_items,
        num_representations=3,
        max_sequence_size=self.max_seq_size)
    item_model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=2)])

    model = density_smoothed_retrieval.DensityWeightedRetrievalModel(
        user_model, item_model, task.MultiShotRetrievalTask(), self.num_items)

    self.assertIsInstance(model, tf.keras.Model)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    def train_fn(train_dataset):
      return model.fit(train_dataset.batch(2), epochs=1)

    training_history = model.iterative_training(train_fn, self.train_dataset,
                                                self.item_dataset,
                                                self.item_count_weights)

    self.assertLen(training_history.history['loss'], 1)


if __name__ == '__main__':
  tf.test.main()
