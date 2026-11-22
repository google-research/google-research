# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Tests for video_structure.datasets."""

import os
from absl import flags
from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf
from video_structure import datasets

FLAGS = flags.FLAGS

TESTDATA_DIR = 'video_structure/testdata'


class GetSequenceDatasetTest(tf.test.TestCase):

  def setUp(self):
    super(GetSequenceDatasetTest, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir, TESTDATA_DIR)
    self.file_glob = 'acrobot*.npz'
    self.batch_size = 4
    self.num_timesteps = 2
    self.num_channels = 3

  def get_dataset(self, batch_size=None, random_offset=True, seed=0):
    return datasets.get_sequence_dataset(
        data_dir=self.data_dir,
        file_glob=self.file_glob,
        batch_size=batch_size or self.batch_size,
        num_timesteps=self.num_timesteps,
        random_offset=random_offset,
        seed=seed)

  def testOutputShapes(self):
    dataset, _ = self.get_dataset()

    expected_keys = {'image', 'true_object_pos'}
    self.assertEqual(
        expected_keys,
        set(dataset.output_shapes.keys()).intersection(expected_keys))

    self.assertEqual(
        dataset.output_shapes['image'],
        [self.batch_size, self.num_timesteps, 64, 64, self.num_channels])

    self.assertEqual(dataset.output_shapes['true_object_pos'],
                     [self.batch_size, self.num_timesteps, 0, 2])

  def testImageRange(self):
    dataset, _ = self.get_dataset()
    dataset_iterator = dataset.make_one_shot_iterator()
    with self.session() as sess:
      batch = sess.run(dataset_iterator.get_next())
    max_val = np.max(batch['image'])
    min_val = np.min(batch['image'])
    self.assertLessEqual(max_val, 0.5)
    self.assertGreaterEqual(min_val, -0.5)
    self.assertGreater(max_val - min_val, 0.25,
                       'Image range is suspiciously small.')

  def testFeedingToModel(self):
    """Build a simple Keras model and test that it trains with the datasets."""

    dataset, _ = self.get_dataset()

    inputs = tf.keras.Input(shape=(self.num_timesteps, 64, 64, 3), name='image')
    conv_layer = tf.keras.layers.Conv2D(
        3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    outputs = tf.keras.layers.TimeDistributed(conv_layer)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.add_loss(tf.nn.l2_loss(inputs - outputs))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2, clipnorm=1.))

    model.fit(dataset, steps_per_epoch=10, epochs=1)

  def testOrderIsDeterministic(self):
    """Tests that data order is deterministic if random_offset is False."""

    def get_new_dataset():
      dataset = self.get_dataset(batch_size=32, random_offset=False)[0]
      return dataset.make_one_shot_iterator()

    with self.session() as sess:
      repeats = [sess.run(get_new_dataset().get_next()) for _ in range(2)]

    # Check that order is reproducible:
    np.testing.assert_array_equal(repeats[0]['filename'],
                                  repeats[1]['filename'])
    np.testing.assert_array_equal(repeats[0]['frame_ind'],
                                  repeats[1]['frame_ind'])

  def testRandomOffset(self):
    """Tests that data order is random if random_offset is True."""

    def get_new_dataset(seed):
      dataset = self.get_dataset(
          batch_size=32, random_offset=True, seed=seed)[0]
      return dataset.make_one_shot_iterator()

    with self.session() as sess:
      repeats = [sess.run(get_new_dataset(seed).get_next()) for seed in [0, 1]]

    # Check that two calls to a fresh dataset pipeline return different orders
    # (this test is technically flaky, but at a very low probability):
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(repeats[0]['frame_ind'],
                                    repeats[1]['frame_ind'])


if __name__ == '__main__':
  absltest.main()
