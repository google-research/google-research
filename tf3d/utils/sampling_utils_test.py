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

"""Tests for tf3d.utils.sampling_utils."""

import numpy as np
import tensorflow as tf

from tf3d.utils import sampling_utils


class SamplingUtilsTest(tf.test.TestCase):

  def _get_instance_id_example(self):
    return tf.constant([[1, 1, 1, 3, 2, 3, 7],
                        [1, 1, 0, 0, 0, 0, 0]], dtype=tf.int32)

  def _get_features_example(self):
    return tf.constant([[[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1],
                         [4, 4, 4],
                         [5, 5, 5],
                         [4, 4, 4],
                         [7, 7, 7]],
                        [[8, 8, 8],
                         [8, 8, 8],
                         [10, 10, 10],
                         [10, 10, 10],
                         [10, 10, 10],
                         [10, 10, 10],
                         [10, 10, 10]]], dtype=tf.float32)

  def _get_valid_mask(self):
    return tf.cast(tf.constant([[1, 1, 1, 1, 1, 0, 0],
                                [1, 0, 1, 1, 1, 0, 0]],
                               dtype=tf.int32), dtype=tf.bool)

  def test_get_instance_id_count(self):
    instance_ids = self._get_instance_id_example()
    instance_id_count = sampling_utils.get_instance_id_count(
        instance_ids=instance_ids)
    np_expected_instance_id_count = np.array(
        [[3, 3, 3, 2, 1, 2, 1],
         [2, 2, 5, 5, 5, 5, 5]], dtype=np.float32)
    self.assertAllClose(instance_id_count.numpy(),
                        np_expected_instance_id_count)

  def test_get_instance_id_count_with_max(self):
    instance_ids = self._get_instance_id_example()
    instance_id_count = sampling_utils.get_instance_id_count(
        instance_ids=instance_ids, max_instance_id=4)
    np_expected_instance_id_count = np.array(
        [[3, 3, 3, 2, 1, 2, 0],
         [2, 2, 5, 5, 5, 5, 5]], dtype=np.float32)
    self.assertAllClose(instance_id_count.numpy(),
                        np_expected_instance_id_count)

  def test_get_instance_id_count_with_valid_mask(self):
    instance_ids = self._get_instance_id_example()
    valid_mask = self._get_valid_mask()
    instance_id_count = sampling_utils.get_instance_id_count(
        instance_ids=instance_ids, valid_mask=valid_mask)
    np_expected_instance_id_count = np.array(
        [[3, 3, 3, 1, 1, 0, 0],
         [1, 0, 3, 3, 3, 0, 0]], dtype=np.float32)
    self.assertAllClose(instance_id_count.numpy(),
                        np_expected_instance_id_count)

  def test_get_balanced_sampling_probability(self):
    instance_ids = self._get_instance_id_example()
    probabilities = sampling_utils.get_balanced_sampling_probability(
        instance_ids=instance_ids)
    expected_probabilities = np.array([[
        1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
        1.0 / 4.0
    ], [1.0 / 4.0, 1.0 / 4.0, 0.1, 0.1, 0.1, 0.1, 0.1]],
                                      dtype=np.float32)
    self.assertAllClose(probabilities.numpy(), expected_probabilities)

  def test_get_balanced_sampling_probability_with_valid_mask(self):
    instance_ids = self._get_instance_id_example()
    valid_mask = self._get_valid_mask()
    probabilities = sampling_utils.get_balanced_sampling_probability(
        instance_ids=instance_ids, valid_mask=valid_mask, max_instance_id=4)
    expected_probabilities = np.array(
        [[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
         [0.5, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]],
        dtype=np.float32)
    self.assertAllClose(probabilities.numpy(), expected_probabilities)

  def test_balanced_sample(self):
    features = self._get_features_example()
    instance_ids = self._get_instance_id_example()
    sampled_features, sampled_instance_ids, sampled_indices = (
        sampling_utils.balanced_sample(
            features=features,
            instance_ids=instance_ids,
            num_samples=20,
            max_instance_id=4))
    self.assertAllEqual(sampled_features.shape, np.array([2, 20, 3]))
    self.assertAllEqual(sampled_instance_ids.shape, np.array([2, 20]))
    self.assertAllEqual(sampled_indices.shape, np.array([2, 20]))

  def test_balanced_sample_with_valid_mask(self):
    features = self._get_features_example()
    valid_mask = self._get_valid_mask()
    instance_ids = self._get_instance_id_example()
    sampled_features, sampled_instance_ids, sampled_indices = (
        sampling_utils.balanced_sample(
            features=features,
            instance_ids=instance_ids,
            num_samples=20,
            valid_mask=valid_mask,
            max_instance_id=4))
    self.assertAllEqual(sampled_features.shape, np.array([2, 20, 3]))
    self.assertAllEqual(sampled_instance_ids.shape, np.array([2, 20]))
    self.assertAllEqual(sampled_indices.shape, np.array([2, 20]))


if __name__ == '__main__':
  tf.test.main()
