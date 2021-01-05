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

"""Tests for ...datasets.scannet_scene."""

from absl.testing import parameterized
import tensorflow as tf

from tf3d import data_provider
from tf3d.datasets import scannet_scene


class ScannetSceneTest(parameterized.TestCase, tf.test.TestCase):

  def test_tf_data_feature_label_keys(self):
    """Tests the ability of a get_tf_data_datasets to have extra labels/key.

    Test is done here because TAP is off in specific dataset tests.
    """
    features_data = data_provider.get_tf_data_dataset(
        dataset_name='scannet_scene',
        split_name='val',
        batch_size=1,
        preprocess_fn=None,
        is_training=True,
        num_readers=2,
        num_parallel_batches=2,
        shuffle_buffer_size=2)
    features = next(iter(features_data))

    self.assertEqual(
        features['mesh/vertices/positions'].get_shape().as_list()[2], 3)
    self.assertEqual(features['mesh/vertices/normals'].get_shape().as_list()[2],
                     3)
    self.assertEqual(features['mesh/vertices/colors'].get_shape().as_list()[2],
                     4)
    self.assertEqual(features['mesh/faces/polygons'].get_shape().as_list()[2],
                     3)
    self.assertEqual(
        features['mesh/vertices/semantic_labels'].get_shape().as_list()[2], 1)
    self.assertEqual(
        features['mesh/vertices/instance_labels'].get_shape().as_list()[2], 1)

  def test_get_feature_keys(self):
    feature_keys = scannet_scene.get_feature_keys()
    self.assertIsNotNone(feature_keys)

  def test_get_label_keys(self):
    label_keys = scannet_scene.get_label_keys()
    self.assertIsNotNone(label_keys)

  def test_get_file_pattern(self):
    file_pattern = scannet_scene.get_file_pattern('train')
    self.assertIsNotNone(file_pattern)

  def test_get_decode_fn(self):
    decode_fn = scannet_scene.get_decode_fn()
    self.assertIsNotNone(decode_fn)


if __name__ == '__main__':
  tf.test.main()
