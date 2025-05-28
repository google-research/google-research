# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests fids.fids_base."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from factors_of_influence.fids import fids_features


class FeaturesTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    feature_args = {
        'box_labels': {'num_box_labels': 5},
        'person_keypoints': {'num_keypoints': 16},
        'person_position': {'dim_per_position': 4},
    }
    self.features = fids_features.FeatureUtils(feature_args)

  def testGetFeatureType(self):
    """Ensures that all tfds types are implemented."""
    for feature_name in self.features.feature_names:
      _ = self.features.get_tfds_type(feature_name)

  def testGetFakeFeature(self):
    """Ensures that all features can return fake examples."""
    for feature_name in self.features.feature_names:
      _ = self.features.get_fake_feature(feature_name)

  @parameterized.named_parameters({
      'testcase_name': 'image',
      'feature_name': 'image',
      'expected': np.zeros([1, 1, 3], dtype=np.uint8)
  }, {
      'testcase_name': 'segmentation',
      'feature_name': 'segmentation',
      'expected': np.zeros([1, 1, 1], dtype=np.uint16)
  }, {
      'testcase_name': 'depth',
      'feature_name': 'depth',
      'expected': np.zeros([1, 1, 1], dtype=np.float32)
  }, {
      'testcase_name': 'boxes',
      'feature_name': 'boxes',
      'expected': [tfds.features.BBox(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)]
  }, {
      'testcase_name': 'box_labels',
      'feature_name': 'box_labels',
      'feature_args': {'box_labels': {'num_box_labels': 3}},
      'expected': [np.array([0, 0, 0], dtype=np.float32)]  # num_box_labels == 3
  })
  def testGetSpecificFakeFeature(
      self, feature_name, expected, feature_args=None):
    feature_utils = fids_features.FeatureUtils(feature_args)
    fake_feature = feature_utils.get_fake_feature(feature_name)
    if isinstance(expected, np.ndarray):
      self.assertAllEqual(fake_feature, expected)
    elif isinstance(expected, list):
      self._assertSequenceEqual(fake_feature, expected)
    else:
      self.assertEqual(fake_feature, expected)

  def testGetTFDSFeaturesDict(self):
    feature_names = ['image', 'boxes']
    feature_types = self.features.get_tfds_features_dict(feature_names)
    for feature_name in feature_names:
      self.assertIn(feature_name, feature_types)

    # feature_types also include 'id' and 'is_present', hence (len + 2).
    self.assertLen(feature_types, len(feature_names) + 2)
    self.assertIn('id', feature_types)
    self.assertIn('is_present', feature_types)

    is_present = feature_types['is_present']
    for feature_name in feature_names:
      self.assertIn(feature_name, is_present)
      self.assertIsInstance(is_present[feature_name], tfds.features.Tensor)
      self.assertEqual(is_present[feature_name].dtype, tf.bool)
    self.assertEqual(len(feature_names), len(is_present))

  def testWrongFeatureNameInFeatureArgsRaisesAttributeError(self):
    with self.assertRaises(AttributeError):
      fids_features.FeatureUtils({'boxlabels': {'num_box_labels': 3}})

  @parameterized.named_parameters({
      'testcase_name': 'misspelled_param',
      'feature_args': {'box_labels': {'nm_box_labels': 3}}
  }, {
      'testcase_name': 'non_existing_param',
      'feature_args': {'id': {'non_existing_param': 1}}
  })
  def testWrongNamedArgsInFeatureArgsRaisesTypeError(self, feature_args):
    with self.assertRaises(TypeError):
      fids_features.FeatureUtils(feature_args)

  def _assertSequenceEqual(self, fake_feature, expected):  # pylint: disable=invalid-name
    for d, e in zip(fake_feature, expected):
      self.assertAllEqual(d, e)


if __name__ == '__main__':
  tf.test.main()
