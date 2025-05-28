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

"""Tests for fids.fids_dataset."""
import tensorflow as tf

from factors_of_influence.fids import fids_dataset

NUM_EXAMPLES_IN_DATASET = 10


class _FIDSDataset(fids_dataset.FIDSDataset):
  """Implements abstract functions for testing."""

  def __init__(self, feature_names, splits, splits_with_missing_features=None):
    super().__init__(
        name='TestDataset',
        config_name='test',
        feature_names=feature_names,
        splits=splits,
        splits_with_missing_features=splits_with_missing_features)

  def _info_features(self):
    info_features = {}
    required_in_meta = self.feature_utils.FEATURES_REQUIRED_IN_METADICT
    for feature in self.feature_names:
      if feature in required_in_meta:
        info_features[feature] = required_in_meta[feature] if required_in_meta[
            feature] else 'feature_meta'

    return info_features

  def get_ids(self, split):
    return [f'{ids}' for ids in range(NUM_EXAMPLES_IN_DATASET)]

  def get_feature(self, split, curr_id, feature_name):
    """Get feature. Assumes non-image features for 'test' are not present."""
    if split == 'test' and feature_name != 'image':
      is_present = False
    else:
      is_present = True

    feature = self.feature_utils.get_fake_feature(feature_name)

    return feature, is_present


class FIDSDatasetTest(tf.test.TestCase):

  def testCreateFIDSDataset(self):
    feature_names = ['image', 'segmentation']
    splits = ['train', 'test']
    splits_with_missing_features = {'test': ['segmentation']}

    dataset = _FIDSDataset(
        feature_names=feature_names,
        splits=splits,
        splits_with_missing_features=splits_with_missing_features)

    self.assertEqual(feature_names, dataset.feature_names)
    self.assertEqual(splits, dataset.splits)
    self.assertEqual(dataset.splits_with_missing_features['train'], [])
    self.assertEqual(dataset.splits_with_missing_features['test'],
                     ['segmentation'])

  def testDatasetWithInvalidFeatureNamesRaisesError(self):
    with self.assertRaises(AssertionError):
      _FIDSDataset(
          feature_names=['invalid_feature_name'], splits=['train'])

  def testDatasetWithInvalidSplitsRaisesError(self):
    with self.assertRaises(AssertionError):
      _FIDSDataset(feature_names=['image'], splits=['invalid_split'])

  def testGenerateExamples(self):
    dataset = _FIDSDataset(
        feature_names=['image', 'segmentation'], splits=['train'])

    num_examples = 0
    for _, example in dataset.generate_examples('train'):
      num_examples += 1

      expected_feature_names = ['id', 'image', 'segmentation', 'is_present']
      for feature_name in example:
        self.assertIn(feature_name, expected_feature_names)
      self.assertEqual(len(example), len(expected_feature_names))

      for is_present, value in example['is_present'].items():
        self.assertIn(is_present, expected_feature_names)
        self.assertTrue(value)
      self.assertLen(
          example['is_present'],
          len(expected_feature_names) - 2)  # minus id, is_present.
    self.assertEqual(num_examples, NUM_EXAMPLES_IN_DATASET)

  def testGenerateExamplesMissingFeatureFailsByDefault(self):
    dataset = _FIDSDataset(
        feature_names=['image', 'segmentation'], splits=['test'])

    with self.assertRaises(AssertionError):
      for _ in dataset.generate_examples('test'):
        pass

  def testGenerateExamplesWithMissingFeatures(self):
    dataset = _FIDSDataset(
        feature_names=['image', 'segmentation'],
        splits=['test'],
        splits_with_missing_features={'test': ['segmentation']})

    num_examples = 0
    for _, example in dataset.generate_examples('test'):
      num_examples += 1
      # Ensure segmentation is missing as intended by _FIDSDataset.
      self.assertFalse(example['is_present']['segmentation'])

    self.assertEqual(num_examples, NUM_EXAMPLES_IN_DATASET)

  def testDatasetWrongSplitInMissingFeaturesRaisesError(self):
    with self.assertRaises(AssertionError):
      _FIDSDataset(
          feature_names=['image'],
          splits=['test'],
          splits_with_missing_features={'train': ['segmentation']
                                       })  # train is not in splits.

  def testDatasetWrongFeatureNameInMissingFeaturesRaisesError(self):
    with self.assertRaises(AssertionError):
      _FIDSDataset(
          feature_names=['image'],
          splits=['test'],
          splits_with_missing_features={
              'test': ['segmentation'],  # segmentation is not in feature_names.
          })

  def testMetadataDictCreation(self):
    dataset = _FIDSDataset(
        feature_names=['image', 'segmentation'], splits=['test'])

    dataset_info = dataset.info()  # checks if required features are present.
    self.assertIn('features', dataset_info)

  def testMetadataDictCreationWithMissingRequiredMetaFeatures(self):

    class _FIDSDatasetWithMissingRequiredMeta(_FIDSDataset):

      def _info_features(self):
        return {'image': 'image_meta'}

    dataset = _FIDSDatasetWithMissingRequiredMeta(
        feature_names=['image', 'segmentation'], splits=['test'])

    with self.assertRaises(AssertionError):
      dataset.info()  # Should fail: segmentation requires a meta-dict entry.

  def testMetadataDictCreationWithTooManyMetaFeatures(self):

    class _FIDSDatasetWithTooManyMeta(_FIDSDataset):

      def _info_features(self):
        return {'segmentation': 'segmentation_meta'}

    dataset = _FIDSDatasetWithTooManyMeta(
        feature_names=['image'], splits=['test'])

    with self.assertRaises(AssertionError):
      dataset.info()  # Should fail: segmentation (in meta) is not a feature.

  def testMetadataDictCreationWithMissingRequiredFields(self):

    class _FIDSDatasetWithMissingFields(_FIDSDataset):

      def _info_features(self):
        return {'depth': dict(default_clip_min=0.00)}

    dataset = _FIDSDatasetWithMissingFields(
        feature_names=['image', 'depth'], splits=['test'])

    with self.assertRaises(AssertionError):
      dataset.info()  # Should fail: field missing in meta for depth

if __name__ == '__main__':
  tf.test.main()
