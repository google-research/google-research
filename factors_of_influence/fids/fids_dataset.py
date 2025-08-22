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

"""Define the FIDS sourceset collection."""
import abc
import re

from typing import Any, Dict, Generator, Iterable, List, Optional, Text, Tuple


import tensorflow_datasets as tfds

from factors_of_influence.fids import fids_features

VALID_SPLITS = ['train', 'validation', 'test']


class FIDSDataset(metaclass=abc.ABCMeta):
  """Base class for FIDS TFDS dataset collection.

  To use this class, implement:
  - get_ids(self, split): return iterable over keys of the examples.
  - get_feature(self, split, curr_id, feature):
      return (feature, feature_is_present). The feature should be of a type
      which can be converted to the corresponding tfds.feature defined in
      fids_features, eg numpy arrays, paths to images, etc., see:
      https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/features/feature.py
  - get_features_args: (optional) if one of the features needs an extra
      argument, such as the number of labels.
  After this, generate_examples will yield key, example pairs.
  """

  def __init__(self,
               name,
               config_name,
               feature_names,
               splits,
               splits_with_missing_features = None,
               feature_args = None):
    """Constructor.

    Args:
      name: name of the dataset.
      config_name: name of the configuration of the dataset
      feature_names: list of features available in this dataset.
      splits: list of splits available in this dataset.
      splits_with_missing_features: Dictionary which maps each split to a list
        of feature_names which may be missing. By default, a missing feature
        will thow an error during dataset creation.
      feature_args: Dictionary, mapping feature_names to named arguments for
          for each feature. Example:
              feature_args={'box_labels': {'num_box_labels': 5}} See TfdsTypes
                in the fids_features module for which feature_name
                consumes which arguments.
    """
    self.name = name
    self.config_name = config_name
    self.feature_utils = fids_features.FeatureUtils(feature_args)
    self.feature_names = feature_names  # Validation depends on feature_utils
    self.splits = splits
    self.splits_with_missing_features = splits_with_missing_features

    # Set to positive value during debugging to limit the number of examples
    # which are created per split.
    self._debug_num_examples_per_split = 0

  @property
  def splits_with_missing_features(self):
    return self._splits_with_missing_features

  @splits_with_missing_features.setter
  def splits_with_missing_features(self, splits_with_missing_features):
    if splits_with_missing_features is None:
      splits_with_missing_features = {}

    for split, missing_features in splits_with_missing_features.items():
      assert split in self.splits
      for feature_name in missing_features:
        assert feature_name in self.feature_names, (
            'Feature_name %s not in self feature_names %s' %
            (feature_name, self.feature_names))

    for split in self.splits:
      if split not in splits_with_missing_features:
        splits_with_missing_features[split] = []

    self._splits_with_missing_features = splits_with_missing_features

  @property
  def config_description(self):
    return f'Dataset {self.name} - config: {self.config_name}'

  @property
  def feature_names(self):
    return self._feature_names

  @feature_names.setter
  def feature_names(self, feature_names):
    for feature in feature_names:
      assert feature in self.feature_utils.feature_names, (
          'Feature %s not in feature_names %s' %
          (feature, self.feature_utils.feature_names))
    self._feature_names = feature_names

  @property
  def splits(self):
    return self._splits

  @splits.setter
  def splits(self, splits):
    for split in splits:
      assert split in VALID_SPLITS, ('Split %s not in VALID_SPLITS %s' %
                                     (split, VALID_SPLITS))
    self._splits = splits

  @abc.abstractmethod
  def _info_features(self):
    """Return feature dictionary with meta-information."""

  def info(self):
    """Return and validate dictionary with meta-information."""
    metadict = {
        'name': self.name,
        'config': self.config_name,
        'description': self.config_description,
        'features': self._info_features()
    }

    provided_features = set(self.feature_names)
    required_meta = provided_features.intersection(
        self.feature_utils.FEATURES_REQUIRED_IN_METADICT.keys())
    provided_meta = set(metadict['features'])

    missing_required = required_meta.difference(provided_meta)
    assert not missing_required, ('Missing required features: %s' %
                                  sorted(missing_required))

    assert provided_meta.issubset(provided_features), (
        f'Features provided in metadict are not features:'
        f'FEATURES: {provided_features}\n META: {provided_meta}')

    for feature in required_meta:
      required_fields = set(
          self.feature_utils.FEATURES_REQUIRED_IN_METADICT[feature])
      if not required_fields: continue
      provided_fields = set(metadict['features'][feature].keys())
      assert provided_fields == required_fields, (
          f'Fields for feature {feature} do not match required fields: '
          f'PROVIDED: {provided_fields}\n REQUIRED: {required_fields}')

    return metadict

  def generate_examples(
      self, split
  ):
    """Yields examples (data with annotations).

    This function assumes that each feature implements its own getter function.
    The getter function returns a dictionary mapping keys to examples (examples
    may be image paths). The getter function is named f'_get_{feature}_dict'.

    Args:
      split: Split of the dataset.

    Yields:
      key, examples pairs.
    """
    assert split in self.splits
    if split in self.splits_with_missing_features:
      possible_missing_features = self.splits_with_missing_features[split]
    else:
      possible_missing_features = []

    ids = self.get_ids(split)
    features = [f for f in self.feature_names if f != 'id']
    for example_nr, curr_id in enumerate(ids):
      if (self._debug_num_examples_per_split and
          example_nr >= self._debug_num_examples_per_split):
        break

      # Build single example.
      plain_id = re.sub(r'\W+', '_', curr_id)  # Plain id replaces non alpha-
      example = {'id': plain_id}  # numeric to '_' to remove path characters.
      feature_present = {}
      for feature_name in features:
        example[feature_name], feature_present[feature_name] = self.get_feature(
            split, curr_id, feature_name)
        if feature_name not in possible_missing_features:
          assert feature_present[feature_name], (
              f'Feature {feature_name} is missing {split} - {curr_id}')
      example['is_present'] = feature_present

      yield curr_id, example  # pytype: disable=bad-return-type  # gen-stub-imports

  @abc.abstractmethod
  def get_ids(self, split):
    """Returns iterable over example ids, which are strings."""
    pass

  @abc.abstractmethod
  def get_feature(self, split, curr_id,
                  feature_name):
    """Return (feature, feature_is_present) given split, id, feature name."""
    pass
