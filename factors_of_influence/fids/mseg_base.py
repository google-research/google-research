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
import functools
import os
from typing import List, Optional, Text

import numpy as np

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import mseg_taxonomy
from factors_of_influence.fids import utils

MSEG_ROOT_DIR = dataset_dirs.MSEG_ROOT_DIR
MSEG_LABEL_DIR = MSEG_ROOT_DIR + 'dataset_lists/'


class MSegBase(
    fids_dataset.FIDSDataset, metaclass=abc.ABCMeta):
  """General class for importing MSeg dataset.

  MSeg datasets provide four different segmentation masks:
  1. segmentation = original annotation (provided by dataset authors).
  2. segmentation_mseg = subset of original segmentation:
      reordering labels and removing unused labels/
  3. segmentation_mseg_relabeled = remapped to the MSEG semantic classes and
      some pixels / segments are re-annotated and assigned to different classes.
      Note: only available for datasets in MSEG train collection.
  4. segmentation_mseg_universal = remapped to the MSEG universal taxonomy which
      allows to relate classes acros different datasets.
      Note: This remaps the semantic labels of 3/ segmentation_mseg_relabeled
  """

  MSEG_FEATURE_NAMES = [
      'image',
      'segmentation',
      'segmentation_mseg',
      'segmentation_mseg_relabeled',
      'segmentation_mseg_universal',
  ]

  def __init__(
      self,
      mseg_name,
      mseg_original_name,
      mseg_base_name,
      mseg_dirname,
      mseg_train_dataset = True,
      mseg_config = 'mseg',
      mseg_segmentation_background_labels = None,
      mseg_use_mapping_for_mseg_segmentation = False,
  ):
    """Constructor for MSeg data.

    Args:
      mseg_name: Descriptive name of the MSEG dataset.
      mseg_original_name: dataset name for the original (raw) annotation.
      mseg_base_name: dataset name for the MSeg annotation.
      mseg_dirname: Directory of images.
      mseg_train_dataset (boolean): Specify if used as mseg train dataset.
      mseg_config (optional): name of the configuration.
      mseg_segmentation_background_labels (optional): List with background
        labels for semantic segmentation feature.
      mseg_use_mapping_for_mseg_segmentation: When False the mseg_segmentation
        feature is created by convert, ie map 255 to 0 (default), if True a
        mapping function is used.
    The used MSeg datasets names can be found here:
    https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/taxonomy_FAQ.md
    """
    self.mseg_name = mseg_name
    self.mseg_original_name = mseg_original_name
    self.mseg_base_name = mseg_base_name
    self.mseg_train_dataset = mseg_train_dataset
    self.mseg_segmentation_background_labels = (
        mseg_segmentation_background_labels or ['unlabeled', 'Unlabeled'])
    self.mseg_use_mapping_for_mseg_segmentation = mseg_use_mapping_for_mseg_segmentation

    if mseg_config == 'mseg' and not self.mseg_train_dataset:
      mseg_config += '_test'

    feature_names = ['image', 'segmentation', 'segmentation_mseg']
    if self.mseg_train_dataset:
      feature_names.extend(
          ['segmentation_mseg_relabeled', 'segmentation_mseg_universal'])

    splits = ['train', 'validation']

    super().__init__(
        name=mseg_name,
        config_name=mseg_config,
        feature_names=feature_names,
        splits=splits)

    self._data_root_dir = os.path.join(MSEG_ROOT_DIR, 'after_remapping',
                                       mseg_dirname)
    self._data_files = {}
    self._taxonomy_converter = None

  @property
  def config_description(self):
    if 'mseg' in self.config_name:
      train_test = 'train' if self.mseg_train_dataset else 'test'
      return f'Dataset {self.name} - MSeg {train_test} collection'

    return super().config_description

  @property
  def taxonomy_converter(self):
    if not self._taxonomy_converter:  # Create here instead of at load time.
      self._taxonomy_converter = mseg_taxonomy.Taxonomy()
    return self._taxonomy_converter

  @functools.lru_cache(maxsize=16)
  def _label_list_from_mapping(self,
                               feature_name = 'segmentation'
                              ):
    """Get segmentation label list from mapping."""
    # This is only used for segmentation and segmentation_mseg iff the option
    # use_mapping_for_mseg_segmentation=True:
    assert feature_name in ['segmentation', 'segmentation_mseg']
    if feature_name == 'segmentation_mseg':
      assert self.mseg_use_mapping_for_mseg_segmentation

    # segmentation_label_mapping can have multiple instances with the same id.
    # For example, from the class void (name=background, original_id=3, id=0),
    # and from the class ignore (name=background, original_id=2, id=0).
    # Create the label list from unique label ids:
    label_mapping = self._segmentation_label_mapping(feature_name)
    label_with_highest_id = max(label_mapping, key=lambda lm: lm.id)
    segmentation_label_list = [None] * (label_with_highest_id.id+1)
    for lm in label_mapping:
      if not segmentation_label_list[lm.id]:
        segmentation_label_list[lm.id] = lm.name
    return segmentation_label_list

  def split_name(self, split):
    """Returns split name."""
    return 'val' if split == 'validation' else split

  def convert_segmentation(self,
                           segmentation,
                           feature_name = 'segmentation'):
    seg_map = self._segmentation_label_mapping(feature_name)
    return utils.convert_segmentation_map(
        segmentation, seg_map)

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if feature_name not in self.MSEG_FEATURE_NAMES:
      raise ValueError(f'Feature {feature_name} not a valid MSEG feature name: '
                       f'{self.MSEG_FEATURE_NAMES}')

    if split not in self._data_files:
      self._create_data_files(split)

    # Deal with missing labels.
    if feature_name not in self._data_files[split]:
      return self.feature_utils.get_fake_feature(feature_name), False
    if curr_id not in self._data_files[split][feature_name]:
      return self.feature_utils.get_fake_feature(feature_name), False

    feature_file = self._data_files[split][feature_name][curr_id]
    if feature_name == 'image':
      # TFDS handles image paths, which are automatically converted to the
      # correct format.
      return feature_file, True

    segmentation = utils.load_png(feature_file)
    if feature_name == 'segmentation_mseg_universal':
      segmentation = self.taxonomy_converter.convert_relabeled_to_universal(
          segmentation, self.mseg_base_name + '-relabeled')
      segmentation = utils.segmentation_set_background_label_to_zero(
          segmentation, old_background_label=255)
    elif feature_name == 'segmentation':
      segmentation = self.convert_segmentation(segmentation, feature_name)
    elif (feature_name == 'segmentation_mseg' and
          self.mseg_use_mapping_for_mseg_segmentation):
      segmentation = self.convert_segmentation(segmentation, feature_name)
    else:
      assert feature_name in [
          'segmentation_mseg_relabeled', 'segmentation_mseg'
      ]
      segmentation = utils.segmentation_set_background_label_to_zero(
          segmentation, old_background_label=255)

    return segmentation, True

  def get_ids(self, split):
    if split not in self._data_files:
      self._create_data_files(split)
    return self._data_files[split]['image'].keys()

  def _label_list_filename(self, feature_name):
    if feature_name == 'segmentation':
      base_name = self.mseg_original_name
    else:
      assert feature_name in [
          'segmentation_mseg', 'segmentation_mseg_relabeled'
      ]
      base_name = self.mseg_base_name
      if feature_name == 'segmentation_mseg_relabeled':
        base_name += '-relabeled'
    return f'{MSEG_LABEL_DIR}/{base_name}/{base_name}_names.txt'

  def _segmentation_label_list(self,
                               feature_name):
    if feature_name == 'segmentation_mseg_universal':
      return ['background'] + [
          self.taxonomy_converter.uid2name[uid]
          for uid in sorted(self.taxonomy_converter.uid2name)
      ]

    if feature_name == 'segmentation_mseg_relabeled':
      label_list_file = self._label_list_filename(feature_name)
      label_list = utils.load_text_to_list(label_list_file)
      return ['background'] + label_list

    if feature_name == 'segmentation_mseg':
      if self.mseg_use_mapping_for_mseg_segmentation:
        return self._label_list_from_mapping(feature_name)
      else:
        label_list_file = self._label_list_filename(feature_name)
        label_list = utils.load_text_to_list(label_list_file)
        return ['background'] + label_list

    assert feature_name == 'segmentation'
    return self._label_list_from_mapping(feature_name)

  @functools.lru_cache(maxsize=16)
  def _segmentation_label_mapping(
      self,
      feature_name = 'segmentation'):
    """Get label mapping for segmentation or segmentation_mseg."""
    assert feature_name in ['segmentation', 'segmentation_mseg']

    original_label_list = utils.load_text_to_list(
        self._label_list_filename(feature_name))

    background_label_list = self.mseg_segmentation_background_labels.copy()
    background_label_list.append('background')

    def _is_background(label):
      return label in background_label_list

    new_label_list = ['background'] + [
        label for label in original_label_list if not _is_background(label)
    ]

    def _new_index(label):
      return 0 if _is_background(label) else new_label_list.index(label)

    # Not all label lists have a background class, add it here explicitly.
    # Not used original_id's are all mapped to 0, ie 255 as implicit background
    # will be mapped to zero by utils.convert_segmentation_map.
    new_label_mapping = [
        utils.LabelMap(name='background', original_id=-1, id=0),
    ]
    for (i, label) in enumerate(original_label_list):
      new_label_mapping.append(
          utils.LabelMap(name=label, original_id=i, id=_new_index(label)))

    return new_label_mapping

  def _info_features(self):
    """Return metadict features for MSeg."""
    return {
        feature_name: self._segmentation_label_list(feature_name)
        for feature_name in self.feature_names
        if feature_name.startswith('segmentation')
    }

  def _create_data_files(self, split):
    """Create data files."""

    filename_and_column_tuples = {  # Keys are equivalent to MSEG_FEATURE_NAMES
        'image': (self.mseg_base_name, 0),
        'segmentation': (self.mseg_original_name, 1),
        'segmentation_mseg': (self.mseg_base_name, 1),
        'segmentation_mseg_relabeled': (self.mseg_base_name + '-relabeled', 1),
        'segmentation_mseg_universal': (self.mseg_base_name + '-relabeled', 1),
    }

    split_name = self.split_name(split)
    data_files_split = {}

    def _feature_path(feature_file):
      return os.path.join(MSEG_LABEL_DIR, feature_file,
                          'list/{}.txt').format(split_name)

    # MSEG files are not cleaned, ie some have multiple entries (non unique
    # values in the first column), this could also be the case for one of the
    # files (eg the segmentation file, and not the image file).
    # We construct keys from the image file, keys are the shortest file path
    # which are unique (starting from filename, then adding parent dir, etc).
    # These keys are used in other files as well.
    feature_file, feature_column = filename_and_column_tuples['image']
    tsv = utils.TSVFileExtractor(
        _feature_path(feature_file), column=feature_column)
    data_files_split['image'] = tsv.to_dict(self._data_root_dir)

    for feature_name in self.feature_names:
      if feature_name not in filename_and_column_tuples:
        continue
      if feature_name in data_files_split:
        continue

      feature_file, feature_column = filename_and_column_tuples[feature_name]
      tsv.set_filename_and_column(_feature_path(feature_file), feature_column)
      data_files_split[feature_name] = tsv.to_dict(self._data_root_dir)

    self._data_files[split] = data_files_split
