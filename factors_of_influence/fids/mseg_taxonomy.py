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

r"""Taxonomy used for the universal segmentation of the MSEG datasets.

The MSEG_LABEL_DIR needs to point to the list of datasets
and their splits, i.e. a copy of this folder:
https://github.com/mseg-dataset/mseg-api/tree/master/mseg/dataset_lists
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from factors_of_influence import dataset_dirs

MSEG_ROOT_DIR = dataset_dirs.MSEG_ROOT_DIR
MSEG_LABEL_DIR = MSEG_ROOT_DIR + 'dataset_lists/'
MSEG_MASTER_FILE_PATH = dataset_dirs.MSEG_MASTER_FILE_PATH

MSEG_DATASETS = [
    'pascal-context-60', 'scannet-20', 'camvid-11', 'voc2012', 'ade20k-150',
    'ade20k-150-relabeled', 'bdd', 'bdd-relabeled', 'sunrgbd-37',
    'sunrgbd-37-relabeled', 'cityscapes-19', 'cityscapes-19-relabeled',
    'coco-panoptic-133', 'coco-panoptic-133-relabeled', 'idd-39',
    'idd-39-relabeled', 'kitti-19', 'mapillary-public65',
    'mapillary-public65-relabeled', 'wilddash-19'
]


class Taxonomy:
  """MSeg taxonomy helper class.

  Allows to map from dataset vocabularies to the universal taxonomy and back.
  This roughly replicates the official converter
  https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py
  but adapted to our setup, simplified and without the dependency to pytorch.
  """

  def __init__(self,
               tsv_fpath=None):
    """Initializes the taxonomy for the datasets specified.

    For creating the taxonomy, one needs to specify the path to a
    Tab Separated Value file, which defined the mapping from dataset specific
    class names to the universal taxonomy. Specifically, the TSV file needs a
    row for each class in the universal taxonomy, whose name is stored in the
    `universal` column. In addition, the TSV file needs a column for each
    dataset, which defines which dataset classes correspond to that universal
    concept (0,1 or several).
    E.g `coffee table` and `table` of ade20k-150 both map to the universal
    concept `table`. Instead, that concept is absent in camvid-11.

    This class is tested in dataset_test.py.

    Args:
      tsv_fpath: path pointing to the mapping file.
    """
    if tsv_fpath is None:
      tsv_fpath = MSEG_MASTER_FILE_PATH

    with tf.io.gfile.GFile(tsv_fpath, 'r') as f:
      self._tsv_data = pd.read_csv(f, sep='\t', keep_default_na=False)

    self._ignore_label = 255
    # Explicit list of available datasets.
    self._available_datasets = MSEG_DATASETS
    self._class_names = self._load_class_names()
    self._uid2uname, self._uname2uid, self._mappings = self._build_mapping()
    self._mapping_indices = {map_name: np.argmax(mapping, axis=1) for
                             map_name, mapping in self._mappings.items()}

  @property
  def available_datasets(self):
    return self._available_datasets

  @property
  def class_names(self):
    return self._class_names

  @property
  def uid2name(self):
    return self._uid2uname

  @property
  def uname2uid(self):
    return self._uname2uid

  @property
  def mappings(self):
    return self._mappings

  def _load_class_names(self):
    """Loads the class names for all datasets."""
    all_class_names = {}
    for dataset_type in self._available_datasets:
      with tf.io.gfile.GFile(
          f'{MSEG_LABEL_DIR}/{dataset_type}/{dataset_type}_names.txt',
          'r') as f:
        all_class_names[dataset_type] = [l.strip('\n') for l in f]
    return all_class_names

  def _get_class_list(self, entry):
    if not entry:
      return []
    elif entry.startswith('{'):
      return [c.strip() for c in entry.strip('{}').split(',')]
    else:
      return [entry.strip()]

  def _build_mapping(self):
    """Builds the mappings between dataset and universal class vocabularies."""
    uid2uname = {}
    uname2uid = {}
    mappings = {}
    for dataset_type in self._available_datasets:
      mappings[dataset_type] = np.zeros((256, 256), np.uint8)
      # Keeps the ignore region.
      mappings[dataset_type][255, 255] = 1

    for uid, row in self._tsv_data.iterrows():
      u_name = row['universal'].strip()
      assert u_name not in uname2uid  # no duplicate names
      if u_name == 'unlabeled':
        uid = self._ignore_label
      for dataset_type in self._available_datasets:
        class_list = self._get_class_list(row[dataset_type])
        if class_list:
          for cls in class_list:
            dataset_id = self._class_names[dataset_type].index(cls)
            mappings[dataset_type][dataset_id, uid] = 1

      uid2uname[uid] = u_name
      uname2uid[u_name] = uid
    return uid2uname, uname2uid, mappings

  def convert_relabeled_to_universal(self, segmentation_mask, dataset_type):
    """Converts an integer segmentation mask to the universal taxonomy.

    Args:
      segmentation_mask: a numpy array with integers.
      dataset_type: string with the dataset. Should end in '-relabeled'.

    Returns:
      converted segmentation_mask with the same shape.
    """
    if not dataset_type.endswith('-relabeled'):
      raise ValueError(f'Can only convert relabeled to universal. '
                       f'dataset_type given: {dataset_type}')
    mapping_indices = self._mapping_indices[dataset_type]
    return mapping_indices[segmentation_mask]

  def convert_to_universal(self, segmentation_mask, dataset_type):
    """Converts an integer segmentation mask to the universal taxonomy.

    Thereby both are represented as images of shape (h,w,1), i.e. the integer
    values indicates the class. This only works for datasets that do NOT have a
    one to many mapping from the dataset vocabulary to the universal taxonomy.
    This corresponds to the training datasets of the MSeg dataset.

    Args:
      segmentation_mask: a segmentation mask in the dataset vocabulary format
        with shape (h, w, 1).
      dataset_type: a string with the dataset name and type (e.g. 'kitti-19').

    Returns:
      segmentation_mask: a segmentation mask in universal taxonomy format,
      with shape (h, w, 1).
    """
    seg_dataset_one_hot = tf.keras.utils.to_categorical(segmentation_mask, 256)

    # Converts it to the universal taxonomy (binary/few hot format).
    seg_universal_binary = self.convert_to_universal_binary(
        seg_dataset_one_hot, dataset_type)
    assert np.sum(
        seg_universal_binary,
        axis=2).max() == 1, ('one to many mapping, likely due to converting a '
                             'test dataset. Cannot assign a single label')

    # Converts to integer label.
    return np.argmax(seg_universal_binary, axis=-1)

  def convert_to_universal_binary(self, segmentation_mask, dataset_type):
    """Converts a binary (one hot) segmentation mask to the universal taxonomy.

    Args:
      segmentation_mask: a segmentation mask in the dataset vocabulary format
        (one hot), with shape (h, w, 256).
      dataset_type: a string with the dataset name and type (e.g. 'kitti-19').

    Returns:
      segmentation_mask: a segmentation mask in universal taxonomy format (one
      hot), with shape (h, w, 256).
    """
    # Maps to universal taxonomy.
    universal_segmentation_mask = np.tensordot(
        segmentation_mask,
        self._mappings[dataset_type].astype(segmentation_mask.dtype),
        axes=([2], [0]))
    return universal_segmentation_mask

  def convert_from_universal_binary(self, segmentation_mask, dataset_type):
    """Converts a segmentation mask from the universal taxonomy.

    Args:
      segmentation_mask: a segmentation mask in universal taxonomy format (one
        hot), with shape (h, w, 256).
      dataset_type: a string with the dataset name and type (e.g. 'kitti-19')

    Returns:
      segmentation_mask: a segmentation mask in dataset vocabulary format (one
      hot), with shape (h, w, 256).
    """
    # Maps from universal to dataset specific mask.
    return np.tensordot(
        segmentation_mask,
        self._mappings[dataset_type].astype(segmentation_mask.dtype),
        axes=([2], [1]))
