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

"""Import Berkely Deep Drive dataset.

URL: https://bdd-data.berkeley.edu/
URL: URL: https://www.bdd100k.com/

Paper:
BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.
Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu,
Vashisht Madhavan, Trevor Darrell. CVPR 2020.

Note on the dataset:
The dataset contains 1100 hour of driving.
10k frames have been annotated with instance segmentation.
100k frames have been annotated with boxes and drivable areas.
The 10k frames are *not* as subset of the 100k frames, but there is
considerable overlap.
"""
import json
import os
from typing import Iterable, Set, Text, Tuple
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_features
from factors_of_influence.fids import mseg_base
from factors_of_influence.fids import utils

BOX_LABELS = [
    'background',  # Manually added. 0 is always background.
    'bike',
    'bus',
    'car',
    'motor',
    'person',
    'rider',
    'traffic light',
    'traffic sign',
    'train',
    'truck'
]

# Drivabale is a 3-class segmentation problem. See Section A.3:
# https://arxiv.org/pdf/1805.04687.pdf
DRIVABLE_LABELS = ['background', 'direct', 'alternative']

IM_WIDTH = 1280
IM_HEIGHT = 720

# Path with MSEG formatted data.
MSEG_DATA_DIR = os.path.join(mseg_base.MSEG_ROOT_DIR,
                             'after_remapping/BDD/bdd100k/')

# Path with original BDD data.
BDD_100K_DIR = dataset_dirs.BDD_100K_DIR
BDD_100K_PAN_SEG = dataset_dirs.BDD_100K_PAN_SEG

# Possible versions of this dataset
ALL_DENSE_3K = 'all_dense_3k'
MSEG = 'mseg'
DETECTION_100K = 'detection_100k'
ALL_SPARSE_107K = 'all_sparse_107k'


class BDD(mseg_base.MSegBase):
  """Import Berkely Deep Drive (BDD) Dataset.

  Notes:
  - The overlap between segmentation and other annotations is:
  -- train: 2976 images.
  -- validation: 0 images.
  -- test: 830 images.
  """
  json_file = 'labels/bdd100k_labels_images_{}.json'
  drivable_file = 'drivable_maps/labels/{}/{}_drivable_id.png'
  instance_seg_file = 'pan_seg/bitmasks/{}/{}.png'

  # Possible versions of this dataset.
  BDD_TYPES = [ALL_DENSE_3K, ALL_SPARSE_107K, MSEG, DETECTION_100K]

  def __init__(self, bdd_type):
    """Constructor.

    Args:
      bdd_type: select version of this dataset. See BDD_TYPES.
    """
    self.bdd_type = bdd_type
    super().__init__(
        mseg_name='Berkeley Deep Drive (BDD)',
        mseg_original_name='bdd',
        mseg_base_name='bdd',
        mseg_dirname='BDD/bdd100k/',
        mseg_train_dataset=True,
        mseg_config=self.bdd_type)

    self.set_features_and_splits()
    self._data = {}
    self._image_path_10k = (mseg_base.MSEG_ROOT_DIR +
                            'after_remapping/BDD/bdd100k/seg/images/{}/{}.jpg')
    self._image_path_100k = BDD_100K_DIR + 'images/100k/{}/{}.jpg'
    self._ids_without_segmentation = set()
    self._ids_without_detection = set()

  @property
  def bdd_type(self):
    return self._bdd_type

  @bdd_type.setter
  def bdd_type(self, bdd_type):
    assert bdd_type in self.BDD_TYPES
    self._bdd_type = bdd_type

  def _info_features(self):
    info_features = super()._info_features()
    if 'box_labels' in self.feature_names:
      info_features['box_labels'] = BOX_LABELS

    if 'drivable' in self.feature_names:
      info_features['drivable'] = DRIVABLE_LABELS

    return info_features

  def set_features_and_splits(self):
    box_feature_args = {'box_labels': {'num_box_labels': len(BOX_LABELS)}}
    self.feature_utils = fids_features.FeatureUtils(box_feature_args)

    mseg_segmentation_features = self.MSEG_FEATURE_NAMES.copy()
    mseg_segmentation_features.remove('image')

    if self.bdd_type == ALL_DENSE_3K:
      self.feature_names = self.MSEG_FEATURE_NAMES + [
          'boxes', 'box_labels', 'drivable', 'instance_segmentation'
      ]
      self.splits = ['train', 'test']
      self.splits_with_missing_features = {
          'test': mseg_segmentation_features + [
              'boxes', 'box_labels', 'drivable',
          ]
      }
    elif self.bdd_type == MSEG:
      self.feature_names = self.MSEG_FEATURE_NAMES + ['instance_segmentation']
      self.splits = ['train', 'validation', 'test']
      self.splits_with_missing_features = {
          'test': mseg_segmentation_features + ['instance_segmentation']
      }
    elif self.bdd_type == DETECTION_100K:
      self.feature_names = ['image', 'boxes', 'box_labels', 'drivable']
      self.splits = ['train', 'validation', 'test']
      self.splits_with_missing_features = {
          'test': ['boxes', 'box_labels', 'drivable']
      }
    else:
      assert self.bdd_type == ALL_SPARSE_107K, (
          f'Unexpected BDD type {self.bdd_type}')

      self.feature_names = self.MSEG_FEATURE_NAMES + [
          'boxes', 'box_labels', 'drivable', 'instance_segmentation',
      ]
      self.splits = ['train', 'validation', 'test']
      missing_features = mseg_segmentation_features + [
          'boxes', 'box_labels', 'drivable', 'instance_segmentation'
      ]
      self.splits_with_missing_features = {
          'train': missing_features,
          'validation': missing_features,
          'test': missing_features
      }

  def get_ids(self, split):
    if self.bdd_type == ALL_DENSE_3K:
      ids_segmentation = self.get_ids_mseg(split)
      ids_detection = self.get_ids_detection(split)
      intersection = ids_segmentation.intersection(ids_detection)
      return intersection
    elif self.bdd_type == MSEG:
      return self.get_ids_mseg(split)
    elif self.bdd_type == DETECTION_100K:
      return self.get_ids_detection(split)
    else:
      assert self.bdd_type == ALL_SPARSE_107K, (
          f'Unexpected BDD type {self.bdd_type}')

      ids_segmentation = self.get_ids_mseg(split)
      ids_detection = self.get_ids_detection(split)
      union = ids_segmentation.union(ids_detection)
      self._ids_without_segmentation = union.difference(ids_segmentation)
      self._ids_without_detection = union.difference(ids_detection)
      return union

  def get_ids_mseg(self, split):
    # MSeg does not contain any test set. Scrape images for separate download.
    if split == 'test':
      test_path = os.path.join(MSEG_DATA_DIR, 'seg/images/test/')
      file_names = tf.io.tf.io.gfile.listdir(test_path)
      image_names = [f[:-4] for f in file_names if f.endswith('.jpg')]
      return set(image_names)
    else:
      return set(super().get_ids(split))

  def get_ids_detection(self, split):
    """Get IDs for the 100k images with detections."""
    if split == 'test':  # test set has no json file. Scrape ids from directory.
      file_names = tf.io.gfile.listdir(
          os.path.dirname(self._image_path_100k.format(split, '')))
      image_names = [f[:-4] for f in file_names if f.endswith('.jpg')]
      return set(image_names)

    if split not in self._data:
      self.process_json(split)
    return self._data[split].keys()

  def get_feature(self, split, curr_id, feature_name):
    if feature_name == 'image':
      split_path = self.split_name(split)
      if tf.io.gfile.exists(self._image_path_100k.format(split_path, curr_id)):
        return self._image_path_100k.format(split_path, curr_id), True
      elif tf.io.gfile.exists(self._image_path_10k.format(split_path, curr_id)):
        return self._image_path_10k.format(split_path, curr_id), True
      else:
        raise ValueError(f'image {curr_id} not found!')

    if split == 'test':  # There are no annotations in the test set.
      return self.feature_utils.get_fake_feature(feature_name), False

    if feature_name in self.MSEG_FEATURE_NAMES and feature_name != 'image':
      return super().get_feature(split, curr_id, feature_name)

    if feature_name == 'instance_segmentation':
      return self.get_feature_instance_segmentation(split, curr_id)

    # At this point, only non-mseg features.
    if curr_id in self._ids_without_detection:
      return self.feature_utils.get_fake_feature(feature_name), False

    if feature_name == 'drivable':
      drivable_file = os.path.join(
          BDD_100K_DIR,
          self.drivable_file.format(self.split_name(split), curr_id))

      drivable_im = utils.load_png(drivable_file)
      drivable_im = drivable_im.astype(np.uint16)
      return drivable_im, True

    if split not in self._data:
      self.process_json(split)
    return self._data[split][curr_id][feature_name], True

  def get_feature_instance_segmentation(
      self, split, curr_id):
    """Get Instance Segmentation from Bitmask file."""

    bitmask_file = os.path.join(
        BDD_100K_PAN_SEG,
        self.instance_seg_file.format(self.split_name(split), curr_id))

    if not tf.io.gfile.exists(bitmask_file):
      return self.feature_utils.get_fake_feature('instance_segmentation'), False

    bit = utils.load_image(bitmask_file)
    # Description: https://doc.bdd100k.com/format.html#bitmask:
    # the B channel and A channel store the “ann_id” for instance segmentation
    # and “ann_id” for segmentation tracking, respectively, which can be
    # computed as (B << 8) + A:
    instance_ids = bit[:, :, 2] * 256 + bit[:, :, 3]
    instance_mask = np.unique(instance_ids, return_inverse=True)[1]
    desired_shape = list(instance_ids.shape[:2]) + [1]
    instance_mask = np.reshape(instance_mask, desired_shape)
    return instance_mask.astype(np.uint16), True

  def label_name_to_one_hot(self, label_name):
    """Convert label name to one_hot vector."""
    label_name_to_int = {n: i for i, n in enumerate(BOX_LABELS)}
    label_id = label_name_to_int[label_name]
    one_hot = np.zeros(shape=[len(BOX_LABELS)], dtype=np.float32)
    one_hot[label_id] = 1
    return one_hot

  def process_json(self, split):
    split_name = self.split_name(split)
    filename = os.path.join(BDD_100K_DIR, self.json_file.format(split_name))
    with tf.io.gfile.GFile(filename, 'rb') as f:
      labels = json.load(f)

    split_data = {}
    for annotation in labels:
      key = annotation['name'][:-4]
      box_annotations = annotation['labels']
      curr_boxes = []
      curr_labels = []
      for box_annotation in box_annotations:
        if 'box2d' in box_annotation:
          curr_boxes.append(
              tfds.features.BBox(
                  xmin=box_annotation['box2d']['x1'] / IM_WIDTH,
                  xmax=box_annotation['box2d']['x2'] / IM_WIDTH,
                  ymin=box_annotation['box2d']['y1'] / IM_HEIGHT,
                  ymax=box_annotation['box2d']['y2'] / IM_HEIGHT))
          curr_labels.append(
              self.label_name_to_one_hot(box_annotation['category']))
      split_data[key] = {'boxes': curr_boxes, 'box_labels': curr_labels}

    self._data[split] = split_data
