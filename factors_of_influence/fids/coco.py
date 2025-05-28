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

"""Defines Coco Panoptic, including the MSeg version.

URL: https://cocodataset.org/

Papers:
- Microsoft COCO: Common objects in context
  T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
  P. Dollar, and C. Zitnick. In ECCV, 2014.
- COCO-Stuff: Thing and stuff classes in context.
  H. Caesar, J. Uijlings, and V. Ferrari. In CVPR, 2018.
- Panoptic segmentation.
  A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollar. In CVPR, 2019.
"""
from typing import Any, Dict, List, Text, Tuple

import numpy as np
import tensorflow_datasets as tfds

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_features
from factors_of_influence.fids import mseg_base
from factors_of_influence.fids import utils
from factors_of_influence.fids.fids_lazy_imports_lib import lazy_imports


BOXES = 'boxes'
KEYPOINTS = 'keypoints'
MSEG = 'mseg'
ALL = 'all'

COCO_ANNOTATION_DIR = dataset_dirs.COCO_ANNOTATION_DIR
COCO_PANOPTIC_DIR = dataset_dirs.COCO_PANOPTIC_DIR

COCO_POSSIBLY_MISSING_FEATURES = [
    'person_keypoints', 'person_boxes', 'boxes', 'box_labels'
]

COCO_BOX_LABELS = [  # Added hardcoded to remove pycoco dependency when loading.
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

COCO_KEYPOINT_LABELS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

# Mapping for changing keypoint annotations for horizontally_flip_image:
#   new_label = KEYPOINT_LABELS[HORIZONTAL_FLIP_MAP[old_label]].
# In words: left/right annotations are swapped after mirroring.
HORIZONTAL_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class COCO(mseg_base.MSegBase):
  """Import COCO Panoptic and Keypoints."""

  def __init__(self, coco_config):
    super().__init__(
        mseg_name='COCO Panoptic',
        mseg_original_name='coco-panoptic-201',
        mseg_base_name='coco-panoptic-133',
        mseg_dirname='COCOPanoptic/',
        mseg_train_dataset=True,
        mseg_config=coco_config)

    self.set_features_and_args_from_config(coco_config)
    self._coco_keypoint_annotations = {}
    self._coco_instance_annotations = {}
    self._coco_box_id_to_tt_id = {}

  def set_features_and_args_from_config(self, coco_config):
    """Set features and arguments based on COCO config."""
    if coco_config == KEYPOINTS:
      feature_names = ['image', 'person_keypoints', 'person_boxes']
    elif coco_config == BOXES:
      feature_names = ['image', 'boxes', 'box_labels']
    elif coco_config == MSEG:
      feature_names = self.MSEG_FEATURE_NAMES + ['instance_segmentation']
    elif coco_config == ALL:
      feature_names = self.MSEG_FEATURE_NAMES + [
          'person_keypoints', 'person_boxes', 'boxes', 'box_labels',
          'instance_segmentation'
      ]
    else:
      raise ValueError(f'COCO config {coco_config} not valid!')

    feature_args = {}
    if 'person_keypoints' in feature_names:
      feature_args['person_keypoints'] = dict(
          num_keypoints=len(COCO_KEYPOINT_LABELS))

    if 'box_labels' in feature_names:
      feature_args['box_labels'] = dict(num_box_labels=len(COCO_BOX_LABELS))
    self.feature_utils = fids_features.FeatureUtils(feature_args)
    self.feature_names = feature_names

    missing_features = [
        feature for feature in COCO_POSSIBLY_MISSING_FEATURES
        if feature in feature_names
    ]
    if missing_features:
      self.splits_with_missing_features = {
          'train': missing_features,
          'validation': missing_features,
      }

  def _info_features(self):
    info_features = super()._info_features()
    if 'person_keypoints' in self.feature_names:
      info_features['person_keypoints'] = dict(
          keypoint_names=COCO_KEYPOINT_LABELS)

    if 'box_labels' in self.feature_names:
      info_features['box_labels'] = COCO_BOX_LABELS

    return info_features

  def _load_pycoco_keypoints(self, split):
    annotation_file = f'{COCO_ANNOTATION_DIR}/person_keypoints_{self.split_name(split)}2017.json'
    self._coco_keypoint_annotations[split] = lazy_imports.pycocotools.coco.COCO(
        annotation_file)

  def _load_pycoco_instances(self, split):
    annotation_file = f'{COCO_ANNOTATION_DIR}/instances_{self.split_name(split)}2017.json'
    self._coco_instance_annotations[split] = lazy_imports.pycocotools.coco.COCO(
        annotation_file)

    # Obtain category names and mapping.
    categories = self._coco_instance_annotations[split].cats
    assert len(categories) == 80
    assert sorted(list(categories.keys())) == list(categories.keys())
    coco_box_categories = ['background']
    coco_box_id_to_tt_id = {}
    for idx, (coco_box_id, category_info) in enumerate(categories.items()):
      coco_box_categories.append(category_info['name'])
      coco_box_id_to_tt_id[coco_box_id] = idx + 1  # plus one for background.

    assert coco_box_categories == COCO_BOX_LABELS
    self._coco_box_id_to_tt_id = coco_box_id_to_tt_id

  def _get_boxes_feature(
      self,
      feature_name,
      annotations,
      image_height,
      image_width):
    feature = []
    if feature_name == 'boxes':
      for annot in annotations:
        bbox_x, bbox_y, bbox_w, bbox_h = annot['bbox']
        feature.append(tfds.features.BBox(
            xmin=bbox_x / image_width,
            ymin=bbox_y / image_height,
            xmax=(bbox_x + bbox_w)/ image_width,
            ymax=(bbox_y + bbox_h) / image_height))
    elif feature_name == 'box_labels':
      for annot in annotations:
        box_label = np.zeros(shape=len(COCO_BOX_LABELS),
                             dtype=np.float32)
        box_label[self._coco_box_id_to_tt_id[annot['category_id']]] = 1.0
        feature.append(box_label)
    else:
      raise ValueError(f'feature {feature_name} not valid!')

    return feature, bool(feature)

  @staticmethod
  def _get_person_feature(feature_name,
                          annotations,
                          image_height,
                          image_width):
    feature = []
    if feature_name == 'person_keypoints':
      for annot in annotations:
        keypoints = np.asarray(annot['keypoints']).astype(np.float32)
        keypoints = np.reshape(keypoints, (-1, 3))
        keypoints[:, 0] /= float(image_width)
        keypoints[:, 1] /= float(image_height)
        keypoints = keypoints[:, [1, 0, 2]]  # Change from (x, y) to (row, col).
        feature.append(keypoints)
    elif feature_name == 'person_boxes':
      for annot in annotations:
        bbox_x, bbox_y, bbox_w, bbox_h = annot['bbox']
        feature.append(tfds.features.BBox(
            xmin=bbox_x / image_width,
            ymin=bbox_y / image_height,
            xmax=(bbox_x + bbox_w)/ image_width,
            ymax=(bbox_y + bbox_h) / image_height))
    else:
      raise ValueError(f'feature {feature_name} not valid!')

    return feature, bool(feature)

  @staticmethod
  def _get_coco_instance_segmentation(split,
                                      curr_id):
    """Get instance segmentation file."""

    split_dir = 'panoptic_train2017' if split == 'train' else 'panoptic_val2017'
    instance_filename = f'{COCO_PANOPTIC_DIR}/{split_dir}/{curr_id}.png'

    instance_segmentation = utils.load_image(instance_filename)
    instance_segmentation_id = np.matmul(
        instance_segmentation, np.array([256**2, 256, 1], dtype=np.uint64))
    instance_mask = np.unique(instance_segmentation_id, return_inverse=True)[1]
    desired_shape = list(instance_segmentation.shape[:2]) + [1]
    instance_mask = np.reshape(instance_mask, desired_shape)
    return instance_mask.astype(np.uint16), True

  @staticmethod
  def get_image_annotations(coco_annotations, image_id):
    image_id = int(image_id)
    annotation_ids = coco_annotations.getAnnIds(imgIds=image_id)
    annotations = coco_annotations.loadAnns(annotation_ids)
    image_height = float(coco_annotations.imgs[image_id]['height'])
    image_width = float(coco_annotations.imgs[image_id]['width'])
    return annotations, image_height, image_width

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if feature_name in self.MSEG_FEATURE_NAMES:
      return super().get_feature(split, curr_id, feature_name)

    if feature_name == 'instance_segmentation':
      return self._get_coco_instance_segmentation(split, curr_id)

    if feature_name in ['person_keypoints', 'person_boxes']:
      if split not in self._coco_keypoint_annotations:
        self._load_pycoco_keypoints(split)
      annotations, image_height, image_width = self.get_image_annotations(
          self._coco_keypoint_annotations[split], curr_id)
      return self._get_person_feature(feature_name, annotations, image_height,
                                      image_width)

    if feature_name in ['boxes', 'box_labels']:
      if split not in self._coco_instance_annotations:
        self._load_pycoco_instances(split)
      annotations, image_height, image_width = self.get_image_annotations(
          self._coco_instance_annotations[split], curr_id)
      return self._get_boxes_feature(
          feature_name, annotations, image_height, image_width)
    raise ValueError(f'Feature {feature_name} not a valid COCO feature: '
                     f'{self.feature_names}')
