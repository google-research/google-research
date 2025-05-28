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

"""Defines StanfordDogs Keypoint (Extra) dataset.

URL 1: https://github.com/benjiebob/StanfordExtra
URL 2: http://vision.stanford.edu/aditya86/ImageNetDogs/

Papers:
- Novel dataset for fine-grained image categorization.
  A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei. In CVPR Workshop, 2011.
- Who left the dogs out?: 3D animal reconstruction
  with expectation maximization in the loop.
  B. Biggs, O. Boyne, J. Charles, A. Fitzgibbon, and R. Cipolla. In ECCV, 2020.

StanfordDogs Keypoints uses a subset of 12K (from 20K) Stanford Dogs images and
provide segmentation masks and 20 keypoint annotations.

NOTE:
  1/ The original stanford dogs dataset contains more images.
  2/ The dataset provide 24 joints, but 4 are never used, these are removed.

Paper Stanford Dogs: http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf
Paper Stanford Keypoints: https://arxiv.org/abs/2007.11110
"""
import collections
import json

import numpy as np
import tensorflow.compat.v2 as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils

STANFORD_DOGS_DIR = dataset_dirs.STANFORD_DOGS_DIR
STANFORD_DOGS_JSON = 'StanfordExtra_v1.json'
STANFORD_DOGS_KEYPOINT = 'keypoint_definitions.csv'

# Label definition (from eypoint_definitions.csv)
LabelDef = collections.namedtuple('LabelDef', ['id', 'name'])
STANFORD_DOGS_KEYPOINT_LABELS = [
    LabelDef(0, 'Left front leg: paw'),
    LabelDef(1, 'Left front leg: middle joint'),
    LabelDef(2, 'Left front leg: top'),
    LabelDef(3, 'Left rear leg: paw'),
    LabelDef(4, 'Left rear leg: middle joint'),
    LabelDef(5, 'Left rear leg: top'),
    LabelDef(6, 'Right front leg: paw'),
    LabelDef(7, 'Right front leg: middle joint'),
    LabelDef(8, 'Right front leg: top'),
    LabelDef(9, 'Right rear leg: paw'),
    LabelDef(10, 'Right rear leg: middle joint'),
    LabelDef(11, 'Right rear leg: top'),
    LabelDef(12, 'Tail start'),
    LabelDef(13, 'Tail end'),
    LabelDef(14, 'Base of left ear'),
    LabelDef(15, 'Base of right ear'),
    LabelDef(16, 'Nose'),
    LabelDef(17, 'Chin'),
    LabelDef(18, 'Left ear tip'),
    LabelDef(19, 'Right ear tip'),
]


class StanfordDogs(fids_dataset.FIDSDataset):
  """Import Stanford Dogs Keypoint dataset."""

  def __init__(self):
    super().__init__(
        name='StanfordDogs',
        config_name='keypoints',
        feature_names=['image', 'person_keypoints', 'scene_class'],
        feature_args={
            'person_keypoints': {
                'num_keypoints': len(STANFORD_DOGS_KEYPOINT_LABELS)
            },
            'scene_class': {
                'num_scene_classes': 120
            }
        },
        splits=['train', 'test'],
        )

    self._json_dict = None

  def _info_features(self):
    return {
        'person_keypoints':
            dict(keypoint_names=[
                kp.name for kp in STANFORD_DOGS_KEYPOINT_LABELS
            ]),
    }

  @staticmethod
  def _convert_joints_to_coco(joints, img_width,
                              img_height):
    """Converts the StanfordExtra joints format to COCO Keypoints format."""
    joints = joints[:20, :]  # StanfordExtra annotate 20 (from 24) keypoint
    joints = joints[:, [1, 0, 2]]  # Convert from x, y, vis to row, col, vis.
    joints[:, 0] /= img_height  # Normalize rows with img height
    joints[:, 1] /= img_width  # Normalize cols with img width
    # Convert visibility encoding from:
    # (0, 0, 0): not annotated, (x, y, 0): not visible, (x, y, 1): visible,
    # to the COCO encoding:
    # 0: not annotated, 1: annotated but not visible, 2: annotated and visible
    joints[:, 2] += np.any(joints[:, :2], axis=1)
    return joints

  def _load_json(self):
    # Load Stanford Dogs Extra annotations: for keypoints (of 12K images).
    json_file = f'{STANFORD_DOGS_DIR}/{STANFORD_DOGS_JSON}'
    with tf.io.gfile.GFile(json_file, mode='r') as f:
      json_data = json.load(f)

    json_dict = {}
    for annotation in json_data:
      key = annotation['img_path']

      img_width = annotation['img_width']
      img_height = annotation['img_height']
      joints = np.asarray(annotation['joints'], dtype=np.float32)
      keypoints = self._convert_joints_to_coco(joints, img_width, img_height)
      json_dict[key] = {'person_keypoints': [keypoints]}  # tfds expects a list

    # Load Stanford Dogs annotations: for scene_class and train/test splits.
    # This is available for a 20K superset of Stanford Dogs Keypoints.
    for split_name in ['train', 'test']:
      split_mat = utils.load_mat(f'{STANFORD_DOGS_DIR}/{split_name}_list.mat')
      for (key, label) in zip(split_mat['file_list'], split_mat['labels']):
        key = str(key[0][0])

        if key not in json_dict: continue  # Skip: not in Keypoint subset.
        json_dict[key]['scene_class'] = int(label)-1  # Labels are 1-120.
        json_dict[key]['split'] = split_name
    self._json_dict = json_dict

  def get_ids(self, split):
    if self._json_dict is None:
      self._load_json()
    return [id for id in self._json_dict
            if self._json_dict[id]['split'] == split]

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if self._json_dict is None:
      self._load_json()

    if feature_name == 'image':
      return f'{STANFORD_DOGS_DIR}/Images/{curr_id}', True

    if feature_name in ['scene_class', 'person_keypoints']:
      return self._json_dict[curr_id][feature_name], True

    raise ValueError(f'Feature {feature_name} not a valid feature: '
                     f'{self.feature_names}')
