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

"""Defines Virtual KITTI 2 (vKITTI2) Dataset.

DATASET URL:
https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/

Virtual KITTI uses the Unity Game engine to create photo realistic scenes of a
driving car. The modalities which are recorded are: rgb, depth, class segmenta-
tion, instance segmentation, bounding boxes, and several optical flow variants.
In this version, we use the imagery, depth and class segmentation.

vKITTI2 is organized along 5 drives / scenes: Scene01, Scene02, Scene06,
Scene18 and Scene20. Scene18 is used for TEST, the others for TRAIN.
Per drive there are scene configurations, like clone (mimic KITTI as much as
possible), rain, fog, morning, sunset, 15-degree-right. In the ALL configuration
of the dataset, these scene configurations are included as scene_class label to
allow for evaluation / training on specific conditions.

References:
- Virtual KITTI 2.
  Cabon, Yohann and Murray, Naila and Humenberger, Martin
  arXiv 2020
  http://arxiv.org/pdf/2001.10773

- Virtual worlds as proxy for multi-object tracking analysis,
  Gaidon, Adrien and Wang, Qiao and Cabon, Yohann and Vig, Eleonora
  CVPR 2016
  https://arxiv.org/pdf/1605.06457
"""
import os

from absl import logging
import numpy as np

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils

VKITTI2_DIR = dataset_dirs.VKITTI2_DIR
VKITTI2_LABELS = [
    utils.LabelColorDef(name='Background', id=0, color=(0, 0, 0)),
    utils.LabelColorDef(name='Terrain', id=1, color=(210, 0, 200)),
    utils.LabelColorDef(name='Sky', id=2, color=(90, 200, 255)),
    utils.LabelColorDef(name='Tree', id=3, color=(0, 199, 0)),
    utils.LabelColorDef(name='Vegetation', id=4, color=(90, 240, 0)),
    utils.LabelColorDef(name='Building', id=5, color=(140, 140, 140)),
    utils.LabelColorDef(name='Road', id=6, color=(100, 60, 100)),
    utils.LabelColorDef(name='GuardRail', id=7, color=(250, 100, 255)),
    utils.LabelColorDef(name='TrafficSign', id=8, color=(255, 255, 0)),
    utils.LabelColorDef(name='TrafficLight', id=9, color=(200, 200, 0)),
    utils.LabelColorDef(name='Pole', id=10, color=(255, 130, 0)),
    utils.LabelColorDef(name='Misc', id=11, color=(80, 80, 80)),
    utils.LabelColorDef(name='Truck', id=12, color=(160, 60, 60)),
    utils.LabelColorDef(name='Car', id=13, color=(255, 127, 80)),
    utils.LabelColorDef(name='Van', id=14, color=(0, 139, 139)),
]

VKITTI2_SCENE_LABELS = [
    '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
    'fog', 'morning', 'overcast', 'rain', 'sunset'
]

ALL = 'all'
CLONE = 'clone'


class VKITTI2(fids_dataset.FIDSDataset):
  """vKITTI2 dataset class."""
  CONFIGS = [ALL, CLONE]

  def __init__(self, config=ALL):
    if config not in self.CONFIGS:
      raise ValueError(f'Config {config} not defined as a valid VKITTI2 config:'
                       f'{self.CONFIGS}')

    feature_names = ['image', 'segmentation', 'depth']
    feature_args = {}
    if config == ALL:
      feature_names.append('scene_class')
      feature_args = {
          'scene_class': {
              'list_scene_classes': VKITTI2_SCENE_LABELS
          }
      }

    super().__init__(
        name='vkitti2',
        config_name=config,
        feature_names=feature_names,
        feature_args=feature_args,
        splits=['train', 'test'],
    )

  def _info_features(self):
    return {
        'segmentation': [label.name for label in VKITTI2_LABELS],
        'depth': dict(default_clip_min=0.94, default_clip_max=80.0),
    }

  def get_ids(self, split):
    # scrape ids from directory:
    split_file = f'{VKITTI2_DIR}/{split}_{self.config_name}.txt'
    logging.info('TT| Load ids from: %s', split_file)
    return utils.load_text_to_list(split_file)

  @staticmethod
  def parse_file_name(example_id):
    """Parse and split filename into scene, setting, modality, ..."""
    # Example filename:
    # Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg
    # Which is splitted as:
    # DRIVE/SCENE/---/---/CAMERA/___FILEID.jpg
    ex_drive, ex_scene, _, _, ex_camera, ex_file_id = example_id.split('/')
    ex_file_id = int(os.path.splitext(ex_file_id)[0].rsplit('_', 1)[1])
    return ex_drive, ex_scene, ex_camera, ex_file_id

  def get_file_name(self, example_id, feature_name):
    if feature_name == 'image':
      return f'{VKITTI2_DIR}/{example_id}'

    ex_drive, ex_scene, ex_camera, ex_fileid = self.parse_file_name(example_id)
    if feature_name == 'segmentation':
      return f'{VKITTI2_DIR}/{ex_drive}/{ex_scene}/frames/classSegmentation/{ex_camera}/classgt_{ex_fileid:05d}.png'

    if feature_name in ['depth']:
      return f'{VKITTI2_DIR}/{ex_drive}/{ex_scene}/frames/depth/{ex_camera}/depth_{ex_fileid:05d}.png'

    return None

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if feature_name in ['image']:
      return self.get_file_name(curr_id, feature_name), True

    if feature_name == 'depth':
      # From documentation: All depth images are encoded as grayscale 16bit PNG
      # files. Max-depth is 655.35 meters (pixels farther away are clipped;
      # however, it is not relevant for this dataset). Depth values are in the
      # range [0;2^16 – 1], such that a pixel intensity of 1 corresponds to a
      # distance of 1cm to the camera plane.
      depth_file_name = self.get_file_name(curr_id, feature_name)
      depth_raw = utils.load_image_cv2_any_color_any_depth(depth_file_name)
      depth_in_meter = depth_raw.astype(np.float32) / 100.0
      return depth_in_meter, True

    if feature_name == 'scene_class':
      _, curr_scene, _, _ = self.parse_file_name(curr_id)
      return curr_scene, True

    if feature_name == 'segmentation':
      segmentation_file = self.get_file_name(curr_id, feature_name)
      segmentation = utils.convert_segmentation_rgb_to_class_id(
          segmentation_file, VKITTI2_LABELS)

      return segmentation, True

    raise ValueError(f'{feature_name} unknown')
