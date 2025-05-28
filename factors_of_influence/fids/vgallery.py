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

"""Defines Virtual Gallery 1 (vGallery) Dataset.

DATASET URL:
https://europe.naverlabs.com/research/3d-vision/virtual-gallery-dataset/

The Virtual Gallery dataset is a synthetic dataset that targets multiple
challenges such as varying lighting conditions and different occlusion levels
for various tasks such as depth estimation, instance segmentation and
visual localization. It consists of a scene containing 3-4 rooms, in which a
total of 42 free-for-use famous paintings are placed on the walls. The virtual
model and the captured images were generated with Unity game engine.

The training/testing scenario simulates the scene captured by a robot
equipped with 6 cameras for training, and photos taken by visitors for testing.

In this version, we use the imagery, depth and class segmentation.

Reference:
- Visual Localization by Learning Objects-of-Interest Dense Match Regression,
  P. Weinzaepfel, G. Csurka, Y. Cabon, and M. Humenberger
  CVPR 2019
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Weinzaepfel_Visual_Localization_by_Learning_Objects-Of-Interest_Dense_Match_Regression_CVPR_2019_paper.pdf
"""

import os

from absl import logging
import numpy as np

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils

VGALLERY_DIR = dataset_dirs.VGALLERY_DIR
VGALLERY_LABELS = [
    utils.LabelColorDef(name='Background', id=0, color=(0, 0, 0)),
    utils.LabelColorDef(name='Wall', id=1, color=(255, 127, 80)),
    utils.LabelColorDef(name='Ceiling', id=2, color=(255, 248, 220)),
    utils.LabelColorDef(name='Sky', id=3, color=(0, 255, 255)),
    utils.LabelColorDef(name='Door', id=4, color=(189, 183, 107)),
    utils.LabelColorDef(name='Light', id=5, color=(230, 230, 250)),
    utils.LabelColorDef(name='Floor', id=6, color=(233, 150, 122)),
    utils.LabelColorDef(name='Misc', id=7, color=(80, 80, 80)),
    utils.LabelColorDef(name='Painting', id=8, color=(128, 0, 0)),
    utils.LabelColorDef(name='Human', id=9, color=(0, 255, 0)),
]

ALL = 'all'


class VGALLERY(fids_dataset.FIDSDataset):
  """vGALLERY dataset class."""
  CONFIGS = [ALL]

  def __init__(self, config=ALL):
    if config not in self.CONFIGS:
      raise ValueError(f'Config {config} not defined as valid VGALLERY config:'
                       f'{self.CONFIGS}')

    super().__init__(
        name='vgallery',
        config_name=config,
        feature_names=['image', 'segmentation', 'depth'],
        splits=['train', 'test'],
    )

  def _info_features(self):
    return {
        'segmentation': [label.name for label in VGALLERY_LABELS],
        'depth': dict(default_clip_min=0.21, default_clip_max=14.89),
    }

  def get_ids(self, split):
    # scrape ids from directory:
    split_file = f'{VGALLERY_DIR}/{split}_{self.config_name}.txt'
    logging.info('TT| Load ids from: %s', split_file)
    return utils.load_text_to_list(split_file)

  @staticmethod
  def parse_file_name(example_id):
    """Parse and split filename into scene, setting, modality, ..."""
    # Example filename:
    # training/gallery_light6_loop5/frames/rgb/camera_4/rgb_00220.jpg
    # Which is splitted as:
    # SPLIT/LOOP/---/---/CAMERA/___FILEID.jpg
    ex_split, ex_loop, _, _, ex_camera, ex_file_id = example_id.split('/')
    ex_file_id = int(os.path.splitext(ex_file_id)[0].rsplit('_', 1)[1])
    return ex_split, ex_loop, ex_camera, ex_file_id

  def get_file_name(self, example_id, feature_name):
    if feature_name == 'image':
      return f'{VGALLERY_DIR}/{example_id}'

    if feature_name in ['segmentation', 'depth', 'depth']:
      if feature_name == 'segmentation':
        feature_dir, feature_prefix = 'classsegmentation', 'classgt'
      else:
        feature_dir, feature_prefix = 'depth', 'depth'

      ex_split, ex_loop, ex_cam, ex_file_id = self.parse_file_name(example_id)
      return (f'{VGALLERY_DIR}/{ex_split}/{ex_loop}/frames/{feature_dir}'
              f'/{ex_cam}/{feature_prefix}_{ex_file_id:05d}.png')

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

    if feature_name == 'segmentation':
      segmentation_file = self.get_file_name(curr_id, feature_name)
      segmentation = utils.convert_segmentation_rgb_to_class_id(
          segmentation_file, VGALLERY_LABELS)
      return segmentation, True

    raise ValueError(f'{feature_name} unknown')
