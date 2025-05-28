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

"""Defines SUNRGBD, segmentation (including Mseg) and depth.

SUN RGB-D Dataset a Scene Understanding Benchmark
Website: https://rgbd.cs.princeton.edu/

Paper:
SUN RGB-D: A RGB-D scene understanding benchmark suite.
S. Song, S. Lichtenberg, and J. Xiao. In CVPR, 2015.

Features/Modalities:
1. RGB image
2. Semantic segmentation
3. Depth image
4. Object detection (2D & 3D)
5. Room layout

Currently only image, semantic segmentation and depth are used.
"""

from typing import Text

import numpy as np

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import mseg_base
from factors_of_influence.fids import utils

DEPTH = 'depth'
MSEG = 'mseg'
ALL = 'all'

DEPTH_FILE_PATTERN = dataset_dirs.SUNRGBD_DEPTH_DIR + '/{}/{:08d}.png'


class SUNRGBD(mseg_base.MSegBase):
  """Import SUNRGBD."""

  def __init__(self, sunrgb_config = MSEG):
    super().__init__(mseg_name='SUNRGB-D',
                     mseg_original_name='sunrgbd-38',
                     mseg_base_name='sunrgbd-37',
                     mseg_dirname='SUNRGBD',
                     mseg_train_dataset=True,
                     mseg_config=sunrgb_config)

    self.feature_names = self.get_features_from_config(sunrgb_config)

  def get_features_from_config(self, sunrgb_config):
    """Return features based on SUNRGBD config."""
    if sunrgb_config == DEPTH:
      return ['image', 'depth']
    elif sunrgb_config == MSEG:
      return self.MSEG_FEATURE_NAMES
    elif sunrgb_config == ALL:
      return self.MSEG_FEATURE_NAMES + ['depth']
    else:
      raise ValueError(f'SUNRGBD config {sunrgb_config} not valid!')

  def _info_features(self):
    info_features = super()._info_features()
    if 'depth' in self.feature_names:
      info_features['depth'] = dict(
          default_clip_min=0.369, default_clip_max=8.0)

    return info_features

  @staticmethod
  def _convert_depth_to_m(depth_raw):
    """Converts depth (uint16) to cm (float)."""
    # Follows the SUNRGBD Matlab Toolbox [SMT]:
    # https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
    # [SMT]: depth = bitor(bitshift(depth,-3), bitshift(depth,16-3));
    # matlab's bitshift(..., -3) is a right shift (of 3); and
    # matlab's bitshift(..., 13) is a left shift:
    depth_raw = np.bitwise_or(np.right_shift(depth_raw, np.uint16(3)),
                              np.left_shift(depth_raw, np.uint16(13)))
    # [SMT]: depth = single(depthInpaint)/1000;
    depth_in_meter = depth_raw.astype(np.float32)/1000.0
    # [SMT]: depth(depth >8)=8;
    # Note practical max is around 5m (given sensors and indoor environments).
    depth_in_meter = np.minimum(depth_in_meter, 8)
    return depth_in_meter

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if feature_name in self.MSEG_FEATURE_NAMES:
      return super().get_feature(split, curr_id, feature_name)

    if feature_name in ['depth']:
      depth_id = int(curr_id.split('-')[1])
      depth_split = 'train' if split == 'train' else 'test'
      depth_file_name = DEPTH_FILE_PATTERN.format(depth_split, depth_id)
      depth_raw = utils.load_image_cv2_any_color_any_depth(depth_file_name)
      return self._convert_depth_to_m(depth_raw), True
