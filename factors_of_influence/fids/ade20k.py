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

"""Import ADE20k, the MSeg version, including instance segmentation masks.

URL: https://groups.csail.mit.edu/vision/datasets/ADE20K/

Scene Parsing through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig,
Sanja Fidler, Adela Barriuso and Antonio Torralba. CVPR, 2017.

Semantic Understanding of Scenes through ADE20K Dataset. Bolei Zhou, Hang Zhao,
Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso and Antonio Torralba.
International Journal on Computer Vision (IJCV).
"""

from typing import Dict, Text

import numpy as np
import tensorflow as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import mseg_base
from factors_of_influence.fids import utils

ADE20K_IMAGES_DIR = dataset_dirs.ADE20K_IMAGES_DIR


def get_instance_filename_dict():
  """Get dictionary with key -> instance segmentation filename."""
  filename_list = []
  for level in range(3, 6):
    glob_pattern = ''.join(['/*'] * level)
    filename_list.extend(
        tf.io.gfile.glob(f'{ADE20K_IMAGES_DIR}{glob_pattern}_seg.png'))

  def _key_from_filename(file_name):
    return file_name.rsplit('/', maxsplit=1)[1].rsplit('_', maxsplit=1)[0]

  return {_key_from_filename(fn): fn for fn in filename_list}


def get_instance_mask(instance_filename):
  """Obtain the instance mask from the blue channel of the segmentation file."""
  # Based on DevKit:
  # https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py

  instance_segmentation = utils.load_image(instance_filename)
  instance_segmentation_blue = instance_segmentation[:, :, 2]
  instance_mask = np.unique(instance_segmentation_blue, return_inverse=True)[1]
  desired_shape = list(instance_segmentation_blue.shape) + [1]
  instance_mask = np.reshape(instance_mask, desired_shape)
  return instance_mask.astype(np.uint16)


class ADE20k(mseg_base.MSegBase):
  """Import ADE20k."""

  def __init__(self):
    super().__init__(
        mseg_name='ADE20k',
        mseg_original_name='ade20k-151',
        mseg_base_name='ade20k-150',
        mseg_dirname='ADE20K/ADEChallengeData2016/',
        mseg_train_dataset=True)

    self.feature_names = self.MSEG_FEATURE_NAMES + ['instance_segmentation']
    self._instance_segmentation_file_from_key = None

  @property
  def instance_segmentation_file_from_key(self):
    if self._instance_segmentation_file_from_key is None:
      self._instance_segmentation_file_from_key = get_instance_filename_dict()
    return self._instance_segmentation_file_from_key

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    if feature_name in self.MSEG_FEATURE_NAMES:
      return super().get_feature(split, curr_id, feature_name)

    if feature_name in ['instance_segmentation']:
      instance_filename = self.instance_segmentation_file_from_key[curr_id]
      instance_mask = get_instance_mask(instance_filename)
      # Instance masks have sometimes different dimensions than image -> resize.
      image_filename = self.get_feature(split, curr_id, 'image')[0]
      image_shape = utils.load_image(image_filename).shape
      instance_mask = utils.resize_image_cv2(instance_mask,
                                             desired_height=image_shape[0],
                                             desired_width=image_shape[1])
      instance_mask = np.reshape(instance_mask, list(image_shape[:2]) + [1])
      return instance_mask, True
