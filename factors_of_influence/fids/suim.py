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

"""Defines SUIM dataset for semantic segmentation of underwater imagery.

URL: http://irvlab.cs.umn.edu/resources/suim-dataset

Paper:
Semantic Segmentation of Underwater Imagery: Dataset and Benchmark.
M. J. Islam, C. Edge, Y. Xiao, P. Luo, M. Mehtaz, C. Morse, S. S. Enan,
and J. Sattar. IROS 2020.

The SUIM dataset contains 1525 train images and 110 test images. Each image
is accompanied with a semantic segmentation mask with 8 classes:
Background (waterbody), Human divers, Aquatic plants and sea-grass, Wrecks and
ruins, Robots, Reefs and invertebrates, Fish and vertebrates, and Sea-floor
and rocks.
"""
import os

import tensorflow.compat.v2 as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils

SUIM_DATASET_DIR = dataset_dirs.SUIM_DATASET_DIR

SUIM_LABELS = [
    utils.LabelColorDef(color=(0, 0, 0), id=0, name='background'),
    utils.LabelColorDef(color=(0, 0, 255), id=1, name='HD: Human divers'),
    utils.LabelColorDef(color=(0, 255, 0), id=2, name='PF: Plants/sea-grass'),
    utils.LabelColorDef(color=(0, 255, 255), id=3, name='WR: Wrecks/ruins'),
    utils.LabelColorDef(color=(255, 0, 0), id=4, name='RO: Robots/instruments'),
    utils.LabelColorDef(
        color=(255, 0, 255), id=5, name='RI: Reefs and invertebrates'),
    utils.LabelColorDef(
        color=(255, 255, 0), id=6, name='FV: Fish and vertebrates'),
    utils.LabelColorDef(
        color=(255, 255, 255), id=7, name='SR: Sand/sea-floor (& rocks)'),
]

SUIM_CORRECT_SEGMENTATION_SHAPE = {
    'f_r_829_': (375, 590),
    'f_r_1151_': (375, 590),
    'w_r_25_': (375, 590),
    'f_r_1866_': (375, 590),
    'f_r_1070_': (375, 590),
    'f_r_1154_': (375, 590),
    'f_r_1515_': (375, 590),
    'w_r_47_': (375, 590),
    'f_r_1570_': (435, 910),
    'f_r_401_': (435, 910),
    'f_r_934_': (375, 590),
    'f_r_1259_': (435, 910),
    'f_r_1289_': (375, 590),
    'f_r_1069_': (375, 590),
    'f_r_1058_': (375, 590),
    'f_r_1133_': (375, 590),
    'f_r_991_': (375, 590),
    'f_r_1812_': (435, 910),
    'f_r_1068_': (375, 590),
    'f_r_1424_': (375, 590),
    'w_r_7_': (375, 590),
    'f_r_1214_': (375, 590),
    'f_r_1302_': (375, 590),
    'f_r_1491_': (375, 590),
    'f_r_1290_': (435, 910),
    'w_r_24_': (375, 590),
    'f_r_1394_': (435, 910),
    'f_r_1142_': (375, 590),
    'w_r_1_': (375, 590),
    'f_r_968_': (375, 590),
    'f_r_1779_': (375, 590),
    'f_r_1233_': (375, 590),
    'f_r_1318_': (375, 590),
    'f_r_1816_': (375, 590),
    'w_r_27_': (375, 590),
    'f_r_1879_': (375, 590),
    'f_r_921_': (375, 590),
}


class SUIM(fids_dataset.FIDSDataset):
  """SUIM dataset class."""

  def __init__(self):
    super().__init__(
        name='SUIM',
        config_name='segmentation',
        feature_names=['image', 'segmentation'],
        splits=['train', 'test'],
    )
    self._img_info = {}

  def _info_features(self):
    return {'segmentation': [label.name for label in SUIM_LABELS]}

  @staticmethod
  def _get_split_dir(split):
    """Return directory for split / test dir."""
    split_dir = 'train_val' if split == 'train' else 'test'
    return f'{SUIM_DATASET_DIR}/{split_dir}/'

  def get_ids(self, split):
    # scrape ids from directory:
    split_pattern = f'{self._get_split_dir(split)}/images/*.jpg'
    img_files = tf.io.gfile.glob(split_pattern)
    ids_list = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    return ids_list

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    split_dir = self._get_split_dir(split)

    if feature_name == 'image':
      img_file = f'{split_dir}/images/{curr_id}.jpg'
      return img_file, True

    if feature_name == 'segmentation':
      segmentation_file = f'{split_dir}/masks/{curr_id}.bmp'
      segmentation = utils.convert_segmentation_rgb_to_class_id(
          segmentation_file, SUIM_LABELS)
      # Fix segmentation mask for few training files by cropping the bottom part
      # of the segmentation masks:
      if split == 'train' and curr_id in SUIM_CORRECT_SEGMENTATION_SHAPE:
        new_seg_shape = SUIM_CORRECT_SEGMENTATION_SHAPE[curr_id]
        segmentation = segmentation[:new_seg_shape[0], :new_seg_shape[1]]
      return segmentation, True

    raise ValueError(f'{feature_name} unknown')
