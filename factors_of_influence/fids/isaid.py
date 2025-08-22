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

"""Defines iSAID Aerial Imagery Segmentation Dataset.

URL: https://captain-whu.github.io/iSAID/
DevKit: https://github.com/CAPTAIN-WHU/iSAID_Devkit

Papers:
- iSAID: A largescale dataset for instance segmentation in aerial images.
  S. Waqas Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. Shahbaz Khan,
  F. Zhu, L. Shao, G.-S. Xia, and X. Bai.  In CVPR Workshops, 2019.
- DOTA: A large-scale dataset for object detection in aerial images.
  G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu,
  M. Pelillo, and L. Zhang. In CVPR 2018.

iSAID contains 655,451 object instances for 15 categories across 2,806 high-
resolution images. Before this dataset can be generated use:
preprocess/isaid_split.py
to split the high-res images into 800 x 800 patches (with some overlap).

The preprocessing tool (isaid_split) creates overlapping patches, eg for an
input image with a width of 1500, with patch_width 800 and patch_overlap 200:
  [0-800, 600-1400, 700-1500]
Note 1: the last patch will always be [image-width - patch-width, image-width],
this could in principle yield a highly overlapping patch.
Note 2: Train and validation splits are not defined on patches, but on tiles.
For height the similar splits are used, independently from the width.

NOTE:
  1/ This iSAID dataset also has instance segmentation annotations
  2/ The related DOTA dataset has 188,282 instances labeled by an arbitrary
     (8 d.o.f.) quadrilateral.
     DOTA: https://captain-whu.github.io/DOTA/index.html
"""

import os
from typing import Text, Tuple

import tensorflow.compat.v2 as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils


ISAID_DATASET_DIR = dataset_dirs.ISAID_DATASET_DIR

# Label definition (from iSAID DevKit)


class ISAIDColorDef(utils.LabelColorDef):
  """Subclass of LabelColorDef to match ISAID label definition."""

  def __init__(self, name, label_id, dataset_id, category,
               category_id, has_instances, ignore_in_eval,
               color, multiplied_color):
    super().__init__(name=name, id=label_id, color=color)


# pylint: disable=line-too-long, bad-whitespace
# pyformat: disable
ISAID_LABELS = [
    #                name                    id   not_used_id  category         catId     hasInstances   ignoreInEval   color          multiplied color
    ISAIDColorDef(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ),
    ISAIDColorDef(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    ISAIDColorDef(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    ISAIDColorDef(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63,  0) , 16128  ),
    ISAIDColorDef(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    ISAIDColorDef(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    ISAIDColorDef(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    ISAIDColorDef(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    ISAIDColorDef(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    ISAIDColorDef(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    ISAIDColorDef(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    ISAIDColorDef(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    ISAIDColorDef(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    ISAIDColorDef(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    ISAIDColorDef(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    ISAIDColorDef(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),
]
# pylint: enable=line-too-long, bad-whitespace
# pyformat: enable


class ISAID(fids_dataset.FIDSDataset):
  """iSAID dataset class."""

  def __init__(self):
    super().__init__(
        name='iSAID',
        config_name='segmentation',
        feature_names=['image', 'segmentation'],
        splits=['train', 'validation', 'test'],
        splits_with_missing_features={'test': ['segmentation']},
    )

  def split_name(self, split):
    """Returns split name."""
    return 'val' if split == 'validation' else split

  def _info_features(self):
    return {'segmentation': [label.name for label in ISAID_LABELS]}

  def get_ids(self, split):
    # scrape ids from directory:
    split_pattern = f'{ISAID_DATASET_DIR}/{self.split_name(split)}/patches/*.png'
    split_list = tf.io.gfile.glob(split_pattern)
    return [
        id_file.replace(ISAID_DATASET_DIR, '') for id_file in split_list
        if not os.path.splitext(id_file)[0].endswith('RGB')
    ]

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    curr_file = f'{ISAID_DATASET_DIR}/{curr_id}'
    if feature_name == 'image':
      return curr_file, True
    if feature_name == 'segmentation':
      if split == 'test':
        return self.feature_utils.get_fake_feature(feature_name), False

      segmentation_base = os.path.splitext(curr_file)[0]
      segmentation_file = f'{segmentation_base}_instance_color_RGB.png'
      segmentation = utils.convert_segmentation_rgb_to_class_id(
          segmentation_file, ISAID_LABELS)
      return segmentation, True

    raise ValueError(f'{feature_name} unknown')
