# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Defines ISPRS Aerial Imagery Segmentation Dataset.

URL: http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html

Paper: F. Rottensteiner, G. Sohn, M. Gerke, and J. D. Wegner. Theme
section - urban object detection and 3d building reconstruction.
ISPRS Journal of Photogrammetry and Remote Sensing, 2014.

ISPRS dataset contains aerial imagery from Potsdam and Vaihingen. The dataset
contains 38 (Potsdam, used as train set) and 33 (Vaihingen, used as val set)
high-res images (6000 x 6000).

Before this dataset can be generated use:
preprocess/isprs_split.py
to split the high-res images into 800 x 800 patches (with some overlap).

The preprocessing tool (isprs_split) creates overlapping patches, eg for an
input image with a width of 1500, with patch_width 800 and patch_overlap 200:
  [0-800, 600-1400, 700-1500]
Note 1: the last patch will always be [image-width - patch-width, image-width],
this could in principle yield a highly overlapping patch.
Note 2: Train and validation splits are not defined on patches, but on the two
cities: train = Potsdam, val = Vaihingen
For height the similar splits are used, independently from the width.

NOTE: ISPRS dataset also contains depth surface maps and near infra red (NIR).
"""

import os
from typing import List, Text, Union

import numpy as np
import tensorflow.compat.v2 as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset
from factors_of_influence.fids import utils
from factors_of_influence.fids.fids_lazy_imports_lib import lazy_imports

ISPRS_DATASET_DIR = dataset_dirs.ISPRS_DATASET_DIR

# Label definition
# https://www2.isprs.org/commissions/comm2/wg4/benchmark/semantic-labeling/
# https://www2.isprs.org/commissions/comm2/wg4/results/
# Note: The clutter/background class is their background class. While it must
# be included for submission, it is not used for ranking in their benchmark
# results.
ISPRS_LABELS = [
    utils.LabelColorDef(name='background', id=0, color=(255, 0, 0)),
    utils.LabelColorDef(
        name='impervious_surfaces', id=1, color=(255, 255, 255)),
    utils.LabelColorDef(name='building', id=2, color=(0, 0, 255)),
    utils.LabelColorDef(name='low_vegetation', id=3, color=(0, 255, 255)),
    utils.LabelColorDef(name='tree', id=4, color=(0, 255, 0)),
    utils.LabelColorDef(name='car', id=5, color=(255, 255, 0)),
]

ISPRS_SETS = {
    'train': dict(city='potsdam', format='RGB'),
    'validation': dict(city='vaihingen', format='IRG')
}


def convert_segmentation_rgb_to_class_id_euclidean(
    segmentation_rgb,
    label_list):
  """Conversion tool for ISPRS to deal with artefacts."""
  # Some GT have artefacts (ie close to green, but not fully green).
  # Therefore we use a conversion based on euclidean distance, instead of
  # equality (as in utils).
  if not isinstance(segmentation_rgb, np.ndarray):
    segmentation_rgb = utils.load_image(segmentation_rgb)

  segmentation_rgb = segmentation_rgb/255.0
  colors = np.asarray([l.color for l in label_list], dtype=np.float32).T
  colors = colors/255.0

  # Compute squared distances
  seg_rgb_norm = lazy_imports.scipy.linalg.norm(segmentation_rgb, axis=-1)
  seg_rgb_norm = seg_rgb_norm[Ellipsis, np.newaxis]
  colors_norm = lazy_imports.scipy.linalg.norm(colors, axis=0)
  seg_label_diff = seg_rgb_norm - 2 * np.matmul(segmentation_rgb,
                                                colors) + colors_norm
  seg_label = seg_label_diff.argmin(axis=-1)
  seg_label = seg_label[Ellipsis, np.newaxis]

  label_map = [
      utils.LabelMap(name=label.name, id=label.id, original_id=idx)
      for idx, label in enumerate(label_list)
  ]
  return utils.convert_segmentation_map(seg_label, label_map)


class ISPRS(fids_dataset.FIDSDataset):
  """ISPRS dataset class."""

  def __init__(self):
    super().__init__(
        name='ISPRS',
        config_name='segmentation',
        feature_names=['image', 'segmentation'],
        splits=list(ISPRS_SETS.keys()),
    )

  def _info_features(self):
    return {'segmentation': [label.name for label in ISPRS_LABELS]}

  @staticmethod
  def _get_split_dir(split):
    """Return directory for split / test dir."""
    return f'{ISPRS_DATASET_DIR}/patches/{ISPRS_SETS[split]["city"]}/'

  def get_ids(self, split):
    # scrape ids from directory:
    split_pattern = f'{self._get_split_dir(split)}/top/*.png'
    split_list = tf.io.gfile.glob(split_pattern)
    ids_list = [os.path.basename(img_file) for img_file in split_list]
    return ids_list

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    split_dir = self._get_split_dir(split)

    if feature_name in ['image', 'image_raw']:
      # Convert RGB (Potsdam) and IRG (Vaihingen) to RG[(R+G)/2].
      img_raw = utils.load_png(f'{split_dir}/top/{curr_id}')
      if feature_name == 'image_raw':
        return img_raw, True
      img_raw_dtype = img_raw.dtype

      img_raw = img_raw.astype(np.float32)
      if ISPRS_SETS[split]['format'] == 'RGB':
        red_channel, green_channel = img_raw[:, :, 0], img_raw[:, :, 1]
      elif ISPRS_SETS[split]['format'] == 'IRG':
        red_channel, green_channel = img_raw[:, :, 1], img_raw[:, :, 2]
      blue_channel = .5 * (red_channel + green_channel)
      img = np.stack([red_channel, green_channel, blue_channel], axis=-1)
      img = img.astype(img_raw_dtype)
      return img, True

    if feature_name == 'segmentation':
      segmentation_file = f'{split_dir}/gt/{curr_id}'
      segmentation = utils.convert_segmentation_rgb_to_class_id(
          segmentation_file, ISPRS_LABELS)
      return segmentation, True

    raise ValueError(f'{feature_name} unknown')
