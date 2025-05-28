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

r"""This is part of the preprocessing scripts / files for the iSAID dataset.

iSAID contains large images (eg 4000 x 4000) and this splits these into patches.
"""
import os
import time
from typing import Text, Tuple

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.preprocess import image_generate_patches

FLAGS = flags.FLAGS

ISPRS_DATASET_DIR = dataset_dirs.ISPRS_DATASET_DIR

_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir', ISPRS_DATASET_DIR, 'Set dataset_dir')
_PATCH_DIR = flags.DEFINE_string('patch_dir', 'patches/', 'Output directory')
_PATCH_WIDTH = flags.DEFINE_integer(
    'patch_width', 800, 'Width of generated patches', lower_bound=0)
_PATCH_HEIGHT = flags.DEFINE_integer(
    'patch_height', 800, 'Height of generated patches', lower_bound=0)
_PATCH_OVERLAP = flags.DEFINE_integer(
    'patch_overlap', 200, 'Default overlap between patches', lower_bound=0)

_DEBUG = flags.DEFINE_integer('debug_number', 0, 'Debug: num of files to use')


def _split_file_name(image_file,
                     split_name):
  """Split filename into image file, gt file, patch dir and patch base name."""
  if _PATCH_DIR.value.startswith('/'):
    patch_dir = f'{_PATCH_DIR.value}/{split_name}'
  else:
    patch_dir = f'{_DATASET_DIR.value}/{_PATCH_DIR.value}/{split_name}'
  img_path, img_name = os.path.split(image_file)
  gt_path = f'{os.path.split(img_path)[0]}/gt'

  gt_name = img_name
  if split_name == 'potsdam':
    gt_base = img_name.rsplit('_', 1)[0]
    gt_name = f'{gt_base}_label.tif'
  gt_file = f'{gt_path}/{gt_name}'

  patch_base = img_name.rsplit('.', 1)[0]
  if split_name == 'potsdam':
    patch_base = img_name.rsplit('_', 1)[0]

  return image_file, gt_file, patch_dir, patch_base


def main(unused_argv):
  for split_set in ['potsdam', 'vaihingen']:
    split_files = tf.io.gfile.glob(
        f'{_DATASET_DIR.value}/{split_set}/top/*.tif')
    split_start = time.time()
    for (i, split_file) in enumerate(split_files):
      image_file, gt_file, patch_dir, patch_base = _split_file_name(
          split_file, split_name=split_set)
      image_generate_patches.create_patches(
          img_file=image_file,
          patch_height=_PATCH_HEIGHT.value,
          patch_width=_PATCH_WIDTH.value,
          patch_overlap=_PATCH_OVERLAP.value,
          patch_dir=f'{patch_dir}/top/',
          patch_base=patch_base,
          load_cv2=False,
      )
      image_generate_patches.create_patches(
          img_file=gt_file,
          patch_height=_PATCH_HEIGHT.value,
          patch_width=_PATCH_WIDTH.value,
          patch_overlap=_PATCH_OVERLAP.value,
          patch_dir=f'{patch_dir}/gt/',
          patch_base=patch_base,
          load_cv2=False,
      )

      if i > 0 and (i % 5) == 0:
        logging.info('%10s %5d / %5d (%5.2f sec/example)', split_set, i,
                     len(split_files), (time.time() - split_start) / float(i))

      if _DEBUG.value > 0 and i > _DEBUG.value:
        break


if __name__ == '__main__':
  app.run(main)
