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
from typing import Optional, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from factors_of_influence import dataset_dirs
from factors_of_influence.preprocess import image_generate_patches

FLAGS = flags.FLAGS

ISAID_DATASET_DIR = dataset_dirs.ISAID_DATASET_DIR

_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir', ISAID_DATASET_DIR, 'Set dataset_dir')
_PATCH_DIR = flags.DEFINE_string('patch_dir', 'patches/', 'Output directory')
_PATCH_WIDTH = flags.DEFINE_integer(
    'patch_width', 800, 'Width of generated patches', lower_bound=0)
_PATCH_HEIGHT = flags.DEFINE_integer(
    'patch_height', 800, 'Height of generated patches', lower_bound=0)
_PATCH_OVERLAP = flags.DEFINE_integer(
    'patch_overlap', 200, 'Default overlap between patches', lower_bound=0)

_DEBUG = flags.DEFINE_integer('debug_number', 0, 'Debug: num of files to use')


def _get_patch_base_names(img_file):
  """Get patch base names from image file."""
  img_dir, img_name = os.path.split(img_file)
  patch_dir = os.path.split(img_dir)[0] + '/' + _PATCH_DIR.value
  if not (_PATCH_WIDTH.value == 800 and _PATCH_HEIGHT.value == 800 and
          _PATCH_OVERLAP.value == 200):
    patch_dir += f'{_PATCH_WIDTH.value}_{_PATCH_HEIGHT.value}_{_PATCH_OVERLAP.value}'

  img_base = os.path.splitext(img_name)[0]
  img_namesplit = img_base.split('_', 1)
  patch_base = img_namesplit[0]
  patch_annotation = img_namesplit[1] if len(img_namesplit) > 1 else None
  return patch_dir, patch_base, patch_annotation


def main(unused_argv):
  for split_set in ['train', 'val', 'test']:
    split_pattern = f'{_DATASET_DIR.value}/{split_set}/images/*.png'
    split_files = tf.io.gfile.glob(split_pattern)
    split_start = time.time()
    for (i, split_file) in enumerate(split_files, start=1):
      # In iSAID devkit - split.py two images are skipped:
      file_name = os.path.split(split_file)[-1]
      if file_name.startswith('P1527') or file_name.startswith('P1530'):
        continue

      patch_dir, patch_base, patch_annot = _get_patch_base_names(split_file)

      image_generate_patches.create_patches(
          img_file=split_file,
          patch_height=_PATCH_HEIGHT.value,
          patch_width=_PATCH_WIDTH.value,
          patch_overlap=_PATCH_OVERLAP.value,
          patch_dir=patch_dir,
          patch_base=patch_base,
          patch_annotation=patch_annot,
          load_cv2=True,
      )

      logging.info('%10s %5d / %5d (%5.2f sec/example)', split_set, i,
                   len(split_files), (time.time() - split_start) / float(i))

      if _DEBUG.value > 0 and i > _DEBUG.value:
        break


if __name__ == '__main__':
  app.run(main)
