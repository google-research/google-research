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

"""Tools for splitting an image into (overlapping) patches."""

from typing import Optional, Text, Tuple

from absl import logging

import cv2
import numpy as np
import tensorflow as tf

from factors_of_influence.fids import utils


def _load_image(image_file, load_cv2 = False):
  """Loads image using utils (PIL Image) or CV2 imdecode."""
  if not load_cv2:
    return utils.load_image(image_file)

  # Load images via cv2:
  with tf.io.gfile.GFile(image_file, 'rb') as f:
    img_bgr = cv2.imdecode(np.fromstring(f.read(), dtype=np.uint8),
                           cv2.IMREAD_COLOR)  # OpenCV reads in BGR format

  return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Return RGB converted image


def _get_patch_start_end(image_length,
                         patch_start,
                         patch_length):
  """Get patch start and end values."""
  patch_end = patch_start + patch_length
  if patch_end > image_length:
    patch_start = image_length - patch_length
    patch_end = image_length
  return patch_start, patch_end


def _create_patch(img, image_width, image_height,
                  start_width, start_height, patch_width,
                  patch_height):
  """Create a specific patch."""
  width_start, width_end = _get_patch_start_end(image_width, start_width,
                                                patch_width)
  height_start, height_end = _get_patch_start_end(image_height, start_height,
                                                  patch_height)

  patch = img[height_start:height_end, width_start:width_end, :]
  patch_name = f'{height_start}_{height_end}_{width_start}_{width_end}'
  return patch, patch_name


def _save_patch(patch,
                patch_dir,
                patch_base,
                patch_name = None,
                patch_annotation = None,
                patch_ext = 'png'):
  """Save an image patch."""
  patch_file = f'{patch_dir}/{patch_base}'
  if patch_name:
    patch_file += f'_{patch_name}'
  if patch_annotation:
    patch_file += f'_{patch_annotation}'
  patch_file += f'.{patch_ext}'

  if not tf.io.gfile.exists(patch_dir):
    logging.info('create dir: %s', patch_dir)
    tf.io.gfile.makedirs(patch_dir)

  utils.save_image(patch_file, patch)
  logging.debug('imwrite %s', patch_file)


def create_patches(img_file,
                   patch_height,
                   patch_width,
                   patch_overlap,
                   patch_dir,
                   patch_base,
                   patch_annotation = None,
                   patch_ext = 'png',
                   load_cv2 = False,
                   ):
  """Create patches from an image file.

  Args:
    img_file: string - Filepath of image to load.
    patch_height: int - Height of the desired patches.
    patch_width: int - Width of the desired patches.
    patch_overlap: int - Overlap between patches.
    patch_dir: Text - Output directory to save the generated patches.
    patch_base: Text - Common name (identifier) to save the patches.
    patch_annotation: Optional[Text] - Common post-fix of the patch name.
    patch_ext: Optional[Text] (default: png) - Extension of patches.
    load_cv2: use cv2 to load image files or utils.load_image.

  This function creates overlapping patches, eg for an input image with a
  width of 1500, using patch_width 800 and patch_overlap 200:
  [0-800, 600-1400, 700-1500]

  Note 1/: If a patch exceeds image boundary, it is repositioned to
    [image-length - patch-length, image-length].
    Image-length could be either width or height. This could create two almost
    overlapping images! For example, consider the worst-case when:
    image-width = patch-width + 1
    Then two patches with just a single pixel difference will be created.
    Negative values can not be obtained, because we require image size to be
    larger or equal to patch size.
  Note 2/: Patch naming is (loosely) as follows:
    {patch_dir}/{patch_base}_{patch_name}_{patch_annotation}{patch_ext}
  """
  img = _load_image(img_file, load_cv2)
  img_height, img_width, _ = img.shape

  if img_height >= patch_height and img_width >= patch_width:
    for start_width in range(0, img_width, patch_width - patch_overlap):
      for start_height in range(0, img_height, patch_height - patch_overlap):
        patch, patch_name = _create_patch(img, img_width, img_height,
                                          start_width, start_height,
                                          patch_width, patch_height)
        _save_patch(patch, patch_dir, patch_base, patch_name, patch_annotation,
                    patch_ext)
  else:
    _save_patch(img, patch_dir, patch_base, None, patch_annotation, patch_ext)
