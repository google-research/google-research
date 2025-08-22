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

"""IO Utility functions."""

from typing import Any, Dict

import cv2
from internal import utils
import numpy as np
from PIL import ExifTags
from PIL import Image


def load_img(pth, is_16bit = False):
  """Load an image and cast to float32."""
  with utils.open_file(pth, 'rb') as f:
    # Use OpenCV for reading 16-bit images, since PIL.Image.open() silently
    # casts those to 8-bit.
    if is_16bit:
      bytes_ = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Read bytes.
      image = np.array(
          cv2.imdecode(bytes_, cv2.IMREAD_UNCHANGED), dtype=np.float32
      )
    else:
      image = np.array(Image.open(f), dtype=np.float32)
  return image


def load_exif(pth):
  """Load EXIF data for an image."""
  with utils.open_file(pth, 'rb') as f:
    image_pil = Image.open(f)
    exif_pil = image_pil._getexif()  # pylint: disable=protected-access
    if exif_pil is not None:
      exif = {
          ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS
      }
    else:
      exif = {}
  return exif


def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  img = Image.fromarray(
      (np.round(np.clip(np.nan_to_num(img), 0.0, 1.0) * 255)).astype(np.uint8)
  )
  with utils.open_file(pth, 'wb') as f:
    img.save(f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  img = Image.fromarray(np.nan_to_num(depthmap).astype(np.float32))
  with utils.open_file(pth, 'wb') as f:
    img.save(f, 'TIFF', compression='tiff_adobe_deflate')
