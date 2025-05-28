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

"""Utility functions."""

import enum
import json
import os
from os import path
from typing import Any, Dict, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import ExifTags
from PIL import Image

_Array = Union[np.ndarray, jnp.ndarray]


@flax.struct.dataclass
class Pixels:
  """All tensors must have the same num_dims and first n-1 dims must match."""

  pix_x_int: _Array
  pix_y_int: _Array
  lossmult: _Array
  near: _Array
  far: _Array
  cam_idx: _Array
  exposure_idx: Optional[_Array] = None
  exposure_values: Optional[_Array] = None


@flax.struct.dataclass
class Rays:
  """All tensors must have the same num_dims and first n-1 dims must match."""

  origins: _Array
  directions: _Array
  viewdirs: _Array
  radii: _Array
  imageplane: _Array
  lossmult: _Array
  near: _Array
  far: _Array
  cam_idx: _Array
  exposure_idx: Optional[_Array] = None
  exposure_values: Optional[_Array] = None


@flax.struct.dataclass
class Batch:
  """Data batch for NeRF training or testing."""

  rays: Any
  rgb: Optional[_Array] = None
  semantic: Optional[_Array] = None
  disps: Optional[_Array] = None
  normals: Optional[_Array] = None
  alphas: Optional[_Array] = None


class DataSplit(enum.Enum):
  """Dataset split."""

  TRAIN = 'train'
  TEST = 'test'


class BatchingMethod(enum.Enum):
  """Draw rays randomly from a single image or all images, in each batch."""

  ALL_IMAGES = 'all_images'
  SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return path.isdir(pth)


def file_exists(pth):
  return path.exists(pth)


def makedirs(pth):
  return os.makedirs(pth, exist_ok=True)


def device_is_tpu():
  return jax.local_devices()[0].platform == 'tpu'


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_util.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs
  )


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def load_img(pth):
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def load_npy(pth):
  """Load an numpy array cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.load(f).astype(np.float32)
  return image


def load_exif(pth):
  """Load EXIF data for an image."""
  with open_file(pth, 'rb') as f:
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
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    ).save(f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')


def save_np(x, pth):
  with open_file(pth, 'wb') as f:
    np.save(f, x)


def load_np(pth):
  with open_file(pth, 'rb') as f:
    x = np.load(f)
  return x


def save_json(x, pth):
  with open_file(pth, 'w') as f:
    json.dump(x, f)
