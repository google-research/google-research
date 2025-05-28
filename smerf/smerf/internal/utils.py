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

from concurrent import futures
import enum
import gzip
import json
import os
from typing import Any, Optional, Union

from camp_zipnerf.internal import image_io as teacher_image_io
from camp_zipnerf.internal import utils as teacher_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np


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
class Rays(teacher_utils.Rays):
  """All tensors must have the same num_dims and first n-1 dims must match."""

  sm_idxs: Optional[_Array] = None  # i32[..., 1]


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


open_file = teacher_utils.open_file
listdir = teacher_utils.listdir
isdir = teacher_utils.isdir
file_exists = teacher_utils.file_exists
makedirs = teacher_utils.makedirs
device_is_tpu = teacher_utils.device_is_tpu
shard = teacher_utils.shard
unshard = teacher_utils.unshard
load_img = teacher_image_io.load_img
load_npy = teacher_utils.load_npy
load_exif = teacher_image_io.load_exif
save_img_u8 = teacher_image_io.save_img_u8
save_img_f32 = teacher_image_io.save_img_f32


def save_json_gz(x, path):
  with open_file(os.fspath(path), 'wb') as f:
    x = json.dumps(x, indent=2, sort_keys=True)
    x = x.encode('utf-8')
    x = gzip.compress(x)
    f.write(x)


def save_json(x, path):
  with open_file(os.fspath(path), 'w') as f:
    json.dump(x, f, indent=2, sort_keys=True)


def load_json(path):
  with open_file(os.fspath(path), 'r') as f:
    x = json.load(f)
  return x


def save_np(x, path):
  with open_file(os.fspath(path), 'wb') as f:
    np.save(f, x)


def load_np(path):
  with open_file(os.fspath(path), 'rb') as f:
    x = np.load(f)
  return x


def pre_pmap(x, ndims, *, xnp=jnp):
  """Prepares an array for pmap.

  Args:
    x: Array to prepare for pmap. Must have at least 'ndim' dimensions.
    ndims: int. Number of dimensions for each data element. All other
      dimensions are batch dimensions.
    xnp: numpy or jax.numpy

  Returns:
    x: 'x' prepared for pmap. x_pmap's first two dimensions correspond to
      local device and per-device batch, respectively.
    state: State to pass to post_pmap() when reconstructing arrays with the
      same batch shape as this array.
  """
  assert len(x.shape) >= ndims
  # Determine the shape of batch and element dimensions.
  num_batch_dims = len(x.shape) - ndims
  batch_shape = x.shape[:num_batch_dims]
  elem_shape = x.shape[num_batch_dims:]

  # Merge all batch dimensions together.
  x = xnp.reshape(x, (-1, *elem_shape))

  # Calculate the required batch size to satisfy pmap.
  actual_batch_size = x.shape[0]
  num_devices = jax.local_device_count()
  padded_batch_size = (
      np.ceil(actual_batch_size / num_devices).astype(int) * num_devices
  )

  # Add padding and introduce a 'devices' dimension.
  x = xnp.resize(x, (padded_batch_size, *elem_shape))
  x = xnp.reshape(x, (num_devices, -1, *elem_shape))

  return x, (actual_batch_size, batch_shape)


def post_pmap(x, state, *, xnp=jnp):
  """Post-processes an array after pmap.

  Args:
    x: Output of a model processed by pmap(). First two dimensions must
      correspond to local device and per-device batch.
    state: See pre_pmap().
    xnp: numpy or jax.numpy

  Returns:
    x: Valid elements of 'x'. Padded elements are discarded and batch shape
      is restored.
  """
  # See pre_pmap() for this definition.
  actual_batch_size, batch_shape = state

  # The first two dimensions are (num_devices, per_device_batch_size)
  elem_shape = x.shape[2:]

  # Drop padding examples.
  x = xnp.reshape(x, (-1, *elem_shape))
  x = x[:actual_batch_size]

  # Reintroduce batch dimensions.
  x = xnp.reshape(x, (*batch_shape, *elem_shape))
  return x


class AsyncThreadPool:
  """A fire-and-forget thread pool."""

  def __init__(self, max_workers=10):
    """Initializes AsyncThreadPool."""
    self._executor = futures.ThreadPoolExecutor(max_workers=max_workers)
    self._futures = []

  def submit(self, fn, *args, **kwargs):
    """Creates a new asynchronous task."""
    future = self._executor.submit(fn, *args, **kwargs)
    self._futures.append(future)

  def flush(self):
    """Flushes all queued futures."""
    n = len(self._futures)
    for future in futures.as_completed(self._futures):
      # This throws an exception if something goes wrong.
      future.result()
    self._futures = self._futures[n:]

