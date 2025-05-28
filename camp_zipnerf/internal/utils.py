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

import concurrent
import enum
import os
import queue
import threading
import time
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

from absl import logging
import flax
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


_Array = Union[np.ndarray, jnp.ndarray]


@flax.struct.dataclass
class Rays:
  """All tensors must have the same num_dims and first n-1 dims must match.

  This dataclass contains spatially meaningful quantities associated with
  the ray that can be calculated by the function casting the ray, as well as
  all metadata necessary for the ray to be rendered by the Model class.
  """

  origins: Optional[_Array] = None
  directions: Optional[_Array] = None
  viewdirs: Optional[_Array] = None
  radii: Optional[_Array] = None
  imageplane: Optional[_Array] = None
  pixels: Optional[_Array] = None
  lossmult: Optional[_Array] = None
  near: Optional[_Array] = None
  far: Optional[_Array] = None
  cam_idx: Optional[_Array] = None
  exposure_idx: Optional[_Array] = None
  exposure_values: Optional[_Array] = None
  device_idx: Optional[_Array] = None


def generate_random_rays(
    rng,
    n,
    origin_lo,
    origin_hi,
    radius_lo,
    radius_hi,
    near_lo,
    near_hi,
    far_lo,
    far_hi,
    include_exposure_idx = False,
    include_exposure_values = False,
    include_device_idx = False,
):
  """Generate a random Rays datastructure."""
  key, rng = random.split(rng)
  origins = random.uniform(
      key, shape=[n, 3], minval=origin_lo, maxval=origin_hi
  )

  key, rng = random.split(rng)
  directions = random.normal(key, shape=[n, 3])
  directions /= jnp.sqrt(
      jnp.maximum(
          jnp.finfo(jnp.float32).tiny,
          jnp.sum(directions**2, axis=-1, keepdims=True),
      )
  )

  viewdirs = directions

  key, rng = random.split(rng)
  radii = random.uniform(key, shape=[n, 1], minval=radius_lo, maxval=radius_hi)

  key, rng = random.split(rng)
  near = random.uniform(key, shape=[n, 1], minval=near_lo, maxval=near_hi)

  key, rng = random.split(rng)
  far = random.uniform(key, shape=[n, 1], minval=far_lo, maxval=far_hi)

  imageplane = jnp.zeros([n, 2])
  lossmult = jnp.zeros([n, 1])

  key, rng = random.split(rng)
  pixels = random.randint(key, shape=[n, 2], minval=0, maxval=1024)

  int_scalar = jnp.int32(jnp.zeros([n, 1]))

  exposure_kwargs = {}
  if include_exposure_idx:
    exposure_kwargs['exposure_idx'] = int_scalar
  if include_exposure_values:
    exposure_kwargs['exposure_values'] = jnp.zeros([n, 1])
  if include_device_idx:
    exposure_kwargs['device_idx'] = int_scalar

  random_rays = Rays(
      origins=origins,
      directions=directions,
      viewdirs=viewdirs,
      radii=radii,
      imageplane=imageplane,
      pixels=pixels,
      lossmult=lossmult,
      near=near,
      far=far,
      cam_idx=int_scalar,
      **exposure_kwargs,
  )
  return random_rays


# Dummy Rays object that can be used to initialize NeRF model.
def dummy_rays(
    include_exposure_idx = False,
    include_exposure_values = False,
    include_device_idx = False,
):
  return generate_random_rays(
      random.PRNGKey(0),
      n=100,
      origin_lo=-1.5,
      origin_hi=1.5,
      radius_lo=1e-5,
      radius_hi=1e-3,
      near_lo=0.0,
      near_hi=1.0,
      far_lo=10,
      far_hi=10000,
      include_exposure_idx=include_exposure_idx,
      include_exposure_values=include_exposure_values,
      include_device_idx=include_device_idx,
  )


@flax.struct.dataclass
class Batch:
  """Data batch for NeRF training or testing.

  This dataclass contains rays and also per-pixel data that is necessary for
  computing the loss term or evaluating metrics but NOT necessary for rendering.
  """

  rays: Rays
  rgb: Optional[_Array] = None
  disps: Optional[_Array] = None
  normals: Optional[_Array] = None
  alphas: Optional[_Array] = None
  masks: Optional[_Array] = None


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


def file_exists(pth):
  return os.path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return os.path.isdir(pth)


def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)


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


def load_npy(pth):
  """Load an numpy array cast to float32."""
  with open_file(pth, 'rb') as f:
    x = np.load(f).astype(np.float32)
  return x


def assert_valid_stepfun(t, y):
  """Assert that step function (t, y) has a valid shape."""
  if t.shape[-1] != y.shape[-1] + 1:
    raise ValueError(
        f'Invalid shapes ({t.shape}, {y.shape}) for a step function.'
    )


def assert_valid_linspline(t, y):
  """Assert that piecewise linear spline (t, y) has a valid shape."""
  if t.shape[-1] != y.shape[-1]:
    raise ValueError(
        f'Invalid shapes ({t.shape}, {y.shape}) for a linear spline.'
    )


_FnT = TypeVar('_FnT', bound=Callable[Ellipsis, Iterable[Any]])


def iterate_in_separate_thread(
    queue_size = 3,
):
  """Decorator factory that iterates a function in a separate thread.

  Args:
    queue_size: Keep at most queue_size elements in memory.

  Returns:
    Decorator that will iterate a function in a separate thread.
  """

  def decorator(
      fn,
  ):
    def result_fn(*args, **kwargs):
      results_queue = queue.Queue(queue_size)
      populating_data = True
      populating_data_lock = threading.Lock()

      def thread_fn():
        # Mark has_data as a variable that's outside of thread_fn
        # Otherwise, `populating_data = True` creates a local variable
        nonlocal populating_data
        try:
          for item in fn(*args, **kwargs):
            results_queue.put(item)
        finally:
          # Set populating_data to False regardless of exceptions to stop
          # iterations
          with populating_data_lock:
            populating_data = False

      # Use executor + futures instead of Thread to propagate exceptions
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        thread_fn_future = executor.submit(thread_fn)

        while True:
          with populating_data_lock:
            if not populating_data and results_queue.empty():
              break
          get_start = time.time()
          try:
            # Set timeout to allow for exceptions to be propagated.
            next_value = results_queue.get(timeout=1.0)
          except queue.Empty:
            continue
          logging.info('Got data in %0.3fs', time.time() - get_start)
          yield next_value

        # Thread exception will be raised here
        thread_fn_future.result()

    return result_fn

  return decorator
