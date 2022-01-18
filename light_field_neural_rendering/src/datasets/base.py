# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""A base class for all the datasets. Based on dataset in jaxnerf."""
INTERNAL = False  # pylint: disable=g-statement-before-imports
import queue
import threading
import jax
import numpy as np

from light_field_neural_rendering.src.utils import data_types
from light_field_neural_rendering.src.utils import data_utils


class BaseDataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, args):
    super(BaseDataset, self).__init__()
    self.queue = queue.Queue(6)  # Set prefetch buffer to 6 batches.
    self.daemon = True
    self.split = split
    self.use_pixel_centers = args.dataset.use_pixel_centers
    if split == "train":
      self._train_init(args)
    elif split == "test":
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"test\", set"
          "to {} here.".format(split))
    self.batch_size = args.dataset.batch_size // jax.host_count()
    self.batching = args.dataset.batching
    self.render_path = args.dataset.render_path
    # Set the image height and width in the config
    args.dataset.image_height = self.h
    args.dataset.image_width = self.w
    self.resolution = self.h * self.w
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: data_types.Batch.
    """
    x = self.queue.get()
    if self.split == "train":
      return data_utils.shard(x)
    else:
      return data_utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: data_types.Batch".
    """
    while self.queue.empty():
      x = None
    # Make a copy of the front of the queue.
    x = jax.tree_map(lambda x: x.copy(), self.queue.queue[0])
    if self.split == "train":
      return data_utils.shard(x)
    else:
      return data_utils.to_device(x)

  def run(self):
    if self.split == "train":
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _load_renderings(self, args):
    raise NotImplementedError

  def _train_init(self, args):
    """Initialize training."""
    self._load_renderings(args)
    self._generate_rays()

    if args.dataset.batching == "all_images":
      # flatten the ray and image dimension together.
      self.images = self.images.reshape([-1, 3])
      self.rays = jax.tree_map(lambda r: r.reshape([-1, r.shape[-1]]),
                               self.rays)
    elif args.dataset.batching in ["single_image", "single_image_per_device"]:
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = jax.tree_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
    else:
      raise NotImplementedError(
          f"{args.dataset.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    self._load_renderings(args)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    if self.batching == "all_images":
      # Sample random ray indices
      # In all_images case, ray elements have a shape of
      # (num_train_images*resolution, _)
      ray_indices = np.random.randint(0, self.rays.batch_shape[0],
                                      (self.batch_size,))
      # Get the rgb value for these rays
      batch_pixels = self.images[ray_indices]
      # Get the ray information
      batch_rays = jax.tree_map(lambda r: r[ray_indices], self.rays)

    elif self.batching == "single_image":
      # Choose a random image
      image_index = np.random.randint(0, self.n_examples, ())
      # Choose ray indices for this image
      # Ray elements have a shape of (num_train_images, resolution, _)
      ray_indices = np.random.randint(0, self.rays.batch_shape[1],
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = jax.tree_map(lambda r: r[image_index][ray_indices],
                                self.rays)

    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")

    target_view = data_types.Views(rays=batch_rays, rgb=batch_pixels)
    return data_types.Batch(target_view=target_view)

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      rays = jax.tree_map(lambda r: r[idx], self.render_rays)
      target_view = data_types.Views(rays=rays)
      return data_types.Batch(target_view=target_view)

    else:
      rays = jax.tree_map(lambda r: r[idx], self.rays)
      pixels = self.images[idx]
      target_view = data_types.Views(rays=rays, rgb=pixels)
      return data_types.Batch(target_view=target_view)

  def _generate_rays(self):
    """Generate rays for all the views."""
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")

    pixels = np.stack((x, y, -np.ones_like(x)), axis=-1)
    inverse_intrisics = np.linalg.inv(self.intrinsic_matrix[Ellipsis, :3, :3])
    camera_dirs = (inverse_intrisics[None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    directions = (self.camtoworlds[:, None, None, :3, :3]
                  @ camera_dirs[None, Ellipsis, None])[Ellipsis, 0]
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=viewdirs)
