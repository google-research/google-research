# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Dataset for forwarding facing scene in NeX(Shiny and Undistorted RFF) with reference view."""

import os
from os import path

import jax
import numpy as np

from light_field_neural_rendering.src.datasets.forward_facing import ForwardFacing
from light_field_neural_rendering.src.utils import data_types


class FFEpipolar(ForwardFacing):
  """Forward Facing epipolar dataset"""

  def __init__(self, split, args, train_ds=None):

    assert args.dataset.batching in [
        "single_image",
    ], "Only single image batching supported for now"

    if split == "test":
      self.train_images = train_ds.images
      self.train_rays = train_ds.rays
      self.train_worldtocamera = train_ds.worldtocamera
      self.train_camtoworlds = train_ds.camtoworlds

    self.num_ref_views = args.dataset.num_interpolation_views

    super(FFEpipolar, self).__init__(split, args)

  def _train_init(self, args):
    super(FFEpipolar, self)._train_init(args)

    #--------------------------------------------------------------------------------------
    # Get world to camera matrices
    # To compute world to camera matrix, conver the 3x4 camtoworld matrix to 4x4
    # matrix.
    bottom = np.tile(
        np.reshape([0, 0, 0, 1.], [1, 1, 4]), (self.camtoworlds.shape[0], 1, 1))
    c2w = np.concatenate([self.camtoworlds, bottom], -2)
    self.worldtocamera = np.linalg.inv(c2w)

    #--------------------------------------------------------------------------------------
    # Compute the nearest neighbors for each train view.
    cam_train_pos = self.camtoworlds[:, :3, -1]  # Camera positions
    train2traincam = np.linalg.norm(
        cam_train_pos[:, None] - cam_train_pos[None, :], axis=-1)
    # To avoid self-mapping
    np.fill_diagonal(train2traincam, np.inf)
    self.sorted_near_cam = np.argsort(train2traincam, axis=-1)[Ellipsis, :-1]

  def _test_init(self, args):
    super(FFEpipolar, self)._test_init(args)

    if args.dataset.render_path:
      # Compute nearest cameras for render poses
      curr_camtoworlds = self.render_poses
    else:
      # Compute nearest cameras for test poses
      curr_camtoworlds = self.camtoworlds

    #--------------------------------------------------------------------------------------
    # Compute the nearest neighbors for each test/render view.
    cam_train_pos = self.train_camtoworlds[:, :3, -1]
    cam_test_pos = curr_camtoworlds[:, :3, -1]
    test2traincam = np.linalg.norm(
        cam_test_pos[:, None] - cam_train_pos[None, :], axis=-1)
    self.sorted_near_cam = np.argsort(
        test2traincam, axis=-1)[Ellipsis, :self.num_ref_views]

  def _next_train(self):
    """Sample batch for training"""
    if self.batching == "single_image":
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays.batch_shape[1],
                                      (self.batch_size,))

      #--------------------------------------------------------------------------------------
      # Get batch pixels and rays
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = jax.tree_map(lambda r: r[image_index][ray_indices],
                                self.rays)

      #--------------------------------------------------------------------------------------
      # Get index of reference views
      # During training for additional regularization we chose a random number of
      # reference view for interpolation
      # Top k number of views to consider when randomly sampling
      # subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
      total_views = 20
      # Number of refernce views to select
      # num_select = self.num_ref_views + np.random.randint(low=-2, high=3)
      num_select = self.num_ref_views

      # Get the set of precomputed nearest camera indices
      batch_near_cam_idx = self.sorted_near_cam[image_index][:total_views]
      batch_near_cam_idx = np.random.choice(
          batch_near_cam_idx,
          min(num_select, len(batch_near_cam_idx)),
          replace=False)

      # Occasionally use input image
      # if np.random.choice([0,1], p=[0.995, .005]):
      #   batch_near_cam_idx[np.random.choice(len(batch_near_cam_idx))] = image_index

      #--------------------------------------------------------------------------------------
      # Get the reference data
      ref_images = self.images[batch_near_cam_idx]
      ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 3)

      ref_cameratoworld = self.camtoworlds[batch_near_cam_idx]
      ref_worldtocamera = self.worldtocamera[batch_near_cam_idx]

      # Each of these reference data need to be shared onto each local device. To
      # support this we replicate the reference data as many times as there are
      # local devices
      l_devices = jax.local_device_count()
      target_view = data_types.Views(rays=batch_rays, rgb=batch_pixels)
      reference_views = data_types.ReferenceViews(
          rgb=np.tile(ref_images, (l_devices, 1, 1, 1)),
          ref_worldtocamera=np.tile(ref_worldtocamera, (l_devices, 1, 1)),
          ref_cameratoworld=np.tile(ref_cameratoworld, (l_devices, 1, 1)),
          intrinsic_matrix=np.tile(self.intrinsic_matrix[None, :],
                                   (l_devices, 1, 1)),
          idx=np.tile(batch_near_cam_idx[None, :],
                      (jax.local_device_count(), 1)),
      )

      return_batch = data_types.Batch(
          target_view=target_view, reference_views=reference_views)

    else:
      raise ValueError("Batching {} not implemented".format(self.batching))

    return return_batch

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      target_view = data_types.Views(
          rays=jax.tree_map(lambda r: r[idx], self.render_rays),)
    else:
      target_view = data_types.Views(
          rays=jax.tree_map(lambda r: r[idx], self.rays), rgb=self.images[idx])

    #--------------------------------------------------------------------------------------
    # Get the reference data
    batch_near_cam_idx = self.sorted_near_cam[idx]
    ref_images = self.train_images[batch_near_cam_idx]
    ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 3)

    ref_cameratoworld = self.train_camtoworlds[batch_near_cam_idx]
    ref_worldtocamera = self.train_worldtocamera[batch_near_cam_idx]

    #--------------------------------------------------------------------------------------
    # Replicate these so that they may be distributed onto several devices for
    # parallel computaion.
    l_devices = jax.local_device_count()
    reference_views = data_types.ReferenceViews(
        rgb=np.tile(ref_images, (l_devices, 1, 1, 1)),
        ref_worldtocamera=np.tile(ref_worldtocamera, (l_devices, 1, 1)),
        ref_cameratoworld=np.tile(ref_cameratoworld, (l_devices, 1, 1)),
        intrinsic_matrix=np.tile(self.intrinsic_matrix[None, :],
                                 (l_devices, 1, 1)),
        idx=np.tile(batch_near_cam_idx[None, :], (jax.local_device_count(), 1)),
    )

    return_batch = data_types.Batch(
        target_view=target_view, reference_views=reference_views)

    return return_batch
