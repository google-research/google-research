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

"""Dataset for forwarding facing scene in NeX(Shiny and Undistorted RFF)."""

import os
from os import path
import tqdm

import jax
import numpy as np
from PIL import Image

from light_field_neural_rendering.src.datasets.base import BaseDataset
from light_field_neural_rendering.src.utils import data_types
from light_field_neural_rendering.src.utils import file_utils


class ForwardFacing(BaseDataset):
  """Dataloader for Shiny / RFF undistorted."""
  camtoworlds: np.ndarray
  render_poses: np.ndarray

  def _get_suffix(self, args):
    """Get the suffix for reshaped image directory."""

    imgdir_suffix = ""
    if args.dataset.factor > 0:
      imgdir_suffix = "_{}".format(int(args.dataset.factor))
      factor = args.dataset.factor
    elif args.dataset.image_height > 0:
      img_path = path.join(args.dataset.data_dir, "images")
      img0 = [
          path.join(args.dataset.data_dir, "images", f)
          for f in sorted(file_utils.listdir(img_path))
          if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
      ][0]
      with file_utils.open_file(img0, "rb") as imgin:
        sh = np.array(Image.open(imgin), dtype=np.uint8).shape
      factor = sh[0] / float(args.dataset.image_height)
      width = int(sh[1] / factor)
      imgdir_suffix = "_{}x{}".format(width, args.dataset.image_height)
    else:
      factor = 1

    return imgdir_suffix, factor

  def _load_images(self, imgdir):
    """Function to load images."""

    if not file_utils.file_exists(imgdir):
      raise ValueError("Image folder {} doesn't exist.".format(imgdir))
    imgfiles = [
        path.join(imgdir, f)
        for f in sorted(file_utils.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    images = []
    for imgfile in tqdm.tqdm(imgfiles):
      with file_utils.open_file(imgfile, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.uint8)
        images.append(image)
    images = np.stack(images, axis=-1)
    return images

  def _load_renderings(self, args):
    """Load images and camera information."""

    #-------------------------------------------
    # Load images.
    #-------------------------------------------
    imgdir_suffix, factor = self._get_suffix(args)
    imgdir = path.join(args.dataset.data_dir, "images" + imgdir_suffix)
    images = self._load_images(imgdir)

    #-------------------------------------------
    # Load poses and bds.
    #-------------------------------------------
    with file_utils.open_file(
        path.join(args.dataset.data_dir, "poses_bounds.npy"), "rb") as fp:
      poses_arr = np.load(fp)

    # Get the intrinsic matrix
    with file_utils.open_file(
        path.join(args.dataset.data_dir, "hwf_cxcy.npy"), "rb") as fp:
      self.intrinsic_arr = np.load(fp)

    # Update the intrinsic matix to accounto for resizing
    self.intrinsic_arr = self.intrinsic_arr * 1. / factor

    # poses_arr contains an array consisting of a 3x4 pose matrices and
    # 2 depth bounds for each image. The pose matrix contain [R t] as the
    # left 3x4 matrix
    # pose_arr has shape (...,14) {3x4 + 2}
    poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Convert R matrix from the form [down right back] to [right up back]
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

    # Transpose such that the first dimension is number of images
    images = np.moveaxis(images, -1, 0)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    scale = 1. / (bds.min() * .75)

    poses[:, :3, 3] *= scale
    bds *= scale
    poses = self._recenter_poses(poses)

    # Get the min and max depth of the scene
    self.min_depth = bds.min()
    self.max_depth = bds.max()
    # Use this to set the near and far plane
    args.model.near = self.min_depth.item()
    args.model.far = self.max_depth.item()

    if self.split == "test":
      self._generate_spiral_poses(poses, bds)

    # Select the split.
    i_test = np.arange(images.shape[0])[::args.dataset.llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    if self.split == "train":
      indices = i_train
    else:
      indices = i_test
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses[:, :3, :4]

    # intrinsic arr has H, W, fx, fy, cx, cy
    self.focal = self.intrinsic_arr[2][0]
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w

    if args.dataset.render_path and self.split == "test":
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

    _, _, fx, fy, cx, cy = self.intrinsic_arr[:, 0]
    self.intrinsic_matrix = np.array([[fx, 0, -cx, 0], [0, -fy, -cy, 0],
                                      [0, 0, 1, 0]]).astype(np.float32)

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    if self.split == "test":
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays()

    # Split poses from the dataset and generated poses
    if self.split == "test":
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split_origins = np.split(self.rays.origins, [n_render_poses], 0)
      split_directions = np.split(self.rays.directions, [n_render_poses], 0)
      if self.rays.base_radius is None:
        split_base_radius = [None, None]
      else:
        split_base_radius = np.split(self.rays.base_radius, [n_render_poses], 0)

      self.render_rays = data_types.Rays(
          origins=split_origins[0],
          directions=split_directions[0],
          base_radius=split_base_radius[0])

      self.rays = data_types.Rays(
          origins=split_origins[1],
          directions=split_directions[1],
          base_radius=split_base_radius[1])

  def _recenter_poses(self, poses):
    """Recenter poses according to the original NeRF code.

    Adopted from JaxNerf
    """
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = self._poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

  def _poses_avg(self, poses):
    """Average poses according to the original NeRF code.

    Adopted from JaxNerf
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = self._normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
    return c2w

  def _viewmatrix(self, z, up, pos):
    """Construct lookat view matrix. Adopted from JaxNerf."""
    vec2 = self._normalize(z)
    vec1_avg = up
    vec0 = self._normalize(np.cross(vec1_avg, vec2))
    vec1 = self._normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

  def _normalize(self, x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

  def _generate_spiral_poses(self, poses, bds):
    """Generate a spiral path for renderin. Adopted from JaxNerf."""
    c2w = self._poses_avg(poses)
    # Get average pose.
    up = self._normalize(poses[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset.
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    # Get radii for spiral path.
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2
    # Generate poses for spiral path.
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    zrate = .5
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
      c = np.dot(c2w[:3, :4], (np.array(
          [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
      z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
      render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
    self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]
