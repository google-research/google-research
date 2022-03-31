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

"""Camera sampling, 3D and scene bound helpers, in Numpy and PyTorch."""

import functools

import numpy as np
import torch

# Height, width and focal length of the image plane in pixels, from NeRF's
# Blender dataset of realistic synthetic objects. The focal length determines
# how zoomed in renderings appear.
_HWF_BLENDER = np.array([800., 800., 1111.1111])


def scale_intrinsics(new_width, hwf=_HWF_BLENDER):
  """Scale camera intrinsics (heigh, width focal) to a desired image width."""
  return hwf * new_width / hwf[1]


def trans_t(t):
  return np.array([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, t],
      [0, 0, 0, 1],
  ],
                  dtype=np.float32)


def rot_phi(phi):
  return np.array([
      [1, 0, 0, 0],
      [0, np.cos(phi), -np.sin(phi), 0],
      [0, np.sin(phi), np.cos(phi), 0],
      [0, 0, 0, 1],
  ],
                  dtype=np.float32)


def rot_theta(th):
  return np.array([
      [np.cos(th), 0, -np.sin(th), 0],
      [0, 1, 0, 0],
      [np.sin(th), 0, np.cos(th), 0],
      [0, 0, 0, 1],
  ],
                  dtype=np.float32)


def pose_spherical(theta, phi, radius):
  c2w = trans_t(radius)
  c2w = np.matmul(rot_phi(phi / 180. * np.pi), c2w)
  c2w = np.matmul(rot_theta(theta / 180. * np.pi), c2w)
  c2w = np.matmul(
      np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), c2w)
  return c2w


def uniform_in_interval(interval):
  return np.random.uniform() * (interval[1] - interval[0]) + interval[0]


def sample_camera(th_range, phi_range, rad_range, focal_mult_range):
  """Sample random camera extrinsics (pose) on a sphere.

  NOTE: This function samples latitude and longitude uniformly in th_range and
        phi_range, respectively, but will oversample points from high latitudes,
        such as near the poles. In experiments, Dream Fields simply use
        phi_range = [-30, -30] to fix the camera to 30 degree elevation, so
        oversampling isn't a problem.

  Args:
    th_range (pair of floats): Camera azimuth range.
    phi_range (pair of floats): Camera elevation range. Negative values are
      above equator.
    rad_range (pair of floats): Distance to center of scene.
    focal_mult_range (pair of floats): Factor to multipy focal range.

  Returns:
    pose (array): Camera to world transformation matrix.
    rad (float): Radius of camera from center of scene.
    focal_mult (float): Value to multiply focal length by.
  """
  th = uniform_in_interval(th_range)
  phi = uniform_in_interval(phi_range)
  rad = uniform_in_interval(rad_range)
  focal_mult = uniform_in_interval(focal_mult_range)
  pose = pose_spherical(th, phi, rad)
  return pose, rad, focal_mult


def sample_cameras(num, th_range, phi_range, rad_range, focal_mult_range):
  cameras = [
      sample_camera(th_range, phi_range, rad_range, focal_mult_range)
      for _ in range(num)
  ]
  poses, rads, focal_mults = zip(*cameras)
  return np.stack(poses), np.array(rads), np.array(focal_mults)


def generate_rays(pixel_coords, pix2cam, cam2world):
  """Generate camera rays from pixel coordinates and poses."""
  homog = np.ones_like(pixel_coords[Ellipsis, :1])
  pixel_dirs = np.concatenate([pixel_coords + .5, homog], axis=-1)[Ellipsis, None]
  cam_dirs = np.matmul(pix2cam, pixel_dirs)
  ray_dirs = np.matmul(cam2world[Ellipsis, :3, :3], cam_dirs)[Ellipsis, 0]
  ray_origins = np.broadcast_to(cam2world[Ellipsis, :3, 3], ray_dirs.shape)

  dpixel_dirs = np.concatenate([pixel_coords + .5 + np.array([1, 0]), homog],
                               axis=-1)[Ellipsis, None]
  ray_diffs = np.linalg.norm(
      (np.matmul(pix2cam, dpixel_dirs) - cam_dirs)[Ellipsis, 0],
      axis=-1,
      keepdims=True) / np.sqrt(12.)
  return ray_origins, ray_dirs, ray_diffs


def pix2cam_matrix(height, width, focal):
  """Inverse intrinsic matrix for a pinhole camera."""
  return np.array([
      [1. / focal, 0, -.5 * width / focal],
      [0, -1. / focal, .5 * height / focal],
      [0, 0, -1.],
  ])


def camera_rays(cam2world, height, width, focal):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  pix2cam = pix2cam_matrix(height, width, focal)
  height, width = height.astype(int), width.astype(int)
  pixel_coords = np.stack(
      np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
  return generate_rays(pixel_coords, pix2cam, cam2world)


def camera_rays_batched(cam2worlds, height, width, focals):
  """Generate rays for a pinhole camera with given extrinsics and intrinsics."""
  rays_batched = []
  for cam2world, focal in zip(cam2worlds, focals):
    rays = camera_rays(cam2world, height, width, focal)
    rays_batched.append(rays)

  ray_origins, ray_dirs, ray_diffs = zip(*rays_batched)
  return np.stack(ray_origins), np.stack(ray_dirs), np.stack(ray_diffs)


def mask_sigma(sigma, pts, mask_radius, mask_rad_norm):
  """Compute and apply mask based on scene bounds."""
  masks = []

  if mask_rad_norm is not None:
    if mask_rad_norm == 'inf':
      bounds = (torch.max(pts.abs(), -1)[0] <= mask_radius)
      masks.append(bounds)
    else:
      norm = torch.linalg.norm(pts, ord=mask_rad_norm, axis=-1)
      bounds = norm <= mask_radius
      masks.append(bounds)

  if masks:
    return sigma * functools.reduce(torch.logical_and, masks)
  else:
    raise ValueError

  return sigma


class EMA:
  """Keep track of the EMA of a single array, on CPU."""

  def __init__(self, value, decay):
    self.value = np.array(value)
    self.decay = decay

  def update(self, new_value):
    self.value = (
        self.value * self.decay + np.array(new_value) * (1 - self.decay))


def pts_center(pts, weights):
  """Estimate the center of mass of a scene."""
  total_weight = torch.sum(weights)
  origin_weights = weights[Ellipsis, None] / total_weight
  origin = origin_weights * pts
  reduce_dims = tuple(range(origin.ndim - 1))
  origin = origin.sum(dim=reduce_dims)  # Sum except last dim.
  return origin, total_weight
