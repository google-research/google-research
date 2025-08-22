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

"""Camera sampling, 3D and scene bound helpers, in Numpy and PyTorch."""

import functools
from typing import Tuple, Union

import numpy as np
import torch

# Height, width and focal length of the image plane in pixels, from NeRF's
# Blender dataset of realistic synthetic objects. The focal length determines
# how zoomed in renderings appear.
_HWF_BLENDER = np.array([800., 800., 1111.1111])

Rays = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Number = Union[int, float]


def scale_intrinsics(new_width, hwf=_HWF_BLENDER):
  """Scale camera intrinsics (heigh, width focal) to a desired image width."""
  return hwf * new_width / hwf[1]


## Numpy extrinsics sampling.
def trans_t(t):
  return np.array([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, t],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def rot_phi(phi):
  return np.array([
      [1, 0, 0, 0],
      [0, np.cos(phi), -np.sin(phi), 0],
      [0, np.sin(phi), np.cos(phi), 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def rot_theta(th):
  return np.array([
      [np.cos(th), 0, -np.sin(th), 0],
      [0, 1, 0, 0],
      [np.sin(th), 0, np.cos(th), 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def pose_spherical(theta, phi, radius):
  c2w = trans_t(radius)
  c2w = np.matmul(rot_phi(phi / 180. * np.pi), c2w)
  c2w = np.matmul(rot_theta(theta / 180. * np.pi), c2w)
  c2w = np.matmul(
      np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), c2w)
  return c2w


def uniform_in_interval(interval, size=None):
  return np.random.uniform(*interval, size=size)


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


## Numpy ray generation.
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


## PyTorch ray generation.
def uniform_in_interval_th(interval, size=None):
  interval_width = interval[1] - interval[0]
  return torch.rand(size=size) * interval_width + interval[0]


@functools.lru_cache
def get_identity_th(d, batch_shape, device, dtype):
  mat = torch.eye(d, dtype=dtype, device=device)
  new_dims = (1,) * len(batch_shape)
  mat = mat.view(*new_dims, d, d)
  mat = mat.expand(*batch_shape, d, d)
  return mat


@functools.lru_cache
def get_flip_axes_th(batch_shape, device, dtype):
  new_dims = (1,) * len(batch_shape)
  flip_axes = torch.tensor(
      [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
      device=device,
      dtype=dtype)
  return flip_axes.view(*new_dims, 4, 4)


def trans_t_th(t):
  c2w = get_identity_th(4, t.shape, t.device, t.dtype).clone()
  c2w[Ellipsis, 2, 3] = t
  return c2w


def rot_phi_th(phi):
  c2w = get_identity_th(4, phi.shape, phi.device, phi.dtype).clone()
  cos_phi = torch.cos(phi)
  sin_phi = torch.sin(phi)
  c2w[Ellipsis, 1, 1] = cos_phi
  c2w[Ellipsis, 1, 2] = -sin_phi
  c2w[Ellipsis, 2, 2] = cos_phi
  c2w[Ellipsis, 2, 1] = sin_phi
  return c2w


def rot_theta_th(theta):
  c2w = get_identity_th(4, theta.shape, theta.device, theta.dtype).clone()
  cos_theta = torch.cos(theta)
  sin_theta = torch.sin(theta)
  c2w[Ellipsis, 0, 0] = cos_theta
  c2w[Ellipsis, 0, 2] = -sin_theta
  c2w[Ellipsis, 2, 2] = cos_theta
  c2w[Ellipsis, 2, 0] = sin_theta
  return c2w


def pose_spherical_th(theta, phi,
                      radius):
  """Construct camera to world matrix with a spherical model.

  The camera is on a sphere looking at the origin.

  Args:
    theta: [*batch] azimuth in radians.
    phi: [*batch] (negative) elevation in radians.
    radius: [*batch] camera radius.

  Returns:
    c2w: [*batch, 4, 4] batched camera-to-world extrinic matrices.
  """
  assert radius.shape == phi.shape == theta.shape
  c2w = trans_t_th(radius)
  c2w = torch.matmul(rot_phi_th(phi), c2w)
  c2w = torch.matmul(rot_theta_th(theta), c2w)
  c2w = torch.matmul(
      get_flip_axes_th(c2w.shape[:-2], c2w.device, c2w.dtype), c2w)
  return c2w


def generate_rays_th(pixel_coords, pix2cam,
                     cam2world):
  """Generate camera rays from pixel coordinates and poses.

  Args:
    pixel_coords: [width, height, 2] pixel-space coordinates.
    pix2cam: [*batch, 3, 3] batched inverse intrinsic matrix.
    cam2world: [*batch, 3+, 4] batched inverse intrinsic matrix.

  Returns:
    ray_origins: [*batch, width, height, 3] origin of each ray.
    ray_dirs: [*batch, width, height, 3] direction vector of each ray.
    ray_diffs: [*batch, width, height, 1] distance between points on each ray.
  """
  batch_shape = pix2cam.shape[:-2]
  new_dims = (1,) * len(batch_shape)

  homog = torch.ones_like(pixel_coords[Ellipsis, :1])  # [width, height, 1].
  pixel_dirs = torch.cat([pixel_coords + .5, homog], dim=-1)
  pix2cam_unsqueezed = pix2cam.view(*batch_shape, 1, 1, 3, 3)
  pixel_dirs_unsqueezed = pixel_dirs.view(*new_dims, *pixel_dirs.shape, 1)
  cam_dirs = pix2cam_unsqueezed.matmul(pixel_dirs_unsqueezed)
  ray_dirs = cam2world[Ellipsis, None, None, :3, :3].matmul(cam_dirs).squeeze(
      -1)  # [*batch, width, height, 3].
  ray_origins = cam2world[Ellipsis, None, None, :3, 3].broadcast_to(
      ray_dirs.shape)  # [*batch, width, height, 3].

  dpixel = torch.tensor([[[1.5, 0.5]]],
                        device=pixel_coords.device)  # [1, 1, 2].
  dpixel_dirs = torch.cat([pixel_coords + dpixel, homog],
                          dim=-1)  # [width, height, 3].
  dpixel_dirs_cam = torch.matmul(
      pix2cam_unsqueezed,
      dpixel_dirs.view(*new_dims, *dpixel_dirs.shape,
                       1)  # [1*batch, width, height, 3, 1].
  )
  ray_diffs = torch.linalg.norm(
      (dpixel_dirs_cam - cam_dirs).squeeze(-1), dim=-1,
      keepdims=True) / np.sqrt(12.)  # [*batch, width, height, 1].
  return ray_origins, ray_dirs, ray_diffs


def pix2cam_matrix_th(height, width,
                      focal):
  """Inverse intrinsic matrix for a pinhole camera."""
  batch_shape = focal.shape
  pix2cam = torch.zeros((*batch_shape, 3, 3), device=focal.device)
  inv_focal = 1. / focal
  pix2cam[Ellipsis, 0, 0] = inv_focal
  pix2cam[Ellipsis, 1, 1] = -inv_focal
  pix2cam[Ellipsis, 2, 2] = -1
  pix2cam[Ellipsis, 0, 2] = -.5 * width * inv_focal
  pix2cam[Ellipsis, 1, 2] = .5 * height * inv_focal
  return pix2cam


def camera_rays_th(cam2world, height, width,
                   focal):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  pix2cam = pix2cam_matrix_th(height, width, focal)
  device = focal.device
  px = torch.arange(width, device=device)
  py = px if height == width else torch.arange(height, device=device)
  pixel_coords = torch.stack(torch.meshgrid(px, py, indexing="xy"), dim=-1)
  return generate_rays_th(pixel_coords, pix2cam, cam2world)


## Other scene utilities.
def mask_sigma(sigma,
               pts,
               mask_radius,
               mask_rad_norm,
               invert_mask=False):
  """Compute and apply mask based on scene bounds."""
  masks = []

  if mask_rad_norm is not None:
    if mask_rad_norm == "inf":
      bounds = (torch.max(pts.abs(), -1)[0] <= mask_radius)
      masks.append(bounds)
    else:
      norm = torch.linalg.norm(pts, ord=mask_rad_norm, axis=-1)
      bounds = norm <= mask_radius
      masks.append(bounds)

  if masks:
    mask = functools.reduce(torch.logical_and, masks)
    if invert_mask:
      mask = torch.logical_not(mask)
    return sigma * mask
  else:
    raise ValueError

  return sigma


class EMA:
  """Keep track of the EMA of a single tensor."""

  @torch.no_grad()
  def __init__(self, value, decay):
    self.value = value.detach()
    self.decay = decay

  @torch.no_grad()
  def update(self, new_value):
    self.value = (
        self.value * self.decay + new_value.detach() * (1 - self.decay))


def pts_center(pts, weights):
  """Estimate the center of mass of a scene."""
  total_weight = torch.sum(weights)
  origin_weights = weights[Ellipsis, None] / total_weight
  origin = origin_weights * pts
  reduce_dims = tuple(range(origin.ndim - 1))
  origin = origin.sum(dim=reduce_dims)  # Sum except last dim.
  return origin, total_weight
