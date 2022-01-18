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

"""Camera sampling, 3D and scene bound helpers."""

import functools

from . import helpers

import jax
from jax import random
import jax.numpy as np
import numpy as onp


# Height, width and focal length of the image plane in pixels, from NeRF's
# Blender dataset of realistic synthetic objects. The focal length determines
# how zoomed in renderings appear.
_HWF_BLENDER = onp.array([800., 800., 1111.1111])


def scale_intrinsics(new_width, hwf=_HWF_BLENDER):
  """Scale camera intrinsics (heigh, width focal) to a desired image width."""
  return hwf * new_width / hwf[1]


def posenc(points, degree):
  """Positional encoding mapping for 3D input points."""
  sh = list(points.shape[:-1])
  points = points[Ellipsis, None] * 2.**np.arange(degree)
  points = np.concatenate([np.cos(points), np.sin(points)], -1)
  return points.reshape(sh + [degree * 2 * 3])


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


@jax.jit
def pose_spherical(theta, phi, radius):
  c2w = trans_t(radius)
  c2w = helpers.matmul(rot_phi(phi / 180. * np.pi), c2w)
  c2w = helpers.matmul(rot_theta(theta / 180. * np.pi), c2w)
  c2w = helpers.matmul(
      np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), c2w)
  return c2w


@jax.jit
def uniform_in_interval(key, interval):
  return random.uniform(key, ()) * (interval[1] - interval[0]) + interval[0]


def sample_camera(key, th_range, phi_range, rad_range, focal_mult_range):
  """Sample random camera extrinsics (pose) on a sphere.

  NOTE: This function samples latitude and longitude uniformly in th_range and
        phi_range, respectively, but will oversample points from high latitudes,
        such as near the poles. In experiments, Dream Fields simply use
        phi_range = [-30, -30] to fix the camera to 30 degree elevation, so
        oversampling isn't a problem.

  Args:
    key: PRNGKey.
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
  key_th, key_phi, key_rad, key_focal = random.split(key, 4)
  th = uniform_in_interval(key_th, th_range)
  phi = uniform_in_interval(key_phi, phi_range)
  rad = uniform_in_interval(key_rad, rad_range)
  focal_mult = uniform_in_interval(key_focal, focal_mult_range)
  pose = pose_spherical(th, phi, rad)
  return pose, rad, focal_mult


def generate_rays(pixel_coords, pix2cam, cam2world):
  """Generate camera rays from pixel coordinates and poses."""
  homog = np.ones_like(pixel_coords[Ellipsis, :1])
  pixel_dirs = np.concatenate([pixel_coords + .5, homog], axis=-1)[Ellipsis, None]
  cam_dirs = helpers.matmul(pix2cam, pixel_dirs)
  ray_dirs = helpers.matmul(cam2world[Ellipsis, :3, :3], cam_dirs)[Ellipsis, 0]
  ray_origins = np.broadcast_to(cam2world[Ellipsis, :3, 3], ray_dirs.shape)

  dpixel_dirs = np.concatenate([pixel_coords + .5 + np.array([1, 0]), homog],
                               axis=-1)[Ellipsis, None]
  ray_diffs = np.linalg.norm(
      (helpers.matmul(pix2cam, dpixel_dirs) - cam_dirs)[Ellipsis, 0], axis=-1,
      keepdims=True) / np.sqrt(12.)
  return ray_origins, ray_dirs, ray_diffs


def pix2cam_matrix(height, width, focal):
  """Inverse intrinsic matrix for a pinhole camera."""
  return np.array([
      [1. / focal, 0, -.5 * width / focal],
      [0, -1. / focal, .5 * height / focal],
      [0, 0, -1.],
  ])


def camera_ray_batch(cam2world, height, width, focal):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  pix2cam = pix2cam_matrix(height, width, focal)
  height, width = height.astype(int), width.astype(int)
  pixel_coords = np.stack(
      np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
  return generate_rays(pixel_coords, pix2cam, cam2world)


def shard_rays(rays, multihost=True):
  ray_origins, ray_dirs, ray_diffs = rays
  if multihost:
    batch_shape = (jax.process_count(), jax.local_device_count(), -1)
  else:
    batch_shape = (jax.local_device_count(), -1)

  return (np.reshape(ray_origins, batch_shape + ray_origins.shape[-1:]),
          np.reshape(ray_dirs, batch_shape + ray_dirs.shape[-1:]),
          np.reshape(ray_diffs, batch_shape + ray_diffs.shape[-1:]))


def padded_shard_rays(rays, multihost=True):
  """Shard camera rays across devices, padding so hosts have equal rays."""
  if multihost:
    n_shard = jax.process_count() * jax.local_device_count()
  else:
    n_shard = jax.local_device_count()

  rays_padded = []
  for x in rays:
    assert x.ndim == 2
    n_ray = x.shape[0]
    n_pad = -n_ray % n_shard
    x_pad = np.pad(x, [(0, n_pad), (0, 0)])
    rays_padded.append(x_pad)

  return shard_rays(rays_padded, multihost)


def gather_and_reshape(pixels_flat, render_width, channels):
  pixels = jax.lax.all_gather(pixels_flat, axis_name='batch')
  pixels = pixels.reshape(render_width, render_width, channels)
  return pixels


def mask_sigma(sigma, pts, mask_radius, config):
  """Compute and apply mask based on scene bounds."""
  masks = []

  if config.mr_norm is not None:
    if config.mr_norm == 'inf':
      bounds = (np.max(np.abs(pts), -1) <= mask_radius)
      masks.append(bounds)
    else:
      norm = np.linalg.norm(pts, ord=config.mr_norm, axis=-1)
      bounds = norm <= mask_radius
      masks.append(bounds)

  if masks:
    return sigma * functools.reduce(np.logical_and, masks)

  return sigma


# Origin tracking
class EMA:
  """Keep track of the EMA of a single array, on CPU."""

  def __init__(self, value, decay):
    self.value = onp.array(value)
    self.decay = decay

  def update(self, new_value):
    self.value = (
        self.value * self.decay + onp.array(new_value) * (1 - self.decay))


def pts_center(pts, weights):
  total_weight = np.sum(weights)
  origin_weights = weights[Ellipsis, None] / total_weight
  origin = origin_weights * pts
  origin = np.array([
      np.sum(origin[Ellipsis, 0]),
      np.sum(origin[Ellipsis, 1]),
      np.sum(origin[Ellipsis, 2]),
  ])  # 3-dim
  return origin, total_weight
