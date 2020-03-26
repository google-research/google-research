# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A collection of projection utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from lighthouse.geometry import sampling


def inv_depths(start_depth, end_depth, num_depths):
  """Returns reversed, sorted inverse interpolated depths.

  Args:
    start_depth: The first depth.
    end_depth: The last depth.
    num_depths: The total number of depths to create, include start_depth and
      end_depth are always included and other depths are interpolated between
      them, in inverse depth space.

  Returns:
    The depths sorted in descending order (so furthest first). This order is
    useful for back to front compositing.
  """

  depths = 1.0 / tf.linspace(1.0 / end_depth, 1.0 / start_depth, num_depths)
  return depths


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates

  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """

  # Derived from code written by Tinghui Zhou and Shubham Tulsiani
  batch = tf.shape(depth)[0]
  height = tf.shape(depth)[1]
  width = tf.shape(depth)[2]
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height * width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords


def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]

  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """

  # Derived from code written by Tinghui Zhou and Shubham Tulsiani
  batch = tf.shape(cam_coords)[0]
  height = tf.shape(cam_coords)[2]
  width = tf.shape(cam_coords)[3]
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def mpi_resample_cube(mpi, tgt, intrinsics, depth_planes, side_length,
                      cube_res):
  """Resample MPI onto cube centered at target point.

  Args:
    mpi: [B,H,W,D,C], input MPI
    tgt: [B,3], [x,y,z] coordinates for cube center (in reference/mpi frame)
    intrinsics: [B,3,3], MPI reference camera intrinsics
    depth_planes: [D] depth values for MPI planes
    side_length: metric side length of cube
    cube_res: resolution of each cube dimension

  Returns:
    resampled: [B, cube_res, cube_res, cube_res, C]
  """

  batch_size = tf.shape(mpi)[0]
  num_depths = tf.shape(mpi)[3]

  # compute MPI world coordinates
  intrinsics_tile = tf.tile(intrinsics, [num_depths, 1, 1])

  # create cube coordinates
  b_vals = tf.to_float(tf.range(batch_size))
  x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
  y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
  z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, cube_res)
  b, y, x, z = tf.meshgrid(b_vals, y_vals, x_vals, z_vals, indexing='ij')

  x = x + tgt[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
  y = y + tgt[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
  z = z + tgt[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]

  ones = tf.ones_like(x)
  coords = tf.stack([x, y, z, ones], axis=1)
  coords_r = tf.reshape(
      tf.transpose(coords, [0, 4, 1, 2, 3]),
      [batch_size * cube_res, 4, cube_res, cube_res])

  # store elements with negative z vals for projection
  bad_inds = tf.less(z, 0.0)

  # project into reference camera to transform coordinates into MPI indices
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size * cube_res, 1, 1])
  intrinsics_tile = tf.tile(intrinsics, [cube_res, 1, 1])
  intrinsics_tile_4 = tf.concat(
      [intrinsics_tile,
       tf.zeros([batch_size * cube_res, 3, 1])], axis=2)
  intrinsics_tile_4 = tf.concat([intrinsics_tile_4, filler], axis=1)
  coords_proj = cam2pixel(coords_r, intrinsics_tile_4)
  coords_depths = tf.transpose(coords_r[:, 2:3, :, :], [0, 2, 3, 1])
  coords_depth_inds = (tf.to_float(num_depths) - 1) * (
      (1.0 / coords_depths) -
      (1.0 / depth_planes[0])) / ((1.0 / depth_planes[-1]) -
                                  (1.0 / depth_planes[0]))
  coords_proj = tf.concat([coords_proj, coords_depth_inds], axis=3)
  coords_proj = tf.transpose(
      tf.reshape(coords_proj, [batch_size, cube_res, cube_res, cube_res, 3]),
      [0, 2, 3, 1, 4])
  coords_proj = tf.concat([b[:, :, :, :, tf.newaxis], coords_proj], axis=4)

  # trilinear interpolation gather from MPI
  # interpolate pre-multiplied RGBAs, then un-pre-multiply
  mpi_alpha = mpi[Ellipsis, -1:]
  mpi_channels_p = mpi[Ellipsis, :-1] * mpi_alpha
  mpi_p = tf.concat([mpi_channels_p, mpi_alpha], axis=-1)

  resampled_p = sampling.trilerp_gather(mpi_p, coords_proj, bad_inds)

  resampled_alpha = tf.clip_by_value(resampled_p[Ellipsis, -1:], 0.0, 1.0)
  resampled_channels = resampled_p[Ellipsis, :-1] / (resampled_alpha + 1e-8)
  resampled = tf.concat([resampled_channels, resampled_alpha], axis=-1)

  return resampled, coords_proj


def spherical_cubevol_resample(vol, env2ref, cube_center, side_length, n_phi,
                               n_theta, n_r):
  """Resample cube volume onto spherical coordinates centered at target point.

  Args:
    vol: [B,H,W,D,C], input volume
    env2ref: [B,4,4], relative pose transformation (transform env to ref)
    cube_center: [B,3], [x,y,z] coordinates for center of cube volume
    side_length: side length of cube
    n_phi: number of samples along vertical spherical coordinate dim
    n_theta: number of samples along horizontal spherical coordinate dim
    n_r: number of samples along radius spherical coordinate dim

  Returns:
    resampled: [B, n_phi, n_theta, n_r, C]
  """

  batch_size = tf.shape(vol)[0]
  height = tf.shape(vol)[1]

  cube_res = tf.to_float(height)

  # create spherical coordinates
  b_vals = tf.to_float(tf.range(batch_size))
  phi_vals = tf.linspace(0.0, np.pi, n_phi)
  theta_vals = tf.linspace(1.5 * np.pi, -0.5 * np.pi, n_theta)

  # compute radii to use
  x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0,
                       tf.to_int32(cube_res))
  y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0,
                       tf.to_int32(cube_res))
  z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0,
                       tf.to_int32(cube_res))
  y_c, x_c, z_c = tf.meshgrid(y_vals, x_vals, z_vals, indexing='ij')
  x_c = x_c + cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
  y_c = y_c + cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
  z_c = z_c + cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]
  cube_coords = tf.stack([x_c, y_c, z_c], axis=4)
  min_r = tf.reduce_min(
      tf.norm(
          cube_coords -
          env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :],
          axis=4),
      axis=[0, 1, 2, 3])  # side_length / cube_res
  max_r = tf.reduce_max(
      tf.norm(
          cube_coords -
          env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :],
          axis=4),
      axis=[0, 1, 2, 3])

  r_vals = tf.linspace(max_r, min_r, n_r)
  b, phi, theta, r = tf.meshgrid(
      b_vals, phi_vals, theta_vals, r_vals,
      indexing='ij')  # currently in env frame

  # transform spherical coordinates into cartesian
  # (currently in env frame, z points forwards)
  x = r * tf.cos(theta) * tf.sin(phi)
  z = r * tf.sin(theta) * tf.sin(phi)
  y = r * tf.cos(phi)

  # transform coordinates into ref frame
  sphere_coords = tf.stack([x, y, z, tf.ones_like(x)], axis=-1)[Ellipsis, tf.newaxis]
  sphere_coords_ref = tfmm(env2ref, sphere_coords)
  x = sphere_coords_ref[Ellipsis, 0, 0]
  y = sphere_coords_ref[Ellipsis, 1, 0]
  z = sphere_coords_ref[Ellipsis, 2, 0]

  # transform coordinates into vol indices
  x_inds = (x - cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis] +
            side_length / 2.0) * ((cube_res - 1) / side_length)
  y_inds = -(y - cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis] -
             side_length / 2.0) * ((cube_res - 1) / side_length)
  z_inds = -(z - cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis] -
             side_length / 2.0) * ((cube_res - 1) / side_length)
  sphere_coords_inds = tf.stack([b, x_inds, y_inds, z_inds], axis=-1)

  # trilinear interpolation gather from volume
  # interpolate pre-multiplied RGBAs, then un-pre-multiply
  vol_alpha = tf.clip_by_value(vol[Ellipsis, -1:], 0.0, 1.0)
  vol_channels_p = vol[Ellipsis, :-1] * vol_alpha
  vol_p = tf.concat([vol_channels_p, vol_alpha], axis=-1)

  resampled_p = sampling.trilerp_gather(vol_p, sphere_coords_inds)

  resampled_alpha = resampled_p[Ellipsis, -1:]
  resampled_channels = resampled_p[Ellipsis, :-1] / (resampled_alpha + 1e-8)
  resampled = tf.concat([resampled_channels, resampled_alpha], axis=-1)

  return resampled, r_vals


def over_composite(rgbas):
  """Combines a list of rgba images using the over operation.

  Combines RGBA images from back to front (where back is index 0 in list)
  with the over operation.

  Args:
    rgbas: A list of rgba images, these are combined from *back to front*.

  Returns:
    Returns an RGB image.
  """

  alphas = rgbas[:, :, :, :, -1:]
  colors = rgbas[:, :, :, :, :-1]
  transmittance = tf.cumprod(
      1.0 - alphas + 1.0e-8, axis=3, exclusive=True, reverse=True) * alphas
  output = tf.reduce_sum(transmittance * colors, axis=3)
  accum_alpha = tf.reduce_sum(transmittance, axis=3)

  return tf.concat([output, accum_alpha], axis=3)


def interleave_shells(shells, radii):
  """Interleave spherical shell tensors out-to-in by radii."""

  radius_order = tf.argsort(radii, direction='DESCENDING')
  shells_interleaved = tf.gather(shells, radius_order, axis=3)
  return shells_interleaved


# To complete this codebase, you must copy lines 6-191 from
# https://github.com/Fyusion/LLFF/blob/master/llff/math/mpi_math.py
# to here. Some incomplete function stubs are provided to suppress python lint
# errors.


def tfmm(a_mat, b_mat):
  """Redefined tensorflow matrix multiply (broken)."""
  return tf.linalg.matmul(a_mat, b_mat)
