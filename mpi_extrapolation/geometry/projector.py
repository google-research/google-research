# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""A collection of projection utility functions.

Modified from code written by Shubham Tulsiani and Tinghui Zhou.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mpi_extrapolation.geometry import homography
from tensorflow.contrib import resampler as contrib_resampler


def projective_forward_homography(src_images, intrinsics, pose, depths):
  """Use homography for forward warping.

  Args:
    src_images: [layers, batch, height, width, channels]
    intrinsics: [batch, 3, 3]
    pose: [batch, 4, 4]
    depths: [layers, batch]
  Returns:
    proj_src_images: [layers, batch, height, width, channels]
  """
  n_layers = tf.shape(src_images)[0]
  n_batch = tf.shape(src_images)[1]
  height = tf.shape(src_images)[2]
  width = tf.shape(src_images)[3]
  # Format for Shubham's planar_transform code:
  # rot: relative rotation, are [...] X 3 X 3 matrices
  # t: B X 3 X 1, translations from source to target camera (R*p_s + t = p_t)
  # n_hat: L X B X 1 X 3, plane normal w.r.t source camera frame [0,0,1]
  #        in our case
  # a: L X B X 1 X 1, plane equation displacement (n_hat * p_src + a = 0)
  rot = pose[:, :3, :3]
  t = pose[:, :3, 3:]
  n_hat = tf.constant([0., 0., 1.], shape=[1, 1, 1, 3])
  n_hat = tf.tile(n_hat, [n_layers, n_batch, 1, 1])
  a = -tf.reshape(depths, [n_layers, n_batch, 1, 1])
  k_s = intrinsics
  k_t = intrinsics
  pixel_coords_trg = tf.transpose(
      meshgrid_abs(n_batch, height, width), [0, 2, 3, 1])
  proj_src_images = homography.planar_transform(
      src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
  return proj_src_images


def meshgrid_abs(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid in the absolute coordinates.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width-1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height-1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords


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
  batch = tf.shape(depth)[0]
  height = tf.shape(depth)[1]
  width = tf.shape(depth)[2]
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
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


def projective_inverse_warp(
    img, depth, pose, intrinsics, ret_flows=False):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
    ret_flows: whether to return the displacements/flows as well
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  num_depths = tf.shape(depth)[0]
  batch = tf.to_int32(tf.shape(img)[0]/num_depths)
  height = tf.shape(img)[1]
  width = tf.shape(img)[2]
  # Construct pixel grid coordinates
  pixel_coords = meshgrid_abs(batch*num_depths, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch * num_depths, 1, 1])
  intrinsics = tf.concat(
      [intrinsics, tf.zeros([batch * num_depths, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  output_img = contrib_resampler.resampler(img, src_pixel_coords)
  if ret_flows:
    return output_img, src_pixel_coords - cam_coords
  else:
    return output_img


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
  transmittance = tf.cumprod(1.0 - alphas + 1.0e-8, axis=3,
                             exclusive=True, reverse=True) * alphas
  output = tf.reduce_sum(transmittance * colors, axis=3)

  return output


def plane_sweep(img, depth_planes, pose, intrinsics):
  """Construct a plane sweep volume.

  Args:
    img: source image [batch, height, width, #channels]
    depth_planes: a list of depth values for each plane
    pose: target to source camera transformation [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    A plane sweep volume [batch, height, width, #planes*#channels]
  """
  num_depths = tf.shape(depth_planes)[0]
  batch = tf.shape(img)[0]
  height = tf.shape(img)[1]
  width = tf.shape(img)[2]
  # use batch dimension for multiple depths
  curr_depths = tf.tile(depth_planes[:, tf.newaxis, tf.newaxis],
                        [batch, height, width])
  img_tile = tf.tile(img, [num_depths, 1, 1, 1])
  pose_tile = tf.tile(pose, [num_depths, 1, 1])
  intrinsics_tile = tf.tile(intrinsics, [num_depths, 1, 1])
  warped_imgs = projective_inverse_warp(img_tile, curr_depths, pose_tile,
                                        intrinsics_tile)
  plane_sweep_volume = tf.reshape(warped_imgs,
                                  [batch, num_depths, height, width, 3])
  plane_sweep_volume = tf.transpose(plane_sweep_volume, [0, 2, 3, 1, 4])

  return plane_sweep_volume


def tgt_coords(depth_planes_tile, pose, intrinsics):
  """Calculate coordinates for MPI voxels in target camera frame.

  Args:
    depth_planes_tile: depth values for each plane
    pose: source to target camera transformation [B, 4, 4]
    intrinsics: camera intrinsics [B, 3, 3]
  Returns:
    coordinates for MPI voxels in target camera frame
  """
  batch = tf.shape(pose)[0]
  num_depths = tf.to_int32(tf.shape(depth_planes_tile)[0]/batch)
  height = tf.shape(depth_planes_tile)[1]
  width = tf.shape(depth_planes_tile)[2]

  pose_tile = tf.tile(pose, [num_depths, 1, 1])
  intrinsics_tile = tf.tile(intrinsics, [num_depths, 1, 1])

  # Construct pixel grid coordinates
  pixel_coords = meshgrid_abs(batch*num_depths, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth_planes_tile, pixel_coords, intrinsics_tile)

  cam_coords = tf.reshape(cam_coords, [batch*num_depths, 4, -1])
  transformed_coords = tf.matmul(pose_tile, cam_coords)
  transformed_coords = tf.reshape(transformed_coords,
                                  [batch, num_depths, 4, height, width])
  transformed_coords = tf.transpose(transformed_coords, [0, 3, 4, 1, 2])
  t_d = transformed_coords[:, :, :, :, 3]
  t_x = transformed_coords[:, :, :, :, 0] / (t_d + 1.0e-10)
  t_y = transformed_coords[:, :, :, :, 1] / (t_d + 1.0e-10)
  t_z = transformed_coords[:, :, :, :, 2] / (t_d + 1.0e-10)
  t_xyz = tf.stack([t_x, t_y, t_z], axis=4)

  return t_xyz


def flow_gather(source_images, flows):
  """Gather from a tensor of images.

  Args:
    source_images: 5D tensor of images [B, H, W, D, 3]
    flows: 5D tensor of x/y offsets to gather for each slice (pixel offsets)
  Returns:
    warped_imgs_reshape: 5D tensor of gathered (warped) images [B, H, W, D, 3]
  """
  batchsize = tf.shape(source_images)[0]
  height = tf.shape(source_images)[1]
  width = tf.shape(source_images)[2]
  num_depths = tf.shape(source_images)[3]
  source_images_reshape = tf.reshape(
      tf.transpose(source_images, [0, 3, 1, 2, 4]),
      [batchsize * num_depths, height, width, 3])
  flows_reshape = tf.reshape(
      tf.transpose(flows, [0, 3, 1, 2, 4]),
      [batchsize * num_depths, height, width, 2])
  _, h, w = tf.meshgrid(
      tf.range(tf.to_float(batchsize * num_depths), dtype=tf.float32),
      tf.range(tf.to_float(height), dtype=tf.float32),
      tf.range(tf.to_float(width), dtype=tf.float32),
      indexing='ij')
  coords_y = tf.clip_by_value(h + flows_reshape[Ellipsis, 0], 0.0,
                              tf.to_float(height))
  coords_x = tf.clip_by_value(w + flows_reshape[Ellipsis, 1], 0.0,
                              tf.to_float(width))
  sampling_coords = tf.stack([coords_x, coords_y], axis=-1)
  warped_imgs = contrib_resampler.resampler(source_images_reshape,
                                            sampling_coords)
  warped_imgs_reshape = tf.transpose(
      tf.reshape(warped_imgs, [batchsize, num_depths, height, width, 3]),
      [0, 2, 3, 1, 4])
  return warped_imgs_reshape
