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

# pylint: skip-file
"""Camera pose and ray generation utility functions."""

import enum
import functools
import types
from typing import Any, List, Mapping, Optional, Text, Tuple, Union

from absl import logging

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from google_research.yobo.internal import configs
from google_research.yobo.internal import math
from google_research.yobo.internal import stepfun
from google_research.yobo.internal import utils


_Array = Union[np.ndarray, jnp.ndarray]

_IDENTIFY_FILE_INDICES_MISSING_FRACTION_ERROR_THRESHOLD = 0.95


def convert_to_ndc(
    origins,
    directions,
    pixtocam,
    near = 1.0,
    xnp = np,
):
  """Converts a set of rays to normalized device coordinates (NDC).

  Args:
    origins: ndarray(float32), [..., 3], world space ray origins.
    directions: ndarray(float32), [..., 3], world space ray directions.
    pixtocam: ndarray(float32), [3, 3], inverse intrinsic matrix.
    near: float, near plane along the negative z axis.
    xnp: either numpy or jax.numpy.

  Returns:
    origins_ndc: ndarray(float32), [..., 3].
    directions_ndc: ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0 (ray direction has a
  negative z-coord); this allows us to share a common NDC space for "forward
  facing" scenes.

  Note that
      projection(origins + t * directions)
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """

  # Shift ray origins to near plane, such that oz = -near.
  # This makes the new near bound equal to 0.
  t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions

  dx, dy, dz = xnp.moveaxis(directions, -1, 0)
  ox, oy, oz = xnp.moveaxis(origins, -1, 0)

  xmult = 1.0 / pixtocam[0, 2]  # Equal to -2. * focal / cx
  ymult = 1.0 / pixtocam[1, 2]  # Equal to -2. * focal / cy

  # Perspective projection into NDC for the t = 0 near points
  #     origins + 0 * directions
  origins_ndc = xnp.stack(
      [xmult * ox / oz, ymult * oy / oz, -xnp.ones_like(oz)], axis=-1
  )

  # Perspective projection into NDC for the t = infinity far points
  #     origins + infinity * directions
  infinity_ndc = xnp.stack(
      [xmult * dx / dz, ymult * dy / dz, xnp.ones_like(oz)], axis=-1
  )

  # directions_ndc points from origins_ndc to infinity_ndc
  directions_ndc = infinity_ndc - origins_ndc

  return origins_ndc, directions_ndc


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses, lock_up = False):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position, lock_up=lock_up)
  return cam2world


def viewmatrix(
    lookdir,
    up,
    position,
    lock_up = False,
):
  """Construct lookat view matrix."""
  orthogonal_dir = lambda a, b: normalize(np.cross(a, b))
  vecs = [None, normalize(up), normalize(lookdir)]
  # x-axis is always the normalized cross product of `lookdir` and `up`.
  vecs[0] = orthogonal_dir(vecs[1], vecs[2])
  # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
  ax = 2 if lock_up else 1
  # Set the not-locked axis to be orthogonal to the other two.
  vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
  m = np.stack(vecs + [position], axis=1)
  return m


def rotation_about_axis(degrees, axis=0):
  """Creates rotation matrix about one of the coordinate axes."""
  radians = degrees / 180.0 * np.pi
  rot2x2 = np.array(
      [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
  )
  r = np.eye(3)
  r[1:3, 1:3] = rot2x2
  r = np.roll(np.roll(r, axis, axis=0), axis, axis=1)
  p = np.eye(4)
  p[:3, :3] = r
  return p


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def focus_point_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


# Constants for generate_spiral_path():
NEAR_STRETCH = 0.9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.0  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = 0.75  # Relative weighting of near, far bounds for render path.


def generate_spiral_path(
    poses,
    bounds,
    n_frames = 120,
    n_rots = 2,
    zrate = 0.5,
):
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of conservative near and far bounds in disparity space.
  near_bound = bounds.min() * NEAR_STRETCH
  far_bound = bounds.max() * FAR_STRETCH
  # All cameras will point towards the world space point (0, 0, -focal).
  focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.0]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = average_pose(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.0]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def transform_poses_pca(poses):
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform


def transform_poses_focus(poses):
  """Transforms poses so that the "focus point" of capture is at the origin.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """

  # Move the focus point to the origin.
  focus_point = focus_point_fn(poses)
  # Use average up vector as the Z axis.
  swap_y_z = np.array([
      [1, 0, 0],
      [0, 0, 1],
      [0, -1, 0.0],
  ])
  rot = average_pose(poses, lock_up=True)[:3, :3] @ swap_y_z
  transform = np.concatenate([rot.T, rot.T @ -focus_point[:, None]], -1)

  poses_recentered = transform @ pad_poses(poses)
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform


def generate_ellipse_path(
    poses,
    n_frames = 120,
    const_speed = True,
    z_variation = 0.0,
    z_phase = 0.0,
    rad_mult_min = 1.0,
    rad_mult_max = 1.0,
    render_rotate_xaxis = 0.0,
    render_rotate_yaxis = 0.0,
    lock_up = False,
):
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    positions = np.stack(
        [
            low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
            low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
            z_variation
            * (
                z_low[2]
                + (z_high - z_low)[2]
                * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
            ),
        ],
        -1,
    )
    # Interpolate between min and max radius multipliers so the camera zooms in
    # and out of the scene center.
    t = np.sin(theta) * 0.5 + 0.5
    rad_mult = rad_mult_min + (rad_mult_max - rad_mult_min) * t
    positions = center + (positions - center) * rad_mult[:, None]
    return positions

  theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  if const_speed:
    # Resample theta angles so that the velocity is closer to constant.
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  poses = np.stack([viewmatrix(p - center, up, p, lock_up) for p in positions])

  poses = poses @ rotation_about_axis(-render_rotate_yaxis, axis=1)
  poses = poses @ rotation_about_axis(render_rotate_xaxis, axis=0)
  return poses


def generate_interpolated_path(
    poses,
    n_interp,
    spline_degree = 5,
    smoothness = 0.03,
    rot_weight = 0.1,
    lock_up = False,
    fixed_up_vector = None,
    lookahead_i = None,
    frames_per_colmap = None,
    const_speed = False,
    n_buffer = None,
    periodic = False,
):
  """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.
    lock_up: if True, forced to use given Up and allow Lookat to vary.
    fixed_up_vector: replace the interpolated `up` with a fixed vector.
    lookahead_i: force the look direction to look at the pose `i` frames ahead.
    frames_per_colmap: conversion factor for the desired average velocity.
    const_speed: renormalize spline to have constant delta between each pose.
    n_buffer: Number of buffer frames to insert at the start and end of the
      path. Helps keep the ends of a spline path straight.
    periodic: make the spline path periodic (perfect loop).

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

  def poses_to_points(poses, dist):
    """Converts from pose matrices to (position, lookat, up) format."""
    pos = poses[:, :3, -1]
    lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
    up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, lookat, up], 1)

  def points_to_poses(points):
    """Converts from (position, lookat, up) format to pose matrices."""
    poses = []
    for i in range(len(points)):
      pos, lookat_point, up_point = points[i]
      if lookahead_i is not None:
        if i + lookahead_i < len(points):
          lookat = pos - points[i + lookahead_i][0]
      else:
        lookat = pos - lookat_point
      up = (up_point - pos) if fixed_up_vector is None else fixed_up_vector
      poses.append(viewmatrix(lookat, up, pos, lock_up=lock_up))
    return np.array(poses)

  def insert_buffer_poses(poses, n_buffer):
    """Insert extra poses at the start and end of the path."""

    def average_distance(points):
      distances = np.linalg.norm(points[1:] - points[0:-1], axis=-1)
      return np.mean(distances)

    def shift(pose, dz):
      result = np.copy(pose)
      z = result[:3, 2]
      z /= np.linalg.norm(z)
      # Move along forward-backward axis. -z is forward.
      result[:3, 3] += z * dz
      return result

    dz = average_distance(poses[:, :3, 3])
    prefix = np.stack([shift(poses[0], (i + 1) * dz) for i in range(n_buffer)])
    prefix = prefix[::-1]  # reverse order
    suffix = np.stack(
        [shift(poses[-1], -(i + 1) * dz) for i in range(n_buffer)]
    )
    result = np.concatenate([prefix, poses, suffix])
    return result

  def remove_buffer_poses(poses, u, n_frames, u_keyframes, n_buffer):
    u_keyframes = u_keyframes[n_buffer:-n_buffer]
    mask = (u >= u_keyframes[0]) & (u <= u_keyframes[-1])
    poses = poses[mask]
    u = u[mask]
    n_frames = len(poses)
    return poses, u, n_frames, u_keyframes

  def interp(points, u, k, s):
    """Runs multidimensional B-spline interpolation on the input points."""
    sh = points.shape
    pts = np.reshape(points, (sh[0], -1))
    k = min(k, sh[0] - 1)
    tck, u_keyframes = scipy.interpolate.splprep(pts.T, k=k, s=s, per=periodic)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (len(u), sh[1], sh[2]))
    return new_points, u_keyframes

  if n_buffer is not None:
    poses = insert_buffer_poses(poses, n_buffer)
  points = poses_to_points(poses, dist=rot_weight)
  n_frames = n_interp * (points.shape[0] - 1)
  u = np.linspace(0, 1, n_frames, endpoint=True)
  new_points, u_keyframes = interp(points, u=u, k=spline_degree, s=smoothness)
  poses = points_to_poses(new_points)
  if n_buffer is not None:
    poses, u, n_frames, u_keyframes = remove_buffer_poses(
        poses, u, n_frames, u_keyframes, n_buffer
    )

  if frames_per_colmap is not None:
    # Recalculate the number of frames to achieve desired average velocity.
    positions = poses[:, :3, -1]
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    total_length_colmap = lengths.sum()
    print('old n_frames:', n_frames)
    print('total_length_colmap:', total_length_colmap)
    n_frames = int(total_length_colmap * frames_per_colmap)
    print('new n_frames:', n_frames)
    u = np.linspace(
        np.min(u_keyframes), np.max(u_keyframes), n_frames, endpoint=True
    )
    new_points, _ = interp(points, u=u, k=spline_degree, s=smoothness)
    poses = points_to_poses(new_points)

  if const_speed:
    # Resample timesteps so that the velocity is nearly constant.
    positions = poses[:, :3, -1]
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    u = stepfun.sample(None, u, np.log(lengths), n_frames + 1)
    new_points, _ = interp(points, u=u, k=spline_degree, s=smoothness)
    poses = points_to_poses(new_points)

  return poses[:-1], u[:-1], u_keyframes


def safe_interpolate_1d(
    x,
    spline_degree,
    smoothness,
    t_input,
    t_output,
):
  """Interpolate 1d signal x (defined at t_input and queried at t_output)."""
  # One needs at least n=k+1 points to fit a polynomial of degree k to n points.
  n = len(x)
  spline_degree = min(spline_degree, n - 1)

  if spline_degree > 0:
    tck = scipy.interpolate.splrep(t_input, x, s=smoothness, k=spline_degree)
    return scipy.interpolate.splev(t_output, tck).astype(x.dtype)
  else:  # n = 0 or 1
    fill_value = x[0] if n else 0.0
    return np.full(t_output.shape, fill_value, dtype=x.dtype)


def identify_file_names(dir_or_text_file):
  """Load filenames from text file or directory."""
  if utils.isdir(dir_or_text_file):
    # If `dir_or_text_file` is a directory, grab the filenames.
    subset_names = sorted(utils.listdir(dir_or_text_file))
  else:
    # If `dir_or_text_file` is a text file, treat each line as a filename.
    with utils.open_file(dir_or_text_file, 'r') as fp:
      # Decode bytes into string and split into lines.
      subset_names = fp.read().decode('utf-8').splitlines()
  return subset_names


def identify_file_indices(
    dir_or_text_file, file_names
):
  """Computes indices for a subset of files out of a larger list."""
  # Load file names.
  subset_names = identify_file_names(dir_or_text_file)

  # COLMAP sometimes doesn't reconstruct all images, which results in some files
  # being missing.
  if not set(subset_names).issubset(file_names):
    subset_names_missing_from_file_names = set(subset_names) - set(file_names)
    logging.warning(
        'Some files from subset are missing in the file names:\n%s',
        ' '.join(str(x) for x in subset_names_missing_from_file_names),
    )
    missing_subset_names_threshold = len(
        subset_names_missing_from_file_names
    ) / len(subset_names)
    if (
        missing_subset_names_threshold
        > _IDENTIFY_FILE_INDICES_MISSING_FRACTION_ERROR_THRESHOLD
    ):
      raise ValueError(
          f'{missing_subset_names_threshold*100}% of subset files is missing'
          f' from file_names: {subset_names_missing_from_file_names}'
      )

  file_names_set = set(file_names)

  # Get indices corresponding to the subset filenames. Ensure that the order
  # used in subset_names is preserved.
  indices = [file_names.index(n) for n in subset_names if n in file_names_set]
  indices = np.array(indices)

  return indices


def get_meters_per_colmap_from_calibration_images(
    config, poses, image_names
):
  """Uses calibration images to get how many meters is a single COLMAP unit."""
  # By default, the input camera poses are scaled to fit in the [-1, 1]^3 cube.
  # This default value implies a scaling of 2 / .25 = 8 meters between the
  # farthest apart camera poses.
  meters_per_colmap = 8.0
  if config.render_calibration_keyframes is not None:
    # Use provided calibration keyframes to determine metric world scale.
    calib_names = identify_file_names(config.render_calibration_keyframes)
    indices = []
    for i in range(0, len(calib_names), 2):
      # Grab pairs of calibration images filenames.
      name0, name1 = calib_names[i : i + 2]
      # Check if both are in the set of colmap-posed images.
      if name0 in image_names and name1 in image_names:
        indices.append((image_names.index(name0), image_names.index(name1)))
    if indices:
      # Extract colmap-space positions from the camera pose matrices.
      positions = poses[indices][Ellipsis, :3, -1]
      # Every pair of calibration keyframes should have world space distance
      # `render_calibration_distance` according to the capture handbook.
      colmap_lengths = np.linalg.norm(
          positions[:, 0] - positions[:, 1], axis=-1
      )
      colmap_length = colmap_lengths.mean(axis=0)
      # Ratio of world distance to colmap distance.
      meters_per_colmap = config.render_calibration_distance / colmap_length
      print('colmap lengths', colmap_lengths)
      print('avg', colmap_length)
      print('meters_per_colmap', meters_per_colmap)
  return meters_per_colmap


def calibrate_spline_speed(
    config, poses, image_names
):
  """Uses input config to determine a conversion factor for the spline speed."""

  if config.render_spline_meters_per_sec is None:
    return None

  meters_per_colmap = get_meters_per_colmap_from_calibration_images(
      config, poses, image_names
  )

  meters_per_sec = config.render_spline_meters_per_sec
  frames_per_sec = config.render_video_fps
  frames_per_colmap = meters_per_colmap / meters_per_sec * frames_per_sec
  print('returning frames_per_colmap', frames_per_colmap)

  return frames_per_colmap


def create_render_spline_path(
    config,
    image_names,
    poses,
    exposures,
):
  """Creates spline interpolation render path from subset of dataset poses.

  Args:
    config: configs.Config object.
    image_names: a list of image filenames.
    poses: [N, 3, 4] array of extrinsic camera pose matrices.
    exposures: optional list of floating point exposure values.

  Returns:
    spline_indices: list of indices used to select spline keyframe poses.
    render_poses: array of interpolated extrinsic camera poses for the path.
    render_exposures: optional list of interpolated exposures for the path.
  """

  def remove_outlier_spline_indices(
      spline_indices, poses, q_max, q_mult
  ):
    """Identify spline indices correspond to inlier poses."""
    poses = poses[spline_indices]
    points = poses[:, :3, -1]
    distances = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    mask = distances < q_mult * np.quantile(distances, q_max)
    mask = np.concatenate([mask, [True]], axis=0)  # Keep the last pose.

    num_inliers = int(np.sum(mask))
    num_total = len(spline_indices)
    print(
        f'remove_outlier_spline_indices: {num_inliers}/{num_total} spline '
        'path poses remaining after outlier removal.'
    )

    return spline_indices[mask]

  # Grab poses corresponding to the image filenames.
  spline_indices = identify_file_indices(
      config.render_spline_keyframes, image_names
  )

  if (
      config.render_spline_outlier_keyframe_quantile is not None
      and config.render_spline_outlier_keyframe_multiplier is not None
  ):
    spline_indices = remove_outlier_spline_indices(
        spline_indices,
        poses,
        q_max=config.render_spline_outlier_keyframe_quantile,
        q_mult=config.render_spline_outlier_keyframe_multiplier,
    )

  keyframes = poses[spline_indices]

  frames_per_colmap = calibrate_spline_speed(config, poses, image_names)

  if config.render_spline_fixed_up:
    # Fix path to use world-space "up" vector instead of "banking" with spline.
    all_up_vectors = poses[:, :3, 1]  # second column of pose matrix is up.
    fixed_up_vector = normalize(all_up_vectors.mean(axis=0))
  else:
    fixed_up_vector = None
  render_poses, frame_timesteps, keyframe_timesteps = (
      generate_interpolated_path(
          keyframes,
          n_interp=config.render_spline_n_interp,
          spline_degree=config.render_spline_degree,
          smoothness=config.render_spline_smoothness,
          rot_weight=config.render_spline_rot_weight,
          lock_up=config.render_spline_lock_up,
          fixed_up_vector=fixed_up_vector,
          lookahead_i=config.render_spline_lookahead_i,
          frames_per_colmap=frames_per_colmap,
          const_speed=config.render_spline_const_speed,
          n_buffer=config.render_spline_n_buffer,
      )
  )
  if config.render_spline_interpolate_exposure:
    if exposures is None:
      raise ValueError(
          'config.render_spline_interpolate_exposure is True but '
          'create_render_spline_path() was passed exposures=None.'
      )
    # Interpolate per-frame exposure value.
    log_exposure = np.log(exposures[spline_indices])
    # Use aggressive smoothing for exposure interpolation to avoid flickering.
    log_exposure_interp = safe_interpolate_1d(
        log_exposure,
        spline_degree=5,
        smoothness=config.render_spline_interpolate_exposure_smoothness,
        t_input=keyframe_timesteps,
        t_output=frame_timesteps,
    )
    render_exposures = np.exp(log_exposure_interp)
  else:
    render_exposures = None
  return spline_indices, render_poses, render_exposures


def intrinsic_matrix(
    fx, fy, cx, cy, xnp = np
):
  """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
  return xnp.array([
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1.0],
  ])


def get_pixtocam(
    focal, width, height, xnp = np
):
  """Inverse intrinsic matrix for a perfect pinhole camera."""
  camtopix = intrinsic_matrix(focal, focal, width * 0.5, height * 0.5, xnp)
  return xnp.linalg.inv(camtopix)


def pixel_coordinates(
    width, height, xnp = np
):
  """Tuple of the x and y integer coordinates for a grid of pixels."""
  return xnp.meshgrid(xnp.arange(width), xnp.arange(height), indexing='xy')


def _radial_and_tangential_distort(
    x,
    y,
    k1 = 0,
    k2 = 0,
    k3 = 0,
    k4 = 0,
    p1 = 0,
    p2 = 0,
):
  """Computes the distorted pixel positions."""
  r2 = x * x + y * y

  radial_distortion = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
  dx_radial = x * radial_distortion
  dy_radial = y * radial_distortion

  dx_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
  dy_tangential = 2 * p2 * x * y + p1 * (r2 + 2 * y * y)

  return x + dx_radial + dx_tangential, y + dy_radial + dy_tangential


def _compute_residual_and_jacobian(
    x,
    y,
    xd,
    yd,
    k1 = 0.0,
    k2 = 0.0,
    k3 = 0.0,
    k4 = 0.0,
    p1 = 0.0,
    p2 = 0.0,
):
  """Auxiliary function of radial_and_tangential_undistort()."""
  # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
  #                   k4 * r(x, y)^4;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd,
    yd,
    k1 = 0,
    k2 = 0,
    k3 = 0,
    k4 = 0,
    p1 = 0,
    p2 = 0,
    eps = 1e-9,
    max_iterations=10,
    xnp = np,
):
  """Computes undistorted (x, y) from (xd, yd)."""
  # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
  # Initialize from the distorted point.
  x = xnp.copy(xd)
  y = xnp.copy(yd)

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2
    )
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = xnp.where(
        xnp.abs(denominator) > eps,
        x_numerator / denominator,
        xnp.zeros_like(denominator),
    )
    step_y = xnp.where(
        xnp.abs(denominator) > eps,
        y_numerator / denominator,
        xnp.zeros_like(denominator),
    )

    x = x + step_x
    y = y + step_y

  return x, y


class ProjectionType(enum.Enum):
  """Camera projection type (perspective pinhole, fisheye, or 360 pano)."""

  PERSPECTIVE = 'perspective'
  FISHEYE = 'fisheye'
  FISHEYE_EQUISOLID = 'fisheye_equisolid'
  PANORAMIC = 'pano'


def pixels_to_rays(
    pix_x_int,
    pix_y_int,
    pixtocams,
    camtoworlds,
    distortion_params = None,
    pixtocam_ndc = None,
    camtype = ProjectionType.PERSPECTIVE,
    rng = None,
    jitter = 0,
    jitter_scale = 1.0,
    xnp = np,
):
  """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

  Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
  inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
  distortion coefficients distortion_params and NDC space projection matrix
  pixtocam_ndc), computes the corresponding 3D camera rays.

  Vectorized over the leading dimensions of the first four arguments.

  Args:
    pix_x_int: int array, shape SH, x coordinates of image pixels.
    pix_y_int: int array, shape SH, y coordinates of image pixels.
    pixtocams: float array, broadcastable to SH + [3, 3], inverse intrinsics.
    camtoworlds: float array, broadcastable to SH + [3, 4], camera extrinsics.
    distortion_params: dict of floats, optional camera distortion parameters.
    pixtocam_ndc: float array, [3, 3], optional inverse intrinsics for NDC.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    xnp: either numpy or jax.numpy.

  Returns:
    origins: float array, shape SH + [3], ray origin points.
    directions: float array, shape SH + [3], ray direction vectors.
    viewdirs: float array, shape SH + [3], normalized ray direction vectors.
    radii: float array, shape SH + [1], ray differential radii.
    imageplane: float array, shape SH + [2], xy coordinates on the image plane.
      If the image plane is at world space distance 1 from the pinhole, then
      imageplane will be the xy coordinates of a pixel in that space (so the
      camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
  """
  if rng is None and jitter > 0:
    rng = jax.random.PRNGKey(0)

  # Must add half pixel offset to shoot rays through pixel centers.
  def pix_to_dir(x, y):
    return xnp.stack([x + 0.5, y + 0.5, xnp.ones_like(x)], axis=-1)


  if jitter > 0:
    key, rng = jax.random.split(rng)
    k1, k2 = jax.random.split(key)

    if jitter == 1:
      dx = (jax.random.uniform(k1, shape=pix_x_int.shape) - 0.5)
      dy = (jax.random.uniform(k2, shape=pix_y_int.shape) - 0.5)
    else:
      dx = jax.random.normal(k1, shape=pix_x_int.shape) * 0.5
      dy = jax.random.normal(k2, shape=pix_y_int.shape) * 0.5

    if jitter_scale > 1.0:
      k1, k2 = jax.random.split(k1)
      dx += (jax.random.uniform(k1, shape=pix_x_int.shape) - 0.5)
      dy += (jax.random.uniform(k2, shape=pix_y_int.shape) - 0.5)
  else:
    dx = 0.0
    dy = 0.0

  # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
  pixel_dirs_stacked = xnp.stack(
      [
          pix_to_dir(pix_x_int + dx, pix_y_int + dy),
          pix_to_dir(pix_x_int + 1 + dx, pix_y_int + dy),
          pix_to_dir(pix_x_int + dx, pix_y_int + 1 + dy),
      ],
      axis=0,
  )

  # For jax, need to specify high-precision matmul.
  matmul = math.matmul if xnp == jnp else xnp.matmul
  mat_vec_mul = lambda A, b: matmul(A, b[Ellipsis, None])[Ellipsis, 0]

  # Apply inverse intrinsic matrices.
  camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

  if distortion_params is not None:
    # Correct for distortion.
    x, y = _radial_and_tangential_undistort(
        camera_dirs_stacked[Ellipsis, 0],
        camera_dirs_stacked[Ellipsis, 1],
        **distortion_params,
        xnp=xnp,
    )
    camera_dirs_stacked = xnp.stack([x, y, xnp.ones_like(x)], -1)

  if camtype in [ProjectionType.FISHEYE, ProjectionType.FISHEYE_EQUISOLID]:
    # See Wikipedia:
    # https://en.wikipedia.org/wiki/Fisheye_lens#Examples_and_specific_models
    # r is image plane radius divided by focal length.
    r = xnp.sqrt(
        xnp.sum(xnp.square(camera_dirs_stacked[Ellipsis, :2]), axis=-1)
    )
    if camtype == ProjectionType.FISHEYE:
      # Equidistant.
      theta = r
      theta = xnp.minimum(xnp.pi, theta)
    else:
      # Equisolid.
      theta = 2. * xnp.arcsin(r / 2.)
    sin_theta_over_r = xnp.sin(theta) / r
    camera_dirs_stacked = xnp.stack(
        [
            camera_dirs_stacked[Ellipsis, 0] * sin_theta_over_r,
            camera_dirs_stacked[Ellipsis, 1] * sin_theta_over_r,
            xnp.cos(theta),
        ],
        axis=-1,
    )

  elif camtype == ProjectionType.PANORAMIC:
    theta = camera_dirs_stacked[Ellipsis, 0]
    phi = camera_dirs_stacked[Ellipsis, 1]
    # Negation on y and z components accounts for expected OpenCV convention.
    camera_dirs_stacked = xnp.stack(
        [
            -xnp.sin(phi) * xnp.sin(theta),
            -xnp.cos(phi),
            -xnp.sin(phi) * xnp.cos(theta),
        ],
        axis=-1,
    )

  # Flip from OpenCV to OpenGL coordinate system.
  camera_dirs_stacked = matmul(
      camera_dirs_stacked, xnp.diag(xnp.array([1.0, -1.0, -1.0]))
  )

  # Extract 2D image plane (x, y) coordinates.
  imageplane = camera_dirs_stacked[0, Ellipsis, :2]

  # Apply camera rotation matrices.
  directions_stacked = mat_vec_mul(
      camtoworlds[Ellipsis, :3, :3], camera_dirs_stacked
  )
  # Extract the offset rays.
  directions, dx, dy = directions_stacked

  origins = xnp.broadcast_to(camtoworlds[Ellipsis, :3, -1], directions.shape)
  viewdirs = directions / xnp.linalg.norm(directions, axis=-1, keepdims=True)

  if pixtocam_ndc is None:
    # Distance from each unit-norm direction vector to its neighbors.
    dx_norm = xnp.linalg.norm(dx - directions, axis=-1)
    dy_norm = xnp.linalg.norm(dy - directions, axis=-1)

  else:
    # Convert ray origins and directions into projective NDC space.
    ndc_fn = functools.partial(convert_to_ndc, pixtocam=pixtocam_ndc, xnp=xnp)
    origins_dx, _ = ndc_fn(origins, dx)
    origins_dy, _ = ndc_fn(origins, dy)
    origins, directions = ndc_fn(origins, directions)

    # In NDC space, we use the offset between origins instead of directions.
    dx_norm = xnp.linalg.norm(origins_dx - origins, axis=-1)
    dy_norm = xnp.linalg.norm(origins_dy - origins, axis=-1)

  # Cut the distance in half, multiply it to match the variance of a uniform
  # distribution the size of a pixel (1/12, see paper).
  radii = (0.5 * (dx_norm + dy_norm))[Ellipsis, None] * 2 / xnp.sqrt(12)

  return origins, directions, viewdirs, radii, imageplane


def points_to_pixels(
    points,
    pixtocams,
    camtoworlds,
    distortion_params = None,
    camtype = ProjectionType.PERSPECTIVE,
    xnp = np,
):
  """Calculates pixel coordinates given 3D points, intrinisics, and extrinsics.

  Given 3D point coordinates points and cameras with inverse intrinsics
  pixtocams and extrinsics camtoworlds (and optional distortion coefficients
  distortion_params), computes the corresponding 2D pixel coordinates.

  Vectorized over the leading dimensions of the first four arguments.

  Args:
    points: float array, [..., 3], 3D coordinates of points to project.
    pixtocams: float array, [..., 3, 3], inverse intrinsics.
    camtoworlds: float array, [..., 3, 4], camera extrinsics.
    distortion_params: dict of floats or float arrays [...], optional camera
      distortion parameters.
    camtype: camera_utils.ProjectionType, type of camera model.
    xnp: either numpy (host compute) or jax.numpy (device compute).

  Returns:
    coordinates: float array, [..., 2], pixel coordinates.
    depth: float array, [...], per-point orthographic depth.
  """

  if camtype != ProjectionType.PERSPECTIVE:
    raise ValueError(f'points_to_pixels only supports perspective projection, '
                     f'not {camtype} mode.')

  # For jax, need to specify high-precision matmul.
  matmul = math.matmul if xnp == jnp else xnp.matmul
  mat_vec_mul = lambda A, b: matmul(A, b[Ellipsis, None])[Ellipsis, 0]

  rotation = camtoworlds[Ellipsis, :3, :3]
  rotation_inv = xnp.swapaxes(rotation, -1, -2)
  translation = camtoworlds[Ellipsis, :3, -1]
  # Points (directions) in the camera coordinate frame.
  points_camera = mat_vec_mul(rotation_inv, points - translation)

  # Projection to image plane by dividing out -z.
  depth = -points_camera[Ellipsis, -1]
  camera_dirs = points_camera / depth[Ellipsis, None]

  # OpenGL to OpenCV coordinates.
  camera_dirs = matmul(camera_dirs, xnp.diag(xnp.array([1.0, -1.0, -1.0])))

  if distortion_params is not None:
    # Correct for distortion.
    x, y = _radial_and_tangential_distort(
        camera_dirs[Ellipsis, 0],
        camera_dirs[Ellipsis, 1],
        **distortion_params,
    )
    camera_dirs = xnp.stack([x, y, xnp.ones_like(x)], -1)

  # Apply intrinsics matrix.
  pixel_dirs = mat_vec_mul(xnp.linalg.inv(pixtocams), camera_dirs)

  # Remove half pixel offset.
  coordinates = pixel_dirs[Ellipsis, :2] - xnp.array([0.5, 0.5])

  return coordinates, depth


def rays_planes_intersection(
    z_min, z_max, origins, directions, xnp = np
):
  """Crops rays to a range of z values.

  This is useful for situations where the scene lies within a range of
  altitudes, but the cameras are very far away, as with aerial data.

  Args:
    z_min: float z value of the lower cropping plane.
    z_max: float z value of the upper cropping plane.
    origins: ray origins points.
    directions: ray direction vectors.
    xnp: either numpy or jax.numpy.

  Returns:
    t_min: parametric location of the cropped ray origins
    t_max: parametric location of the ends of the cropped rays
  """
  t1 = (z_min - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  t2 = (z_max - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  t_min = xnp.minimum(t1, t2)
  t_max = xnp.maximum(t1, t2)
  return t_min, t_max


def ray_box_intersection(ray_o, ray_d, corners, xnp=np):
  """Returns enter/exit distances along the ray for box defined by `corners`."""
  t1 = (corners[0] - ray_o) / ray_d
  t2 = (corners[1] - ray_o) / ray_d
  t_min = xnp.minimum(t1, t2).max(axis=-1)
  t_max = xnp.maximum(t1, t2).min(axis=-1)
  return t_min, t_max


def modify_rays_with_bbox(rays, corners, xnp=np):
  """Sets near/far by bbox intersection and multiplies lossmult by mask."""
  lossmult = rays.lossmult
  near = rays.near
  far = rays.far

  t_min, t_max = ray_box_intersection(
      rays.origins, rays.directions, corners, xnp=xnp
  )
  valid = (t_min <= t_max)[Ellipsis, None]
  if lossmult is None:
    lossmult = valid.astype(xnp.float32)
  else:
    lossmult = xnp.where(valid, lossmult, 0.0)
  near = xnp.where(valid, t_min[Ellipsis, None], near)
  far = xnp.where(valid, t_max[Ellipsis, None], far)

  return rays.replace(lossmult=lossmult, near=near, far=far)


def gather_cameras(cameras, cam_idx, xnp=np):
  """Gathers relevant camera parameters for each ray."""
  pixtocams, camtoworlds, distortion_params = cameras[:3]

  if pixtocams.ndim > 2:
    pixtocams_idx = pixtocams[cam_idx]
  else:
    pixtocams_idx = pixtocams

  if camtoworlds.ndim > 2:
    camtoworlds_idx = camtoworlds[cam_idx]
  else:
    camtoworlds_idx = camtoworlds

  if distortion_params is not None:
    distortion_params_idx = {}
    for k, v in distortion_params.items():  # pytype: disable=attribute-error  # jax-ndarray
      if not xnp.isscalar(v):
        distortion_params_idx[k] = v[cam_idx]
      else:
        distortion_params_idx[k] = v
  else:
    distortion_params_idx = None

  return (
      pixtocams_idx,
      camtoworlds_idx,
      distortion_params_idx,
  )


def cast_ray_batch(
    cameras,
    pixels,
    camtype = ProjectionType.PERSPECTIVE,
    rng = None,
    jitter = 0,
    jitter_scale = 1.0,
    scene_bbox = None,
    xnp = np,
):
  """Maps from input cameras and Pixel batch to output Ray batch.

  `cameras` is a Tuple of five sets of camera parameters.
    pixtocams: 1 or N stacked [3, 3] inverse intrinsic matrices.
    camtoworlds: 1 or N stacked [3, 4] extrinsic pose matrices.
    distortion_params: optional, dict[str, float] containing pinhole model
      distortion parameters.
    pixtocam_ndc: optional, [3, 3] inverse intrinsic matrix for mapping to NDC.
    z_range: optional range of Z values

  Args:
    cameras: described above.
    pixels: integer pixel coordinates and camera indices, plus ray metadata.
      These fields can be an arbitrary batch shape.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    scene_bbox: min and max corner of scene bounding box, if applicable.
    xnp: either numpy or jax.numpy.

  Returns:
    rays: Rays dataclass with computed 3D world space ray data.
  """

  # pixels.cam_idx has shape [..., 1], remove this hanging dimension.
  cam_idx = pixels.cam_idx[Ellipsis, 0]
  cameras_idx = gather_cameras(cameras, cam_idx, xnp=xnp)
  pixtocams, camtoworlds, distortion_params = cameras_idx
  pixtocam_ndc, z_range = cameras[3:5]

  # Compute rays from pixel coordinates.
  origins, directions, viewdirs, radii, imageplane = pixels_to_rays(
      pixels.pix_x_int,
      pixels.pix_y_int,
      pixtocams,
      camtoworlds,
      distortion_params=distortion_params,
      pixtocam_ndc=pixtocam_ndc,
      camtype=camtype,
      rng=rng,
      jitter=jitter,
      jitter_scale=jitter_scale,
      xnp=xnp,
  )

  if z_range is not None:
    t_min, t_max = rays_planes_intersection(
        z_range[0], z_range[1], origins, directions, xnp
    )

    t_min = xnp.broadcast_to(t_min[Ellipsis, None], origins.shape)
    t_max = xnp.broadcast_to(t_max[Ellipsis, None], origins.shape)
    hit_mask = t_max < t_min

    origins = xnp.where(hit_mask, origins, origins + directions * t_min)
    directions = xnp.where(hit_mask, directions, directions * (t_max - t_min))

  rays = utils.Rays(
      origins=origins,
      directions=directions,
      viewdirs=viewdirs,
      radii=radii,
      imageplane=imageplane,
      lossmult=pixels.lossmult,
      near=pixels.near,
      far=pixels.far,
      cam_idx=pixels.cam_idx,
      exposure_idx=pixels.exposure_idx,
      exposure_values=pixels.exposure_values,
      pix_x_int=pixels.pix_x_int,
      pix_y_int=pixels.pix_y_int,
  )

  if scene_bbox is not None:
    rays = modify_rays_with_bbox(rays, scene_bbox, xnp=xnp)

  return rays


def cast_general_rays(
    camtoworld,
    pixtocam,
    height,
    width,
    near,
    far,
    distortion_params = None,
    pixtocam_ndc = None,
    camtype = ProjectionType.PERSPECTIVE,
    rng = None,
    jitter = 0,
    jitter_scale = 1.0,
    cam_idx = 0,
    xnp = np,
):
  """Wrapper for generating a general ray batch."""

  pix_x_int, pix_y_int = pixel_coordinates(width, height, xnp=xnp)

  ray_args = pixels_to_rays(
      pix_x_int,
      pix_y_int,
      pixtocam,
      camtoworld,
      distortion_params=distortion_params,
      pixtocam_ndc=pixtocam_ndc,
      camtype=camtype,
      rng=rng,
      jitter=jitter,
      jitter_scale=jitter_scale,
      xnp=xnp,
  )

  broadcast_scalar = lambda x: xnp.broadcast_to(x, pix_x_int.shape)[Ellipsis, None]
  ray_kwargs = {
      'lossmult': broadcast_scalar(1.0),
      'near': broadcast_scalar(near),
      'far': broadcast_scalar(far),
      'cam_idx': broadcast_scalar(1) * cam_idx,
  }

  return utils.Rays(
      *ray_args,
      **ray_kwargs,
      pix_x_int=pix_x_int,
      pix_y_int=pix_y_int,
  )


def cast_pinhole_rays(
    camtoworld,
    height,
    width,
    focal,
    near,
    far,
    rng = None,
    jitter = 0,
    jitter_scale = 1.0,
    xnp = np,
):
  """Generates a pinhole camera ray batch (w/o distortion)."""

  return cast_general_rays(
      camtoworld,
      get_pixtocam(focal, width, height, xnp=xnp),
      height,
      width,
      near,
      far,
      camtype=ProjectionType.PERSPECTIVE,
      rng=rng,
      jitter=jitter,
      jitter_scale=jitter_scale,
      xnp=xnp,
  )


def cast_spherical_rays(
    camtoworld,
    height,
    width,
    near,
    far,
    rng = None,
    jitter = 0,
    jitter_scale = 1.0,
    xnp = np,
):
  """Generates a spherical camera ray batch."""

  return cast_general_rays(
      camtoworld,
      xnp.diag(xnp.array([2.0 * np.pi / width, np.pi / height, 1.0])),
      height,
      width,
      near,
      far,
      camtype=ProjectionType.PANORAMIC,
      rng=rng,
      jitter=jitter,
      jitter_scale=jitter_scale,
      xnp=xnp,
  )


def gather_features(points, feature_images, feature_cameras):
  """Projects 3D points to gather features from N input images."""
  # Input shapes should be [..., 3] for points, [N, H, W, C] for feature_images,
  # and feature_cameras is a tuple of N stacked pixtocams, camtoworlds, and
  # distortion params.
  # Pad all the camera params so they broadcast across the 3D points.
  pad_dims = (1,) * len(points.shape[:-1])
  feature_cameras = jax.tree_util.tree_map(
      lambda x: x.reshape((-1,) + pad_dims + x.shape[1:]), feature_cameras
  )
  # Project points into each camera.
  coordinates, _ = points_to_pixels(points, *feature_cameras, xnp=jnp)
  # Sample the features at the projected pixel coordinates.
  # resample_2d expects at least 3 dimensions, so we add one extra in case the
  # input points array only has 2 dims.
  f = jax_resample.resample_2d(feature_images, coordinates[:, None])[:, 0]
  # Return should have shape (N,) + points.shape[:-1] + (C,).
  return f


def plane_sweep_volume(
    images,
    cameras,
    target_camera,
    height,
    width,
    near,
    far,
    n_planes=120,
    n_views=4,
    lindisp=True,
    xnp=jnp,
):
  """Creates a plane sweep volume given input images/poses and a target pose.

  Cameras are specified as stacked pytrees of (pixtocam, camtoworld, distortion)
  parameters.

  Args:
    images: stack of input images.
    cameras: tuple of corresponding camera params.
    target_camera: tuple of target camera params (frame of the PSV).
    height: output PSV height.
    width: output PSV width.
    near: near plane for sampling planes in PSV.
    far: far plane for sampling planes in PSV.
    n_planes: number of planes (in depth) in PSV.
    n_views: number of nearest views to sample.
    lindisp: if True, sample planes linearly in disparity.
    xnp: np or jnp module (note resampling function is currently always jnp).

  Returns:
    A plane sweep volume with shape (n_views, height, width, n_planes, C) where
    C = images.shape[-1].
  """

  def get_nearest_pose(poses, pose):
    # Get nearest pose, measured by translation.
    t = pose[:3, -1]
    ts = poses[:, :3, -1]
    d = xnp.linalg.norm(ts - t, axis=-1)
    return xnp.argsort(d)

  # Cast rays for target camera.
  t_p2c, t_c2w, t_dist = target_camera
  rays = cast_general_rays(
      t_c2w, t_p2c, height, width, near, far, t_dist, xnp=xnp
  )
  # Sample n_planes 3D points along rays.
  if lindisp:
    t = 1.0 / xnp.linspace(1.0 / near, 1.0 / far, n_planes)
  else:
    t = xnp.linspace(near, far, n_planes)
  points = rays.origins + rays.directions * t[:, None, None, None]

  # Get n_views nearest target cameras/images.
  inds = get_nearest_pose(cameras[1], t_c2w)[:n_views]
  images, cameras = jax.tree_util.tree_map(lambda x: x[inds], (images, cameras))

  # Resample images to get PSV.
  out = gather_features(points, images, cameras)
  out = xnp.array(out)
  return out


def jax_camera_from_tuple(
    camera_tuple,
    image_size,
    projection_type,
):
  """Converts a camera tuple into a JAX camera.

  Args:
    camera_tuple: A tuple containing `inv_intrinsics`, the inverse intrinsics
      matrix; `extrinsics`, the camera to world matrix; and `distortion_params`,
      the dictionary of distortion parameters.
    image_size: An array containing the (width, height) image size.
    projection_type: The projection type of the camera.

  Returns:
    A JAX camera class instance encoding the same camera information.
  """
  if projection_type.value not in {
      ProjectionType.PERSPECTIVE.value,
      ProjectionType.FISHEYE.value,
      ProjectionType.FISHEYE_EQUISOLID.value,
  }:
    raise ValueError(f'Projection {projection_type} is not supported.')

  inv_intrinsics, extrinsic, distortion_params = camera_tuple[:3]
  intrinsics = jnp.linalg.inv(inv_intrinsics)
  focal_length = intrinsics[0, 0]
  principal_point = intrinsics[:2, 2]
  pixel_aspect_ratio = intrinsics[1, 1] / intrinsics[0, 0]

  radial_distortion = None
  tangential_distortion = None
  if distortion_params is not None:
    if (
        'k1' in distortion_params
        and 'k2' in distortion_params
        and 'k3' in distortion_params
    ):
      radial_keys = ['k1', 'k2', 'k3', 'k4']
      radial_distortion = jnp.array(
          [distortion_params[k] for k in radial_keys if k in distortion_params]
      )
    if 'p1' in distortion_params and 'p2' in distortion_params:
      tangential_distortion = jnp.array([
          distortion_params['p1'],
          distortion_params['p2'],
      ])

  extrinsic = jnp.concatenate(
      [extrinsic[:3, :4], jnp.array([[0, 0, 0, 1]])], axis=0
  )
  # Convert to OpenCV coordinates.
  extrinsic = math.matmul(extrinsic, jnp.diag(jnp.array([1, -1, -1, 1])))
  world_to_cam = jnp.linalg.inv(extrinsic)
  camera = camera_lib.create(
      focal_length=focal_length,
      pixel_aspect_ratio=pixel_aspect_ratio,
      radial_distortion=radial_distortion,
      tangential_distortion=tangential_distortion,
      principal_point=principal_point,
      image_size=image_size,
      is_fisheye=(projection_type.value == ProjectionType.FISHEYE.value),
  )
  camera = camera_lib.update_world_to_camera_matrix(camera, world_to_cam)
  return camera


def tuple_from_jax_camera(
    jax_camera,
):
  """Converts a JAX camera into a camera tuple."""
  focal_x = jax_camera.focal_length
  focal_y = jax_camera.focal_length * jax_camera.pixel_aspect_ratio
  intrinsic = jnp.block([
      [focal_x, jax_camera.skew, jax_camera.principal_point[0]],
      [0, focal_y, jax_camera.principal_point[1]],
      [0, 0, 1],
  ])
  pix_to_cam = jnp.linalg.inv(intrinsic)
  world_to_cam = camera_lib.world_to_camera_matrix(jax_camera)
  cam_to_world = jnp.linalg.inv(world_to_cam)
  # Convert back to OpenGL coordinates.
  cam_to_world = math.matmul(cam_to_world, jnp.diag(jnp.array([1, -1, -1, 1])))
  cam_to_world = cam_to_world[:3, :]
  distortion_params = None
  if jax_camera.has_distortion:
    distortion_params = {}
    if jax_camera.has_radial_distortion:
      distortion_params.update({
          'k1': jax_camera.radial_distortion[0],
          'k2': jax_camera.radial_distortion[1],
          'k3': jax_camera.radial_distortion[2],
          'k4': jax_camera.radial_distortion[3],
      })
    if jax_camera.has_tangential_distortion:
      distortion_params.update({
          'p1': jax_camera.tangential_distortion[0],
          'p2': jax_camera.tangential_distortion[1],
      })

  return pix_to_cam, cam_to_world, distortion_params


def rotation_distance(
    rotation_mat1, rotation_mat2
):
  """Computes the angle between two rotation matrices in degrees.

  Args:
    rotation_mat1: (3, 3) The first batch of rotation matrix.
    rotation_mat2: (3, 3) The second batch of rotation matrix.

  Returns:
    The angle in degrees between 0 and 180.
  """
  axis_angle1 = rigid_body.log_so3(rotation_mat1)
  axis_angle2 = rigid_body.log_so3(rotation_mat2)
  orientation_error_deg = jnp.degrees(
      jnp.linalg.norm(axis_angle1 - axis_angle2, axis=-1)
  )
  return jnp.where(
      orientation_error_deg < 180,
      orientation_error_deg,
      360 - orientation_error_deg,
  )


def compute_camera_metrics(
    cameras_gt, cameras_pred
):
  """Computes the metrics between two cameras."""
  orientation_diffs = jax.vmap(rotation_distance)(
      cameras_pred.orientation, cameras_gt.orientation
  )
  translation_diffs = jnp.abs(cameras_pred.translation - cameras_gt.translation)
  diffs = {
      'focal_length': jnp.abs(
          cameras_pred.focal_length - cameras_gt.focal_length
      ),
      'position': jnp.linalg.norm(
          cameras_pred.position - cameras_gt.position, axis=-1
      ),
      'translation_x': translation_diffs[Ellipsis, 0],
      'translation_y': translation_diffs[Ellipsis, 1],
      'translation_z': translation_diffs[Ellipsis, 2],
      'orientation': jnp.abs(orientation_diffs),
      'principal_points': jnp.linalg.norm(
          cameras_pred.principal_point - cameras_gt.principal_point,
          axis=-1,
      ),
  }
  if cameras_pred.radial_distortion is not None:
    radial_distortion_gt = jnp.zeros(4)
    if cameras_gt.has_radial_distortion:
      radial_distortion_gt = cameras_gt.radial_distortion
    for i in range(cameras_pred.radial_distortion.shape[-1]):
      diffs[f'radial_distortion_{i}'] = jnp.abs(
          cameras_pred.radial_distortion[Ellipsis, i] - radial_distortion_gt[Ellipsis, i]
      )
  if cameras_pred.tangential_distortion is not None:
    tangential_distortion_gt = jnp.zeros(2)
    if cameras_gt.has_tangential_distortion:
      tangential_distortion_gt = cameras_gt.radial_distortion
    for i in range(cameras_pred.tangential_distortion.shape[-1]):
      diffs[f'tangential_distortion_{i}'] = jnp.abs(
          cameras_pred.tangential_distortion[Ellipsis, i]
          - tangential_distortion_gt[Ellipsis, i]
      )

  return diffs