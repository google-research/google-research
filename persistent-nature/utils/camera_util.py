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

# pylint: disable=invalid-name,g-importing-member,g-multiple-import
"""Utilities for camera manipulation."""
from collections import namedtuple
from math import cos, sin, pi
import random

import numpy as np
import torch


####### camera utils

# Tuple to represent user camera position
Camera = namedtuple(
    'Camera',
    [
        'x',
        'y',
        'z',  # position
        'theta',  # horizontal direction to look, in degrees. (0 = positive x)
        'psi',  # up/down angle, in degrees (0 = level)
    ],
)


def initial_camera():
  return Camera(0.0, 0.0, 0.0, 0.0, 0.0)


# Camera movement constants
ROTATION_HORIZONTAL_DEGREES = 5
ROTATION_UPDOWN_DEGREES = 5
UPDOWN_MIN = -90
UPDOWN_MAX = 90
FORWARD_SPEED = 1 / 2
SIDEWAYS_SPEED = FORWARD_SPEED / 2
VERTICAL_SPEED = FORWARD_SPEED / 2
INITIAL_CAMERA = None


def pose_from_camera(camera):
  """A 4x4 pose matrix mapping world to camera space.

  Args:
    camera: camera object

  Returns:
    world2cam matrix
  """
  cos_theta = cos((camera.theta + 90) * pi / 180)
  sin_theta = sin((camera.theta + 90) * pi / 180)
  cos_psi = cos(camera.psi * pi / 180)
  sin_psi = sin(camera.psi * pi / 180)
  Ry = torch.tensor([
      [cos_theta, 0, sin_theta, 0],
      [0, 1, 0, 0],
      [-sin_theta, 0, cos_theta, 0],
      [0, 0, 0, 1],
  ])
  Rx = torch.tensor([
      [1, 0, 0, 0],
      [0, cos_psi, sin_psi, 0],
      [0, -sin_psi, cos_psi, 0],
      [0, 0, 0, 1],
  ])
  T = torch.tensor([
      [1, 0, 0, -camera.x],
      [0, 1, 0, -camera.y],
      [0, 0, 1, -camera.z],
      [0, 0, 0, 1],
  ])
  return torch.mm(torch.mm(Rx, Ry), T)


def camera_from_pose(Rt):
  """Solve for camera variables from world2cam pose.

  Args:
    Rt: 4x4 torch.Tensor, world2cam pose

  Returns:
    camera object
  """
  assert list(Rt.shape) == [4, 4]

  # solve for theta
  cos_theta = Rt[0, 0]  # x
  sin_theta = Rt[0, 2]  # y
  theta = torch.atan2(sin_theta, cos_theta)  # y, x
  theta = theta * 180 / pi  # convert to deg
  theta = (theta - 90) % 360  # 90 degree rotation

  # solve for psi
  cos_psi = Rt[1, 1]
  sin_psi = -Rt[2, 1]
  psi = torch.atan(sin_psi / cos_psi)
  psi = psi * 180 / pi

  # Rx @ Ry
  R = pose_from_camera(Camera(0.0, 0.0, 0.0, theta.item(), psi.item()))
  T = torch.mm(R.inverse(), Rt.cpu())
  camera = Camera(
      -T[0, 3].item(),
      -T[1, 3].item(),
      -T[2, 3].item(),
      theta.item(),
      psi.item(),
  )
  return camera


def get_full_image_parameters(
    layout_model,
    nerf_render_size,
    batch_size,
    device='cuda',
    Rt=None,
    sample_fov=False,
):
  """Construct intrisics for image of size nerf_render_size."""
  camera_params = {}
  if sample_fov:
    fov = layout_model.fov_mean + layout_model.fov_std * np.random.randn(
        batch_size
    )
  else:
    # use the mean FOV rather than sampling
    fov = layout_model.fov_mean + 0.0 * np.random.randn(batch_size)

  sampled_size = np.array([nerf_render_size] * batch_size)
  focal = (sampled_size / 2) / np.tan(np.deg2rad(fov) / 2)
  K = np.zeros((batch_size, 3, 3))
  K[:, 0, 0] = focal
  K[:, 1, 1] = -focal
  K[:, 2, 2] = -1  # Bx3x3
  K = torch.from_numpy(K).float().to(device)

  camera_params['K'] = K
  camera_params['global_size'] = torch.from_numpy(sampled_size).float()
  camera_params['fov'] = torch.from_numpy(fov).float()

  if Rt is not None:
    if Rt.ndim == 4:
      assert Rt.shape[1] == 1
      Rt = Rt[:, 0, :, :]
    camera_params['Rt'] = Rt  # Bx4x4
  return camera_params


# --------------------------------------------------------------------
# camera motion utils


def update_camera(camera, key, auto_adjust_height_and_tilt=True):
  """move camera according to key pressed."""
  if key == 'x':
    # Reset
    if INITIAL_CAMERA is not None:
      return INITIAL_CAMERA
    return initial_camera()  # camera at origin

  if auto_adjust_height_and_tilt:
    # ignore additional controls
    if key in ['r', 'f', 't', 'g']:
      return camera

  x = camera.x
  y = camera.y
  z = camera.z
  theta = camera.theta
  psi = camera.psi
  cos_theta = cos(theta * pi / 180)
  sin_theta = sin(theta * pi / 180)

  # Rotation left and right
  if key == 'a':
    theta -= ROTATION_HORIZONTAL_DEGREES
  if key == 'd':
    theta += ROTATION_HORIZONTAL_DEGREES
  theta = theta % 360

  # Looking up and down
  if key == 'r':
    psi += ROTATION_UPDOWN_DEGREES
  if key == 'f':
    psi -= ROTATION_UPDOWN_DEGREES
  psi = max(UPDOWN_MIN, min(UPDOWN_MAX, psi))

  # Movement in 3 dimensions
  if key == 'w':
    # Go forward
    x += cos_theta * FORWARD_SPEED
    z += sin_theta * FORWARD_SPEED
  if key == 's':
    # Go backward
    x -= cos_theta * FORWARD_SPEED
    z -= sin_theta * FORWARD_SPEED
  if key == 'q':
    # Move left
    x -= -sin_theta * SIDEWAYS_SPEED
    z -= cos_theta * SIDEWAYS_SPEED
  if key == 'e':
    # Move right
    x += -sin_theta * SIDEWAYS_SPEED
    z += cos_theta * SIDEWAYS_SPEED
  if key == 't':
    # Move up
    y += VERTICAL_SPEED
  if key == 'g':
    # Move down
    y -= VERTICAL_SPEED
  return Camera(x, y, z, theta, psi)


def move_camera(camera, forward_speed, rotation_speed):
  x = camera.x
  y = camera.y
  z = camera.z
  theta = camera.theta + rotation_speed
  psi = camera.psi
  cos_theta = cos(theta * pi / 180)
  sin_theta = sin(theta * pi / 180)
  x += cos_theta * forward_speed
  z += sin_theta * forward_speed
  return Camera(x, y, z, theta, psi)


# --------------------------------------------------------------------
# camera balancing utils

# How far up the image should the horizon be, ideally.
# Suggested range: 0.5 to 0.7.
horizon_target = 0.65

# What proportion of the depth map should be "near" the camera, ideally.
# The smaller the number, the higher up the camera will fly.
# Suggested range: 0.05 to 0.2
near_target = 0.2

tilt_velocity_scale = 0.3
offset_velocity_scale = 0.5


def land_fraction(sky_mask):
  return torch.mean(sky_mask).item()


def near_fraction(depth, near_depth=0.3, near_spread=0.1):
  near = torch.clip((depth - near_depth) / near_spread, 0.0, 1.0)
  return torch.mean(near).item()


def adjust_camera_vertically(camera, offset, tilt):
  return Camera(
      camera.x, camera.y + offset, camera.z, camera.theta, camera.psi + tilt
  )


# layout model: adjust tilt and offset parameters based
# on near and land fraction
def update_tilt_and_offset(
    outputs,
    tilt,
    offset,
    horizon_target=horizon_target,
    near_target=near_target,
    tilt_velocity_scale=tilt_velocity_scale,
    offset_velocity_scale=offset_velocity_scale,
):  # pylint: disable=redefined-outer-name
  """Adjust tilt and offest based on geometry."""
  depth = (
      outputs['depth_up'][0]
      if outputs['depth_up'] is not None
      else outputs['depth_thumb']
  )
  sky_mask = outputs['sky_mask'][0]
  horizon = land_fraction(sky_mask)
  near = near_fraction(depth)
  tilt += tilt_velocity_scale * (horizon - horizon_target)
  offset += offset_velocity_scale * (near - near_target)
  return tilt, offset


# --------------------------------------------------------------------
# camera interpolation utils


# Interpolate between random points
def interpolate_camera(start, end, l):
  def i(a, b):
    return b * l + a * (1 - l)

  end_theta = end.theta
  if end.theta - start.theta > 180:
    end_theta -= 360
  if start.theta - end.theta > 180:
    end_theta += 360
  return Camera(
      i(start.x, end.x),
      i(start.y, end.y),
      i(start.z, end.z),
      i(start.theta, end_theta),
      i(start.psi, end.psi),
  )


def ease(x):
  if x < 0.5:
    return 2 * x * x
  return 1 - 2 * (1 - x) * (1 - x)


def lerp(a, b, l):
  return a * (1 - l) + b * l


def random_camera(tlim=16, psi_multiplier=20):
  height = random.uniform(0, 2)
  psi = -psi_multiplier * height
  return Camera(
      random.uniform(-tlim, tlim),
      height,
      random.uniform(-tlim, tlim),
      random.uniform(0, 360),
      psi,
  )


def visualize_rays(G_terrain, Rt, xyz, layout, display_size, cam_grid=None):
  """Return an image showing the camera rays projected onto X-Z plane."""
  # Rt = world2cam matrix

  if hasattr(G_terrain, 'layout_generator'):
    # layout model
    global_feat_res = G_terrain.layout_decoder.global_feat_res
    coordinate_scale = G_terrain.coordinate_scale
  else:
    # triplane model
    global_feat_res = G_terrain.backbone_xz.img_resolution
    coordinate_scale = G_terrain.rendering_kwargs['box_warp']

  inference_feat_res = layout.shape[-1]

  # compute pixel locations for camera points
  cam_frustum = xyz / (coordinate_scale / 2)  # normalize to [-1, 1]
  cam_frustum = (
      cam_frustum * global_feat_res / inference_feat_res
  )  # rescale for extended spatial grid
  cam_frustum = (cam_frustum + 1) / 2  # normalize to [0, 1]
  cam_frustum = (
      (cam_frustum * display_size).long().clamp(0, display_size - 1)
  )  # convert to [0, display size]

  # compute pixel locations for camera center
  tform_cam2world = Rt.inverse()
  cam_center = tform_cam2world[0, :3, -1]
  cam_center = cam_center / (coordinate_scale / 2)
  cam_center = (
      cam_center * global_feat_res / inference_feat_res
  )  # rescale for extended spatial grid
  cam_center = (cam_center + 1) / 2
  cam_center = (cam_center * display_size).long().clamp(0, display_size - 1)

  # compute pixel locations for white box representing training region
  orig_grid_offset = torch.Tensor([
      -1,
  ]) * (global_feat_res / inference_feat_res)
  orig_grid_offset = (orig_grid_offset + 1) / 2
  orig_grid_offset = (
      (orig_grid_offset * display_size).long().clamp(0, display_size - 1)
  )  # convert to [0, display size]

  if cam_grid is None:
    cam_grid = torch.zeros(3, display_size, display_size)
  else:
    cam_grid = cam_grid.clone().cpu()

  # plot everything on image
  cam_grid[
      1, cam_frustum[Ellipsis, 2].reshape(-1), cam_frustum[Ellipsis, 0].reshape(-1)
  ] = 1
  cam_grid[
      0, cam_frustum[Ellipsis, 2].reshape(-1), cam_frustum[Ellipsis, 0].reshape(-1)
  ] = 0.5
  cam_grid[:, orig_grid_offset, orig_grid_offset:-orig_grid_offset] = 1
  cam_grid[:, -orig_grid_offset, orig_grid_offset:-orig_grid_offset] = 1
  cam_grid[:, orig_grid_offset:-orig_grid_offset, orig_grid_offset] = 1
  cam_grid[:, orig_grid_offset:-orig_grid_offset, -orig_grid_offset] = 1
  cam_grid[
      0,
      cam_center[2] - 2 : cam_center[2] + 2,
      cam_center[0] - 2 : cam_center[0] + 2,
  ] = 1
  return cam_grid
