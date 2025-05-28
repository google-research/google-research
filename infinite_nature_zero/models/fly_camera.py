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

# -*- coding: utf-8 -*-
"""Library for autopliot algorithm."""
import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1e-8  # small number to avoid numerical issues


def normalize(x):
  return x / np.linalg.norm(x)


def pose_from_look_direction_np(camera_pos, camera_dir, camera_up):
  """Computes poses given camera parameters, in numpy.

  Args:
    camera_pos: camera position.
    camera_dir: camera looking direction
    camera_up: camera up vector.

  Returns:
    (R, t) rotation and translation of given camera parameters
  """

  camera_right = normalize(np.cross(camera_up, camera_dir))
  rot = np.zeros((4, 4))
  rot[0, 0:3] = normalize(camera_right)
  rot[1, 0:3] = normalize(np.cross(camera_dir, camera_right))
  rot[2, 0:3] = normalize(camera_dir)
  rot[3, 3] = 1
  trans_matrix = np.array([[1.0, 0.0, 0.0, -camera_pos[0]],
                           [0.0, 1.0, 0.0, -camera_pos[1]],
                           [0.0, 0.0, 1.0, -camera_pos[2]],
                           [0.0, 0.0, 0.0, 1.0]])
  tmp = rot @ trans_matrix
  return tmp[:3, :3], tmp[:3, 3]


def pose_from_look_direction(camera_pos, camera_dir, down_direction):
  """Computes poses given camera parameters, in Pytorch.

  Args:
    camera_pos: camera position.
    camera_dir: camera looking direction
    down_direction: camera down vector.

  Returns:
    (R, t) rotation and translation of given camera parameters
  """

  camera_right = F.normalize(
      torch.cross(down_direction, camera_dir, dim=1), dim=-1)
  rot = torch.eye(4).unsqueeze(0).repeat(camera_pos.shape[0], 1, 1)
  rot[:, 0, 0:3] = (camera_right)
  rot[:, 1, 0:3] = F.normalize(
      torch.cross(camera_dir, camera_right, dim=1), dim=-1)
  rot[:, 2, 0:3] = F.normalize(camera_dir, dim=-1)

  # this is equivalent to inverse matrix
  trans_matrix = torch.eye(4).unsqueeze(0).repeat(camera_pos.shape[0], 1, 1)
  trans_matrix[:, 0:3, 3] = -camera_pos[:, 0:3]

  pose_inv = rot @ trans_matrix
  return pose_inv[:, :3, :3], pose_inv[:, :3, 3]


def skyline_balance(disparity,
                    horizon,
                    sky_threshold,
                    near_fraction):
  """Computes movement parameters from a disparity image.

  Args:
    disparity: current disparity image
    horizon: how far down the image the horizon should ideally be.
    sky_threshold: target sky percentage
    near_fraction: target near content percentage

  Returns:
    (x, y, h) where x and y are where in the image we want to be looking
    and h is how much we want to move upwards.
  """
  sky = torch.clamp(20.0 * (sky_threshold - disparity), 0.0, 1.0)
  # How much of the image is sky?
  sky_fraction = torch.mean(sky, dim=[1, 2])
  y = torch.clamp(0.5 + sky_fraction - horizon, 0.0, 1.0)

  # The balance of sky in the left and right half of the image.
  w2 = disparity.shape[-1] // 2
  sky_left = torch.mean(sky[:, :, :w2], dim=[1, 2])
  sky_right = torch.mean(sky[:, :, w2:], dim=[1, 2])
  # Turn away from mountain:
  x = (sky_right + EPSILON) / (sky_left + sky_right + 2 * EPSILON)

  # Now we try to measure how "near the ground" we are, by looking at how
  # much of the image has disparity > 0.4 (ramping to max at 0.5)
  ground_t = 0.4
  ground = torch.clamp(10.0 * (disparity - ground_t), 0.0, 1.0)
  ground_fraction = torch.mean(ground, dim=[1, 2])
  h = horizon + (near_fraction - ground_fraction)
  return x, y, h


def auto_pilot(intrinsic,
               disp,
               speed,
               look_dir,
               move_dir,
               position,
               camera_down,
               looklerp=0.05,
               movelerp=0.2,
               horizon=0.4,
               sky_fraction=0.1,
               near_fraction=0.2):
  """Auto-pilot algorithm that determines the next pose to sample.

  Args:
   intrinsic: Intrinsic matrix
   disp: disparity map
   speed: moving speed
   look_dir: look ahead direction
   move_dir: camera moving direction
   position: camera position
   camera_down: camera down vector (opposite of up vector)
   looklerp: camera viewing direction moving average interpolation ratio
   movelerp: camera translation moving average interpolation ratio
   horizon: predefined ratio of hozion in the image
   sky_fraction: predefined ratio of sky content
   near_fraction: predefined ratio of near content

  Returns:
   next_rot: next rotation to sample
   next_t: next translation to sample
   look_dir: next look ahead direction
   move_dir: next moving translation vector
   position: next camera position
  """
  img_w, img_h = disp.shape[-1], disp.shape[-2]

  x, y, h = skyline_balance(
      disp,
      horizon=horizon,
      sky_threshold=sky_fraction,
      near_fraction=near_fraction)

  look_uv = torch.stack([x, y], dim=-1)
  move_uv = torch.stack([torch.zeros_like(h) + 0.5, h], dim=-1)
  uvs = torch.stack([look_uv, move_uv], dim=1)

  fx, fy = intrinsic[0, 0], intrinsic[1, 1]
  px, py = intrinsic[0, 2], intrinsic[1, 2]
  c_x = (uvs[Ellipsis, 0] * img_w - px) / fx
  c_y = (uvs[Ellipsis, 1] * img_h - py) / fy
  new_coords = torch.stack([c_x, c_y], dim=-1)
  coords_h = torch.cat(
      [new_coords, torch.ones_like(new_coords[Ellipsis, 0:1])], dim=-1)

  new_look_dir = F.normalize(coords_h[:, 0, :], dim=-1)
  new_move_dir = F.normalize(coords_h[:, 1, :], dim=-1)
  look_dir = look_dir * (1.0 - looklerp) + new_look_dir * looklerp
  move_dir = move_dir * (1.0 - movelerp) + new_move_dir * movelerp
  move_dir_ = move_dir * speed.unsqueeze(-1)
  position = position + move_dir_

  # world to camera
  next_rot, next_t = pose_from_look_direction(
      position, look_dir, camera_down)  # look from pos in direction dir

  return {
      'next_rot': next_rot,
      'next_t': next_t,
      'look_dir': look_dir,
      'move_dir': move_dir,
      'position': position
  }
