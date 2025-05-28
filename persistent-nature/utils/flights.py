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

# pylint: disable=g-importing-member,g-multiple-import
"""utils for camera flights."""
from math import cos, sin, pi, atan2

import numpy as np
from utils import camera_util


def fly_figure8(initial_camera, size, frames):
  """Figure 8 position as function of t from 0 to 1."""
  def xy(t, theta):
    theta = theta / 180 * pi
    t = t * 2.0 * pi
    x = 0.5 * sin(2.0 * t)
    y = sin(t)
    return (x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta))

  cameras = []
  for i in range(frames):
    t = i / frames
    (x, y) = xy(t, initial_camera.theta - 90)

    epsilon = 1e-4
    (x0, y0) = xy(t - epsilon, initial_camera.theta - 90)
    (x1, y1) = xy(t + epsilon, initial_camera.theta - 90)
    theta = atan2(y1 - y0, x1 - x0) * (180 / pi)
    cameras.append(
        camera_util.Camera(
            initial_camera.x + x * size,
            initial_camera.y,
            initial_camera.z + y * size,
            theta,
            0,
        )
    )
  return cameras


def fly_forward(initial_camera, frames):
  forward_speed = 0.025 * 7
  camera = initial_camera
  cameras = [initial_camera]
  for _ in range(frames):
    camera = camera_util.move_camera(camera, forward_speed, 0, 0)
    cameras.append(camera)
  return cameras


def fly_rotate(initial_camera, frames):
  camera = initial_camera
  cameras = [initial_camera]
  for _ in range(frames):
    camera = camera_util.move_camera(camera, 0, 360 / frames)
    cameras.append(camera)
  return cameras


def fly_forward_backward(initial_camera, frames):
  forward_speed = 0.025 * 7
  camera = initial_camera
  cameras = [initial_camera]
  for _ in range(frames // 2):
    camera = camera_util.move_camera(camera, forward_speed, 0, 0)
    cameras.append(camera)
  for _ in range(frames // 2):
    camera = camera_util.move_camera(camera, -forward_speed, 0, 0)
    cameras.append(camera)
  return cameras


def fly_aerial(initial_camera):
  """generate 300 frames of camera moving up while turning down."""
  camera = initial_camera
  up_frames = 120
  ys = np.linspace(initial_camera.y, 8, up_frames)
  psis = np.linspace(initial_camera.psi, -90, up_frames)
  cameras = []
  for i in range(up_frames):
    cameras.append(
        camera_util.Camera(
            initial_camera.x,
            float(ys[i]),
            initial_camera.z,
            initial_camera.theta,
            float(psis[i]),
        )
    )
  camera = cameras[-1]
  rotate_frames = 181
  for frames in range(rotate_frames):
    camera = camera_util.move_camera(camera, 0, 360 / frames, 0)
    cameras.append(camera)
  return cameras
