# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Functions for heuristically flying a camera through a generated scene."""

import math

import geometry
import tensorflow as tf


def camera_with_look_direction(position, look_direction, down_direction):
  """A camera pose specified by where it is and what direction it looks in.

  Args:
    position: [..., 3] position of camera in world.
    look_direction: [..., 3] direction of optical axis (need not be normalised).
    down_direction: [..., 3] a direction that should project to down (+ve Y).
  Returns:
    [..., 3, 4] Camera pose.
  """
  # We construct world vectors that correspond to the three axis in camera
  # space.
  # look_direction is like Z, down_direction is like Y.
  # Y cross Z = X (right-hand rule).
  vector_z = tf.math.l2_normalize(look_direction, axis=-1)
  vector_x = tf.math.l2_normalize(
      tf.linalg.cross(down_direction, vector_z), axis=-1)
  vector_y = tf.linalg.cross(vector_z, vector_x)
  # With these three vectors and the pose, we can build the camera matrix:
  camera_to_world = tf.stack([vector_x, vector_y, vector_z, position], axis=-1)
  return geometry.mat34_pose_inverse(camera_to_world)


def skyline_balance(disparity, horizon=0.3, near_fraction=0.2):
  """Computes movement parameters from a disparity image.

  Args:
    disparity: [H, W, 1] disparity image.
    horizon: how far down the image the horizon should ideally be.
    near_fraction: how much of the image should be "near".

  Returns:
    (x, y, h) where x and y are where in the image we want to be looking (as
    texture coordinates) and h is how much we want to move upwards.
  """
  # Experiment shows that the skyline boundary is somewhere between disparity
  # 0.05 and disparity 0.1. So scale and clip to give a soft sky mask.
  sky = tf.clip_by_value(20.0 * (0.1 - disparity), 0.0, 1.0)

  # How much of the image is sky?
  sky_fraction = tf.reduce_mean(sky)
  y = 0.5 + sky_fraction - horizon

  # The balance of sky in the left and right half of the image.
  w2 = disparity.shape[-2] // 2
  sky_left = tf.reduce_mean(sky[Ellipsis, :w2, :])
  sky_right = tf.reduce_mean(sky[Ellipsis, w2:, :])
  # Turn away from mountain:
  epsilon = 1e-4
  x = (sky_right + epsilon) / (sky_left + sky_right + 2 * epsilon)

  # Now we try to measure how "near the ground" we are, by looking at how
  # much of the image has disparity > 0.5 (ramping to max at 0.6)
  ground = tf.clip_by_value(10.0 * (disparity - 0.5), 0.0, 1.0)
  ground_fraction = tf.reduce_mean(ground)
  h = horizon + (near_fraction - ground_fraction)
  return x, y, h


def fly_dynamic(
    intrinsics, initial_pose,
    speed=0.2, lerp=0.05, movelerp=0.05,
    horizon=0.3, near_fraction=0.2,
    meander_x_period=100, meander_x_magnitude=0.0,
    meander_y_period=100, meander_y_magnitude=0.0,
    turn_function=None):
  """Return a function for flying a camera heuristically.

  This flying function looks at the disparity as it goes and decides whether
  to look more up/down or left/right, and also whether to try to fly further
  away from or nearer to the ground.

  Args:
    intrinsics: [4] Camera intrinsics.
    initial_pose: [3, 4] Initial camera pose.
    speed: How far to move per step.
    lerp: How fast to converge look direction to target.
    movelerp: How fast to converge movement to target.
    horizon: What fraction of the image should lie above the horizon
    near_fraction:
    meander_x_period: Number of frames to produce a cyclic meander in the
      horizontal direction
    meander_x_magnitude: How far to meander horizontally
    meander_y_period: Number of frames to produce a cyclic meander in the
      vertical direciton
    meander_y_magnitude: How far to meander vertically
    turn_function: A function which returns an x, y position to turn towards

  Returns:
    a function fly_step which takes an rgbd image and returns the pose for the
    the next camera. Call fly_step repeatedly to generate a series of poses.
    This is a stateful function and will internally keep track of camera
    position and velocity. Can only operate in eager mode.
  """
  # Where is the camera looking, and which way is down:
  camera_to_world = geometry.mat34_pose_inverse(initial_pose)
  look_dir = camera_to_world[:, 2]
  move_dir = look_dir  # Begin by moving forwards.
  down = camera_to_world[:, 1]
  position = camera_to_world[:, 3]
  t = 0

  reverse = (speed < 0)

  def fly_step(rgbd):
    nonlocal camera_to_world
    nonlocal look_dir
    nonlocal move_dir
    nonlocal down
    nonlocal position
    nonlocal t

    if turn_function:
      (xoff, yoff) = turn_function(t)
    else:
      (xoff, yoff) = (0.0, 0.0)

    xoff += math.sin(t * 2.0 * math.pi/ meander_x_period) * meander_x_magnitude
    yoff += math.sin(t * 2.0 * math.pi/ meander_y_period) * meander_y_magnitude
    t = t + 1

    down = camera_to_world[:, 1]  # Comment this out for fixed down
    disparity = rgbd[Ellipsis, 3:]
    x, y, h = skyline_balance(
        disparity, horizon=horizon, near_fraction=near_fraction)
    if reverse:
      h = 1.0 - h
      x = 1.0 - x
    look_uv = tf.stack([x + xoff, y + yoff])
    move_uv = tf.stack([0.5, h])
    uvs = tf.stack([look_uv, move_uv], axis=0)

    # Points in world
    points = geometry.mat34_transform(
        camera_to_world,
        geometry.texture_to_camera_coordinates(uvs, intrinsics))
    new_look_dir = tf.math.l2_normalize(points[0] - position)
    new_move_dir = tf.math.l2_normalize(points[1] - position)

    # Very simple smoothing
    look_dir = look_dir * (1.0 - lerp) + new_look_dir * lerp
    move_dir = move_dir * (1.0 - movelerp) + new_move_dir * movelerp
    position = position + move_dir * speed

    # Next pose
    pose = camera_with_look_direction(position, look_dir, down)
    camera_to_world = geometry.mat34_pose_inverse(pose)
    return pose

  return fly_step
