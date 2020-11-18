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

#!/usr/bin/env python
"""Base Task class."""

import collections
import os
import random
import string

import cv2
import numpy as np
import pybullet as p
from ravens import cameras
from ravens import primitives
from ravens import utils
from ravens.grippers import Suction


class Task():
  """Base Task class."""

  def __init__(self):
    self.ee = Suction
    self.mode = 'train'
    self.sixdof = False
    self.primitive = primitives.PickPlace()
    self.oracle_cams = cameras.Oracle.CONFIG

    # Evaluation epsilons (for pose evaluation metric).
    self.pos_eps = 0.01
    self.rot_eps = np.deg2rad(15)

    # Workspace bounds.
    self.pix_size = 0.003125
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

  def reset(self, env):  # pylint: disable=unused-argument
    self.goals = []
    self.progress = 0  # Task progression metric in range [0, 1].
    self._rewards = 0  # Cumulative returned rewards.

  #-------------------------------------------------------------------------
  # Oracle Agent
  #-------------------------------------------------------------------------

  def oracle(self, env):
    """Oracle agent."""
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])

    def act(obs, info):  # pylint: disable=unused-argument
      """Calculate action."""

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      _, hmap, obj_mask = self.get_true_image(env)

      # Unpack next goal step.
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

      # Match objects to targets without replacement.
      if not replace:

        # Modify a copy of the match matrix.
        matches = matches.copy()

        # Ignore already matched objects.
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            if self.is_match(pose, targs[j], symmetry):
              matches[i, :] = 0
              matches[:, j] = 0

      # Get objects to be picked (prioritize farthest from nearest neighbor).
      nn_dists = []
      nn_targets = []
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        xyz, _ = p.getBasePositionAndOrientation(object_id)
        targets_i = np.argwhere(matches[i, :]).reshape(-1)
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
          targets_xyz = np.float32([targs[j][0] for j in targets_i])
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
          nn = np.argmin(dists)
          nn_dists.append(dists[nn])
          nn_targets.append(targets_i[nn])

        # Handle ignored objects.
        else:
          nn_dists.append(0)
          nn_targets.append(-1)
      order = np.argsort(nn_dists)[::-1]

      # Filter out matched objects.
      order = [i for i in order if nn_dists[i] > 0]

      for pick_i in order:
        pick_mask = np.uint8(obj_mask == objs[pick_i][0])

        # Erode to avoid picking on edges.
        # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

        if np.sum(pick_mask) > 0:
          break

      # Trigger task reset if no object is visible.
      if np.sum(pick_mask) == 0:
        self.goals = []
        print('Object for pick is not visible. Skipping demonstration.')
        return

      # Get picking pose.
      pick_prob = np.float32(pick_mask)
      pick_pix = utils.sample_distribution(pick_prob)
      # For "deterministic" demonstrations on insertion-easy, use this:
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size)
      pick_pose = (pick_pos, (0, 0, 0, 1))

      # Get placing pose.
      targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable
      obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable
      if not self.sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)
      world_to_pick = utils.invert(pick_pose)
      obj_to_pick = utils.multiply(world_to_pick, obj_pose)
      pick_to_obj = utils.invert(obj_to_pick)
      place_pose = utils.multiply(targ_pose, pick_to_obj)

      # Rotate end effector?
      if not rotations:
        place_pose = (place_pose[0], (0, 0, 0, 1))

      return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act)

  #-------------------------------------------------------------------------
  # Reward Function and Task Completion Metrics
  #-------------------------------------------------------------------------

  def reward(self):
    """Get delta rewards for current timestep.

    Returns:
      A tuple consisting of the scalar (delta) reward, plus `extras`
        dict which has extra task-dependent info from the process of
        computing rewards that gives us finer-grained details. Use
        `extras` for further data analysis.
    """
    reward, info = 0, {}

    # Unpack next goal step.
    objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]

    # Evaluate by matching object poses.
    if metric == 'pose':
      step_reward = 0
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        pose = p.getBasePositionAndOrientation(object_id)
        targets_i = np.argwhere(matches[i, :]).reshape(-1)
        for j in targets_i:
          target_pose = targs[j]
          if self.is_match(pose, target_pose, symmetry):
            step_reward += max_reward / len(objs)
            break

    # Evaluate by measuring object intersection with zone.
    elif metric == 'zone':
      zone_pts, total_pts = 0, 0
      obj_pts, zones = params
      for zone_pose, zone_size in zones:

        # Count valid points in zone.
        for obj_id in obj_pts:
          pts = obj_pts[obj_id]
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_zone = utils.invert(zone_pose)
          obj_to_zone = utils.multiply(world_to_zone, obj_pose)
          pts = np.float32(utils.apply(obj_to_zone, pts))
          if len(zone_size) > 1:
            valid_pts = np.logical_and.reduce([
                pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                pts[2, :] < self.bounds[2, 1]])

          zone_pts += np.sum(np.float32(valid_pts))
          total_pts += pts.shape[1]
      step_reward = max_reward * (zone_pts / total_pts)

    # Get cumulative rewards and return delta.
    reward = self.progress + step_reward - self._rewards
    self._rewards = self.progress + step_reward

    # Move to next goal step if current goal step is complete.
    if np.abs(max_reward - step_reward) < 0.01:
      self.progress += max_reward  # Update task progress.
      self.goals.pop(0)

    return reward, info

  def done(self):
    """Check if the task is done or has failed.

    Returns:
      True if the episode should be considered a success, which we
        use for measuring successes, which is particularly helpful for tasks
        where one may get successes on the very last time step, e.g., getting
        the cloth coverage threshold on the last alllowed action.
        However, for bag-items-easy and bag-items-hard (which use the
        'bag-items' metric), it may be necessary to filter out demos that did
        not attain sufficiently high reward in external code. Currently, this
        is done in `main.py` and its ignore_this_demo() method.
    """

    # # For tasks with self.metric == 'pose'.
    # if hasattr(self, 'goal'):
    # goal_done = len(self.goal['steps']) == 0  # pylint:
    # disable=g-explicit-length-test
    return (len(self.goals) == 0) or (self._rewards > 0.99)  # pylint: disable=g-explicit-length-test
    # return zone_done or defs_done or goal_done

  #-------------------------------------------------------------------------
  # Environment Helper Functions
  #-------------------------------------------------------------------------

  def is_match(self, pose0, pose1, symmetry):
    """Check if pose0 and pose1 match within a threshold."""

    # Get translational error.
    diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
    dist_pos = np.linalg.norm(diff_pos)

    # Get rotational error around z-axis (account for symmetries).
    diff_rot = 0
    if symmetry > 0:
      rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
      rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
      diff_rot = np.abs(rot0 - rot1) % symmetry
      if diff_rot > (symmetry / 2):
        diff_rot = symmetry - diff_rot

    return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps)

  def get_true_image(self, env):
    """Get RGB-D orthographic heightmaps and segmentation masks."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render(self.oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = utils.reconstruct_heightmaps(
        [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
    return cmap, hmap, mask

  def get_random_pose(self, env, obj_size):
    """Get random collision-free object pose within workspace bounds."""

    # Get erosion size of object in pixels.
    max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
    erode_size = int(np.round(max_size / self.pix_size))

    _, hmap, obj_mask = self.get_true_image(env)

    # Randomly sample an object pose within free-space pixels.
    free = np.ones(obj_mask.shape, dtype=np.uint8)
    for obj_ids in env.obj_ids.values():
      for obj_id in obj_ids:
        free[obj_mask == obj_id] = 0
    free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    if np.sum(free) == 0:
      return
    pix = utils.sample_distribution(np.float32(free))
    pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
    pos = (pos[0], pos[1], obj_size[2] / 2)
    theta = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    return pos, rot

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def fill_template(self, template, replace):
    """Read a file and replace key strings."""
    filepath = os.path.dirname(os.path.abspath(__file__))
    template = os.path.join(filepath, '..', template)
    with open(template, 'r') as file:
      fdata = file.read()
    for field in replace:
      for i in range(len(replace[field])):
        fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    fname = f'{template}.{rname}'
    with open(fname, 'w') as file:
      file.write(fdata)
    return fname

  def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
    """Get random box size."""
    size = np.random.rand(3)
    size[0] = size[0] * (max_x - min_x) + min_x
    size[1] = size[1] * (max_y - min_y) + min_y
    size[2] = size[2] * (max_z - min_z) + min_z
    return tuple(size)

  def get_object_points(self, obj):
    obj_shape = p.getVisualShapeData(obj)
    obj_dim = obj_shape[0][3]
    xv, yv, zv = np.meshgrid(
        np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
        np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
        np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
        sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def color_random_brown(self, obj):
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    p.changeVisualShape(obj, -1, rgbaColor=color)
