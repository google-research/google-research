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

"""Put Blocks in Zone Task.

Task specification for Put blocks in Zone task. The zone is defined by a small
rectangular region, in which blocks have to be placed. Blocks can be of seen or
unseen colors.
"""

import random

from cliport.utils import utils
import cv2
import numpy as np
import pybullet as p
from tasks.pick_and_place import determine_region
from tasks.pick_and_place import PickAndPlaceTask
from tasks.pick_and_place import SPATIAL_RELATIONS


class PutInZone(PickAndPlaceTask):
  """Put Blocks in Zone task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 10
    self.lang_template = "put the {pick} block in the {place} zone"
    self.task_completed_desc = "done placing block in the bowl."
    self.n_distractors = 7

  def build_language_goal(self, reward):
    zone_obj_info, _, block_obj_info = reward["all_object_ids"][-3:]
    return self.lang_template.format(
        pick=block_obj_info["color"], place=zone_obj_info["color"])

  def random_pose_next_to_goal(self, goal_pose, spat_rel, env):
    """Get a pose next to the goal consistent with the spatial relationship.

    Args:
      goal_pose: (location, orientation) of the goal.
      spat_rel: string defining the relative spatial location.
      env: current Environment class object.

    Returns:
      sampled_pose that is consistent with the spatial relationship.
    """
    goal_pose = utils.xyz_to_pix(goal_pose[0], self.bounds, self.pix_size)
    obj_size = (0.04, 0.04, 0.04)
    erode_size = self.get_erode_size(obj_size)

    _, hmap, obj_mask = self.get_true_image(env)
    free = self.compute_free_space(env, obj_mask)

    # Find valid pose
    def compute_angle(i, j):
      theta = np.arctan2(goal_pose[0] - i, j - goal_pose[1])
      return np.rad2deg(theta)

    def compute_dist(i, j):
      dist = np.sqrt((goal_pose[0] - i)**2 + (j - goal_pose[1])**2)
      return dist

    angle_from_goal = np.fromfunction(compute_angle, free.shape)
    dist_from_goal = np.fromfunction(compute_dist, free.shape)
    is_valid_dist = np.vectorize(lambda x: x < erode_size * 3)
    is_valid_dist_near = np.vectorize(lambda x: x > erode_size * 2)
    is_valid = self.find_valid_region(spat_rel)

    # For each occupied region, expand the region a little bit more to avoid
    # placing objects too close by.
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    free[~is_valid(angle_from_goal)] = 0
    free[~is_valid_dist(dist_from_goal)] = 0
    free[~is_valid_dist_near(dist_from_goal)] = 0
    (
        free[0:erode_size, :],
        free[:, 0:erode_size],
        free[-erode_size:, :],
        free[:, -erode_size:],
    ) = (0, 0, 0, 0)

    if np.sum(free) == 0:
      print("There is no free space!!")
      return None, None

    pos, rot = self.sample_pos_in_free_space(free, hmap, obj_size)
    return pos, rot

  def add_zone(self, env, zone_color, width, height):
    """Add zone into the environment.

    Args:
      env: Pybullet environment
      zone_color: name of the zone color
      width: width of the tabletop to determine spatial location of block.
      height: height of the tabletop to determine spatial location of block.

    Returns:
      A dictionary containing all the important info pertaining to this object.
    """
    zone_size = (0.15, 0.15, 0)
    zone_pose = self.get_random_pose(env, zone_size)
    zone_pix = utils.xyz_to_pix(zone_pose[0], self.bounds, self.pix_size)
    zone_obj_id = env.add_object("zone/zone.urdf", zone_pose, "fixed")
    p.changeVisualShape(
        zone_obj_id, -1, rgbaColor=utils.COLORS[zone_color] + [1])
    zone_obj_info = {
        "obj_id": zone_obj_id,
        "pose": zone_pose,
        "size": zone_size,
        "urdf": "zone/zone.urdf",
        "color": zone_color,
        "pix": zone_pix,
        "unknown_color": zone_color in utils.EVAL_COLORS,
        "region": determine_region(zone_pix[0], zone_pix[1], width, height)
    }
    return zone_obj_info

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
    _, _, _, _, _, _, params, max_reward = self.goals[0]

    # Evaluate by measuring object intersection with zone.
    zone_pts, total_pts = 0, 0
    obj_pts = params["obj_pts"]
    zone_poses = params["ref_pose"]
    zones_sizes = params["zone_size"]

    for _, (zone_pose, zone_size) in enumerate(zip(
        zone_poses, zones_sizes)):

      # Count valid points in zone.
      for _, obj_id in enumerate(obj_pts):
        pts = obj_pts[obj_id]
        obj_pose = p.getBasePositionAndOrientation(obj_id)
        world_to_zone = utils.invert(zone_pose)
        obj_to_zone = utils.multiply(world_to_zone, obj_pose)
        pts = np.float32(utils.apply(obj_to_zone, pts))
        if len(zone_size) > 1:
          valid_pts = np.logical_and.reduce([
              pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
              pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
              pts[2, :] < self.zone_bounds[2, 1]
          ])

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
      if self.lang_goals:
        self.lang_goals.pop(0)

    return reward, info

  def reset(self, env):
    super().reset(env)

    _, hmap, _ = self.get_true_image(env)
    width, height = hmap  # hmap is transposed.

    color_names = self.get_colors()
    zone_color, block_color, target_block_color = random.sample(
        color_names, k=3)

    zone_obj_info = self.add_zone(env, zone_color, width, height)
    block_obj_info = self.add_block(env, block_color, width, height)

    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = "bowl/bowl.urdf"
    block_size = block_obj_info["size"]
    block_urdf = block_obj_info["urdf"]

    # Add an object fairly close to the zone, to learn spatial relationships.
    is_block = np.random.rand() > 0.35
    target_urdf = block_urdf if is_block else bowl_urdf
    target_size = block_size if is_block else bowl_size

    spat_rel = np.random.choice(list(SPATIAL_RELATIONS.keys()))
    target_pose = self.random_pose_next_to_goal(zone_obj_info["pose"], spat_rel,
                                                env)
    if target_pose[0] is None:
      return
    target_id = env.add_object(target_urdf, target_pose)
    p.changeVisualShape(
        target_id, -1, rgbaColor=utils.COLORS[target_block_color] + [1])
    # (0, None): 0 means that the block is symmetric.
    # TODO(hagrawal): Not sure what None means. Update.
    target_block_pix = utils.xyz_to_pix(target_pose[0], self.bounds,
                                        self.pix_size)
    target_block_obj_info = {
        "obj_id":
            target_id,
        "pose":
            target_pose,
        "size":
            target_size,
        "urdf":
            target_urdf,
        "color":
            target_block_color,
        "unknown_color":
            target_block_color in utils.EVAL_COLORS,
        "pix":
            target_block_pix,
        "region":
            determine_region(target_block_pix[0], target_block_pix[1], width,
                             height),
    }

    # Colors of distractor objects.
    block_color = block_obj_info["color"]
    distractor_block_colors = [
        c for c in utils.COLORS if c != target_block_color or c != block_color
    ]

    def select_colors(is_block):
      colors = distractor_block_colors if is_block else list(utils.COLORS)
      return colors

    bowl_obj_info = {
        "urdf": bowl_urdf,
        "size": bowl_size,
    }
    distractor_obj_info = self.add_distractor_objects(env, block_obj_info,
                                                      bowl_obj_info,
                                                      select_colors, width,
                                                      height)

    obj_pts = {}
    block_id = block_obj_info["obj_id"]
    zone_pose = zone_obj_info["pose"]
    zone_size = zone_obj_info["size"]
    obj_pts[block_id] = self.get_box_object_points(block_id)
    reward_params = {
        "spat_rel":
            spat_rel,
        "obj_pts":
            obj_pts,
        "ref_pose": [zone_pose],
        "zone_size": [zone_size],
        "goal_id":
            block_id,
        "ref_pose_pix":
            utils.xyz_to_pix(zone_pose[0], self.bounds, self.pix_size),
        "all_object_ids":
            distractor_obj_info +
            [zone_obj_info, target_block_obj_info, block_obj_info],
    }

    # Build the text description of the environment.
    lang_goal = self.build_language_goal(reward_params)
    reward_params["env_desc"] = None

    # Goal: Place
    # Tuple Definition:
    # ```objs, matches, targs, replace, rotations, metric, params, max_reward```
    # - `objs` are the object ids that need to be picked up.
    # - `matches` is the mapping between objs and targs
    # - `targs` are the target ids
    # - `replace` objects are picked with replacement or without replacement.
    # - `rotations` whether the End-effector rotation is required or not.
    # - `metric`: `pose` measure similarity to pose, `zone` measure whether it's
    #    within zone.
    # - `params`: None when metric is `pose`. When metric is `zone`, it's a
    #    tuple of `obj_pts`, `zone_pose`. `obj_pts` is a list of points
    #    associated with the point-cloud of all the objects. `zone_pose` is the
    #    pose of the zone. This is used for all calculating partial reward.
    # - `max_reward`: Maximum reward allowed. This is typically 1
    self.goals.append((self.blocks, np.ones(
        (1, 1)), [zone_pose], True, False, "zone", reward_params, 1))
    self.lang_goals.append(lang_goal)

    # Only one mistake allowed.
    self.max_steps = len(self.blocks) + 1


class PutInZoneFullColor(PutInZone):
  """Put in Zone task where blocks are of seen and unseen color."""

  def get_colors(self):
    if self.mode == "train":
      all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
    else:
      all_colors = utils.EVAL_COLORS
    return all_colors


class PutInZoneUnseenColor(PutInZone):
  """Put in Zone task where blocks are of unseen color."""

  def __init__(self):
    super().__init__()
    self.use_spat_rel = False

  def get_colors(self):
    all_colors = utils.EVAL_COLORS
    return all_colors


class PutInZoneSeenColor(PutInZone):
  """Put in Zone task where blocks are of seen color."""

  def __init__(self):
    super().__init__()
    self.use_spat_rel = False
    self.drop_prob = 0.0

  def get_colors(self):
    all_colors = utils.TRAIN_COLORS
    return all_colors
