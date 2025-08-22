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

"""Put Blocks in Different Colored Bowl Task.

Contains the task definition of Put Blocks in the Bowl Task. The task instructs
the robot to put blocks of seen or unseen color in bowls of different colors.
"""

import random

from cliport.utils import utils
import numpy as np
from tasks.pick_and_place import PickAndPlaceTask
from tasks.pick_and_place import SPATIAL_RELATIONS


class PutBlockInBowlUnseenColors(PickAndPlaceTask):
  """Put Blocks in Bowl base class and task."""

  def __init__(self):
    super().__init__()
    self.lang_template = "put the {pick} block in the {place} bowl"
    self.task_completed_desc = "done placing blocks in bowls."
    self.bowl_color = None

  def build_language_goal(self, reward):
    bowl_obj_info, block_obj_info = reward["all_object_ids"][-2:]
    return self.lang_template.format(
        pick=block_obj_info["color"], place=bowl_obj_info["color"])

  def reset(self, env):
    self.goals = []
    self.lang_goals = []
    self.progress = 0  # Task progression metric in range [0, 1].
    self._rewards = 0  # Cumulative returned rewards.

    _, hmap, _ = self.get_true_image(env)
    width, height = hmap.shape  # hmap is transposed.

    all_color_names = self.get_colors()
    selected_color_names = random.sample(all_color_names, 2)

    # Pick color for the block and the bowl.
    block_color, bowl_color = selected_color_names
    self.bowl_color = bowl_color
    bowl_obj_info = self.add_bowl(env, bowl_color, width, height)
    block_obj_info = self.add_block(env, block_color, width, height)

    # Goal: Place
    # Tuple Definition:
    # ```objs, matches, targs, replace, rotations, metric, params, max_reward```
    # - `objs` are the object ids that need to be picked up.
    # - `matches` is the mapping between objs and targs
    # - `targs` are the target ids
    # - `replace` objects are picked with replacement or without replacement.
    # - `rotations` whether the End-effector rotation is required or not.
    # - `metric`: `pose` measure similarity to pose, `zone` measure whether it's
    #             within zone.
    # - `params`: None when metric is `pose`. When metric is `zone`, it's a
    #             tuple of `obj_pts`, `zone_pose`. `obj_pts` is a list of points
    #             associated with the point-cloud of all the objects.
    #             `zone_pose` is the pose of the zone. This is used for all
    #             calculating partial reward.
    # - `max_reward`: Maximum reward allowed. This is typically 1

    # Get goal pose based on the spatial relationship
    spat_rel = np.random.choice(list(SPATIAL_RELATIONS.keys()))

    bowl_color = bowl_obj_info["color"]
    block_color = block_obj_info["color"]
    distractor_bowl_colors = [c for c in utils.COLORS if c != bowl_color]
    distractor_block_colors = [c for c in utils.COLORS if c != block_color]

    def select_colors(is_block):
      colors = distractor_block_colors if is_block else distractor_bowl_colors
      return colors

    # Add distractors.
    distractor_obj_info = self.add_distractor_objects(
        env, block_obj_info, bowl_obj_info, select_colors, width, height)

    reward_params = {
        "spat_rel":
            spat_rel,
        "ref_pose":
            bowl_obj_info["pose"],
        "goal_id":
            block_obj_info["obj_id"],
        "ref_pose_pix":
            utils.xyz_to_pix(
                bowl_obj_info["pose"][0], self.bounds, self.pix_size),
        "all_object_ids":
            distractor_obj_info + [bowl_obj_info, block_obj_info],
    }

    # Build the text description of the environment.
    lang_goal = self.build_language_goal(reward_params)
    # Keep env_desc empty. Descriptions will be generated dynamically.
    reward_params["env_desc"] = ""

    self.goals.append((
        self.blocks,
        np.ones((len(self.blocks), 1)),  # length of target bowls = 1
        [bowl_obj_info["pose"]],
        False,
        True,
        "pose",
        reward_params,
        1,
    ))
    self.lang_goals.append(lang_goal)

    # Only one mistake allowed.
    self.max_steps = len(self.blocks) + 1


class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
  """Put Blocks in Bowl task but only use seen colors.
  """

  def get_colors(self):
    return utils.TRAIN_COLORS


class PutBlockInBowlFullColors(PutBlockInBowlUnseenColors):
  """Put Blocks in Bowl task but use all colors in train mode.
  """

  def get_colors(self):
    if self.mode == "train":
      all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
    else:
      all_colors = utils.EVAL_COLORS

    return all_colors
