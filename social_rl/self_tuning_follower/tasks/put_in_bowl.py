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

"""Put Block in Bowl Task.

Contains the task definition of Put Blocks in the Bowl Task. The task instructs
the robot to put blocks of seen or unseen color in the bowl of a single color.
"""

from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from tasks.pick_and_place import PickAndPlaceTask
from tasks.pick_and_place import SPATIAL_RELATIONS


class PutInBowl(PickAndPlaceTask):
  """Put Blocks in Bowl base class and task."""

  def __init__(self, bowl_color="blue"):
    super().__init__()
    self.bowl_color = bowl_color
    self.task_completed_desc = f"done placing block in the {self.bowl_color} bowl."

  def reset(self, env):
    super().reset(env)
    _, hmap, _ = self.get_true_image(env)
    width, height = hmap.shape  # hmap is transposed.
    all_color_names = self.get_colors()

    bowl_obj_info = self.add_bowl(env, self.bowl_color, width, height)

    # Pick color for the block, and add it to the environment.
    block_color = np.random.choice(all_color_names)
    block_obj_info = self.add_block(env, block_color, width, height)

    # Get goal pose based on the spatial relationship
    spat_rel = np.random.choice(list(SPATIAL_RELATIONS.keys()))

    # Add distractors.
    bowl_color = bowl_obj_info["color"]
    block_color = block_obj_info["color"]
    distractor_bowl_colors = [c for c in utils.COLORS if c != bowl_color]
    distractor_block_colors = [c for c in utils.COLORS if c != block_color]

    def select_colors(is_block):
      colors = distractor_block_colors if is_block else distractor_bowl_colors
      return colors

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

    # Build the text description and language goal for the environment.
    self.env_description, lang_goal = self.build_language_description_and_goal(
        reward_params)
    reward_params["env_desc"] = self.env_description

    # Goal: Place
    # Tuple Definition:
    # ```objs, matches, targs, replace, rotations, metric, params, max_reward```
    # - `objs` are the object ids that need to be picked up.
    # - `matches` is the mapping between objs and targs
    # - `targs` are the target ids
    # - `replace` objects are picked with replacement or without replacement.
    # - `rotations` whether the End-effector rotation is required or not.
    # - `metric`: `pose` measure similarity to pose, `zone` measure
    #    whether it's within zone.
    # - `params`: a bunch of information required to compute reward.
    # - `max_reward`: Maximum reward allowed. This is typically 1
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


class PutInBowlSimple(Task):
  """Put Block in Bowl task but language goal doesn't use spatial relationships.
  """

  def __init__(self):
    super().__init__()
    self.drop_prob = 0.0
    self.use_spat_rel = False


class PutInBowlFullColor(PutInBowl):
  """Put Blocks in Bowl task but use all the colors during training."""

  def get_colors(self):
    if self.mode == "train":
      all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
    else:
      all_colors = utils.EVAL_COLORS
    return all_colors


class PutInBowlUnseenColor(PutInBowl):
  """Put Blocks in Bowl task but use unseen colors for both training and testing.
  """

  def __init__(self):
    super().__init__()
    self.use_spat_rel = False

  def get_colors(self):
    all_colors = utils.EVAL_COLORS
    return all_colors


class PutInBowlSeenColor(PutInBowl):
  """Put Blocks in Bowl task but use seen colors for both training and testing.
  """

  def __init__(self):
    super().__init__()
    self.use_spat_rel = False
    self.drop_prob = 0.0

  def get_colors(self):
    all_colors = utils.TRAIN_COLORS
    return all_colors
