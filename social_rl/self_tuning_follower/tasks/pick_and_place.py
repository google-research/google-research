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

"""Generic Pick and Place Task Template.

Contains the task definition of a generic Pick and Place Task template used
by other tasks. The task instruct the robot to pick and place a certain object
of seen or unseen color in presence of multiple distractor objects.
"""

import functools
import os
from cliport.tasks.task import Task
from cliport.utils import utils
import cv2
import numpy as np
import pandas as pd
import pybullet as pb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import euclidean_distances

SPATIAL_RELATIONS = {
    "left": "to the left of",
    "above-left": "above and to the left of",
    "above-right": "above and to the right of",
    "above": "above",
    "below-left": "below and to the left of",
    "below-right": "below and to the right of",
    "below": "below",
    "right": "to the right of",
}

TABLETOP_REGIONS = {
    0: "top-left",
    1: "top-center",
    2: "top-right",
    3: "left-center",
    4: "center",
    5: "right-center",
    6: "bottom-left",
    7: "bottom-center",
    8: "bottom-right",
}


def compute_angle(x1, y1, x2, y2):
  """Computes the angle between two 2D points (x1,y1) and (x2, y2)."""
  # +ve y-axis is downwards, hence the negative sign
  theta = np.arctan2(x2 - x1, -(y2 - y1))
  angle_from_goal = np.rad2deg(theta)
  # The top-down view is transposed.
  if (angle_from_goal > 157.5) or (angle_from_goal <= -157.5):
    spat_rel = "above"
  elif (angle_from_goal > -157.5) and (angle_from_goal <= -112.5):
    spat_rel = "above-right"
  elif (angle_from_goal > -112.5) and (angle_from_goal <= -67.5):
    spat_rel = "right"
  elif (angle_from_goal > -67.5) and (angle_from_goal <= -22.5):
    spat_rel = "below-right"
  elif (angle_from_goal > -22.5) and (angle_from_goal <= 22.5):
    spat_rel = "below"
  elif (angle_from_goal > 22.5) and (angle_from_goal <= 67.5):
    spat_rel = "below-left"
  elif (angle_from_goal > 67.5) and (angle_from_goal <= 112.5):
    spat_rel = "left"
  elif (angle_from_goal > 112.5) and (angle_from_goal <= 157.5):
    spat_rel = "above-left"
  return spat_rel


def get_ref_obj_color(ref_obj):
  """Get the reference object color."""
  ref_obj_color = (
      ref_obj["color"] if ref_obj["color"] in utils.TRAIN_COLORS else "unknown")
  return ref_obj_color


def determine_region(x, y, width, height):
  """Find which region does the object belong to."""
  xs = [0, width / 3, 2 * width / 3, width]
  ys = [0, height / 3, 2 * height / 3, height]
  for i in range(3):
    for j in range(3):
      if (x >= xs[j] and x < xs[j + 1]) and (y >= ys[i] and y < ys[i + 1]):
        return i * 3 + j


class PickAndPlaceTask(Task):
  """Pick and Place base class and task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 10
    self.pos_eps = 0.05
    self.task_completed_desc = "done placing block."
    # probability to drop color info.
    self.drop_prob = 0.6
    # should use spatial locations to describe objects.
    self.use_spat_rel = True
    self.n_distractors = 10
    self.blocks = []

  def get_erode_size(self, obj_size):
    """Get the kernel size for erode operator in pixels."""
    max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
    erode_size = int(np.round(max_size / self.pix_size))
    return erode_size

  def compute_free_space(self, env, obj_mask):
    """Randomly sample an object pose within free-space pixels."""
    free = np.ones(obj_mask.shape, dtype=np.uint8)
    for obj_ids in env.obj_ids.values():
      for obj_id in obj_ids:
        free[obj_mask == obj_id] = 0
    return free

  def find_valid_region(self, spat_rel):
    """Find valid regions that satisfy the spatial relationship."""
    # The top-down view is transposed.
    if spat_rel == "above":
      is_valid = np.vectorize(lambda x: (x > 157.5) or (x <= -157.5))
    elif spat_rel == "above-right":
      is_valid = np.vectorize(lambda x: (x > -157.5) and (x <= -112.5))
    elif spat_rel == "right":
      is_valid = np.vectorize(lambda x: (x > -112.5 and x <= -67.5))
    elif spat_rel == "below-right":
      is_valid = np.vectorize(lambda x: (x > -67.5) and (x <= -22.5))
    elif spat_rel == "below":
      is_valid = np.vectorize(lambda x: (x > -22.5) and (x <= 22.5))
    elif spat_rel == "below-left":
      is_valid = np.vectorize(lambda x: (x > 22.5) and (x <= 67.5))
    elif spat_rel == "left":
      is_valid = np.vectorize(lambda x: (x > 67.5) and (x <= 112.5))
    elif spat_rel == "above-left":
      is_valid = np.vectorize(lambda x: (x > 112.5) and (x <= 157.5))
    return is_valid

  def sample_pos_in_free_space(self, free, hmap, obj_size):
    """Sample a point in the free space."""
    pix = utils.sample_distribution(np.float32(free))
    pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
    pos = (pos[0], pos[1], obj_size[2] / 2)
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, 0))
    return pos, rot

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
    compute_angle_wrt_goal = functools.partial(
        compute_angle, x2=goal_pose[0], y2=goal_pose[1])

    def compute_dist(i, j):
      dist = np.sqrt((goal_pose[0] - i)**2 + (j - goal_pose[1])**2)
      return dist

    angle_from_goal = np.fromfunction(compute_angle_wrt_goal, free.shape)
    dist_from_goal = np.fromfunction(compute_dist, free.shape)
    is_valid_dist = np.vectorize(lambda x: x < erode_size * 2)
    is_valid = self.find_valid_region(spat_rel)

    # For each occupied region, expand the region a little bit more to avoid
    # placing objects too close by.
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    free[~is_valid(angle_from_goal)] = 0
    free[~is_valid_dist(dist_from_goal)] = 0
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

  def fill_template_description(self,
                                obj,
                                ref_obj=None,
                                spat_rel=None,
                                drop_color=False):
    """Generates a description of the object.

    The object is either descripted using its location on the table, or using
    its spatial relationship with respect to some other object.

    Args:
      obj: object that needs to be described.
      ref_obj: reference object with respect to which obj has to be described.
        When None, obj is described in isolation.
      spat_rel: spatial relation between obj and ref_obj
      drop_color: if True, then the color info is dropped from the
        description.

    Returns:
      returns a string containing the description of an object.
    """
    obj_urdf = os.path.splitext(os.path.basename(obj["urdf"]))[0]
    region = obj["region"]
    if obj["color"] in utils.TRAIN_COLORS and not drop_color:
      obj_color = obj["color"]
    else:
      obj_color = "unknown"
    if ref_obj is not None:
      ref_obj_color = get_ref_obj_color(ref_obj)
      ref_obj_urdf = os.path.splitext(os.path.basename(ref_obj["urdf"]))[0]

    if not spat_rel:
      return f"There is a {obj_color} {obj_urdf} in the {TABLETOP_REGIONS[region]} region."
    else:
      return f"There is a {obj_color} {obj_urdf} which is {SPATIAL_RELATIONS[spat_rel]} the {ref_obj_color} {ref_obj_urdf}."

  def fill_template_instruction(self,
                                obj,
                                ref_obj=None,
                                spat_rel=None,
                                drop_color=False):
    """Build the language goal.

    Instructions can be of two types. 1. Put the <seen-color> block in
    the <bowl-color> bowl. 2. Put the block <spatial-location like to the left
    of> <seen-color> block in the <bowl-color> bowl.
    Which of these two are used is determined by spat_rel and drop_color args.
    Args:
      obj: object that needs to be described.
      ref_obj: reference object with respect to which obj has to be described.
        When None, obj is described in isolation.
      spat_rel: spatial relation between obj and ref_obj
      drop_color: if True, then the color info is dropped from the
        description.

    Returns:
      returns a string containing the description of an object.
    """
    obj_urdf = os.path.splitext(os.path.basename(obj["urdf"]))[0]
    if not drop_color:
      obj_color = obj["color"] + " "
    else:
      obj_color = ""
    if ref_obj is not None:
      ref_obj_color = get_ref_obj_color(ref_obj)
      ref_obj_urdf = os.path.splitext(os.path.basename(ref_obj["urdf"]))[0]

    if not spat_rel:
      return f"Put the {obj_color}{obj_urdf} in the {self.bowl_color} bowl."
    else:
      return f"Put the {obj_color}{obj_urdf} which is {SPATIAL_RELATIONS[spat_rel]} {ref_obj_color} {ref_obj_urdf} in the {self.bowl_color} bowl."

  def build_language_description_and_goal(self, reward):
    """Build the language description and goal of the environment.

    Args:
      reward: Dict containing the necessary information to build the
        description.

    Returns:
      a tuple containing language description and goal
    """
    obj_infos = reward["all_object_ids"]
    goal_obj_id = reward["goal_id"]

    # Start writing the description. First sort the objects. Objects with
    # known colors occur first. Then they are sorted based on region to group
    # them up. Finally they are sorted by y-coordinate, and then x-coordinate.
    sorted_obj_infos = sorted(
        obj_infos,
        key=lambda k:  # pylint: disable=g-long-lambda
        (k["unknown_color"], k["region"], k["pix"][1], k["pix"][0]),
    )

    obj_df = pd.DataFrame.from_dict(sorted_obj_infos)
    ins_list = ""
    lang_goal = None
    drop_prob_flag = False

    # Should the color info be dropped
    if np.random.rand() < self.drop_prob:
      drop_prob_flag = True

    # Group by regions, and describe objects in each region.
    # Build a minimum spanning tree to connect all objects in a region.
    # Use DFS to describe objects in a fixed order.
    for region, region_df in obj_df.groupby("region"):
      # weigh edges with unknown objects higher.
      unknown_mat = np.array(region_df["unknown_color"].tolist()).reshape(-1, 1)
      unknown_dist_mul = 100 * unknown_mat | unknown_mat.T
      region_df = region_df.reset_index()
      dist = euclidean_distances(region_df["pix"].tolist())
      dist += unknown_dist_mul
      mat = csr_matrix(dist)
      mst = minimum_spanning_tree(mat).toarray().astype(int)

      # convert sparse graph representation to dense.
      mst = mst | mst.T

      # Preferably always start with known object.
      root_id = 0
      # pylint: disable=g-explicit-bool-comparison,singleton-comparison
      known_object_df = region_df[region_df["unknown_color"] ==
                                  False]

      if known_object_df.shape[0] > 0:
        root_id = known_object_df.index[0]

      # determine the order of objects.
      order, predecessors = depth_first_order(mst, root_id)
      ins = f"{region+1}. "

      for idx in order:
        drop_color = False
        drop_color_ins = False
        # If train mode, drop random color descriptions in the environment
        # description. Same for instruction. Or,
        # ff the selected object color is unknown, but the mode is training,
        # drop the color from the description and instruction.
        if (self.mode == "train" and
            drop_prob_flag) or (self.mode == "train" and
                                region_df["unknown_color"][idx]):
          drop_color = True
          drop_color_ins = True
        # If the mode is not training, don't drop the color from instruction
        # but drop the color from description only if unknown color.
        elif self.mode != "train":
          drop_color = region_df["unknown_color"][idx]
          drop_color_ins = False
        if predecessors[idx] == -9999:
          # If current object is unknown, drop color from description.
          # Otherwise, don't because this is the root object.
          drop_color = region_df["unknown_color"][idx]
          drop_color_ins &= drop_color
          ins += self.fill_template_description(
              region_df.iloc[idx], drop_color=drop_color)
          # If this object is the goal object, then build language
          # instructions as well.
          if region_df["obj_id"][idx] == goal_obj_id:
            lang_goal = self.fill_template_instruction(
                region_df.iloc[idx], drop_color=drop_color_ins)
        else:
          # If current object is unknown, drop color from description.
          # Otherwise, only drop if drop_color is set to True previously.
          drop_color |= region_df["unknown_color"][idx]
          drop_color_ins &= drop_color
          spat_rel = compute_angle(
              *region_df.iloc[idx]["pix"],
              *region_df.iloc[predecessors[idx]]["pix"],
          )
          ins += " " + self.fill_template_description(
              region_df.iloc[idx],
              region_df.iloc[predecessors[idx]],
              spat_rel,
              drop_color=drop_color,
          )
          # If this object is the goal object, then build language
          # instructions as well.
          if region_df["obj_id"][idx] == goal_obj_id:
            ref_obj = None
            ref_spat_rel = None
            if self.use_spat_rel:
              ref_obj = region_df.iloc[predecessors[idx]]
              ref_spat_rel = spat_rel
            lang_goal = self.fill_template_instruction(
                region_df.iloc[idx],
                ref_obj,
                ref_spat_rel,
                drop_color=drop_color_ins,
            )
      ins_list += ins + "\n"
    return ins_list, lang_goal

  def add_bowl(self, env, bowl_color, width, height):
    """Add bowl into the environment.

    Args:
      env: Pybullet environment
      bowl_color: name of the bowl color
      width: width of the tabletop to determine spatial location of block.
      height: height of the tabletop to determine spatial location of block.
    Returns:
      A dictionary containing all the important info pertaining to this object.
    """

    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = "bowl/bowl.urdf"
    bowl_pose = self.get_random_pose(env, bowl_size)
    bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
    pb.changeVisualShape(
        bowl_id, -1, rgbaColor=utils.COLORS[bowl_color] + [1])
    bowl_pix = utils.xyz_to_pix(bowl_pose[0], self.bounds, self.pix_size)
    bowl_obj_info = {
        "obj_id": bowl_id,
        "pose": bowl_pose,
        "size": bowl_size,
        "urdf": bowl_urdf,
        "color": bowl_color,
        "pix": bowl_pix,
        "unknown_color": bowl_color in utils.EVAL_COLORS,
        "region": determine_region(bowl_pix[0], bowl_pix[1], width, height),
    }

    return bowl_obj_info

  def add_block(self, env, block_color, width, height):
    """Add block into the environment.

    Args:
      env: Pybullet environment
      block_color: name of the block color
      width: width of the tabletop to determine spatial location of block.
      height: height of the tabletop to determine spatial location of block.
    Returns:
      A dictionary containing all the important info pertaining to this object.
    """

    block_size = (0.04, 0.04, 0.04)
    block_urdf = "stacking/block.urdf"
    block_pose = self.get_random_pose(env, block_size)
    block_id = env.add_object(block_urdf, block_pose)
    pb.changeVisualShape(
        block_id, -1, rgbaColor=utils.COLORS[block_color] + [1])
    # (0, None): 0 means that the block is symmetric.
    # TODO(hagrawal): Not sure what None means. Update. This is kept
    # for CLIPort compatibility. We don't use it.
    self.blocks.append((block_id, (0, None)))
    block_pix = utils.xyz_to_pix(block_pose[0], self.bounds, self.pix_size)
    block_obj_info = {
        "obj_id": block_id,
        "pose": block_pose,
        "size": block_size,
        "urdf": block_urdf,
        "color": block_color,
        "unknown_color": block_color in utils.EVAL_COLORS,
        "pix": block_pix,
        "region": determine_region(block_pix[0], block_pix[1], width, height),
    }
    return block_obj_info

  def add_distractor_objects(self, env, block_obj_info,
                             bowl_obj_info, select_colors, width, height):
    """Add distractor objects into the environment.

    Args:
      env: PyBullet environment.
      block_obj_info: Dictionary containing all the info about blocks.
      bowl_obj_info: Dictionary containing all the info about bowls.
      select_colors: Function to select distractor color
      width: Width of the tabletop.
      height: Height of the tabletop.
    Returns:
    A list of dictionaries containing all the info about distractor objects.
    """
    n_distractors = 0
    max_distractors = self.n_distractors
    distractor_obj_info = []
    while n_distractors < max_distractors:
      is_block = np.random.rand() > 0.35
      urdf = block_obj_info["urdf"] if is_block else bowl_obj_info["urdf"]
      size = block_obj_info["size"] if is_block else bowl_obj_info["size"]
      colors = select_colors(is_block)
      pose = self.get_random_pose(env, size)
      if not pose:
        continue
      obj_id = env.add_object(urdf, pose)
      color = np.random.choice(colors)
      if not obj_id:
        continue
      pb.changeVisualShape(obj_id, -1, rgbaColor=utils.COLORS[color] + [1])
      n_distractors += 1
      object_pix = utils.xyz_to_pix(pose[0], self.bounds, self.pix_size)
      distractor_obj_info.append({
          "obj_id":
              obj_id,
          "pose":
              pose,
          "size":
              size,
          "urdf":
              urdf,
          "color":
              color,
          "unknown_color":
              color in utils.EVAL_COLORS,
          "pix":
              object_pix,
          "region":
              determine_region(object_pix[0], object_pix[1], width,
                               height),  # hmap is transposed.
      })
      return distractor_obj_info

  def get_colors(self):
    return utils.TRAIN_COLORS if self.mode == "train" else utils.EVAL_COLORS
