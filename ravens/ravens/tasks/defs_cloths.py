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

"""Cloth related tasks.

Interpreting self.zone_pose[0] = (x,y,z) where x and y are the VERTICAL and
HORIZONTAL (resp.) ranges in the diagram, in METERS:

  -0.5 .... +0.5
  ------------  0.3
  |          |   :
  |          |   :
  |          |   :
  ------------  0.7

The self.zone_pose corresponds to the colored yellow lines in the GUI. The
(0,0,0) location corresponds to the base of the robot.

Notes from Xuchen Han:
- if reducing mass of cloth, reduce springElasticStiffness.
- recommends springDampingAllDirections=0 for realistic damping.

Notes on the cloth:
- basePosition for the cloth is the CENTER of the cloth.
- started w/mass=1.0 and elastic/damping stiffness of 40 and 0.1.
- NxN cloth means the edge length is "cloth length" divided by N-1, not N.

With 10x10 (therefore, there are 9x9=81 actual _squares_ in the cloth mesh),
indices are:

  90  -->  99
  80  -->  89
  ..  ...  ..
  10  -->  19
   0  -->   9

For 5x5 (so 4x4=16 squares) the indices are:

  20  -->  24
  15  -->  19
  10  -->  14
   5  -->   9
   0  -->   4

For the corner-pulling demonstrator (see implementation in tasks/task.py), I
recommend gripping 'one corner inwards' instead of using the _actual_
corners, except for the 5x5 case which might be too coarse-grained.

We can adjust the size of both the zone and the cloth. The provided zone.obj
ranges from (-10,10) whereas cloth files from Blender scale from (-1,1),
hence the cloth scale needs to be 10x larger than the zone scale. If this
convention changes with any of the assets, adjust sizes accordingly.

Reminder: add any soft body IDs to `self.def_ids`.
"""
import os
import time

import cv2
import numpy as np
import pybullet as p
from ravens import utils as U
from ravens.tasks import Task


class ClothEnv(Task):
  """Superclass for tasks that use cloth."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.primitive = 'pick_place'
    self.max_steps = 11
    self._settle_secs = 0

    # Gripping parameters. Empirically, 0.025 works well.
    self._def_threshold = 0.025
    self._def_nb_anchors = 1

    # See scaling comments above.
    self._zone_scale = 0.01
    self._cloth_scale = 0.10
    self._cloth_length = (2.0 * self._cloth_scale)
    self._zone_length = (20.0 * self._zone_scale)
    self.zone_size = (20.0 * self._zone_scale, 20.0 * self._zone_scale, 0)
    self._cloth_size = (self._cloth_length, self._cloth_length, 0.01)
    assert self._cloth_scale == self._zone_scale * 10, self._cloth_scale

    # Cloth resolution and corners (should be clockwise).
    self.n_cuts = 10
    if self.n_cuts == 5:
      self.corner_indices = [0, 20, 24, 4]  # actual corners
    elif self.n_cuts == 10:
      self.corner_indices = [11, 81, 88, 18]  # one corner inwards
    else:
      raise NotImplementedError(self.n_cuts)
    self._f_cloth = 'assets/cloth/bl_cloth_{}_cuts.obj'.format(
        str(self.n_cuts).zfill(2))

    # Other cloth parameters.
    self._mass = 0.5
    self._edge_length = (2.0 * self._cloth_scale) / (self.n_cuts - 1)
    self._collision_margin = self._edge_length / 5.0

    # IoU/coverage rewards (both w/zone or goal images). Pixels w/255 are
    # targets.
    self.target_hull_bool = None
    self.zone_id = -1

  def get_target_zone_corners(self):
    """Determine corners of target zone.

    Assumes we follow this structure in some order:
      c2 --- c3
      |       |
      |       |
      c1 --- c4
    The actual xyz positions depend on the sampled `self.zone_pose`.

    We use this for (among other things) the target points for the
    ClothPickPlace demonstrator. Make sure the clockwise ordering is
    consistent with labels, for the reward and demonstrator. NOTE: the
    zone stays clockwise in terms of our c1 -> c2 -> c3 -> c4 ordering
    (from a top down view) but the CLOTH may have been flipped. For now I
    assume that can be handled in the action by considering the possible
    counter-clockwise map.
    """
    el2 = self._zone_length / 2
    self.c1_position = U.apply(self.zone_pose, (-el2, -el2, 0))
    self.c2_position = U.apply(self.zone_pose, (-el2, el2, 0))
    self.c3_position = U.apply(self.zone_pose, (el2, el2, 0))
    self.c4_position = U.apply(self.zone_pose, (el2, -el2, 0))
    self._corner_targets_xy = np.array([
        self.c1_position[:2],
        self.c2_position[:2],
        self.c3_position[:2],
        self.c4_position[:2],
    ])

  def get_masks_target(self, env, zone_id):
    """For getting a mask of the cloth's target zone.

    We can then use this as a target, and compute a simple, pixel-based
    IoU or coverage metric for the reward.

    To make things easier for goal-conditioned envs, we should also save
    the `target_hull_bool` with each episode -- put in `last_info`.

    Args:
      env: a ravens environment.
      zone_id: id of the target zone.
    """
    _, _, object_mask = self.get_object_masks(env)

    # Check object_mask for all pixels corresponding to the zone ID.
    mask = np.float32(object_mask == zone_id)
    mask = np.uint8(mask * 255)

    # Find contours of the `mask` image, combine all to get shape (N,1,2).
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = [np.concatenate(list(contours))]

    # Find the convex hull object for that combined contour.
    hull_list = [cv2.convexHull(c) for c in contours_list]

    # Make an RGB image, then draw the filled-in area of all items in
    # `hull_list`.
    hull = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(hull, hull_list, -1, (255, 255, 255), thickness=-1)

    # Assign to `target_hull_bool` so we continually check it each time step.
    target_hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)
    self.target_hull_bool = np.array(target_hull, dtype=bool)

  def compute_pixel_IoU_coverage(self):
    """Computes IoU and coverage based on pixels.

    Use: `self.target_hull` and `self.current_hull`. For the former:
    values of 255 refer to the area contained within the zone (and
    includes the zone itself, FWIW). For the latter: segment to detect
    the workspace OR the zone ID. Then use numpy and/or operators:

    https://stackoverflow.com/questions/49338166/python-intersection-over-union
    https://stackoverflow.com/questions/14679595/vs-and-vs

    NOTE I: assumes cloth can be segmented by detecting the workspace and
    zone lines, and that any pixel OTHER than those belongs to cloth.

    NOTE II: IoU and coverage are computed in the same way, except that
    the former divides by the union, the latter divides by just the goal.

    Returns:
      (pixel_iou, pixel_coverage)
    """
    _, _, object_mask = self.get_object_masks(self.env)

    # Check object_mask for all pixels OTHER than the cloth.
    ids = [1, self.zone_id]  # 1 = workspace_ID
    mask = np.isin(object_mask, test_elements=ids)

    # Flip items so that 1 = cloth (then 255 when scaled).
    idx_0s = (mask == 0)
    idx_1s = (mask == 1)
    mask[idx_0s] = 1
    mask[idx_1s] = 0
    cloth_mask_bool = np.array(mask, dtype=bool)

    # Compute pixel-wise IoU and coverage, using two bool dtype arrays.
    overlap = self.target_hull_bool & cloth_mask_bool  # Logical AND
    union = self.target_hull_bool | cloth_mask_bool  # Logical OR
    overlap_count = np.count_nonzero(overlap)
    union_count = np.count_nonzero(union)
    goal_count = np.count_nonzero(self.target_hull_bool)
    pixel_iou = overlap_count / float(union_count)
    pixel_coverage = overlap_count / float(goal_count)
    return (pixel_iou, pixel_coverage)

  def is_item_covered(self):
    """For cloth-cover, if it's covered, it should NOT be in the mask."""
    _, _, object_mask = self.get_object_masks(self.env)
    assert len(self.block_ids) == 1, self.block_ids
    block = self.block_ids[0]
    return 1 - float(block in object_mask)

  def add_zone(self, env, zone_pose=None):
    """Adds green square target zone, size based on zone_scale.

    To handle goal-conditioned cloth flattening, save `zone_pose` and
    provide it as input, to avoid re-sampling. Thus, starting cloth
    states are sampled at a valid distance from target zones.

    Args:
      env: a ravens environment.
      zone_pose: pose of the target zone.
    """
    zone_template = 'assets/zone/zone-template.urdf'
    replace = {'LENGTH': (self._zone_scale, self._zone_scale)}
    zone_urdf = self.fill_template(zone_template, replace)

    # Pre-assign zone pose _only_ if loading goal-conditioned policies.
    if zone_pose is not None:
      self.zone_pose = zone_pose
    else:
      self.zone_pose = self.random_pose(env, self.zone_size)

    # Add objects in a consistent manner.
    zone_id = env.add_object(zone_urdf, self.zone_pose, fixed=True)
    os.remove(zone_urdf)

    # Get pixel-based mask to detect target, for coverage/IoU later.
    self.get_masks_target(env, zone_id)

    # To reference it later for IoUs/coverage, or to remove if needed.
    self.zone_id = zone_id

  def add_cloth(self, env, base_pos, base_orn):
    """Adding a cloth from an .obj file.

    Since this is a soft body, need to add this ID to `self.def_ids.

    Args:
      env: a ravens environment.
      base_pos: Position of the cloth.
      base_orn: Orientation of the cloth.

    Returns:
      bullet id of the cloth.
    """
    cloth_id = p.loadSoftBody(
        fileName=self._f_cloth,
        basePosition=base_pos,
        baseOrientation=base_orn,
        collisionMargin=self._collision_margin,
        scale=self._cloth_scale,
        mass=self._mass,
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=40,
        springDampingStiffness=0.1,
        springDampingAllDirections=0,
        useSelfCollision=1,
        frictionCoeff=1.0,
        useFaceContact=1)

    # Add objects in a consistent manner.
    self.object_points[cloth_id] = np.float32((0, 0, 0)).reshape(3, 1)
    env.objects.append(cloth_id)
    self.def_ids.append(cloth_id)

    # Sanity checks.
    nb_vertices, _ = p.getMeshData(bodyUniqueId=cloth_id)
    assert nb_vertices == self.n_cuts * self.n_cuts
    return cloth_id

  def _sample_cloth_orientation(self):
    """Sample the bag (and let it drop) to get interesting starting states."""
    orn = [
        self._base_orn[0] + np.random.normal(loc=0.0, scale=self._scalex),
        self._base_orn[1] + np.random.normal(loc=0.0, scale=self._scaley),
        self._base_orn[2] + np.random.normal(loc=0.0, scale=self._scalez),
    ]
    return p.getQuaternionFromEuler(orn)

  @property
  def coverage_threshold(self):
    return self._coverage_thresh

  @property
  def corner_targets_xy(self):
    return self._corner_targets_xy

  @property
  def def_threshold(self):
    return self._def_threshold

  @property
  def def_nb_anchors(self):
    return self._def_nb_anchors


class ClothFlat(ClothEnv):
  """Flat cloth environment.

  Start with a flat cloth that starts relatively close to a visible target
  zone, and usually overlaps (but not enough to trigger the coverage
  threshold at the start, and we can detect that in any case in main.py).
  """

  def __init__(self):
    super().__init__()
    self.max_steps = 11
    self.metric = 'cloth-coverage'
    self._name = 'cloth-flat'
    self._coverage_thresh = 0.85

    # Env reference so we can call Task.get_object_masks(env)
    self.env = None

    # Cloth sampling. Max zone distance (cloth-to-zone) heavily tuned.
    self._scalex = 0.0
    self._scaley = 0.0
    self._scalez = 0.5
    self._base_orn = [np.pi / 2.0, 0, 0]
    self._drop_height = 0.01
    self._max_zone_dist = 0.34

    # Action parameters.
    self.primitive_params = {
        1: {
            'speed': 0.002,
            'delta_z': -0.0010,
            'postpick_z': 0.05,
            'preplace_z': 0.05,
            'pause_place': 0.0,
        }
    }
    self.task_stage = 1

  def reset(self, env, zone_pose=None):
    self.total_rewards = 0
    self.object_points = {}
    self.t = 0
    self.task_stage = 1
    self.def_ids = []
    self.env = env

    # Add square target zone, determine corners, handle reward function.
    self.add_zone(env, zone_pose=zone_pose)
    self.get_target_zone_corners()

    # Used to sample a pose that is sufficiently close to the zone center.
    def cloth_to_zone(bpos):
      p1 = np.float32(bpos)
      p2 = np.float32(self.zone_pose[0])
      return np.linalg.norm(p1 - p2)

    # Sample a flat cloth position somewhere on the workspace.
    bpos, _ = self.random_pose(env, self._cloth_size)
    while cloth_to_zone(bpos) > self._max_zone_dist:
      bpos, _ = self.random_pose(env, self._cloth_size)

    # Make cloth closer to the zone, sample orientation, and create it.
    alpha = 0.6
    bpos_x = (bpos[0] * alpha) + (self.zone_pose[0][0] * (1 - alpha))
    bpos_y = (bpos[1] * alpha) + (self.zone_pose[0][1] * (1 - alpha))
    self.base_pos = [bpos_x, bpos_y, self._drop_height]
    self.base_orn = self._sample_cloth_orientation()
    self.cloth_id = self.add_cloth(env, self.base_pos, self.base_orn)

    env.start()
    time.sleep(self._settle_secs)
    env.pause()


class ClothFlatNoTarget(ClothFlat):
  """Like ClothFlat, except no visible targets."""

  def __init__(self):
    super().__init__()
    self._name = 'cloth-flat-notarget'

  def reset(self, env, last_info=None):
    """Reset to start an episode.

    Call the superclass to generate as usual, and then remove the zone
    here. Requires care in `environment.py` to avoid iterating over
    invalid IDs, and we need the zone in the superclass for many reasons.

    If loading goal images, we cannot just override self.target_hull_bool
    because that means the cloth isn't sampled the same way as if the
    target was visible. We need to first decide on the pose, THEN sample
    the cloth. Easiest solution: load the sampled `zone_pose`, then pass
    that into the reset() call so that we don't re-sample `zone_pose`.
    Everything is reconsructed from there.

    Args:
      env: A ravens environment.
      last_info: Last info dictionary.
    """
    zone_pose = None
    if last_info is not None:
      zone_pose = last_info['sampled_zone_pose']
    super().reset(env, zone_pose=zone_pose)
    p.removeBody(self.zone_id)


class ClothCover(ClothEnv):
  """Cloth and a single item, which must be covered."""

  def __init__(self):
    super().__init__()
    self.max_steps = 2 + 1
    self.metric = 'cloth-cover-item'
    self._name = 'cloth-cover'

    # Env reference so we can call Task.get_object_masks(env)
    self.env = None

    # Cloth sampling.
    self._scalex = 0.0
    self._scaley = 0.0
    self._scalez = 0.5
    self._base_orn = [np.pi / 2.0, 0, 0]
    self._drop_height = 0.01

    # Action parameters. Make postpick_z a bit higher vs cloth-flat.
    self.primitive_params = {
        1: {
            'speed': 0.003,
            'delta_z': -0.0005,
            'postpick_z': 0.10,
            'preplace_z': 0.10,
            'pause_place': 0.0,
        },
        2: {
            'speed': 0.003,
            'delta_z': -0.0005,
            'postpick_z': 0.10,
            'preplace_z': 0.10,
            'pause_place': 0.0,
        },
    }
    self.task_stage = 1

    # Extra non-cloth items. For now keep number of blocks at 1.
    self._nb_blocks = 1
    self.block_ids = []

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.t = 0
    self.task_stage = 1
    self.def_ids = []
    self.block_ids = []
    self.env = env

    # Add blocks (following sorting environment).
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'assets/stacking/block.urdf'
    for _ in range(self._nb_blocks):
      block_pose = self.random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      self.object_points[block_id] = np.float32((0, 0, 0)).reshape(3, 1)
      self.block_ids.append(block_id)

    # Sample a flat cloth arbitrarily on the workspace.
    bpos, _ = self.random_pose(env, self._cloth_size)
    self.base_pos = [bpos[0], bpos[1], self._drop_height]
    self.base_orn = self._sample_cloth_orientation()
    self.cloth_id = self.add_cloth(env, self.base_pos, self.base_orn)

    env.start()
    time.sleep(self._settle_secs)
    env.pause()
