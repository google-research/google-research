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

"""A set of bag-related robotics tasks.

Bag type and radii at the start, with no perturbations on the spheres.

- bag 1, radius at very start: 0.1000 (as expected)
- bag 2, radius at very start: 0.0981
- bag 3, radius at very start: 0.0924
- bag 4, radius at very start: 0.0831
- bag 5, radius at very start: 0.0707

The gripping threshold, `self._def_threshold`, is slightly lower compared to
cloth tasks, because with bags that combine beads and vertices, it's normally
better (for physics purposes) to grip beads instead of vertices.

Reminder: add any soft body IDs to `self.def_ids`.
"""
import os
import time

import cv2
import numpy as np
import pybullet as p
from ravens import utils as U
from ravens.tasks import Task

BAGS_TO_FILES = {
    1: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.1_numV_257.obj',
    2: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.3_numV_289.obj',
    3: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.4_numV_321.obj',
    4: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.6_numV_353.obj',
    5: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.8_numV_385.obj',
}

# An identity pose we can use to gracefully handle failure cases.
IDENTITY = {
    'pose0': ((0.3, 0, 0.3), (0, 0, 0, 1)),
    'pose1': ((0.3, 0, 0.3), (0, 0, 0, 1))
}  # TODO(daniel) remove
BEAD_THRESH = 0.33  # TODO(daniel) make cleaner


class BagEnv(Task):
  """Superclass for tasks that use bags."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.primitive = 'pick_place'
    self.max_steps = 11
    self._settle_secs = 0

    # Gripping parameters. Empirically, 0.020 works well.
    self._def_threshold = 0.020
    self._def_nb_anchors = 1

    # Scale the bag / zone. The zone.obj ranges from (-20,20).
    self._zone_scale = 0.0130
    self._bag_scale = 0.10
    self._zone_length = (20. * self._zone_scale)
    self.zone_size = (20. * self._zone_scale, 20. * self._zone_scale, 0.0)
    self._bag_size = (1. * self._bag_scale, 1. * self._bag_scale, 0.01)

    # Bag type (or resolution?) and parameters.
    self._bag = 4
    self._mass = 1.0
    self._scale = 0.25
    self._collision_margin = 0.003
    self._base_orn = [np.pi / 2.0, 0.0, 0.0]
    self._f_bag = BAGS_TO_FILES[self._bag]
    self._drop_height = 0.15

  def add_zone(self, env):
    """Adds green square target zone, size based on zone_scale.

    Similar to add_zone for the cloth tasks, except it's not necessary to
    pre-assign zone poses (no goal-conditioning) or to find the zone mask.

    Args:
      env: A ravens environment.
    """
    zone_template = 'assets/zone/zone-template.urdf'
    replace = {'LENGTH': (self._zone_scale, self._zone_scale)}
    zone_urdf = self.fill_template(zone_template, replace)
    self.zone_pose = self.random_pose(env, self.zone_size)

    # Add objects in a consistent manner.
    zone_id = env.add_object(zone_urdf, self.zone_pose, fixed=True)
    os.remove(zone_urdf)

    # As in cloth tasks, assign zone to reference it later, for removal.
    self.zone_id = zone_id

  def add_cube(self, env, pose, global_scaling=1.0):
    """Andy's ravens/block's default size should be (0.04, 0.04, 0.04)."""
    cube_id = p.loadURDF(
        fileName='assets/block/block_for_anchors.urdf',
        basePosition=pose[0],
        baseOrientation=pose[1],
        globalScaling=global_scaling,
        useMaximalCoordinates=True)

    # Add objects in a consistent manner.
    self.object_points[cube_id] = np.float32((0, 0, 0)).reshape(3, 1)
    env.objects.append(cube_id)
    return cube_id

  def add_random_box(self, env, max_total_dims):
    """Generate randomly shaped box, adapted from the aligning task.

    Make rand_x and rand_y add up to the max_total. The aligning env uses
    a box with mass 0.1, but this one can be lighter. Heavier boxes mean
    the robot will not fully lift the bag off the ground.

    Args:
      env: Environment, used for the add_object convenience method.
      max_total_dims: To control dimensions of boxes. Recommended to keep
        this value at a level making these boxes comparable to the cubes,
        if not smaller, used in bag-items-easy.

    Returns:
      Tuple with The PyBullet integer ID for the box, and the box size,
        which is randomly drawn (in case we use it later).
    """
    min_val = 0.015
    assert min_val * 2 <= max_total_dims, min_val
    rand_x = np.random.uniform(min_val, max_total_dims - min_val)
    rand_y = max_total_dims - rand_x
    box_size = (rand_x, rand_y, 0.03)

    # Add box. See tasks/aligning.py.
    box_template = 'assets/box/box-template.urdf'
    box_urdf = self.fill_template(box_template, {'DIM': box_size})
    box_pose = self.random_pose(env, box_size)
    box_id = env.add_object(box_urdf, box_pose)
    os.remove(box_urdf)
    self.color_random_brown(box_id)
    self.object_points[box_id] = np.float32(
        (0, 0, 0)).reshape(3, 1)  # TODO(daniel) remove?
    return (box_id, box_size)

  def add_cable_ring(self, env):
    """Make the cable beads coincide with the vertices of the top ring.

    Should lead to better physics and will make it easy for an algorithm
    to see the bag's top ring. Notable differences between this and the
    cables: (1) we don't need to discretize rotations and manually
    compute bead positions, because the previously created bag does it
    for us, (2) Beads have anchors with vertices, in ADDITION to
    constraints with adjacent beads.

    Args:
      env: A ravens environment.
    """
    num_parts = len(self._top_ring_idxs)
    radius = 0.005
    color = U.COLORS['blue'] + [1]
    beads = []
    bead_positions_l = []
    part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
    part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)

    # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
    _, verts_l = p.getMeshData(self.bag_id)

    # Iterate through parts and create constraints as needed.
    for i in range(num_parts):
      bag_vidx = self._top_ring_idxs[i]
      bead_position = np.float32(verts_l[bag_vidx])
      part_id = p.createMultiBody(
          0.01, part_shape, part_visual, basePosition=bead_position)
      p.changeVisualShape(part_id, -1, rgbaColor=color)

      if i > 0:
        parent_frame = bead_position - bead_positions_l[-1]
        constraint_id = p.createConstraint(
            parentBodyUniqueId=beads[-1],
            parentLinkIndex=-1,
            childBodyUniqueId=part_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=(0, 0, 0),
            parentFramePosition=parent_frame,
            childFramePosition=(0, 0, 0))
        p.changeConstraint(constraint_id, maxForce=100)

      # Make a constraint with i=0. Careful with `parent_frame`!
      if i == num_parts - 1:
        parent_frame = bead_positions_l[0] - bead_position
        constraint_id = p.createConstraint(
            parentBodyUniqueId=part_id,
            parentLinkIndex=-1,
            childBodyUniqueId=beads[0],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=(0, 0, 0),
            parentFramePosition=parent_frame,
            childFramePosition=(0, 0, 0))
        p.changeConstraint(constraint_id, maxForce=100)

      # Create constraint between a bead and certain bag vertices.
      _ = p.createSoftBodyAnchor(
          softBodyBodyUniqueId=self.bag_id,
          nodeIndex=bag_vidx,
          bodyUniqueId=part_id,
          linkIndex=-1)

      # Track beads.
      beads.append(part_id)
      bead_positions_l.append(bead_position)

      # Add objects in a consistent manner.
      self.cable_bead_ids.append(part_id)
      env.objects.append(part_id)
      self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

  def add_bag(self, env, base_pos, base_orn):  # pylint: disable=g-doc-args
    """Adding a bag from an .obj file.

    Since this is a soft body, need to add this ID to `self.def_ids.

    Returns:
      bullet object id for the bag.
    """
    bag_id = p.loadSoftBody(
        fileName=self._f_bag,
        basePosition=base_pos,
        baseOrientation=base_orn,
        collisionMargin=self._collision_margin,
        scale=self._bag_scale,
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
    self.object_points[bag_id] = np.float32((0, 0, 0)).reshape(3, 1)
    env.objects.append(bag_id)
    self.def_ids.append(bag_id)
    return bag_id

  def _sample_bag_orientation(self):
    """Sample the bag (and let it drop) to get interesting starting states."""
    orn = [
        self._base_orn[0] + np.random.normal(loc=0.0, scale=self._scale),
        self._base_orn[1] + np.random.normal(loc=0.0, scale=self._scale),
        self._base_orn[2] + np.random.normal(loc=0.0, scale=self._scale),
    ]
    return p.getQuaternionFromEuler(orn)

  @property
  def circle_area(self):
    return self._circle_area

  @property
  def area_thresh(self):
    """Testing with bag-alone-open, similar to cable-ring, slightly lower?"""
    return 0.70

  @property
  def circle_target_positions(self):
    return self._target_positions

  @property
  def circle_target_center(self):
    return self._circle_center

  @property
  def top_ring_idxs(self):
    return self._top_ring_idxs

  @property
  def def_threshold(self):
    return self._def_threshold

  @property
  def def_nb_anchors(self):
    return self._def_nb_anchors

  def understand_bag_top_ring(self, env, base_pos):  # pylint: disable=g-doc-args
    """By our circular bag design, there exists a top ring file.

    Reading it gives us several important pieces of information. We assign
    to:

        _top_ring_idxs: indices of the vertices (out of entire bag).
        _top_ring_posi: their starting xyz positions (BEFORE simulation
            or applying pose transformations). This way we can get the
            area of the circle. We can't take the rotated bag and map
            vertices to the xy plane, because any rotation will make the
            area artificially smaller.

    The .txt file saves in (x,y,z) order but the .obj files put z second.
    Make sure vertex indices are MONOTONICALLY INCREASING since I use
    that assumption to 'assign' vertex indices in order to targets.

    Input: base_pos, the center of the bag's sphere.
    """
    self._top_ring_f = (self._f_bag).replace('.obj', '_top_ring.txt')
    self._top_ring_f = os.path.join('ravens', self._top_ring_f)
    self._top_ring_idxs = []  # is this the same as p.getMeshData?
    self._top_ring_posi = []  # for raw, non-scaled bag
    with open(self._top_ring_f, 'r') as fh:
      for line in fh:
        ls = (line.rstrip()).split()
        vidx = int(ls[0])
        vx, vy, vz = float(ls[1]), float(ls[2]), float(ls[3])
        if len(self._top_ring_idxs) >= 1:
          assert vidx > self._top_ring_idxs[-1], \
                  f'Wrong: {vidx} vs {self._top_ring_idxs}'
        self._top_ring_idxs.append(vidx)
        self._top_ring_posi.append((vx, vy, vz))

    # Next, define a target zone. This makes a bunch of plus signs in a
    # circular fashion from the xy projection of the ring.
    self._target_positions = []
    for item in self._top_ring_posi:
      sx, sy, _ = item
      sx = sx * self._bag_scale + base_pos[0]
      sy = sy * self._bag_scale + base_pos[1]
      self._target_positions.append((sx, sy, 0))
      if self._targets_visible:
        square_pose = ((sx, sy, 0.001), (0, 0, 0, 1))
        square_template = 'assets/square/square-template-allsides-green.urdf'
        replace = {'DIM': (0.004,), 'HALF': (0.004 / 2,)}
        urdf = self.fill_template(square_template, replace)
        env.add_object(urdf, square_pose, fixed=True)
        os.remove(urdf)

    # Fit a circle and print some statistics, can be used by demonstrator.
    # We should be careful to consider nonplanar cases, etc.
    xc, yc, rad, _ = U.fit_circle(
        self._top_ring_posi, self._bag_scale, debug=False)
    self._circle_area = np.pi * (rad**2)
    self._circle_center = (xc * self._bag_scale + base_pos[0],
                           yc * self._bag_scale + base_pos[1])

  def _apply_small_force(self, num_iters, fx=10, fy=10, fz=8, debug=False):
    """A small force to perturb the starting bag."""
    bead_idx = np.random.randint(len(self.cable_bead_ids))
    bead_id = self.cable_bead_ids[bead_idx]
    fx = np.random.randint(low=-fx, high=fx + 1)
    fy = np.random.randint(low=-fy, high=fy + 1)
    for _ in range(num_iters):
      p.applyExternalForce(
          bead_id,
          linkIndex=-1,
          forceObj=[fx, fy, fz],
          posObj=[0, 0, 0],
          flags=p.LINK_FRAME)
      if debug:
        print(f'Perturbing {bead_id}: [{fx:0.2f}, {fy:0.2f}, {fz:0.2f}]')


class BagAloneOpen(BagEnv):
  """Given a single perturbed bag, the objective is to open it.

  This task is similar to cable-ring wrt bead opening. The zone is created
  and then deleted, so that the bag color is the same as in
  bag-items-{easy,hard}, at least for PyBullet 2.8.4.
  """

  def __init__(self):
    super().__init__()
    self.max_steps = 11
    self.metric = 'bag-alone-open'
    self._name = 'bag-alone-open'
    self._settle_secs = 5
    self._targets_visible = False

    # Make the scale small as it's just to make the bag a certain color.
    self._zone_scale = 0.004

    # Higher means more forces applied to the bag.
    self.num_force_iters = 12

    # Parameters for pick_place primitive. Setting prepick_z to be <0.3
    # because it takes a while to check against gripping deformables.
    self.primitive_params = {
        1: {
            'speed': 0.003,
            'delta_z': -0.001,
            'prepick_z': 0.10,
            'postpick_z': 0.05,
            'preplace_z': 0.05,
            'pause_place': 0.5,
        },
    }
    self.task_stage = 1

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.t = 0
    self.task_stage = 1
    self.cable_bead_ids = []
    self.def_ids = []

    # Add square target zone only to get the bag a certain color.
    self.add_zone(env)

    # Pose of the bag, sample mid-air to let it drop naturally.
    bpos, _ = self.random_pose(env, self._bag_size)
    self.base_pos = [bpos[0], bpos[1], self._drop_height]
    self.base_orn = self._sample_bag_orientation()

    # Add the bag, load info about top ring, and make a cable.
    self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
    self.understand_bag_top_ring(env, self.base_pos)
    self.add_cable_ring(env)

    # Env must begin before we can apply forces to perturb the bag.
    env.start()
    self._apply_small_force(num_iters=self.num_force_iters)

    # Remove the zone ID -- only had this to make the bag color the same.
    p.removeBody(self.zone_id)

    time.sleep(self._settle_secs)
    env.pause()


class BagItemsEasy(BagEnv):
  """Like BagAlone except we add other stuff.

  Right now I'm trying to make the demonstrator follow one of three stages,
  where the stages are bag opening, item insertion, and bag moving. For a
  consistant API among other 'bag-items' environments, please put all items
  to be inserted in `self.item_ids[]` and use `self.items_in_bag_ids` to
  track those IDs which are already inserted (or at least, which the
  demonstrator thinks is inserted).
  """

  def __init__(self):
    super().__init__()
    self.max_steps = 11
    self.metric = 'bag-items'
    self._name = 'bag-items-easy'
    self._settle_secs = 5
    self._targets_visible = False

    # Can make this smaller compared to bag-items-alone.
    self.num_force_iters = 8

    # Extra items, in addition to the bag.
    self._nb_items = 1

    # Env reference so we can call Task.get_object_masks(env)
    self.env = None

    # Parameters for pick_place primitive, which is task dependent.
    # stage 1: bag opening. [Copying params from bag-alone-open]
    # stage 2: item insertion.
    # stage 3: bag pulling.
    self.primitive_params = {
        1: {
            'speed': 0.003,
            'delta_z': -0.001,
            'prepick_z': 0.10,
            'postpick_z': 0.05,
            'preplace_z': 0.05,
            'pause_place': 0.5,
        },
        2: {
            'speed': 0.010,
            'delta_z': -0.001,
            'prepick_z': 0.10,  # hopefully makes it faster
            'postpick_z': 0.30,
            'preplace_z': 0.30,
            'pause_place': 0.0,
        },
        3: {
            'speed': 0.002,  # Will this slow bag movement?
            'delta_z': -0.001,
            'prepick_z': 0.08,  # hopefully makes it faster
            'postpick_z': 0.40,
            'preplace_z': 0.40,
            'pause_place': 2.0,
        },
    }
    self.task_stage = 1

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.t = 0
    self.task_stage = 1
    self.cable_bead_ids = []
    self.def_ids = []
    self.env = env

    # New stuff versus bag-alone-open, to better track stats.
    self.item_ids = []
    self.items_in_bag_ids = []
    self.item_sizes = []

    # Add square target zone.
    self.add_zone(env)

    # Pose of the bag, sample mid-air to let it drop naturally.
    bpos, _ = self.random_pose(env, self._bag_size)
    self.base_pos = [bpos[0], bpos[1], self._drop_height]
    self.base_orn = self._sample_bag_orientation()

    # Add the bag, load info about top ring, and make a cable.
    self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
    self.understand_bag_top_ring(env, self.base_pos)
    self.add_cable_ring(env)

    # Add cube(s). The size is straight from the urdf.
    item_size = (0.04, 0.04, 0.04)
    for _ in range(self._nb_items):
      item_pose = self.random_pose(env, item_size)
      item_id = self.add_cube(env, pose=item_pose, global_scaling=1.0)
      self.item_ids.append(item_id)
      self.item_sizes.append(item_size)

    # Env must begin before we can apply forces to perturb the bag.
    env.start()
    self._apply_small_force(num_iters=self.num_force_iters)
    time.sleep(self._settle_secs)
    env.pause()

  # TODO(daniel) clean up method?
  def determine_task_stage(self,
                           colormap=None,
                           heightmap=None,
                           object_mask=None,
                           visible_beads=None):  # pylint: disable=g-doc-args
    """Get the task stage in a consistent manner among different policies.

    When training an oracle policy, we can determine the training stage,
    which is critical because of this task's particular quirks in
    requiring different action parameters (particularly height of the
    pull) for each stage. One option is to use this method to determine
    the hard-coded task stage for each task. This does depend on the
    learned policy inferring when to switch among task stages?

    Returns:
        Tuple, first item is False if the task is almost certainly going
        to fail, and the second provides valid placing pixels (locations)
        if it's relevant to the task stage.
    """
    if self.task_stage == 2 and (len(self.items_in_bag_ids) == len(
        self.item_ids)):
      self.task_stage = 3
      return (True, None)
    elif self.task_stage == 3:
      return (True, None)

    # Hand-tuned, seems reasonable to use.
    buf = 0.025

    # Check object_mask for all IDs that correspond to the cable ring.
    cable_ids = np.array(self.cable_bead_ids)
    bead_mask = np.isin(object_mask, test_elements=cable_ids)

    # Threshold image to get 0s and 255s (255s=bead pixels) and find its
    # contours.
    bead_mask = np.uint8(bead_mask * 255)
    contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # If only a few beads are visible (or no contours detected) exit early.
    frac_visible = len(visible_beads) / len(self.cable_bead_ids)
    if not contours or frac_visible <= BEAD_THRESH:
      return (False, None)

    # Combine contours via concatenation (shape=(N,1,2)) and get the convex
    # hull.
    allc = np.concatenate(list(contours))
    contours_list = [allc]
    hull_list = [cv2.convexHull(c) for c in contours_list]

    # Make an RGB image, then draw the filled-in area of all items in
    # `hull_list`.
    hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(hull, hull_list, -1, (255, 255, 255), thickness=-1)
    hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

    # Following task.random_pose, use object_size to find placing points.
    # Assumes object sizes are same. We add a buffer since convex hulls inflate
    # area.
    object_size = self.item_sizes[0]
    object_size = (object_size[0] + buf, object_size[1] + buf, object_size[2])
    max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
    erode_size = int(np.round(max_size / self.pixel_size))

    # Use cv2.erode to find place pixels on the hull, converted to grayscale.
    place_pixels = np.uint8(hull == 255)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    place_pixels_eroded = cv2.erode(place_pixels, kernel)

    # If on stage 1, and there exists any possible placing point, go to stage 2.
    if self.task_stage == 1:
      if np.sum(place_pixels_eroded) > 0:
        self.task_stage = 2
    return (True, place_pixels_eroded)


class BagItemsHard(BagEnv):
  """The harder version of BagItemsEasy, where items are randomized."""

  def __init__(self):
    super().__init__()
    self.max_steps = 11
    self.metric = 'bag-items'
    self._name = 'bag-items-hard'
    self._settle_secs = 5
    self._targets_visible = False

    # Could make this smaller compared to bag-alone-open?
    self.num_force_iters = 8

    # Extra items, in addition to the bag.
    self._nb_items = 2
    self._max_total_dims = 0.08

    # Env reference so we can call Task.get_object_masks(env)
    self.env = None

    # Exactly the same as BagItemsEasy.
    self.primitive_params = {
        1: {
            'speed': 0.003,
            'delta_z': -0.001,
            'prepick_z': 0.10,
            'postpick_z': 0.05,
            'preplace_z': 0.05,
            'pause_place': 0.5,
        },
        2: {
            'speed': 0.010,
            'delta_z': -0.001,
            'prepick_z': 0.10,  # hopefully makes it faster
            'postpick_z': 0.30,
            'preplace_z': 0.30,
            'pause_place': 0.0,
        },
        3: {
            'speed': 0.002,  # Will this slow bag movement?
            'delta_z': -0.001,
            'prepick_z': 0.08,  # hopefully makes it faster
            'postpick_z': 0.40,
            'preplace_z': 0.40,
            'pause_place': 2.0,
        },
    }
    self.task_stage = 1

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.t = 0
    self.task_stage = 1
    self.cable_bead_ids = []
    self.def_ids = []
    self.env = env

    # New stuff versus bag-alone-open, to better track stats.
    self.item_ids = []
    self.items_in_bag_ids = []
    self.item_sizes = []

    # Add square target zone.
    self.add_zone(env)

    # Pose of the bag, sample mid-air to let it drop naturally.
    bpos, _ = self.random_pose(env, self._bag_size)
    self.base_pos = [bpos[0], bpos[1], self._drop_height]
    self.base_orn = self._sample_bag_orientation()

    # Add the bag, load info about top ring, and make a cable.
    self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
    self.understand_bag_top_ring(env, self.base_pos)
    self.add_cable_ring(env)

    # Add randomly-shaped boxes.
    for _ in range(self._nb_items):
      box_id, box_size = self.add_random_box(env, self._max_total_dims)
      self.item_ids.append(box_id)
      self.item_sizes.append(box_size)

    # Env must begin before we can apply forces to perturb the bag.
    env.start()
    self._apply_small_force(num_iters=self.num_force_iters)
    time.sleep(self._settle_secs)
    env.pause()

  # TODO(daniel) clean up method?
  def determine_task_stage(self,
                           colormap=None,
                           heightmap=None,
                           object_mask=None,
                           visible_beads=None):  # pylint: disable=g-doc-args
    """Get the task stage in a consistent manner among different policies.

    Similar (but not quite the same) as in bag-items-easy.

    Returns:
      Tuple, first item is False if the task is almost certainly going
        to fail, and the second provides valid placing pixels (locations)
        if it's relevant to the task stage.
    """
    if self.task_stage == 2 and (len(self.items_in_bag_ids) == len(
        self.item_ids)):
      self.task_stage = 3
      return (True, None)
    elif self.task_stage == 3:
      return (True, None)

    # Hand-tuned, if too small the agent won't open the bag enough ...
    buf = 0.025

    # But we can decrease it if we're on task stage 2 and have to put in more
    # items.
    if self.task_stage == 2:
      buf = 0.015

    # Check object_mask for all IDs that correspond to the cable ring.
    cable_ids = np.array(self.cable_bead_ids)
    bead_mask = np.isin(object_mask, test_elements=cable_ids)

    # Threshold image to get 0s and 255s (255s=bead pixels) and find its
    # contours.
    bead_mask = np.uint8(bead_mask * 255)
    contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # If only a few beads are visible (or no contours detected) exit early.
    frac_visible = len(visible_beads) / len(self.cable_bead_ids)
    if not contours or frac_visible <= BEAD_THRESH:
      return (False, None)

    # Combine contours via concatenation (shape=(N,1,2)) and get the convex
    # hull.
    allc = np.concatenate(list(contours))
    contours_list = [allc]
    hull_list = [cv2.convexHull(c) for c in contours_list]

    # Make an RGB image, then draw the filled-in area of all items in
    # `hull_list`.
    hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(hull, hull_list, -1, (255, 255, 255), thickness=-1)
    hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

    # Following task.random_pose, use object_size to find placing points.
    # Assumes object sizes are same. We add a buffer since convex hulls inflate
    # area.
    object_size = self.item_sizes[0]
    object_size = (object_size[0] + buf, object_size[1] + buf, object_size[2])
    max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
    erode_size = int(np.round(max_size / self.pixel_size))

    if self.task_stage == 2 and self.items_in_bag_ids:
      # For the hard bag-items version, get array of 0s = items in bag (hence,
      # invalid placing points) and 1s = all other points (could be valid).
      pixels_bag_items = np.ones((hull.shape[0], hull.shape[1]), dtype=np.uint8)
      for item_id in self.items_in_bag_ids:
        item_pix = np.uint8(item_id == object_mask)
        pixels_bag_items = pixels_bag_items & item_pix  # Logical AND
      pixels_no_bag_items = np.uint8(1 - pixels_bag_items)
    else:
      # Make it all 1s so it's safe to apply logical AND with hull pixels.
      pixels_no_bag_items = np.ones((hull.shape[0], hull.shape[1]),
                                    dtype=np.uint8)

    # Combine the hull and pixel conditions.
    place_pixels_hull = np.uint8(hull == 255)
    place_pixels = place_pixels_hull & pixels_no_bag_items

    # Use cv2.erode to find valid place pixels.
    kernel = np.ones((erode_size, erode_size), np.uint8)
    place_pixels_eroded = cv2.erode(place_pixels, kernel)

    # If we're in task stage 2 and there's nothing, let's revert back to
    # original.
    if self.task_stage == 2 and np.sum(place_pixels_eroded) == 0:
      place_pixels_eroded = cv2.erode(place_pixels_hull, kernel)

    # Keep this debugging code to make it easier to inspect.
    if False:  # pylint: disable=using-constant-test
      heightmap = heightmap / np.max(heightmap) * 255
      place_rgb = cv2.cvtColor(hull.copy(), cv2.COLOR_GRAY2BGR)
      place_rgb[place_pixels_eroded > 0] = 127  # gray
      print(f'max_size: {max_size:0.3f}, erode_size: {erode_size}')
      print(f'number of pixels for placing: {np.sum(place_pixels)}')
      print(
          f'number of pixels for placing (after eroding): {np.sum(place_pixels_eroded)}'
      )
      nb = len([x for x in os.listdir('tmp/') if 'color' in x and '.png' in x])
      cv2.imwrite(f'tmp/img_{nb}_colormap.png',
                  cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR).astype(np.uint8))
      cv2.imwrite(f'tmp/img_{nb}_heightmap.png', heightmap.astype(np.uint8))
      cv2.imwrite(f'tmp/img_{nb}_bead_mask.png', bead_mask)
      cv2.imwrite(f'tmp/img_{nb}_place_rgb.png',
                  cv2.cvtColor(place_rgb, cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'tmp/img_{nb}_place_pixels_eroded.png',
                  np.uint8(place_pixels_eroded * 255))
      if self.task_stage == 2 and self.items_in_bag_ids:
        pixels_no_bag_items *= 255
        cv2.imwrite(f'tmp/img_{nb}_pixels_no_bag_items.png',
                    np.uint8(pixels_no_bag_items))

    # If on stage 1, and there exists any possible placing point, go to stage 2.
    if self.task_stage == 1:
      if np.sum(place_pixels_eroded) > 0:
        self.task_stage = 2
    return (True, place_pixels_eroded)
