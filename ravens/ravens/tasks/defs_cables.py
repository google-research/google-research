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

"""A set of tasks that use cables.

Design philosophy is to have a generic CableEnv with common functionality,
and then make new tasks subclass CableEnv with their specific versions of
inits and resets.
"""
import os
import time
import numpy as np
import pybullet as p
from ravens import utils as U
from ravens.tasks import Task


class CableEnv(Task):
  """Superclass for tasks that use a cable."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.primitive = 'pick_place'
    self.max_steps = 11
    self._settle_secs = 2
    self._name = None

    # Scaling the zone as needed.
    self.zone_scale = 0.01
    self.zone_length = (20.0 * self.zone_scale)
    self.zone_size = (20.0 * self.zone_scale, 20.0 * self.zone_scale, 0)

    # Target zone and debug marker visibility. Set both False for goal-based
    # tasks.
    self.target_zone_visible = True
    self.target_debug_markers = False

    # Cable-related parameters, can override in subclass.
    self.num_parts = 24
    self.radius = 0.005
    self.length = 2 * self.radius * self.num_parts * np.sqrt(2)
    self.color_bead = U.COLORS['blue'] + [1]
    self.color_end = U.COLORS['yellow'] + [1]

    # Put cable bead IDs here, so we don't count non cable IDs for targets.
    self.cable_bead_ids = []

  def add_zone(self, env):
    """Adds a green target zone."""
    zone_template = 'assets/zone/zone-template.urdf'
    replace = {'LENGTH': (self.zone_scale, self.zone_scale)}
    zone_urdf = self.fill_template(zone_template, replace)
    self.zone_pose = self.random_pose(env, self.zone_size)
    zone_id = env.add_object(zone_urdf, self.zone_pose, fixed=True)
    os.remove(zone_urdf)
    return zone_id

  def add_cable(self, env, size_range, info):
    """Add a cable to the env, consisting of rigids beads.

    Add each bead ID to (a) env.objects, (b) object_points, and (c)
    cable_bead_ids. Use (b) because the demonstrator checks it to pick
    the bead farthest from a goal, and it is also used to tally up beads
    within the zone (to compute reward). Use (c) to distinguish between
    bead vs non-bead objects in case we add other items.

    Args:
      env: A ravens environment.
      size_range: Used to indicate the area of the target, so the beads
        avoid spawning there.
      info: Stores relevant stuff, such as for ground-truth targets.
    """
    num_parts = self.num_parts
    radius = self.radius
    length = self.length

    # Add beaded cable.
    distance = length / num_parts
    position, _ = self.random_pose(env, size_range)
    position = np.float32(position)
    part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
    part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)

    # Iterate through parts and create constraints as needed.
    for i in range(num_parts):
      position[2] += distance
      parent_frame = (0, 0, distance)
      part_id = p.createMultiBody(
          0.1, part_shape, part_visual, basePosition=position)
      if i > 0:
        constraint_id = p.createConstraint(
            parentBodyUniqueId=env.objects[-1],
            parentLinkIndex=-1,
            childBodyUniqueId=part_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=(0, 0, 0),
            parentFramePosition=parent_frame,
            childFramePosition=(0, 0, 0))
        p.changeConstraint(constraint_id, maxForce=100)

      # Colors
      if (i > 0) and (i < num_parts - 1):
        p.changeVisualShape(part_id, -1, rgbaColor=self.color_bead)
      elif i == num_parts - 1:
        p.changeVisualShape(part_id, -1, rgbaColor=self.color_end)

      # Add objects in a consistent manner.
      self.cable_bead_ids.append(part_id)
      env.objects.append(part_id)
      self.object_points[part_id] = np.float32(([0], [0], [0]))

      # Get target placing positions for each cable bead, if applicable.
      if (self._name == 'cable-shape' or self._name == 'cable-shape-notarget' or
          self._name == 'cable-line-notarget'):
        # ----------------------------------------------------------- #
        # Here, zone_pose = square_pose, unlike Ravens cable, where the
        # zone_pose is shifted so that its center matches the straight
        # line segment center. For `true_position`, we use `zone_pose`
        # but apply the correct offset to deal with the sides. Note
        # that `length` is the size of a fully smoothed cable, BUT we
        # made a rectangle with each side <= length.
        # ----------------------------------------------------------- #
        lx = info['lengthx']
        ly = info['lengthy']
        r = radius

        if info['nb_sides'] == 1:
          # Here it's just a straight line on the 'lx' side.
          x_coord = lx / 2 - (distance * i)
          y_coord = 0
          true_position = (x_coord - r, y_coord, 0)

        elif info['nb_sides'] == 2:
          # Start from lx side, go 'left' to the pivot point, then on
          # the ly side, go 'upwards' but offset by `i`. For radius
          # offset, I just got this by tuning. XD
          if i < info['cutoff']:
            x_coord = lx / 2 - (distance * i)
            y_coord = -ly / 2
            true_position = (x_coord - r, y_coord, 0)
          else:
            x_coord = -lx / 2
            y_coord = -ly / 2 + (distance * (i - info['cutoff']))
            true_position = (x_coord, y_coord + r, 0)

        elif info['nb_sides'] == 3:
          # Start from positive lx, positive ly, go down to first
          # pivot. Then go left to the second pivot, then up again.
          # For v1, division by two is because we assume BOTH of the
          # 'ly edges' were divided by two.
          v1 = (self.num_parts - info['cutoff']) / 2
          v2 = self.num_parts - v1
          if i < v1:
            x_coord = lx / 2
            y_coord = ly / 2 - (distance * i)
            true_position = (x_coord, y_coord - r, 0)
          elif i < v2:
            x_coord = lx / 2 - (distance * (i - v1))
            y_coord = -ly / 2
            true_position = (x_coord - r, y_coord, 0)
          else:
            x_coord = -lx / 2
            y_coord = -ly / 2 + (distance * (i - v2))
            true_position = (x_coord, y_coord + r, 0)

        elif info['nb_sides'] == 4:
          # I think this is similar to the 2-side case: we start in
          # the same direction and go counter-clockwise.
          v1 = info['cutoff'] / 2
          v2 = num_parts / 2
          v3 = (num_parts + info['cutoff']) / 2
          if i < v1:
            x_coord = lx / 2 - (distance * i)
            y_coord = -ly / 2
            true_position = (x_coord, y_coord, 0)
          elif i < v2:
            x_coord = -lx / 2
            y_coord = -ly / 2 + (distance * (i - v1))
            true_position = (x_coord, y_coord, 0)
          elif i < v3:
            x_coord = -lx / 2 + (distance * (i - v2))
            y_coord = ly / 2
            true_position = (x_coord, y_coord, 0)
          else:
            x_coord = lx / 2
            y_coord = ly / 2 - (distance * (i - v3))
            true_position = (x_coord, y_coord, 0)

        # Map true_position onto the workspace from zone_pose.
        true_position = U.apply(self.zone_pose, true_position)

        # See `cable.py`: just get the places and steps set.
        self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
        symmetry = 0
        self.goal['steps'][0][part_id] = (symmetry, [part_id])

        # Debugging target zones.
        if self.target_debug_markers:
          sq_pose = ((true_position[0], true_position[1], 0.002), (0, 0, 0, 1))
          sq_template = 'assets/square/square-template-allsides-blue.urdf'
          replace = {'DIM': (0.003,), 'HALF': (0.003 / 2,)}
          urdf = self.fill_template(sq_template, replace)
          env.add_object(urdf, sq_pose, fixed=True)
          os.remove(urdf)
      else:
        print(f'Warning, env {self._name} will not have goals.')

  def add_cable_ring(self, env, info):
    """Add a cable, but make it connected at both ends to form a ring.

    For consistency, add each `part_id` to various information tracking
    lists and dictionaries (see `add_cable` documentation).

    Args:
      env: A ravens environment.
      info: Stores relevant stuff, such as for ground-truth targets.
    """

    def rad_to_deg(rad):
      return (rad * 180.0) / np.pi

    def get_discretized_rotations(i, num_rotations):
      # counter-clockwise
      theta = i * (2 * np.pi) / num_rotations
      return (theta, rad_to_deg(theta))

    # Bead properties.
    num_parts = self.num_parts
    radius = self.radius
    color = self.color_bead

    # The `ring_radius` (not the bead radius!) has to be tuned somewhat.
    # Try to make sure the beads don't have notable gaps between them.
    ring_radius = info['ring_radius']
    beads = []
    bead_positions_l = []

    # Add beaded cable. Here, `position` is the circle center.
    position = np.float32(info['center_position'])
    part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
    part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)

    # Iterate through parts and create constraints as needed.
    for i in range(num_parts):
      angle_rad, _ = get_discretized_rotations(i, num_parts)
      px = ring_radius * np.cos(angle_rad)
      py = ring_radius * np.sin(angle_rad)
      bead_position = np.float32([position[0] + px, position[1] + py, 0.01])
      part_id = p.createMultiBody(
          0.1, part_shape, part_visual, basePosition=bead_position)
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

      # Track beads.
      beads.append(part_id)
      bead_positions_l.append(bead_position)

      # Add objects in a consistent manner.
      self.cable_bead_ids.append(part_id)
      env.objects.append(part_id)
      self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

      if self._name == 'cable-ring' or self._name == 'cable-ring-notarget':
        # We assume the starting position gives us the targets.
        true_position = (bead_position[0], bead_position[1], 0)
        self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
        symmetry = 0
        self.goal['steps'][0][part_id] = (symmetry, [part_id])

        # Make the true positions visible if desired.
        if info['targets_visible']:
          sq_pose = ((true_position[0], true_position[1], 0.002), (0, 0, 0, 1))
          sq_template = 'assets/square/square-template-allsides-green.urdf'
          replace = {'DIM': (0.003,), 'HALF': (0.003 / 2,)}
          urdf = self.fill_template(sq_template, replace)
          env.add_object(urdf, sq_pose, fixed=True)
          os.remove(urdf)
      else:
        print(f'Warning, env {self._name} will not have goals.')

  @property
  def circle_area(self):
    """Only applies to cable-ring and cable-ring-notarget."""
    return np.pi * self.ring_radius**2

  @property
  def area_thresh(self):
    """Only applies to cable-ring and cable-ring-notarget.

    Using >= 0.8 might be too hard because moving beads to targets causes
    other beads to move, thus potentially decreasing the area of the
    convex hull of the beads. 0.75 strikes a reasonable balance.
    """
    return 0.75


class CableShape(CableEnv):
  """A single cable, and manipulating to a complex target.

  Application inspiration: moving a cable towards a target is commonly done
  in cases such as knot-tying and rearranging stuff on a surface, and more
  generally it's a common robotics benchmark.

  For now we are using targets based on line segments stacked with each
  other. This means we have to change the normal zone metric because it
  assumes a linear target, but shouldn't be too difficult. Also, because
  this involves just a simple cable, we are going to use the same
  pick_place demonstrator.

  Remember that the UR5 is at zone (0,0,0) and the 'square' is like this:

    |     |   xxxx
    |  o  |   xxxx
    |     |   xxxx
    -------

  where `o` is the center of the robot, and `x`'s represent the workspace
  (horizontal axis is x). Then the 'line' has to fill in the top part of
  the square. Each edge has length `length` in code. We generalize this for
  targets of between 1 and 4 connected line segments. Use `length_x` and
  `length_y` to determine the lengths of the sides. With two sides, we get:

    |         xxxx
    |  o      xxxx  length_y
    |         xxxx
    -------
    length_x

  where `length_x + length_y = length`, or one of the original square
  sides. Thus, the square essentially defines where the target can be
  sampled. Also keep in mind that we actually use a rectangle, not a
  square; the square_pose is just used for a pose and sampling bounds.
  """

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.max_steps = 21
    self.metric = 'cable-target'
    self.primitive = 'pick_place'
    self._name = 'cable-shape'

    # Target zone and the debug marker visibility.
    self.target_zone_visible = True
    self.target_debug_markers = False

    # Cable parameters.
    self.num_parts = 24
    self.radius = 0.005
    self.length = 2 * self.radius * self.num_parts * np.sqrt(2)
    self.num_sides_low = 2
    self.num_sides_high = 4

    # Parameters for pick_place primitive.
    self.primitive_params = {
        1: {
            'speed': 0.001,
            'delta_z': -0.001,
            'postpick_z': 0.04,
            'preplace_z': 0.04,
            'pause_place': 0.0,
        },
    }
    self.task_stage = 1

    # To see if performance varies as a function of the number of sides.
    self.nb_sides = None

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.task_stage = 1
    self.cable_bead_ids = []

    # Use this for the built-in pick_place demonstrator in `task.py`.
    self.goal = {'places': {}, 'steps': [{}]}

    # Sample the 'square pose' which is the center of a rectangle.
    square_size = (self.length, self.length, 0)
    square_pose = self.random_pose(env, square_size)
    assert square_pose is not None, 'Cannot sample a pose.'

    # Be careful. We deduce ground-truth pose labels from zone_pose.
    self.zone_pose = square_pose
    zone_range = (self.length / 10, self.length / 10, 0)

    # Sample the number of sides to preserve from the rectangle.
    low, high = self.num_sides_low, self.num_sides_high
    self.nb_sides = nb_sides = np.random.randint(
        low=low, high=high + 1)  # note +1
    template = f'assets/rectangle/rectangle-template-sides-{nb_sides}.urdf'

    if nb_sides == 1:
      # One segment target: a straight cable should be of length `length`.
      lengthx = self.length
      lengthy = 0
      cutoff = 0
    elif nb_sides == 2:
      # Two segment target: length1 + length2 should equal a straight cable.
      cutoff = np.random.randint(0, self.num_parts + 1)
      alpha = cutoff / self.num_parts
      lengthx = self.length * alpha
      lengthy = self.length * (1 - alpha)
    elif nb_sides == 3:
      # Three segment target: remove length1, but need to remove a bit more.
      offset = 4  # avoid 'extremes'
      cutoff = np.random.randint(offset, self.num_parts + 1 - offset)
      alpha = cutoff / self.num_parts
      lengthx = self.length * alpha
      lengthy = (self.length * (1 - alpha)) / 2
    elif nb_sides == 4:
      # Four segment target, divide by two to make the cable 'fit'.
      offset = 4  # avoid 'extremes'
      cutoff = np.random.randint(offset, self.num_parts + 1 - offset)
      alpha = cutoff / self.num_parts
      lengthx = (self.length * alpha) / 2
      lengthy = (self.length * (1 - alpha)) / 2

    # I deduced DIM & HALF from rectangle template through trial & error.
    dim = (lengthx, lengthy)
    half = (dim[1] / 2, dim[0] / 2)
    if self.target_zone_visible:
      replace = {'DIM': dim, 'HALF': half}
      urdf = self.fill_template(template, replace)
      env.add_object(urdf, square_pose, fixed=True)
      os.remove(urdf)

    # Add cable.
    info = {
        'nb_sides': nb_sides,
        'cutoff': cutoff,
        'lengthx': lengthx,
        'lengthy': lengthy,
        'DIM': dim,
        'HALF': half,
    }
    self.add_cable(env, size_range=zone_range, info=info)

    env.start()
    time.sleep(self._settle_secs)
    env.pause()


class CableShapeNoTarget(CableShape):
  """CableShape, but without a target, so we need a goal image."""

  def __init__(self):
    super().__init__()
    self._name = 'cable-shape-notarget'

    # Target zone and the debug marker visibility.
    self.target_zone_visible = False
    self.target_debug_markers = False

  def reset(self, env, last_info=None):
    """Reset to start an episode.

    If generating training data for goal-conditioned Transporters with
    `main.py` or goal images using `generate_goals.py`, then call the
    superclass. The code already puts the bead poses inside `info`. For
    this env it's IDs 4 through 27 (for 24 beads) but I scale it based on
    num_parts in case we change this value.

    If loading using `load.py` (detect with self.goal_cond_testing) then
    must make targets based on loaded info. However, we still have to
    randomly create the cable, so the easiest way might be to make the
    cable as usual, and then just override the 'places' key later.

    Args:
      env: A ravens environment.
      last_info: Last info dictionary.

    Returns:
      places:
    """
    super().reset(env)
    if self.goal_cond_testing:
      assert last_info is not None
      self.goal['places'] = self._get_goal_info(last_info)

  def _get_goal_info(self, last_info):
    """Used to determine the goal given the last `info` dict."""
    start_id = 4
    end_id = start_id + self.num_parts
    places = {}
    for id_ in range(start_id, end_id):
      assert id_ in last_info, f'something went wrong with ID={id_}'
      position, _, _ = last_info[id_]
      places[id_] = (position, (0, 0, 0, 1.))
    return places


class CableLineNoTarget(CableShape):
  """Like CableShapeNoTarget, but only straight lines (no visible targets)."""

  def __init__(self):
    super().__init__()
    self._name = 'cable-line-notarget'

    # Major change, only considering straight lines.
    self.num_sides_low = 1
    self.num_sides_high = 1

    # Target zone and the debug marker visibility.
    self.target_zone_visible = False
    self.target_debug_markers = False

  def reset(self, env, last_info=None):
    """See `CableShapeNoTarget.reset()`."""
    super().reset(env)
    if self.goal_cond_testing:
      assert last_info is not None
      self.goal['places'] = self._get_goal_info(last_info)

  def _get_goal_info(self, last_info):
    """See `CableShapeNoTarget._get_goal_info()`."""
    start_id = 4
    end_id = start_id + self.num_parts
    places = {}
    for id_ in range(start_id, end_id):
      assert id_ in last_info, f'something went wrong with ID={id_}'
      position, _, _ = last_info[id_]
      places[id_] = (position, (0, 0, 0, 1.))
    return places


class CableRing(CableEnv):
  """Cable as a ring.

  This differs from CableShape in that (1) the cable is a ring and
  continuously connected, and (2) the target is also a ring.

  We need good parameters for num_parts, radius, and ring_radius. So far I
  like these combinations: (24, 0.005, 0.06), (32, 0.005, 0.075), (36,
  0.005, 0.09)... using 32 parts to the bead is ideal, given that it's the
  same number as what the bag uses. The postpick and preplace should be
  just high enough to let one layer of cables go above another, to avoid
  the demonstrator engaging in back-and-forth actions.
  """

  def __init__(self):
    super().__init__()
    self.metric = 'cable-ring'
    self.max_steps = 21
    self.primitive = 'pick_place'
    self._name = 'cable-ring'

    # Cable parameters. We use ring_radius to determine sampling bounds.
    self.num_parts = 32
    self.radius = 0.005
    self.ring_radius = 0.075
    self.targets_visible = True

    # Parameters for pick_place primitive.
    self.primitive_params = {
        1: {
            'speed': 0.001,
            'delta_z': -0.001,
            'postpick_z': 0.04,
            'preplace_z': 0.04,
            'pause_place': 0.0,
        },
    }
    self.task_stage = 1

  def reset(self, env):
    self.total_rewards = 0
    self.object_points = {}
    self.task_stage = 1
    self.cable_bead_ids = []

    # We need this to use the built-in pick_place demonstrator in `task.py`.
    self.goal = {'places': {}, 'steps': [{}]}

    # Sample the center of the ring, increasing size to allow for random force.
    boundary_size = (self.ring_radius * 3, self.ring_radius * 3, 0)
    boundary_pose = self.random_pose(env, boundary_size)
    self.zone_pose = (boundary_pose[0], (0, 0, 0, 1))

    # Add cable ring.
    info = {
        'center_position': self.zone_pose[0],
        'ring_radius': self.ring_radius,
        'targets_visible': self.targets_visible,
    }
    self.add_cable_ring(env, info=info)

    # Env must begin before we can apply forces.
    env.start()

    # Add a small force to perturb the cable. Pick a bead at random.
    bead_idx = np.random.randint(len(self.cable_bead_ids))
    bead_id = self.cable_bead_ids[bead_idx]
    fx = np.random.randint(low=-20, high=20 + 1)
    fy = np.random.randint(low=-20, high=20 + 1)
    fz = 40
    for _ in range(20):
      p.applyExternalForce(
          bead_id,
          linkIndex=-1,
          forceObj=[fx, fy, fz],
          posObj=[0, 0, 0],
          flags=p.LINK_FRAME)

    time.sleep(self._settle_secs)
    env.pause()


class CableRingNoTarget(CableRing):
  """Cable as a ring, but no target, so it subclasses CableRing."""

  def __init__(self):
    super().__init__()
    self._name = 'cable-ring-notarget'
    self.targets_visible = False

  def reset(self, env):  # pylint: disable=useless-super-delegation
    super().reset(env)
