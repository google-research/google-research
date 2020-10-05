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

"""Classes to handle gripper dynamics."""

import time

import numpy as np
import pybullet as p


class Gripper:
  """Base gripper class."""

  def __init__(self):
    self.activated = False

  def step(self):
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


#-----------------------------------------------------------------------------
# Suction-Based Gripper
#-----------------------------------------------------------------------------


class Suction(Gripper):
  """Simulate simple suction dynamics."""

  def __init__(self, robot_id, tool_link):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      robot_id: int representing PyBullet ID of robot.
      tool_link: int representing PyBullet ID of tool link.
    """
    super().__init__()

    position = (0.487, 0.109, 0.351)
    rotation = p.getQuaternionFromEuler((np.pi, 0, 0))
    urdf = 'assets/ur5/suction/suction-head.urdf'
    self.body = p.loadURDF(urdf, position, rotation)

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=tool_link,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.07))
    p.changeConstraint(constraint_id, maxForce=50)

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self, possible_objects, def_ids):
    """Simulate suction.

    Simulates suction by creating rigid fixed constraint between suction
    gripper and contacted object.

    If gripping a rigid item, record the object ID and distance from
    gripper to object. This way, we can then track if this gripped item
    is touching a deformable. That causes the `distances` to shrink.
    Assumes gripper suctions one rigid item (init_grip_item), at a time.

    Args:
      possible_objects: list of objects involved in suction.
      def_ids: placeholder.
    """
    del def_ids
    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)

      if points:
        # Handle contact between suction with a rigid object.
        for point in points:
          object_id, contact_link = point[2], point[4]
        if object_id in possible_objects:
          body_pose = p.getLinkState(self.body, 0)
          object_pose = p.getBasePositionAndOrientation(object_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          object_to_body = p.multiplyTransforms(world_to_body[0],
                                                world_to_body[1],
                                                object_pose[0], object_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=object_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=object_to_body[0],
              parentFrameOrientation=object_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))
          # Record gripping information!
          distance = np.linalg.norm(
              np.float32(body_pose[0]) - np.float32(object_pose[0]))
          self.init_grip_distance = distance
          self.init_grip_item = object_id
        # print(f'Gripping a rigid item!')

      elif self.def_grip_item is not None:
        # Handle contact between suction and a deformable (soft body).
        info = self.activate_def(self.def_grip_item)
        self.def_grip_anchors = info['anchors']
        self.def_min_vertex = info['closest_vertex']
        self.def_min_distance = info['min_distance']
        # print(f'Gripping a deformable!')

      self.activated = True

  def activate_def(self, def_id):
    """Simulates suction by anchoring vertices of the deformable object.

    Get distance values in `distances`, get indices for argsort, then
    resulting indices in `distances_sort` correspond _exactly_ to vertex
    indices arranged from nearest to furthest to the gripper.

    Args:
      def_id: bullet id for object to anchor.

    Returns:
      info dict containing activation information.
    """
    _, vert_pos_l = p.getMeshData(bodyUniqueId=def_id)
    gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
    distances = []
    for v_position in vert_pos_l:
      d = gripper_position - np.float32(v_position)
      distances.append(np.linalg.norm(d))
    distances_sort = np.argsort(distances)

    anchors = []
    for i in range(self.def_nb_anchors):
      # For each vertex close enough (under threshold), create anchor(s).
      v_index = distances_sort[i]
      if distances[v_index] > self.def_threshold:
        pass
      # This should prevent us from gripping if the suction didn't grip
      # anything.
      if distances[v_index] > self.def_ignore:
        print(f'WARNING, dist={distances[v_index]:0.4f} > thresh '
              f'{self.def_ignore:0.4f}. No points are close to the suction')
        break
      anchor_id = p.createSoftBodyAnchor(
          softBodyBodyUniqueId=def_id,
          nodeIndex=v_index,
          bodyUniqueId=self.body,
          linkIndex=-1,
      )
      anchors.append(anchor_id)

    info = {
        'anchors': anchors,
        'closest_vertex': distances_sort[0],
        'min_distance': np.min(distances),
    }
    return info

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False

      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self, def_ids):
    """Detects a contact for either gripping or releasing purposes.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    After checking if contact with a rigid item, then proceed to checking
    for contacts with deformable. TODO(daniel) see if we can cache
    computation or speed it up in some way, however my profiling using
    cProfile did not indicate that this was a bottleneck.

    Args:
      def_ids: List of IDs of deformables (if any). It may be
        computationally intensive to keep querying distances, so if we
        have any contact points with rigid objects, we return right away
        without checking for contact with deformables.

    Returns:
      bool indicating contact.
    """
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    # If there's no deformables, return False, there can't be any contact.
    if not def_ids:
      return False

    # If suction off (not activated) and len(points)==0, check contact w/defs.
    if not self.activated:
      gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
      for id_ in def_ids:
        if self.detect_contact_def(gripper_position, id_):
          self.def_grip_item = id_
          return True
      return False

    # Now either (a) gripping a def., or (b) gripping a rigid item that is
    # NOT touching another rigid item, but _may_ be touching a def.
    assert self.activated

    if self.init_grip_item is not None:
      # If suction on, check if a gripped RIGID item touches a deformable.
      # When this happens, the suction 'goes through' its gripped item,
      # and hence the distance between it and the object decreases.
      object_pose = p.getBasePositionAndOrientation(self.init_grip_item)
      gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
      d = gripper_position - np.float32(object_pose[0])
      distance = np.linalg.norm(d)
      fraction = distance / self.init_grip_distance
      if fraction < 0.92 or fraction > 2.0:
        self.init_grip_distance = None
        self.init_grip_item = None
      return fraction < 0.92 or fraction > 2.0

    elif self.def_grip_item is not None:
      # This is for making deformable-deformable contact, but I dont
      # think we ever trigger this condition since something else kicks
      # in first (e.g., height of gripper, or touching rigid).
      # TODO(daniel) confirm.
      # TODO(daniel) I was going to use self.def_min_vertex here I think.
      return False

    # We should always be gripping a rigid or a def., so code shouldn't reach
    # here.
    return False

  def detect_contact_def(self, gripper_position, def_id):
    """Detect contact, when dealing with deformables.

    Args:
      gripper_position: gripper position.
      def_id: Bullet id of deformable.

    Returns:
      bool indicating if there exists _any_ vertex within the distance
        threshold.
    """
    _, vert_pos_l = p.getMeshData(bodyUniqueId=def_id)
    distances = []
    for v_position in vert_pos_l:
      d = gripper_position - np.float32(v_position)
      distances.append(np.linalg.norm(d))
    return np.min(distances) < self.def_threshold

  def check_grasp(self):
    """Check a grasp for picking success.

    If picking fails, then robot doesn't do the place action. For rigid
    items: index 2 in getConstraintInfo returns childBodyUniqueId. For
    deformables, check the length of the anchors.

    Returns:
      bool indicating if contact constraint is not None (gripped a rigid item)
        or if there exists at least one grip anchor (gripped a soft body).
    """

    # TODO(daniel) I left this code untouched but we never use suctioned_object?
    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    del suctioned_object

    pick_deformable = False
    if self.def_grip_anchors is not None:
      pick_deformable = len(self.def_grip_anchors) > 0  # pylint: disable=g-explicit-length-test
    return (self.contact_constraint is not None) or pick_deformable

  def set_def_threshold(self, threshold):
    self.def_threshold = threshold

  def set_def_nb_anchors(self, nb_anchors):
    self.def_nb_anchors = nb_anchors


#-----------------------------------------------------------------------------
# Parallel-Jaw Two-Finger Gripper (TODO: fix)
#-----------------------------------------------------------------------------


class Robotiq2F85:
  """Gripper handling for Robotiq 2F85."""

  def __init__(self, robot, tool):
    self.robot = robot
    self.tool = tool
    pos = [0.487, 0.109, 0.421]
    rot = p.getQuaternionFromEuler([np.pi, 0, 0])
    urdf = 'assets/ur5/gripper/robotiq_2f_85.urdf'
    self.body = p.loadURDF(urdf, pos, rot)
    self.n_joints = p.getNumJoints(self.body)
    self.activated = False

    # Connect gripper base to robot tool
    p.createConstraint(
        self.robot,
        tool,
        self.body,
        0,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, -0.05])

    # Set friction coefficients for gripper fingers
    for i in range(p.getNumJoints(self.body)):
      p.changeDynamics(
          self.body,
          i,
          lateralFriction=1.5,
          spinningFriction=1.0,
          rollingFriction=0.0001,
          # rollingFriction=1.0,
          frictionAnchor=True)  # contactStiffness=0.0, contactDamping=0.0

    # Start thread to handle additional gripper constraints
    self.motor_joint = 1
    # self.constraints_thread = threading.Thread(target=self.step)
    # self.constraints_thread.daemon = True
    # self.constraints_thread.start()

  # Control joint positions by enforcing hard contraints on gripper behavior
  # Set one joint as the open/close motor joint (other joints should mimic)
  def step(self):
    # while True:
    currj = [p.getJointState(self.body, i)[0] for i in range(self.n_joints)]
    indj = [6, 3, 8, 5, 10]
    targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
    p.setJointMotorControlArray(
        self.body, indj, p.POSITION_CONTROL, targj, positionGains=np.ones(5))
    # time.sleep(0.001)

  # Close gripper fingers and check grasp success (width between fingers
  # exceeds some threshold)
  def activate(self, valid_obj=None):
    """Simulate suction."""
    del valid_obj
    p.setJointMotorControl2(
        self.body,
        self.motor_joint,
        p.VELOCITY_CONTROL,
        targetVelocity=1,
        force=100)
    if not self.external_contact():
      while self.moving():
        time.sleep(0.001)
    self.activated = True

  # Open gripper fingers
  def release(self):
    p.setJointMotorControl2(
        self.body,
        self.motor_joint,
        p.VELOCITY_CONTROL,
        targetVelocity=-1,
        force=100)
    while self.moving():
      time.sleep(0.001)
    self.activated = False

  # If activated and object in gripper: check object contact
  # If activated and nothing in gripper: check gripper contact
  # If released: check proximity to surface
  def detect_contact(self):
    obj, _, ray_frac = self.check_proximity()
    if self.activated:
      empty = self.grasp_width() < 0.01
      cbody = self.body if empty else obj
      if obj == self.body or obj == 0:
        return False
      return self.external_contact(cbody)
    else:
      return ray_frac < 0.14 or self.external_contact()

  # Return if body is in contact with something other than gripper
  def external_contact(self, body=None):
    if body is None:
      body = self.body
    pts = p.getContactPoints(bodyA=body)
    pts = [pt for pt in pts if pt[2] != self.body]
    return len(pts) > 0  # pylint: disable=g-explicit-length-test

  # Check grasp success
  def check_grasp(self):
    while self.moving():
      time.sleep(0.001)
    success = self.grasp_width() > 0.01
    return success

  def grasp_width(self):
    lpad = np.array(p.getLinkState(self.body, 4)[0])
    rpad = np.array(p.getLinkState(self.body, 9)[0])
    dist = np.linalg.norm(lpad - rpad) - 0.047813
    return dist

  # Helper functions

  def moving(self):
    v = [
        np.linalg.norm(p.getLinkState(self.body, i, computeLinkVelocity=1)[6])
        for i in [3, 8]
    ]
    return any(np.array(v) > 1e-2)

  def check_proximity(self):
    ee_pos = np.array(p.getLinkState(self.robot, self.tool)[0])
    tool_pos = np.array(p.getLinkState(self.body, 0)[0])
    vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
    ee_targ = ee_pos + vec
    ray_data = p.rayTest(ee_pos, ee_targ)[0]
    obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
    return obj, link, ray_frac
