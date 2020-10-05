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

"""Base environment class."""

import os
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data  # pylint: disable=unused-import

from ravens import tasks
from ravens import utils
from ravens.gripper import Gripper
from ravens.gripper import Suction

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005


class Environment():
  """Base environment class."""

  def __init__(self, disp=False, hz=240):
    """Creates OpenAI gym-style env with support for PyBullet threading.

    Args:
      disp: Whether or not to use PyBullet's built-in display viewer. Use
        this either for local inspection of PyBullet, or when using any
        soft body (cloth or bags), because PyBullet's TinyRenderer
        graphics (used if disp=False) will make soft bodies invisible.
      hz: Parameter used in PyBullet to control the number of physics
        simulation steps. Higher values lead to more accurate physics at
        the cost of slower computaiton time. By default, PyBullet uses
        240, but for soft bodies we need this to be at least 480 to avoid
        cloth intersecting with the plane.
    """
    self.ee = None
    self.task = None
    self.objects = []
    self.running = False
    self.fixed_objects = []
    self.pix_size = 0.003125
    self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
    self.primitives = {
        'push': self.push,
        'sweep': self.sweep,
        'pick_place': self.pick_place,
        'pick_place_6dof': self.pick_place_6dof
    }

    # Set default movej timeout limit. For most tasks, 15 is reasonable.
    self.t_lim = 15

    # Need these settings for simulating cloth or bags.
    self.use_new_deformable = True
    self.hz = hz

    # Start PyBullet.
    p.connect(p.GUI if disp else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    assets_path = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(assets_path)

    # If using --disp, move default camera a little closer to the scene.
    if disp:
      _, _, _, _, _, _, _, _, _, _, _, target = p.getDebugVisualizerCamera()
      p.resetDebugVisualizerCamera(
          cameraDistance=1.0,
          cameraYaw=90,
          cameraPitch=-25,
          cameraTargetPosition=target,
      )

    # Control PyBullet simulation steps.
    self.step_thread = threading.Thread(target=self.step_simulation)
    self.step_thread.daemon = True
    self.step_thread.start()

  def step_simulation(self):
    """Controls PyBullet simulation stepping.

    From Erwin: the default value for p.setTimeStep is 1/240, and the
    time.sleep(0.001) call is mainly for us to visualize in the GUI.
    """
    assert self.hz > 0
    p.setTimeStep(1.0 / self.hz)
    while True:
      if self.running:
        p.stepSimulation()
      if self.ee is not None:
        self.ee.step()
      time.sleep(0.001)

  def stop(self):
    p.disconnect()
    del self.step_thread

  def start(self):
    self.running = True

  def pause(self):
    self.running = False

  def is_static(self):
    """Checks if env is static, used for checking if action finished.

    However, this won't quite work in PyBullet (at least v2.8.4) since
    soft bodies will make this extend cloth objects cause this code to
    hand. Therefore, to get around this, look at the task's `def_ids`
    list, which by design will have all IDs of soft bodies. Furthermore,
    for the bag tasks, the beads generally move around, so for those,
    just use a hard cutoff limit (outside this method).

    Returns:
      bool if environment is static.
    """
    if self.is_softbody_env():
      assert self.task.def_ids, 'Did we forget to add to def_ids?'
      v = [
          np.linalg.norm(p.getBaseVelocity(i)[0])
          for i in self.objects
          if i not in self.task.def_ids
      ]
    else:
      v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.objects]
    return all(np.array(v) < 1e-2)

  def add_object(self, urdf, pose, fixed=False):
    fixed_base = 1 if fixed else 0
    object_id = p.loadURDF(urdf, pose[0], pose[1], useFixedBase=fixed_base)
    if fixed:
      self.fixed_objects.append(object_id)
    else:
      self.objects.append(object_id)
    return object_id

  #-------------------------------------------------------------------------
  # Standard Gym Functions
  #-------------------------------------------------------------------------

  def reset(self, task, last_info=None, disable_render_load=True):
    """Performs common reset functionality for all supported tasks.

    Args:
      task: a standard or deformable simulated robotics setting, where the
        type is subclass of `ravens.tasks.Task`.
      last_info: For goal-conditioned learning during TEST time. Our
        dataset.py provides a target image, but we need final object poses
        as well, if we want to provide pose-based targets (usually a good
        idea). We can put all this in an `info` dict, and load it in a
        task.reset() call.
      disable_render_load: Need this as True to avoid `p.loadURDF`
        becoming a time bottleneck, judging from my profiling.

    Returns:
      obs: ew obs after reset.
    """
    self.pause()
    self.task = task
    self.objects = []
    self.fixed_objects = []
    if self.use_new_deformable:
      p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    else:
      p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    # Slightly increase default movej timeout for the more demanding tasks.
    if (isinstance(self.task, tasks.names['bag-items-easy']) or
        isinstance(self.task, tasks.names['bag-items-hard'])):
      self.t_lim = 25

    # Empirically, this makes loading URDFs faster w/remote displays.
    if disable_render_load:
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])
    p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])

    # Load UR5 robot arm equipped with task-specific end effector.
    self.ur5 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf')
    self.ee_tip_link = 12
    if self.task.ee == 'suction':
      self.ee = Suction(self.ur5, 11)
    # elif self.task.ee == 'gripper':
    #   self.ee = Robotiq2F85(self.ur5, 9)
    #   self.ee_tip_link = 10
    else:
      self.ee = Gripper()

    # Get revolute joint indices of robot (skip fixed joints).
    num_joints = p.getNumJoints(self.ur5)
    joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
    self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home joint configuration.
    for i in range(len(self.joints)):
      p.resetJointState(self.ur5, self.joints[i], self.homej[i])

    # Get end effector tip pose in home configuration.
    ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
    self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])

    # Reset end effector.
    self.ee.release()

    # Reset task.
    if last_info is not None:
      task.reset(self, last_info)
    else:
      task.reset(self)

    # Assign deformable params, and check the Hz value.
    if self.is_softbody_env():
      self.ee.set_def_threshold(threshold=self.task.def_threshold)
      self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
      assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

    # Restart simulation.
    self.start()
    if disable_render_load:
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    (obs, _, _, _) = self.step()
    return obs

  def step(self, act=None):
    """Execute action with specified primitive.

    Exit early if action failed at any point along the sequence of
    waypoint movements, or if exiting gracefully. The latter is used for
    bag tasks which can reach irrecoverable states (with low amounts of
    beads visible), so it's faster to exit early. For this add
    task.done=False and other stuff to the 'extras' dict.

    Args:
      act: action dict to execute.

    Returns:
      (obs, reward, done, info) tuple containing MDP step data.
    """
    if act and act['primitive']:
      success = self.primitives[act['primitive']](**act['params'])

      # Exit early if action failed, or if exiting gracefully.
      if (not success) or self.task.exit_gracefully:
        _, reward_extras = self.task.reward()
        info = self.info
        reward_extras['task.done'] = False
        if self.task.exit_gracefully:
          reward_extras['exit_gracefully'] = True
          self.task.exit_gracefully = False

        # Still want info['extras'] for consistency.
        info['extras'] = reward_extras
        return {}, 0, True, info

    # Wait for objects to settle, with a hard exit for bag tasks.
    start_t = time.time()
    while not self.is_static():
      if self.is_bag_env() and (time.time() - start_t > 3.0):
        break
      time.sleep(0.001)

    # Compute task rewards.
    reward, reward_extras = self.task.reward() if act else (0, {})
    done = self.task.done()

    # Pass ground truth robot state as info.
    info = self.info

    # Pass extra stuff in info; task.done detects if succeeded on LAST action.
    reward_extras['task.done'] = done
    info['extras'] = reward_extras
    if isinstance(self.task, tasks.names['cloth-flat-notarget']):
      info['sampled_zone_pose'] = self.task.zone_pose

    # Get camera observations per specified config.
    obs = {}
    if act and 'camera_config' in act:
      obs['color'], obs['depth'] = [], []
      for config in act['camera_config']:
        color, depth, _ = self.render(config)
        obs['color'].append(color)
        obs['depth'].append(depth)

    return obs, reward, done, info

  def render(self, config):
    """Render RGB-D image with specified configuration."""

    # Compute OpenGL camera settings.
    lookdir = np.array([0, 0, 1]).reshape(3, 1)
    updir = np.array([0, -1, 0]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.array(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = config['position'] + lookdir
    focal_length = config['intrinsics'][0]
    znear, zfar = config['zrange']
    viewm = p.computeViewMatrix(config['position'], lookat, updir)
    fovh = (np.arctan(
        (config['image_size'][0] / 2) / focal_length) * 2 / np.pi) * 180

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = config['image_size'][1] / config['image_size'][0]
    projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = p.getCameraImage(
        width=config['image_size'][1],
        height=config['image_size'][0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (config['image_size'][0], config['image_size'][1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if config['noise']:
      color = np.int32(color)
      color += np.int32(np.random.normal(0, 3, config['image_size']))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (config['image_size'][0], config['image_size'][1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    depth = (2. * znear * zfar) / depth
    if config['noise']:
      depth += np.random.normal(0, 0.003, depth_image_size)

    # Get segmentation image.
    segm = np.uint8(segm).reshape(depth_image_size)

    return color, depth, segm

  @property
  def info(self):
    """Environment info dictionary."""
    # Some tasks create and remove zones, so ignore those IDs.
    removed_ids = []
    if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        isinstance(self.task, tasks.names['bag-alone-open'])):
      removed_ids.append(self.task.zone_id)

    info = {}  # object id : (position, rotation, dimensions)
    for object_id in self.fixed_objects + self.objects:
      if object_id in removed_ids:
        continue
      position, rotation = p.getBasePositionAndOrientation(object_id)
      dimensions = p.getVisualShapeData(object_id)[0][3]
      info[object_id] = (position, rotation, dimensions)
    return info

  #-------------------------------------------------------------------------
  # Robot Movement Functions
  #-------------------------------------------------------------------------

  def movej(self, targj, speed=0.01, t_lim=20):
    """Move UR5 to target joint configuration."""
    t0 = time.time()
    while (time.time() - t0) < t_lim:
      currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
      currj = np.array(currj)
      diffj = targj - currj
      if all(np.abs(diffj) < 1e-2):
        return True

      # Move with constant velocity
      norm = np.linalg.norm(diffj)
      v = diffj / norm if norm > 0 else 0
      stepj = currj + v * speed
      gains = np.ones(len(self.joints))
      p.setJointMotorControlArray(
          bodyIndex=self.ur5,
          jointIndices=self.joints,
          controlMode=p.POSITION_CONTROL,
          targetPositions=stepj,
          positionGains=gains)
      time.sleep(0.001)
    print(f'Warning: movej exceeded {t_lim} second timeout. Skipping.')
    return False

  def movep(self, pose, speed=0.01):
    """Move UR5 to target end effector pose."""
    # # Keep joint angles between -180/+180
    # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
    targj = self.solve_ik(pose)
    return self.movej(targj, speed, self.t_lim)

  def solve_ik(self, pose):
    """Calculate joint configuration."""
    homej_list = np.array(self.homej).tolist()
    joints = p.calculateInverseKinematics(
        bodyUniqueId=self.ur5,
        endEffectorLinkIndex=self.ee_tip_link,
        targetPosition=pose[:3],
        targetOrientation=pose[3:],
        lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
        upperLimits=[17, 0, 17, 17, 17, 17],
        jointRanges=[17] * 6,
        restPoses=homej_list,
        maxNumIterations=100,
        residualThreshold=1e-5)
    joints = np.array(joints)
    joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
    joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
    return joints

  #-------------------------------------------------------------------------
  # Motion Primitives
  #-------------------------------------------------------------------------

  def pick_place_6dof(self, pose0, pose1):
    return self.pick_place(pose0, pose1, place_6dof=True)

  def pick_place(self, pose0, pose1, place_6dof=False):
    """Execute pick and place primitive.

    Standard ravens tasks use the `delta` vector to lower the gripper
    until it makes contact with something. With deformables, however, we
    need to consider cases when the gripper could detect a rigid OR a
    soft body (cloth or bag); it should grip the first item it touches.
    This is handled in the Gripper class.

    Different deformable ravens tasks use slightly different parameters
    for better physics (and in some cases, faster simulation). Therefore,
    rather than make special cases here, those tasks will define their
    own action parameters, which we use here if they exist. Otherwise, we
    stick to defaults from standard ravens. Possible action parameters a
    task might adjust:

    speed: how fast the gripper moves.
    delta_z: how fast the gripper lowers for picking / placing.
    prepick_z: height of the gripper when it goes above the target
        pose for picking, just before it lowers.
    postpick_z: after suction gripping, raise to this height, should
        generally be low for cables / cloth.
    preplace_z: like prepick_z, but for the placing pose.
    pause_place: add a small pause for some tasks (e.g., bags) for
        slightly better soft body physics.
    final_z: height of the gripper after the action. Recommended to
        leave it at the default of 0.3, because it has to be set high
        enough to avoid the gripper occluding the workspace when
        generating color/depth maps.

    Args:
      pose0: picking pose.
      pose1: placing pose.
      place_6dof: Boolean to check if we're using 6 DoF placing.

    Returns:
      A bool indicating whether the action succeeded or not, via
      checking the sequence of movep calls. If any movep failed, then
      self.step() will terminate the episode after this action.
    """

    # Defaults used in the standard Ravens environments.
    speed = 0.01
    delta_z = -0.001
    prepick_z = 0.3
    postpick_z = 0.3
    preplace_z = 0.3
    pause_place = 0.0
    final_z = 0.3

    # Parameters if task provides them, which depends on the task stage.
    if hasattr(self.task, 'primitive_params'):
      ts = self.task.task_stage
      if 'prepick_z' in self.task.primitive_params[ts]:
        prepick_z = self.task.primitive_params[ts]['prepick_z']
      speed = self.task.primitive_params[ts]['speed']
      delta_z = self.task.primitive_params[ts]['delta_z']
      postpick_z = self.task.primitive_params[ts]['postpick_z']
      preplace_z = self.task.primitive_params[ts]['preplace_z']
      pause_place = self.task.primitive_params[ts]['pause_place']

    # Used to track deformable IDs, so that we can get the vertices.
    def_ids = []
    if self.is_softbody_env():
      assert hasattr(self.task, 'def_ids'), 'Soft bodies need def_ids'
      def_ids = self.task.def_ids

    # Now proceed as usual with given action parameters.
    success = True
    pick_position = np.array(pose0[0])
    pick_rotation = np.array(pose0[1])

    prepick_position = pick_position.copy()
    prepick_position[2] = prepick_z

    # Execute picking motion primitive.
    prepick_pose = np.hstack((prepick_position, pick_rotation))
    success &= self.movep(prepick_pose)
    target_pose = prepick_pose.copy()
    delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

    # Lower gripper until (a) touch object (rigid OR def), or (a) hit ground.
    while not self.ee.detect_contact(def_ids) and target_pose[2] > 0:
      target_pose += delta
      success &= self.movep(target_pose)

      # Might need this?
      if not success:
        return False

    # Create constraint (rigid objects) or anchor (deformable).
    self.ee.activate(self.objects, def_ids)

    # Increase z slightly (or hard-code it) and check picking success.
    if self.is_softbody_env() or self.is_new_cable_env():
      prepick_pose[2] = postpick_z
      success &= self.movep(prepick_pose, speed=speed)
      time.sleep(pause_place)  # extra rest for bags
    elif isinstance(self.task, tasks.names['cable']):
      prepick_pose[2] = 0.03
      success &= self.movep(prepick_pose, speed=0.001)
    else:
      prepick_pose[2] += pick_position[2]
      success &= self.movep(prepick_pose)
    pick_success = self.ee.check_grasp()

    if pick_success:
      place_position = np.array(pose1[0])
      place_rotation = np.array(pose1[1])

      if not place_6dof:
        preplace_position = place_position.copy()
        preplace_position[2] = 0.3 + pick_position[2]
        preplace_rotation = place_rotation.copy()
      else:
        t_world_place = pose1
        t_place_preplace = (np.array([0, 0, 0.3]),
                            utils.get_pybullet_quaternion_from_rot((0, 0, 0)))
        t_world_preplace = utils.multiply(t_world_place, t_place_preplace)

        preplace_position = np.array(t_world_preplace[0])
        preplace_rotation = np.array(t_world_preplace[1])

      # Execute placing motion primitive if pick success.
      preplace_pose = np.hstack((preplace_position, preplace_rotation))

      if self.is_softbody_env() or self.is_new_cable_env():
        preplace_pose[2] = preplace_z
        success &= self.movep(preplace_pose, speed=speed)
        time.sleep(pause_place)  # extra rest for bag
      elif isinstance(self.task, tasks.names['cable']):
        preplace_pose[2] = 0.03
        success &= self.movep(preplace_pose, speed=0.001)
      else:
        success &= self.movep(preplace_pose)

      # Might need this?
      # if not success:
      #    return False

      max_place_steps = 1e3

      if not place_6dof:
        # Lower the gripper. TODO(daniel) test on cloth/bags.
        target_pose = preplace_pose.copy()
        place_steps = 0
        while not self.ee.detect_contact(def_ids) and target_pose[2] > 0:
          place_steps += 1

          if place_steps > max_place_steps:
            print('Catching -- I was stuck.')
            return False

          target_pose += delta
          success &= self.movep(target_pose)
          # Might need this?
          if not success:
            return False

      else:
        current_pose = preplace_pose.copy()
        place_pose = np.hstack((place_position, place_rotation))

        place_delta = np.linalg.norm(place_pose[0:3] - current_pose[0:3])
        place_direction = (place_pose[0:3] - current_pose[0:3]) / place_delta
        place_steps = 0
        while not self.ee.detect_contact(
        ) and place_delta > PLACE_DELTA_THRESHOLD:
          place_steps += 1
          if place_steps > max_place_steps:
            print('Catching -- I was stuck.')
            return False

          current_pose[0:3] += place_direction * PLACE_STEP
          success &= self.movep(current_pose)

          # Might need this?
          if not success:
            return False
          place_delta = np.linalg.norm(place_pose[0:3] - current_pose[0:3])
          place_direction = (place_pose[0:3] - current_pose[0:3]) / place_delta

      # Release and move gripper up (if not 6 DoF) to clear the view for images.
      self.ee.release()
      if not place_6dof:
        preplace_pose[2] = final_z
      success &= self.movep(preplace_pose)
    else:
      # Release and move gripper up to clear the view for images.
      self.ee.release()
      prepick_pose[2] = final_z
      success &= self.movep(prepick_pose)
    return success

  def sweep(self, pose0, pose1):
    """Execute sweeping primitive."""
    success = True
    position0 = np.float32(pose0[0])
    position1 = np.float32(pose1[0])
    position0[2] = 0.001
    position1[2] = 0.001
    direction = position1 - position0
    length = np.linalg.norm(position1 - position0)
    if length == 0:
      direction = np.float32([0, 0, 0])
    else:
      direction = (position1 - position0) / length

    theta = np.arctan2(direction[1], direction[0])
    rotation = utils.get_pybullet_quaternion_from_rot((0, 0, theta))

    over0 = position0.copy()
    over0[2] = 0.3
    over1 = position1.copy()
    over1[2] = 0.3

    success &= self.movep(np.hstack((over0, rotation)))
    success &= self.movep(np.hstack((position0, rotation)))

    num_pushes = np.int32(np.floor(length / 0.01))
    for _ in range(num_pushes):
      target = position0 + direction * num_pushes * 0.01
      success &= self.movep(np.hstack((target, rotation)), speed=0.003)

    success &= self.movep(np.hstack((position1, rotation)), speed=0.003)
    success &= self.movep(np.hstack((over1, rotation)))
    return success

  def push(self, pose0, pose1):
    """Execute pushing primitive."""
    success = True
    p0 = np.float32(pose0[0])
    p1 = np.float32(pose1[0])
    p0[2], p1[2] = 0.025, 0.025
    if np.sum(p1 - p0) == 0:
      push_direction = 0
    else:
      push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
    p1 = p0 + push_direction * 0.01
    success &= self.movep(np.hstack((p0, self.home_pose[3:])))
    success &= self.movep(np.hstack((p1, self.home_pose[3:])), speed=0.003)
    return success

  #-------------------------------------------------------------------------
  # Checking if the env's task is part of deformable ravens.
  #-------------------------------------------------------------------------

  def is_softbody_env(self):
    """Checks whether the env's task uses a soft body.

    This counts cloth and bags, but not cables, since those are composed
    of a series of rigid bodies.

    Returns:
      bool indicating the environment uses a soft body.
    """
    return self.is_cloth_env() or self.is_bag_env()

  def is_new_cable_env(self):
    """Helps to track any new cable-related tasks."""
    return (isinstance(self.task, tasks.names['cable-shape']) or
            isinstance(self.task, tasks.names['cable-ring']) or
            isinstance(self.task, tasks.names['cable-ring-notarget']) or
            isinstance(self.task, tasks.names['cable-shape-notarget']) or
            isinstance(self.task, tasks.names['cable-line-notarget']))

  def is_cloth_env(self):
    return (isinstance(self.task, tasks.names['cloth-flat']) or
            isinstance(self.task, tasks.names['cloth-flat-notarget']) or
            isinstance(self.task, tasks.names['cloth-cover']))

  def is_bag_env(self):
    return (isinstance(self.task, tasks.names['bag-alone-open']) or
            isinstance(self.task, tasks.names['bag-items-easy']) or
            isinstance(self.task, tasks.names['bag-items-hard']))
