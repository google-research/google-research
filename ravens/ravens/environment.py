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

"""Environment class."""

import os
import time

import numpy as np
import pybullet as p
import pybullet_data  # pylint: disable=unused-import

from ravens import cameras


class Environment():
  """OpenAI Gym-style environment class."""

  def __init__(self, disp=False, hz=240):
    """Creates OpenAI Gym-style environment with PyBullet.

    Args:
      disp: show environment with PyBullet's built-in display viewer
      hz: PyBullet physics simulation step speed. Set to 480 for deformables.
    """
    self.pix_size = 0.003125
    self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
    self.agent_cams = cameras.RealSenseD415.CONFIG

    # Start PyBullet.
    p.connect(p.GUI if disp else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    assets_path = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(assets_path)
    p.setTimeStep(1. / hz)

    # If using --disp, move default camera closer to the scene.
    if disp:
      target = p.getDebugVisualizerCamera()[11]
      p.resetDebugVisualizerCamera(
          cameraDistance=1.1,
          cameraYaw=90,
          cameraPitch=-25,
          cameraTargetPosition=target)

  @property
  def is_static(self):
    """Return true if objects are no longer moving."""
    v = [np.linalg.norm(p.getBaseVelocity(i)[0])
         for i in self.obj_ids['rigid']]
    return all(np.array(v) < 5e-3)

  def add_object(self, urdf, pose, category='rigid'):
    """List of (fixed, rigid, or deformable) objects in env."""
    fixed_base = 1 if category == 'fixed' else 0
    obj_id = p.loadURDF(urdf, pose[0], pose[1], useFixedBase=fixed_base)
    self.obj_ids[category].append(obj_id)
    return obj_id

  #---------------------------------------------------------------------------
  # Standard Gym Functions
  #---------------------------------------------------------------------------

  def reset(self, task):
    """Performs common reset functionality for all supported tasks."""
    self.task = task
    self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.8)

    # Temporarily disable rendering to load scene faster.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    p.loadURDF('assets/plane/plane.urdf', (0, 0, -0.001))
    p.loadURDF('assets/ur5/workspace.urdf', (0.5, 0, 0))

    # Load UR5 robot arm equipped with suction end effector.
    # TODO(andyzeng): add back parallel-jaw grippers.
    self.ur5 = p.loadURDF('assets/ur5/ur5.urdf')
    self.ee = self.task.ee(self.ur5, 9, self.obj_ids)
    self.ee_tip = 10  # Link ID of suction cup.

    # Get revolute joint indices of robot (skip fixed joints).
    n_joints = p.getNumJoints(self.ur5)
    joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
    self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home joint configuration.
    for i in range(len(self.joints)):
      p.resetJointState(self.ur5, self.joints[i], self.homej[i])

    # Reset end effector.
    self.ee.release()

    # Reset task.
    self.task.reset(self)

    # Re-enable rendering.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return self.step()

  def step(self, act=None):
    """Execute action with specified primitive.

    Args:
      act: action to execute.

    Returns:
      (obs, reward, done, info) tuple containing MDP step data.
    """

    if act:
      timeout = self.task.primitive(self.movej, self.movep, self.ee, **act)

      # Exit early if action times out.
      if timeout:
        return {}, 0, True, self.info

    # Step simulator asynchronously until objects settle.
    while not self.is_static:
      p.stepSimulation()

    # Get task rewards.
    reward, info = self.task.reward() if act else (0, {})
    done = self.task.done()

    # Add ground truth robot state into info.
    info.update(self.info)

    # Get RGB-D camera image observations.
    obs = {'color': [], 'depth': []}
    for config in self.agent_cams:
      color, depth, _ = self.render(config)
      obs['color'].append(color)
      obs['depth'].append(depth)

    return obs, reward, done, info

  def render(self, config):
    """Render RGB-D image with specified camera configuration."""

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = config['position'] + lookdir
    focal_len = config['intrinsics'][0]
    znear, zfar = config['zrange']
    viewm = p.computeViewMatrix(config['position'], lookat, updir)
    fovh = (config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

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
    """Environment info variable with object poses, dimensions, and colors."""

    # Some tasks create and remove zones, so ignore those IDs.
    # removed_ids = []
    # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
    #         isinstance(self.task, tasks.names['bag-alone-open'])):
    #   removed_ids.append(self.task.zone_id)

    info = {}  # object id : (position, rotation, dimensions)
    for obj_ids in self.obj_ids.values():
      for obj_id in obj_ids:
        pos, rot = p.getBasePositionAndOrientation(obj_id)
        dim = p.getVisualShapeData(obj_id)[0][3]
        info[obj_id] = (pos, rot, dim)
    return info

  #---------------------------------------------------------------------------
  # Robot Movement Functions
  #---------------------------------------------------------------------------

  def movej(self, targj, speed=0.01, timeout=5):
    """Move UR5 to target joint configuration."""
    t0 = time.time()
    while (time.time() - t0) < timeout:
      currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
      currj = np.array(currj)
      diffj = targj - currj
      if all(np.abs(diffj) < 1e-2):
        return False

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
      p.stepSimulation()
    print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
    return True

  def movep(self, pose, speed=0.01):
    """Move UR5 to target end effector pose."""
    targj = self.solve_ik(pose)
    return self.movej(targj, speed)

  def solve_ik(self, pose):
    """Calculate joint configuration with inverse kinematics."""
    joints = p.calculateInverseKinematics(
        bodyUniqueId=self.ur5,
        endEffectorLinkIndex=self.ee_tip,
        targetPosition=pose[0],
        targetOrientation=pose[1],
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
        restPoses=np.float32(self.homej).tolist(),
        maxNumIterations=100,
        residualThreshold=1e-5)
    joints = np.float32(joints)
    joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
    return joints
