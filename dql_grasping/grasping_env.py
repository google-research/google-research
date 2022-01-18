# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python2, python3
"""Simplified grasping environment using PyBullet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import time

from absl import logging
import gin
import gym
from gym import spaces
import numpy as np
from PIL import Image
from six.moves import range

import pybullet
from dql_grasping import kuka

INTERNAL_BULLET_ROOT = None
if INTERNAL_BULLET_ROOT is None:
  import pybullet_data
  OSS_DATA_ROOT = pybullet_data.getDataPath()
else:
  OSS_DATA_ROOT = ''
# pylint: enable=bad-import-order
# pylint: enable=g-import-not-at-top


@gin.configurable
class KukaGraspingProceduralEnv(gym.Env):
  """Simplified grasping environment with discrete and continuous actions.
  """

  def __init__(
      self,
      block_random=0.3,
      camera_random=0,
      simple_observations=False,
      continuous=False,
      remove_height_hack=False,
      urdf_list=None,
      render_mode='GUI',
      num_objects=5,
      dv=0.06,
      target=False,
      target_filenames=None,
      non_target_filenames=None,
      num_resets_per_setup=1,
      render_width=128,
      render_height=128,
      downsample_width=64,
      downsample_height=64,
      test=False,
      allow_duplicate_objects=True,
      max_num_training_models=900,
      max_num_test_models=100):
    """Creates a KukaGraspingEnv.

    Args:
      block_random: How much randomness to use in positioning blocks.
      camera_random: How much randomness to use in positioning camera.
      simple_observations: If True, observations are the position and
        orientation of end-effector and closest block, rather than images.
      continuous: If True, actions are continuous, else discrete.
      remove_height_hack: If True and continuous is True, add a dz
                          component to action space.
      urdf_list: List of objects to populate the bin with.
      render_mode: GUI, DIRECT, or TCP.
      num_objects: The number of random objects to load.
      dv: Velocity magnitude of cartesian dx, dy, dz actions per time step.
      target: If True, then we receive reward only for grasping one "target"
        object.
      target_filenames: Objects that we want to grasp.
      non_target_filenames: Objects that we dont want to grasp.
      num_resets_per_setup: How many env resets before calling setup again.
      render_width: Width of camera image to render with.
      render_height: Height of camera image to render with.
      downsample_width: Width of image observation.
      downsample_height: Height of image observation.
      test: If True, uses test split of objects.
      allow_duplicate_objects: If True, samples URDFs with replacement.
      max_num_training_models: The number of distinct models to choose from when
        selecting the num_objects placed in the tray for training.
      max_num_test_models: The number of distinct models to choose from when
        selecting the num_objects placed in the tray for testing.
    """
    self._time_step = 1. / 200.
    self._max_steps = 15

    # Open-source search paths.
    self._urdf_root = OSS_DATA_ROOT
    self._models_dir = os.path.join(self._urdf_root, 'random_urdfs')

    self._action_repeat = 200
    self._env_step = 0
    self._renders = render_mode in ['GUI', 'TCP']
    # Size we render at.
    self._width = render_width
    self._height = render_height
    # Size we downsample to.
    self._downsample_width = downsample_width
    self._downsample_height = downsample_height
    self._target = target
    self._num_objects = num_objects
    self._dv = dv
    self._urdf_list = urdf_list
    if target_filenames:
      target_filenames = [self._get_urdf_path(f) for f in target_filenames]
    if non_target_filenames:
      non_target_filenames = [
          self._get_urdf_path(f) for f in non_target_filenames]
    self._object_filenames = (target_filenames or []) + (
        non_target_filenames or [])
    self._target_filenames = target_filenames or []
    self._block_random = block_random
    self._cam_random = camera_random
    self._simple_obs = simple_observations
    self._continuous = continuous
    self._remove_height_hack = remove_height_hack
    self._resets = 0
    self._num_resets_per_setup = num_resets_per_setup
    self._test = test
    self._allow_duplicate_objects = allow_duplicate_objects
    self._max_num_training_models = max_num_training_models
    self._max_num_test_models = max_num_test_models

    if render_mode == 'GUI':
      self.cid = pybullet.connect(pybullet.GUI)
      pybullet.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    elif render_mode == 'DIRECT':
      self.cid = pybullet.connect(pybullet.DIRECT)
    elif render_mode == 'TCP':
      self.cid = pybullet.connect(pybullet.TCP, 'localhost', 6667)

    self.setup()
    if self._continuous:
      self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
      if self._remove_height_hack:
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(5,))  # dx, dy, dz, da, close
    else:
      self.action_space = spaces.Discrete(8)
      if self._remove_height_hack:
        self.action_space = spaces.Discrete(10)

    if self._simple_obs:
      # (3 pos + 4 quat) x 2
      self.observation_space = spaces.Box(low=-100, high=100, shape=(14,))
    else:
      # image (self._height, self._width, 3) x position of the gripper (3,)
      img_space = spaces.Box(
          low=0,
          high=255,
          shape=(self._downsample_height, self._downsample_width, 3))
      pos_space = spaces.Box(low=-5, high=5, shape=(3,))
      self.observation_space = spaces.Tuple((img_space, pos_space))
    self.viewer = None

  def setup(self):
    """Sets up the robot + tray + objects.
    """
    test = self._test
    if not self._urdf_list:  # Load from procedural random objects.
      if not self._object_filenames:
        self._object_filenames = self._get_random_objects(
            num_objects=self._num_objects,
            test=test,
            replace=self._allow_duplicate_objects,
        )
      self._urdf_list = self._object_filenames
    logging.info('urdf_list %s', self._urdf_list)
    pybullet.resetSimulation(physicsClientId=self.cid)
    pybullet.setPhysicsEngineParameter(
        numSolverIterations=150, physicsClientId=self.cid)
    pybullet.setTimeStep(self._time_step, physicsClientId=self.cid)
    pybullet.setGravity(0, 0, -10, physicsClientId=self.cid)
    plane_path = os.path.join(self._urdf_root, 'plane.urdf')
    pybullet.loadURDF(plane_path, [0, 0, -1], physicsClientId=self.cid)
    table_path = os.path.join(self._urdf_root, 'table/table.urdf')
    pybullet.loadURDF(
        table_path, [0.5, 0.0, -.82], [0., 0., 0., 1.],
        physicsClientId=self.cid)
    self._kuka = kuka.Kuka(
        urdfRootPath=self._urdf_root,
        timeStep=self._time_step,
        clientId=self.cid)
    self._block_uids = []
    for urdf_name in self._urdf_list:
      xpos = 0.4 + self._block_random * random.random()
      ypos = self._block_random * (random.random() - .5)
      angle = np.pi / 2 + self._block_random * np.pi * random.random()
      ori = pybullet.getQuaternionFromEuler([0, 0, angle])
      uid = pybullet.loadURDF(
          urdf_name, [xpos, ypos, .15], [ori[0], ori[1], ori[2], ori[3]],
          physicsClientId=self.cid)
      self._block_uids.append(uid)
      for _ in range(500):
        pybullet.stepSimulation(physicsClientId=self.cid)

  def reset(self):
    self._resets += 1
    if self._resets % self._num_resets_per_setup == 0:
      self.setup()

    self._attempted_grasp = False

    look = [0.23, 0.2, 0.54]
    distance = 1.
    pitch = -56 + self._cam_random * np.random.uniform(-3, 3)
    yaw = 245 + self._cam_random * np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cam_random * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.1
    far = 10
    self._proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov, aspect, near, far)
    self._env_step = 0

    for i in range(len(self._urdf_list)):
      xpos = 0.4 + self._block_random * random.random()
      ypos = self._block_random * (random.random() - .5)
      # random angle
      angle = np.pi / 2 + self._block_random * np.pi * random.random()
      ori = pybullet.getQuaternionFromEuler([0, 0, angle])
      pybullet.resetBasePositionAndOrientation(
          self._block_uids[i], [xpos, ypos, .15],
          [ori[0], ori[1], ori[2], ori[3]],
          physicsClientId=self.cid)
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(500):
        pybullet.stepSimulation(physicsClientId=self.cid)

    # Let the blocks settle and move arm down into a closer approach pose.
    self._kuka.reset()
    # note the velocity continues throughout the grasp.
    self._kuka.applyAction([0, 0, -0.3, 0, 0.3])
    for i in range(100):
      pybullet.stepSimulation(physicsClientId=self.cid)
    return self._get_observation()

  def __del__(self):
    pybullet.disconnect(physicsClientId=self.cid)

  def _get_observation(self):
    if self._simple_obs:
      return self._get_simple_observation()
    else:
      return self._get_image_observation()

  def _get_image_observation(self):
    results = pybullet.getCameraImage(width=self._width,
                                      height=self._height,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      physicsClientId=self.cid)
    rgba = results[2]
    np_img_arr = np.reshape(rgba, (self._height, self._width, 4))
    # Extract RGB components only.
    img = Image.fromarray(np_img_arr[:, :, :3].astype(np.uint8))
    shape = (self._downsample_width, self._downsample_height)
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img)
    return img

  def _get_simple_observation(self):
    """Observations for simplified observation space.

    Returns:
      Numpy array containing location and orientation of nearest block and
      location of end-effector.
    """
    state = pybullet.getLinkState(
        self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex,
        physicsClientId=self.cid)
    end_effector_pos = np.array(state[0])
    end_effector_ori = np.array(state[1])

    distances = []
    pos_and_ori = []
    for uid in self._block_uids:
      pos, ori = pybullet.getBasePositionAndOrientation(
          uid, physicsClientId=self.cid)
      pos, ori = np.array(pos), np.array(ori)
      pos_and_ori.append((pos, ori))
      distances.append(np.linalg.norm(end_effector_pos - pos))
    pos, ori = pos_and_ori[np.argmin(distances)]
    return np.concatenate((pos, ori, end_effector_pos, end_effector_ori))

  def step(self, action):
    dv = self._dv  # velocity per physics step.
    if self._continuous:
      dx = dv * action[0]
      dy = dv * action[1]
      if self._remove_height_hack:
        dz = dv * action[2]
        da = 0.25 * action[3]
      else:
        dz = -dv
        da = 0.25 * action[2]
    else:
      # Static type assertion for integers.
      assert isinstance(action, int)
      if self._remove_height_hack:
        dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
        dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
        da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
      else:
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = -dv
        da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
    return self._step_continuous([dx, dy, dz, da, 0.3])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    self._env_step += 1
    self._kuka.applyAction(action)
    for _ in range(self._action_repeat):
      pybullet.stepSimulation(physicsClientId=self.cid)
      if self._renders:
        time.sleep(self._time_step)
      if self._termination():
        break

    # If we are close to the bin, attempt grasp.
    state = pybullet.getLinkState(self._kuka.kukaUid,
                                  self._kuka.kukaEndEffectorIndex,
                                  physicsClientId=self.cid)
    end_effector_pos = state[0]
    if end_effector_pos[2] <= 0.1:
      finger_angle = 0.3
      for _ in range(1000):
        grasp_action = [0, 0, 0.001, 0, finger_angle]
        self._kuka.applyAction(grasp_action)
        pybullet.stepSimulation(physicsClientId=self.cid)
        finger_angle -= 0.3/100.
        if finger_angle < 0:
          finger_angle = 0
      self._attempted_grasp = True
    observation = self._get_observation()
    done = self._termination()
    reward = self._reward()

    debug = {
        'grasp_success': self._grasp_success
    }
    return observation, reward, done, debug

  def _render(self, mode='human'):
    return

  def _termination(self):
    return self._attempted_grasp or self._env_step >= self._max_steps

  def _reward(self):
    reward = 0
    self._grasp_success = 0

    if self._target:
      target_uids = self._block_uids[0:len(self._target_filenames)]
    else:
      target_uids = self._block_uids

    for uid in target_uids:
      pos, _ = pybullet.getBasePositionAndOrientation(
          uid, physicsClientId=self.cid)
      # If any block is above height, provide reward.
      if pos[2] > 0.2:
        self._grasp_success = 1
        reward = 1
        break
    return reward

  def close_display(self):
    pybullet.disconnect()
    self.cid = pybullet.connect(pybullet.DIRECT)
    self._setup()

  def _get_urdf_path(self, filename):
    """Resolve urdf path of filename."""
    d = os.path.splitext(filename)[0]
    return os.path.join(self._models_dir, d, filename)

  def _get_random_objects(self, num_objects, test, replace=True):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects: Number of graspable objects.
      test: Whether to use the training or test pool of objects.
      replace: Whether to allow choosing the same object twice.
    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._models_dir, '*0/*.urdf')
      max_num_objects = self._max_num_test_models
    else:
      urdf_pattern = os.path.join(self._models_dir, '*[^0]/*.urdf')
      max_num_objects = self._max_num_training_models
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    if total_num_objects > max_num_objects:
      total_num_objects = max_num_objects
    selected_objects = np.random.choice(
        np.arange(total_num_objects), num_objects, replace=replace)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    logging.info('selected_objects_filenames %s', selected_objects_filenames)
    return selected_objects_filenames
