# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Environments used in the DARC experiments."""
import tempfile
import gin
import gym
from gym import utils
from gym.envs.mujoco import ant
from gym.envs.mujoco import half_cheetah
from gym.envs.mujoco import hopper
from gym.envs.mujoco import humanoid
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import walker2d
import gym.spaces
import numpy as np
import reacher_7dof
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


XML = """
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>

    <geom name="obstacle" type="box" pos="-3 0 %f" size="1 10 10" rgba="0.2 0.5 0.2 1" conaffinity="1"/>

  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""


class HalfCheetahDirectionEnv(half_cheetah.HalfCheetahEnv):
  """Variant of half-cheetah that includes an obstacle."""

  def __init__(self, use_obstacle):
    self._tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".xml")
    if use_obstacle:
      obstacle_height = 1.0
    else:
      obstacle_height = -50
    self._tempfile.write(XML % (obstacle_height))
    self._tempfile.flush()
    mujoco_env.MujocoEnv.__init__(self, self._tempfile.name, 5)
    utils.EzPickle.__init__(self)
    self.observation_space = gym.spaces.Box(
        low=self.observation_space.low,
        high=self.observation_space.high,
        dtype=np.float32,
    )

  def step(self, action):
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(action).sum()
    reward_run = abs(xposafter - xposbefore) / self.dt
    reward = reward_ctrl + reward_run
    done = False
    return ob, reward, done, dict(
        reward_run=reward_run, reward_ctrl=reward_ctrl)

  def camera_setup(self):
    super(HalfCheetahDirectionEnv, self).camera_setup()
    self.camera._render_camera.distance = 5.0  # pylint: disable=protected-access


def get_half_cheetah_direction_env(mode):
  if mode == "sim":
    env = HalfCheetahDirectionEnv(use_obstacle=False)
  else:
    assert mode == "real"
    env = HalfCheetahDirectionEnv(use_obstacle=True)
  env = suite_gym.wrap_env(env, max_episode_steps=1000)
  return tf_py_environment.TFPyEnvironment(env)


class BrokenReacherEnv(reacher_7dof.Reacher7DofEnv):
  """Variant of the 7DOF reaching environment with a broken joint.

  I implemented the BrokenReacherEnv before implementing the more general
  BrokenJoint wrapper. While in theory they should do the same thing, I haven't
  confirmed this yet, so I'm keeping BrokenReacherEnv separate for now.
  """

  def __init__(self, broken_joint=2, state_includes_action=True):
    self._broken_joint = broken_joint
    self._state_includes_action = state_includes_action
    super(BrokenReacherEnv, self).__init__()
    if state_includes_action:
      obs_dim = len(self.observation_space.low)
      action_dim = len(self.action_space.low)
      self._observation_space = gym.spaces.Box(
          low=np.full(obs_dim + action_dim, -np.inf, dtype=np.float32),
          high=np.full(obs_dim + action_dim, np.inf, dtype=np.float32),
      )

  def reset(self):
    s = super(BrokenReacherEnv, self).reset()
    a = np.zeros(len(self.action_space.low))
    if self._state_includes_action:
      s = np.concatenate([s, a])
    return s

  def step(self, action):
    action = action.copy()
    if self._broken_joint is not None:
      action[self._broken_joint] = 0.0
    ns, r, done, info = super(BrokenReacherEnv, self).step(action)
    if self._state_includes_action:
      ns = np.concatenate([ns, action])
    return ns, r, done, info


class BrokenJoint(gym.Wrapper):
  """Wrapper that disables one coordinate of the action, setting it to 0."""

  def __init__(self, env, broken_joint):
    super(BrokenJoint, self).__init__(env)
    # Change dtype of observation to be float32
    self.observation_space = gym.spaces.Box(
        low=self.observation_space.low,
        high=self.observation_space.high,
        dtype=np.float32,
    )
    if broken_joint is not None:
      assert 0 <= broken_joint <= len(self.action_space.low) - 1
    self.broken_joint = broken_joint

  def step(self, action):
    action = action.copy()
    if self.broken_joint is not None:
      action[self.broken_joint] = 0
    return super(BrokenJoint, self).step(action)


@gin.configurable
def get_broken_joint_env(mode, env_name, broken_joint=0):
  """Creates an environment with a broken joint."""
  if env_name == "ant":
    env = ant.AntEnv()
  elif env_name == "half_cheetah":
    env = half_cheetah.HalfCheetahEnv()
  elif env_name == "reacher_7dof":
    env = reacher_7dof.Reacher7DofEnv()
  else:
    raise NotImplementedError
  if mode == "sim":
    env = BrokenJoint(env, broken_joint=None)
  else:
    assert mode == "real"
    env = BrokenJoint(env, broken_joint=broken_joint)
  env = suite_gym.wrap_env(env, max_episode_steps=1000)
  return tf_py_environment.TFPyEnvironment(env)


class FallingEnv(gym.Wrapper):
  """Wrapper that disables the termination condition."""

  def __init__(self, env, ignore_termination=False):
    self._ignore_termination = ignore_termination
    super(FallingEnv, self).__init__(env)

  def step(self, a):
    ns, r, done, info = super(FallingEnv, self).step(a)
    r = 0.0
    if self._ignore_termination:
      done = False
    return ns, r, done, info


def get_falling_env(mode, env_name):
  """Creates an environment with the termination condition disabled."""
  if env_name == "hopper":
    env = hopper.HopperEnv()
  elif env_name == "humanoid":
    env = humanoid.HumanoidEnv()
  elif env_name == "walker":
    env = walker2d.Walker2dEnv()
  else:
    raise NotImplementedError
  if mode == "sim":
    env = FallingEnv(env, ignore_termination=True)
  else:
    assert mode == "real"
    env = FallingEnv(env, ignore_termination=False)

  env = suite_gym.wrap_env(env, max_episode_steps=300)
  return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def get_broken_reacher_env(mode, broken_joint=0):
  if mode == "sim":
    env = BrokenReacherEnv(broken_joint=None)
  else:
    assert mode == "real"
    env = BrokenReacherEnv(broken_joint=broken_joint)
  env = suite_gym.wrap_env(env, max_episode_steps=100)
  return tf_py_environment.TFPyEnvironment(env)
