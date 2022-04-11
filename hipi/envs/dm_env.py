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

"""Dynamics and TaskDistributions for dm_control environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_control
from dm_control.rl import control
from dm_control.suite import cheetah
from dm_control.suite import common
from dm_control.suite import finger
from dm_control.suite import fish
from dm_control.suite import hopper
from dm_control.suite import humanoid
from dm_control.suite import point_mass
from dm_control.suite import quadruped
from dm_control.suite import reacher
from dm_control.suite import stacker
from dm_control.suite import swimmer
from dm_control.suite import walker
from dm_control.utils import rewards
from envs import multitask
import gin
import gym
import numpy as np
import tensorflow as tf


def gaussian_sigmoid(x, value_at_margin):
  scale = tf.sqrt(-2 * tf.log(value_at_margin))
  return tf.exp(-0.5 * (x * scale)**2)


def long_tail_sigmoid(x, value_at_margin):
  scale = tf.sqrt(1 / value_at_margin - 1)
  return 1 / ((x * scale)**2 + 1)


def linear_sigmoid(x, value_at_margin):
  scale = 1 - value_at_margin
  scaled_x = x * scale
  return tf.where(tf.abs(scaled_x) < 1, 1 - scaled_x, tf.zeros_like(scaled_x))


def quadratic_sigmoid(x, value_at_margin):
  scale = tf.sqrt(1.0 - value_at_margin)
  scaled_x = x * scale
  return tf.where(tf.abs(scaled_x) < 1, 1 - scaled_x**2, tf.zeros_like(x))


def tf_tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    value_at_margin=rewards._DEFAULT_VALUE_AT_MARGIN,  # pylint: disable=protected-access
    sigmoid="gaussian",
):
  """Computes a reward based on distance from the desired position."""
  if sigmoid == "gaussian":
    sigmoid_fn = gaussian_sigmoid
  elif sigmoid == "long_tail":
    sigmoid_fn = long_tail_sigmoid
  elif sigmoid == "linear":
    sigmoid_fn = linear_sigmoid
  elif sigmoid == "quadratic":
    sigmoid_fn = quadratic_sigmoid
  else:
    raise NotImplementedError
  lower, upper = bounds
  in_bounds = tf.logical_and(lower <= x, x <= upper)
  d = tf.where(x < lower, lower - x, x - upper) / margin
  value = tf.where(in_bounds, tf.ones_like(d), sigmoid_fn(d, value_at_margin))
  return value


class DMDynamics(multitask.Dynamics):
  """Wrapper around the dynamics for the dm_control environments."""

  def __init__(self, env, task_key):
    """Wrapper for converting a DM environment into a Dynamics instance.

    Args:
      env: the DM environment
      task_key: (str or list of str): observation keys to omit from the
        observation
    """
    self._env = env
    self._action_space = gym.spaces.Box(
        low=self._env.action_spec().minimum,
        high=self._env.action_spec().maximum,
        dtype=np.float32,
    )
    if not isinstance(task_key, list):
      task_key = [task_key]
    self._obs_keys = sorted(
        [k for k in self._env.observation_spec().keys() if k not in task_key])
    obs_spec = self._env.observation_spec()
    # If the obs_spec = () for some key, np.prod will return 1.0. We therefore
    # need to cast the result to be an integer.
    num_dim = (
        int(np.sum([np.prod(obs_spec[k].shape) for k in self._obs_keys])) + 1
    )  # Add one for the done flag
    self._observation_space = gym.spaces.Box(
        low=np.full((num_dim,), -np.inf),
        high=np.full((num_dim,), np.inf),
        dtype=np.float32,
    )

  def reset(self):
    ts = self._env.reset()
    return self._ts_to_obs(ts)

  def step(self, action):
    try:
      ts = self._env.step(action)
    except dm_control.rl.control.PhysicsError:
      print("DM Physics Error! Restarting episode")
      ts = self._env.reset()
    return self._ts_to_obs(ts)

  def _ts_to_obs(self, ts):
    return np.concatenate([ts.observation[k].flatten()
                           for k in self._obs_keys] +
                          [[float(ts.last())]]).astype(np.float32)


class DMTaskDistribution(multitask.TaskDistribution):
  """General class for implementing tasks for dm_control environments."""

  def __init__(
      self,
      dynamics,
      task_key=None,
      relabel_task_key=None,
      relabel_dim=None,
      relabel_index_offset=0,
  ):
    """Initialize the task distribution.

    Args:
      dynamics: An instance of multitask.Dynamics.
      task_key: This is the part of the observation that indicates the goal.
        The dynamic should remove this part of the observation, and it will be
        added later by the environment wrapper.
      relabel_task_key: This is the part of the observation that indicates the
        goal actually reached. For example, in a manipulation task, this would
        be the current position of the object.
      relabel_dim: (int) When converting a state to a task
        start at an offset of relabel_index_offset, and only take the first
        relabel_dim coordinates.
      relabel_index_offset: (int) See above.
    """
    self._dynamics = dynamics
    self._task_key = task_key
    if relabel_dim is None:
      self._relabel_dim = np.prod(
          self._dynamics._env.observation_spec()[relabel_task_key].shape)
    else:
      self._relabel_dim = relabel_dim
    self._task_space = gym.spaces.Box(
        low=np.full((self._relabel_dim,), -np.inf),
        high=np.full((self._relabel_dim,), np.inf),
    )
    self._relabel_index = (
        self._get_obs_index(relabel_task_key) + relabel_index_offset)

  def _get_obs_index(self, key):
    key_index = self._dynamics._obs_keys.index(key)  # pylint: disable=protected-access
    keys_before = self._dynamics._obs_keys[:key_index]  # pylint: disable=protected-access
    obs_spec = self._dynamics._env.observation_spec()  # pylint: disable=protected-access
    return int(np.sum([np.prod(obs_spec[k].shape) for k in keys_before]))

  def sample(self):
    task = self._dynamics._env.task  # pylint: disable=protected-access
    physics = self._dynamics._env.physics  # pylint: disable=protected-access
    if isinstance(self._dynamics._env,   # pylint: disable=protected-access
                  dm_control.composer.environment.Environment):
      obs = self._dynamics._env.reset().observation   # pylint: disable=protected-access
    else:
      obs = task.get_observation(physics)
    target_pos = obs[self._task_key].flatten()
    return np.copy(target_pos).astype(np.float32)

  def _evaluate(self, states, actions, tasks):
    raise NotImplementedError

  def state_to_task(self, states):
    task = states[:,
                  self._relabel_index:self._relabel_index + self._relabel_dim]
    return task


@gin.configurable
class CheetahDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.01
    time_limit = max_episode_steps * dt
    super(CheetahDynamics, self).__init__(
        env=cheetah.run(time_limit=time_limit), task_key=None)


@gin.configurable
class CheetahTaskDistribution(DMTaskDistribution):
  """Implements tasks for the cheetah environment."""

  def __init__(self, cheetah_dynamics, vel_range=(0.0, 3.0), margin=None):
    # Only take the first component of the velocity as the task
    super(CheetahTaskDistribution, self).__init__(
        cheetah_dynamics, relabel_dim=1, relabel_task_key="velocity")
    self._vel_index = self._get_obs_index("velocity")
    self._vel_range = vel_range
    self._margin = margin

  def _evaluate(self, states, actions, tasks):
    # X velocity is the first component of the velocity vector.
    vel = states[:, self._vel_index]
    target_vel = tf.squeeze(tasks, axis=1)
    if self._margin is None:
      vel_rew = -1.0 * tf.abs(target_vel - vel)
    else:
      vel_rew = tf.cast(tf.abs(target_vel - vel) < self._margin, tf.float32)
    ctrl_cost = 0.05 * tf.reduce_sum(actions**2, axis=1)
    rew = vel_rew - ctrl_cost
    dones = tf.cast(states[:, -1], tf.bool)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    # Make sure this is an array.
    min_vel, max_vel = self._vel_range
    return np.random.uniform(min_vel, max_vel, [1]).astype(np.float32)

### Quadruped Environment


def run_task(time_limit=quadruped._DEFAULT_TIME_LIMIT,  # pylint: disable=protected-access
             random=None,
             environment_kwargs=None):
  """Returns the Run task."""
  xml_string = quadruped.make_model(floor_size=quadruped._DEFAULT_TIME_LIMIT *  # pylint: disable=protected-access
                                    quadruped._RUN_SPEED)  # pylint: disable=protected-access
  physics = quadruped.Physics.from_xml_string(xml_string, common.ASSETS)
  task = Move(desired_speed=quadruped._RUN_SPEED, random=random)  # pylint: disable=protected-access
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics,
      task,
      time_limit=time_limit,
      control_timestep=quadruped._CONTROL_TIMESTEP,  # pylint: disable=protected-access
      **environment_kwargs)


class Move(quadruped.Move):
  """Implements the quadruped task of moving."""

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.
    """
    # Initial configuration.
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    orientation /= np.linalg.norm(orientation)
    quadruped._find_non_contacting_height(physics, orientation)   # pylint: disable=protected-access
    ## Make sure to call the grandparent class, not the parent class. Calling
    # the parent class will re-initialize the agent with a random quaternion.
    super(quadruped.Move, self).initialize_episode(physics)


class QuadrupedRunDynamics(DMDynamics):
  """Implements the quadruped running task."""

  def __init__(self, max_episode_steps=1000, add_xy=True):
    dt = 0.02
    time_limit = max_episode_steps * dt
    self._add_xy = add_xy
    env = run_task(time_limit=time_limit)

    super(QuadrupedRunDynamics, self).__init__(env=env, task_key=None)
    obs_spec = self._env.observation_spec()
    num_dim = (
        int(np.sum([np.prod(obs_spec[k].shape) for k in self._obs_keys])) + 1
    )  # Add one for the done flag
    if add_xy:
      # Add 2 for the XY position
      num_dim += 2
    self._observation_space = gym.spaces.Box(
        low=np.full((num_dim,), -np.inf),
        high=np.full((num_dim,), np.inf),
        dtype=np.float32,
    )

  def _ts_to_obs(self, ts):
    # Modify to include the XYZ position
    if self._add_xy:   # pylint: disable=protected-access
      xy = self._env.physics.named.data.xpos["torso"][:2]
      return np.concatenate(
          [xy] + [ts.observation[k].flatten() for k in self._obs_keys] +
          [[float(ts.last())]]).astype(np.float32)
    else:
      return super(QuadrupedRunDynamics, self)._ts_to_obs(ts)


@gin.configurable
def quadruped_reward(states,
                     actions,
                     goals,
                     sparse=False,
                     goal_distance=None,
                     goal_radius=1.0):
  """Reward function for the quadruped.

  Args:
    states: the current state.
    actions: the current action.
    goals: the goal state.
    sparse: whether the reward function is sparse or dense.
    goal_distance: *initial* distance to the goal.
    goal_radius: *desired* distance to the goal.
  Returns:
    reward: the reward.
  """
  assert goal_distance is not None
  pos = states[:, :2]
  dist_to_goal = tf.linalg.norm(pos - goals, axis=1)
  ctrl_cost = 0.1 * tf.reduce_sum(actions**2, axis=1)
  stay_alive_bonus = 1.0
  if sparse:
    rew = tf.cast(dist_to_goal < goal_radius, tf.float32) - 1.0
    dones = tf.cast(tf.zeros_like(rew), tf.bool)
  else:
    rew = (goal_distance - dist_to_goal) - ctrl_cost + stay_alive_bonus
    dones = tf.cast(states[:, -1], tf.bool)
  assert rew.shape == dones.shape
  return rew, dones


@gin.configurable
class QuadrupedContinuousTaskDistribution(DMTaskDistribution):
  """Implements the task distribution for the quadruped environment."""

  def __init__(self, quadruped_dynamics, goal_distance=3.0):  # pylint: disable=super-init-not-called
    self._dynamics = quadruped_dynamics
    self._goal_distance = goal_distance
    self._task_space = gym.spaces.Box(
        low=np.full(2, -goal_distance),
        high=np.full(2, goal_distance),
        dtype=np.float32,
    )

  def _evaluate(self, states, actions, tasks):
    goals = tasks
    return quadruped_reward(
        states, actions, goals, goal_distance=self._goal_distance)

  def sample(self):
    theta = np.random.uniform(0, 2 * np.pi)
    goal = self._goal_distance * np.array([np.cos(theta), np.sin(theta)])
    return goal.astype(np.float32)

  def state_to_task(self, states):
    return states[:, :2]


@gin.configurable
class HopperDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.02
    time_limit = max_episode_steps * dt
    super(HopperDynamics, self).__init__(
        env=hopper.hop(time_limit=time_limit), task_key=None)


@gin.configurable
class HopperTaskDistribution(DMTaskDistribution):
  """Implements the tasks for the hopper environment."""

  def __init__(self, hopper_dynamics, version=0):  # pylint: disable=super-init-not-called
    self._dynamics = hopper_dynamics
    self._touch_index = self._get_obs_index("touch")
    self._pos_index = self._get_obs_index("position")
    self._vel_index = self._get_obs_index("velocity")

    ## [touch front, touch back, z pos, x vel, action norm
    if version == 0:
      low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
      high = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    elif version == 1:
      low = np.array([-10.0, -10.0, 0.0, -10.0, 0.0])
      high = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    elif version == 2:
      low = np.array([-10.0, -10.0, 1.0, -10.0, 0.0])
      high = np.array([0.0, 0.0, 1.0, 10.0, 0.0])
    elif version == 3:
      low = np.array([-10.0, -10.0, 0.0, -10.0, 0.0])
      high = np.array([10.0, 10.0, 10.0, 10.0, 0.0])
    self._task_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

  def _evaluate(self, states, actions, tasks):
    # Toe and Heel forces
    touch = states[:, self._touch_index:self._touch_index + 2]
    # Z is vertical motion, Y is rotation
    z_pos = states[:, self._pos_index:self._pos_index + 1]
    x_vel = states[:, self._vel_index:self._vel_index + 1]
    ctrl = tf.reduce_sum(actions**2, axis=1)
    features = tf.concat([touch, z_pos + 0.5, x_vel, ctrl[:, None]], axis=1)
    assert features.shape == tasks.shape
    rew = tf.reduce_sum(features * tasks, axis=1)
    # dones = tf.cast(states[:, -1], tf.bool)
    dones = tf.fill((states.shape[0],), False)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    task = np.random.uniform(self.task_space.low, self.task_space.high)
    return task.astype(np.float32)

  def state_to_task(self, states):
    raise NotImplementedError


class HopperDiscreteTaskDistribution(HopperTaskDistribution):
  """Implements a discrete task distribution for the hopper environment."""

  def __init__(self, hopper_dynamics):
    super(HopperDiscreteTaskDistribution, self).__init__(hopper_dynamics)
    ## [touch front, touch back, z pos, x vel, action norm
    self._task_vec = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],  # run forward
            [0.0, 0.0, 1.0, -1.0, 1.0],  # run backwards
            [0.0, 0.0, -1.0, 1.0, 1.0],  # crawl forwards
            [0.0, 0.0, -1.0, -1.0, 1.0],  # crawl forwards
            [1.0, 1.0, 1.0, 1.0, 1.0],  # run forward with minimal impact
            [1.0, 1.0, 1.0, -1.0, 1.0],  # run forward with minimal impact
        ],
        dtype=np.float32,
    )
    self._num_tasks = len(self._task_vec)
    self._task_space = gym.spaces.Box(
        low=np.zeros(self._num_tasks),
        high=np.ones(self._num_tasks),
        dtype=np.float32,
    )

  def _evaluate(self, states, actions, one_hot_tasks):
    tasks = tf.matmul(one_hot_tasks, self._task_vec)
    assert tasks.shape == (states.shape[0], self._task_vec.shape[1])
    return super(HopperDiscreteTaskDistribution,
                 self)._evaluate(states, actions, tasks)

  def sample(self):
    task = np.zeros(self._num_tasks)
    task[np.random.choice(self._num_tasks)] = 1.0
    return task.astype(np.float32)

  @property
  def tasks(self):
    return np.eye(self._num_tasks, dtype=np.float32)


def walker_run(time_limit=walker._DEFAULT_TIME_LIMIT,  # pylint: disable=protected-access
               random=None,
               environment_kwargs=None):
  """Returns the Run task."""
  physics = walker.Physics.from_xml_string(*walker.get_model_and_assets())
  task = PlanarWalker(move_speed=walker._RUN_SPEED, random=random)   # pylint: disable=protected-access
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics,
      task,
      time_limit=time_limit,
      control_timestep=walker._CONTROL_TIMESTEP,   # pylint: disable=protected-access
      **environment_kwargs)


class PlanarWalker(walker.PlanarWalker):

  def initialize_episode(self, physics):
    ### Modified to not land on the head
    super(walker.PlanarWalker, self).initialize_episode(physics)


@gin.configurable
class WalkerDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.025
    time_limit = max_episode_steps * dt
    super(WalkerDynamics, self).__init__(
        env=walker_run(time_limit=time_limit), task_key=None)


@gin.configurable
class WalkerTaskDistribution(DMTaskDistribution):
  """Implements the tasks for the walker environment."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      walker_dynamics,
      vel_coef=1.0,
      stay_alive_bonus=0.0,
      terminate_on_fall=False,
      fall_height=1.2,
      new_bounds=False,
  ):
    self._dynamics = walker_dynamics
    self._vel_coef = vel_coef
    self._fall_height = fall_height
    self._stay_alive_bonus = stay_alive_bonus
    self._terminate_on_fall = terminate_on_fall
    self._orientation_index = self._get_obs_index("orientations")
    self._height_index = self._get_obs_index("height")
    self._vel_index = self._get_obs_index("velocity")

    ## [height, xvel, L ft X, L ft Z, R ft X, R ft Z, action norm]
    if new_bounds:
      # Modified so the height coefficient is zero and the X velocity is always
      # positive.
      low = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0])
      high = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    else:
      low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
      high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    self._task_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

  def _evaluate(self, states, actions, tasks):
    height = states[:, self._height_index]
    x_vel = states[:, self._vel_index + 1]
    left_foot_index = 2 * 3 + self._orientation_index
    right_foot_index = 2 * 6 + self._orientation_index
    xz_left_foot = states[:, left_foot_index:left_foot_index + 2]
    xz_right_foot = states[:, right_foot_index:right_foot_index + 2]
    ctrl = tf.reduce_sum(actions**2, axis=1)
    height_offset = 0.5
    foot_offset = np.array([0.0, -0.7])[None]
    features = tf.concat(
        [
            height[:, None] - height_offset,
            self._vel_coef * x_vel[:, None],
            xz_left_foot - foot_offset,
            xz_right_foot - foot_offset,
            ctrl[:, None],
        ],
        axis=1,
    )
    assert features.shape == tasks.shape
    rew = tf.reduce_sum(features * tasks, axis=1) + self._stay_alive_bonus
    if self._terminate_on_fall:  # DM control uses 1.2 as the stand height
      dones = height < self._fall_height
    else:
      dones = tf.cast(states[:, -1], tf.bool)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    task = np.random.uniform(self.task_space.low, self.task_space.high)
    return task.astype(np.float32)

  def state_to_sk(self, states):
    raise NotImplementedError


@gin.configurable
class HumanoidDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.025
    time_limit = max_episode_steps * dt
    super(HumanoidDynamics, self).__init__(
        env=humanoid.run(time_limit=time_limit), task_key=None)


@gin.configurable
class HumanoidTaskDistribution(DMTaskDistribution):
  """Implements the tasks for the humanoid environment."""

  def __init__(self, humanoid_dynamics, fall_penalty=0.0):  # pylint: disable=super-init-not-called
    assert fall_penalty >= 0
    self._fall_penalty = fall_penalty
    self._dynamics = humanoid_dynamics
    self._head_height_index = self._get_obs_index("head_height")
    self._extremities_index = self._get_obs_index("extremities")
    self._com_vel_index = self._get_obs_index("com_velocity")

    ## [height, ex1, ex2, ex3, ex4, com1, com2, com3, ctrl]
    ## NB: formerly we used a range of [-1, 1] for the head coefficient.
    low = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    self._task_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

  def _evaluate(self, states, actions, tasks):
    height = states[:, self._head_height_index]
    extremities = states[:,
                         self._extremities_index:self._extremities_index + 12]
    extremities_dist = tf.norm(
        tf.reshape(extremities, (-1, 4, 3)),
        axis=-1)  # distance to the 4 extremities
    com_vel = states[:, self._com_vel_index:self._com_vel_index + 3]
    ctrl = tf.reduce_sum(actions**2, axis=1)
    height_offset = 1.0  # DM uses 1.4 as the standing threshold
    extremities_offset = 0.5
    com_vel_offset = 0.0
    features = tf.concat(
        [
            height[:, None] - height_offset,  # 1
            extremities_dist - extremities_offset,  # 4
            com_vel - com_vel_offset,  # 3
            ctrl[:, None],
        ],
        axis=1,
    )  # 1
    fall_penalty = -1.0 * self._fall_penalty * tf.cast(height < 1.4, tf.float32)
    assert features.shape == tasks.shape
    rew = tf.reduce_sum(features * tasks, axis=1) + fall_penalty
    dones = tf.cast(states[:, -1], tf.bool)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    task = np.random.uniform(self.task_space.low, self.task_space.high)
    return task.astype(np.float32)

  def state_to_task(self, states):
    raise NotImplementedError


### SPARSE SUITE


@gin.configurable
class FingerDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.02
    time_limit = max_episode_steps * dt
    super(FingerDynamics, self).__init__(
        env=finger.turn_hard(time_limit=time_limit),
        task_key=["target_position", "dist_to_target"],
    )


@gin.configurable
class FingerTaskDistribution(DMTaskDistribution):
  """Implements the task for the finger environment.

  For a random policy, the distance is typically around 0.25, gets down to
  0.02 occasionally.
  """

  def __init__(self, finger_dynamics, margin=0.1, use_neg_rew=True):
    super(FingerTaskDistribution, self).__init__(
        finger_dynamics,
        task_key="target_position",
        relabel_task_key="position",
        relabel_dim=2,
        relabel_index_offset=2,
    )  # position[2:4] is the tip position
    self._margin = margin
    self._use_neg_rew = use_neg_rew

  def _evaluate(self, states, actions, tasks):
    del actions
    tip_pos = self.state_to_task(states)
    target_tip_pos = tasks
    dist = tf.norm(tip_pos - target_tip_pos, axis=1)
    dones = dist < self._margin
    if self._use_neg_rew:
      rew = tf.cast(dones, tf.float32) - 1.0
    else:
      rew = tf.cast(dones, tf.float32)

    assert rew.shape == dones.shape
    return rew, dones


@gin.configurable
class PointMassDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.02
    time_limit = max_episode_steps * dt
    super(PointMassDynamics, self).__init__(
        env=point_mass.hard(time_limit=time_limit), task_key=None)


class PointMassTaskDistribution(DMTaskDistribution):
  """Under random policy, distance is usually in [0.1, 0.4].

  Recommended range
  to try would be [0.1, 0.03, 0.01, 0.003].
  """

  def __init__(self, point_mass_dynamics, margin=0.1, use_neg_rew=True):
    super(PointMassTaskDistribution, self).__init__(
        point_mass_dynamics, task_key=None, relabel_task_key="position")
    self._margin = margin
    self._use_neg_rew = use_neg_rew
    self._pos_index = self._get_obs_index("position")

  def _evaluate(self, states, actions, tasks):
    del actions
    pos = states[:, self._pos_index:self._pos_index + 2]
    target_pos = tasks
    dist = tf.norm(pos - target_pos, axis=1)
    dones = dist < self._margin
    if self._use_neg_rew:
      rew = tf.cast(dones, tf.float32) - 1.0
    else:
      rew = tf.cast(dones, tf.float32)

    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    return np.random.uniform(-0.3, 0.3, size=2).astype(np.float32)


@gin.configurable
class StackerDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.01
    time_limit = max_episode_steps * dt
    super(StackerDynamics, self).__init__(
        env=stacker.stack_4(time_limit=time_limit), task_key=None)


class StackerDistribution(DMTaskDistribution):
  pass


@gin.configurable
class ReacherDynamics(DMDynamics):

  def __init__(self, max_episode_steps=1000):
    dt = 0.02
    time_limit = max_episode_steps * dt
    super(ReacherDynamics, self).__init__(
        env=reacher.hard(time_limit=time_limit), task_key=None)


class ReacherTaskDistribution(DMTaskDistribution):
  """Implements the reacher task.

  Under random policy, distance is usually in [0.1, 0.3], sometimes down to
  0.03. Recommended range to try would be [0.03, 0.01, 0.003, 0.001].
  """

  def __init__(self, reacher_dynamics, margin=0.1, use_neg_rew=True):
    super(ReacherTaskDistribution, self).__init__(
        reacher_dynamics, task_key=None, relabel_task_key="to_target")
    self._margin = margin
    self._use_neg_rew = use_neg_rew
    self._pos_index = self._get_obs_index("position")

  def _evaluate(self, states, actions, tasks):
    del actions
    pos = states[:, self._pos_index:self._pos_index + 2]
    target_pos = tasks
    dist = tf.norm(pos - target_pos, axis=1)
    dones = dist < self._margin
    if self._use_neg_rew:
      rew = tf.cast(dones, tf.float32) - 1.0
    else:
      rew = tf.cast(dones, tf.float32)

    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0.05, 0.20)
    return radius * np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)


def sample_circle(min_radius, max_radius):
  while True:
    x, y = np.random.uniform(-max_radius, max_radius, 2)
    r = np.sqrt(x**2 + y**2)
    if min_radius <= r <= max_radius:
      return [x, y]


@gin.configurable
class ReacherDiscreteTaskDistribution(DMTaskDistribution):
  """Implements a discrete set of tasks for the reacher environment."""

  def __init__(self, reacher_dynamics, margin=0.1, num_tasks=32, linear=False):
    super(ReacherDiscreteTaskDistribution, self).__init__(
        reacher_dynamics, task_key=None, relabel_task_key="to_target")
    self._pos_index = self._get_obs_index("position")
    self._margin = margin
    self._linear = linear
    np.random.seed(0)
    self._goal_vec = np.array(
        [sample_circle(0.05, 0.2) for _ in range(num_tasks)])
    low = np.zeros(num_tasks)
    high = np.ones(num_tasks)
    self._task_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

  def _evaluate(self, states, actions, tasks):
    del actions
    pos = states[:, self._pos_index:self._pos_index + 2]
    dist = tf.norm(
        pos[:, None] - self._goal_vec[None], axis=-1)  # B x num_tasks
    if self._linear:
      at_goals = tf.cast(dist < self._margin, tf.float32)
      rew = tf.reduce_sum(at_goals * tasks, axis=1)
      dones = tf.cast(tf.zeros_like(rew), tf.bool)
    else:
      dist = tf.reduce_sum(dist * tasks, axis=1)
      dones = dist < self._margin
      rew = tf.cast(dones, tf.float32) - 1.0
    assert rew.shape == dones.shape
    return rew, dones

  @property
  def tasks(self):
    if self._linear or len(self._goal_vec) > 32:
      return None
    else:
      return np.eye(len(self._goal_vec))

  def sample(self):
    num_tasks = len(self._goal_vec)
    if self._linear:
      task = np.random.uniform(self._task_space.low, self._task_space.high)
    else:
      task = np.zeros(num_tasks)
      task[np.random.choice(num_tasks)] = 1.0
    return task.astype(np.float32)


@gin.configurable
class SwimmerDynamics(DMDynamics):
  """Wrapper around the dm_control swimmer dynamics."""

  def __init__(self, max_episode_steps=1000):
    dt = 0.03
    time_limit = max_episode_steps * dt
    super(SwimmerDynamics, self).__init__(
        env=swimmer.swimmer6(time_limit=time_limit), task_key=None)

    # Modify observation space to include position of nose
    num_dim = 2 + len(self._observation_space.low)
    self._observation_space = gym.spaces.Box(
        low=np.full((num_dim,), -np.inf),
        high=np.full((num_dim,), np.inf),
        dtype=np.float32,
    )

  def _ts_to_obs(self, ts):
    obs = super(SwimmerDynamics, self)._ts_to_obs(ts)
    nose_pos = self._env.physics.named.data.geom_xpos[
        "nose"][:2]  # Ignore Z dimension
    return np.concatenate([nose_pos, obs]).astype(np.float32)


@gin.configurable
class SwimmerTaskDistribution(DMTaskDistribution):
  """Implements the task for the swimmer environment.

  Under random policy, distance is usually in [0.4, 0.9].
  Recommended range to try would be [0.1, 0.03, 0.01, 0.003].
  """

  def __init__(self, swimmer_dynamics):
    super(SwimmerTaskDistribution, self).__init__(
        swimmer_dynamics, task_key=None, relabel_task_key="to_target")

  def _evaluate(self, states, actions, tasks):
    del actions
    pos = states[:, :2]
    target_pos = tasks
    dist = tf.norm(pos - target_pos, axis=1)
    target_size = 0.1
    rew = tf_tolerance(
        dist,
        bounds=(0, target_size),
        margin=5 * target_size,
        sigmoid="long_tail")
    dones = tf.cast(states[:, -1], tf.bool)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    return np.random.uniform(-2.0, 2.0, size=2).astype(np.float32)


@gin.configurable
class FishDynamics(DMDynamics):
  """Wrapper around the dm_control fish dynamics."""

  def __init__(self, max_episode_steps=1000):
    dt = 0.04
    time_limit = max_episode_steps * dt
    super(FishDynamics, self).__init__(
        env=fish.swim(time_limit=time_limit), task_key=None)

    # Modify observation space to include mouth position
    num_dim = 3 + len(self._observation_space.low)
    self._observation_space = gym.spaces.Box(
        low=np.full((num_dim,), -np.inf),
        high=np.full((num_dim,), np.inf),
        dtype=np.float32,
    )

  def _ts_to_obs(self, ts):
    obs = super(FishDynamics, self)._ts_to_obs(ts)
    mouth_pos = self._env.physics.named.data.geom_xpos["mouth"]
    return np.concatenate([mouth_pos, obs]).astype(np.float32)


@gin.configurable
class FishTaskDistribution(DMTaskDistribution):
  """Under random policy, distance is usually in [0.1, 0.3].

  Recommended range to try would be [0.03, 0.01, 0.003, 0.001].
  """

  def __init__(self, fish_dynamics):
    super(FishTaskDistribution, self).__init__(
        fish_dynamics, task_key=None, relabel_task_key="target")
    self._upright_index = (self._get_obs_index("upright") + 3
                          )  # because we add three dimensions for the mouth

  def _evaluate(self, states, actions, tasks):
    del actions
    pos = states[:, :3]
    upright = states[:, self._upright_index]
    target_pos = tasks
    dist = tf.norm(pos - target_pos, axis=1)

    radii = 0.005 + 0.04
    is_upright = 0.5 * (upright + 0.5)
    in_target = tf_tolerance(
        dist, bounds=(0, radii), margin=2 * radii, sigmoid="gaussian")
    rew = (7 * in_target + is_upright) / 8
    dones = tf.cast(states[:, -1], tf.bool)
    assert rew.shape == dones.shape
    return rew, dones

  def sample(self):
    low = np.array([-0.4, -0.4, 0.1])
    high = np.array([0.4, 0.4, 0.3])
    return np.random.uniform(low, high).astype(np.float32)
