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

# Lint as: python3
"""Utilities for the soft relabelling project.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from envs import dm_env
from envs import multitask
from envs import point_env
import gin
from gym.envs import registration
from gym.wrappers import TimeLimit
import numpy as np
import tensorflow as tf
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import gym_wrapper
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import normal_projection_network
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

# The Sawyer and dm_control environments don't play nicely together.
# If imported together we get a segfault.
try:
  from envs import sawyer_env  # pylint: disable=g-import-not-at-top
except ImportError:
  print("The sawyer_env and dm_control environments don\'t play nicely "
        "together. If imported together we get a segfault.")


@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      std_bias_initializer_value=std_bias_initializer_value,
      scale_distribution=True,
  )


@gin.configurable
def get_py_env(env_name,
               max_episode_steps=None,
               constant_task=None,
               use_neg_rew=None,
               margin=None):
  """Load an environment.

  Args:
    env_name: (str) name of the environment.
    max_episode_steps: (int) maximum number of steps per episode. Set to None
      to not include a limit.
    constant_task: specifies a fixed task to use for all episodes. Set to None
      to use tasks sampled from the task distribution.
    use_neg_rew: (bool) For the goal-reaching tasks, indicates whether to use
      a (-1, 0) sparse reward (use_neg_reward = True) or a (0, 1) sparse reward.
    margin: (float) For goal-reaching tasks, indicates the desired distance
      to the goal.
  Returns:
    env: the environment, build from a dynamics and task distribution
    task_distribution: the task distribution used for the environment.
  """
  if "sawyer" in env_name:
    print(("ERROR: Modify utils.py to import sawyer_env and not dm_env. "
           "Currently the sawyer_env import is commented out to prevent "
           "a segfault from occuring when trying to import both sawyer_env "
           "and dm_env"))
    assert False

  if env_name.split("_")[0] == "point":
    _, walls, resize_factor = env_name.split("_")
    dynamics = point_env.PointDynamics(
        walls=walls, resize_factor=int(resize_factor))
    task_distribution = point_env.PointGoalDistribution(dynamics)
  elif env_name.split("_")[0] == "pointTask":
    _, walls, resize_factor = env_name.split("_")
    dynamics = point_env.PointDynamics(
        walls=walls, resize_factor=int(resize_factor))
    task_distribution = point_env.PointTaskDistribution(dynamics)

  elif env_name == "quadruped-run":
    dynamics = dm_env.QuadrupedRunDynamics()
    task_distribution = dm_env.QuadrupedRunTaskDistribution(dynamics)
  elif env_name == "quadruped":
    dynamics = dm_env.QuadrupedRunDynamics()
    task_distribution = dm_env.QuadrupedContinuousTaskDistribution(dynamics)
  elif env_name == "hopper":
    dynamics = dm_env.HopperDynamics()
    task_distribution = dm_env.HopperTaskDistribution(dynamics)
  elif env_name == "hopper-discrete":
    dynamics = dm_env.HopperDynamics()
    task_distribution = dm_env.HopperDiscreteTaskDistribution(dynamics)
  elif env_name == "walker":
    dynamics = dm_env.WalkerDynamics()
    task_distribution = dm_env.WalkerTaskDistribution(dynamics)
  elif env_name == "humanoid":
    dynamics = dm_env.HumanoidDynamics()
    task_distribution = dm_env.HumanoidTaskDistribution(dynamics)

  ### sparse tasks
  elif env_name == "finger":
    dynamics = dm_env.FingerDynamics()
    task_distribution = dm_env.FingerTaskDistribution(
        dynamics, use_neg_rew=use_neg_rew, margin=margin)
  elif env_name == "manipulator":
    dynamics = dm_env.ManipulatorDynamics()
    task_distribution = dm_env.ManipulatorTaskDistribution(dynamics)
  elif env_name == "point-mass":
    dynamics = dm_env.PointMassDynamics()
    task_distribution = dm_env.PointMassTaskDistribution(
        dynamics, use_neg_rew=use_neg_rew, margin=margin)
  elif env_name == "stacker":
    dynamics = dm_env.StackerDynamics()
    task_distribution = dm_env.FingerStackerDistribution(
        dynamics, use_neg_rew=use_neg_rew, margin=margin)
  elif env_name == "swimmer":
    dynamics = dm_env.SwimmerDynamics()
    task_distribution = dm_env.SwimmerTaskDistribution(dynamics)
  elif env_name == "fish":
    dynamics = dm_env.FishDynamics()
    task_distribution = dm_env.FishTaskDistribution(dynamics)

  elif env_name == "sawyer-reach":
    dynamics = sawyer_env.SawyerDynamics()
    task_distribution = sawyer_env.SawyerReachTaskDistribution(dynamics)
  else:
    raise NotImplementedError("Unknown environment: %s" % env_name)
  gym_env = multitask.Environment(
      dynamics, task_distribution, constant_task=constant_task)
  if max_episode_steps is not None:
    # Add a placeholder spec so the TimeLimit wrapper works.
    gym_env.spec = registration.EnvSpec("env-v0")
    gym_env = TimeLimit(gym_env, max_episode_steps)
  wrapped_env = gym_wrapper.GymWrapper(gym_env, discount=1.0, auto_reset=True)
  return wrapped_env, task_distribution


def get_env(env_name,
            max_episode_steps=None,
            constant_task=None,
            num_parallel_environments=1):
  """Loads the environment.

  Args:
    env_name: (str) name of the environment.
    max_episode_steps: (int) maximum number of steps per episode. Set to None
      to not include a limit.
    constant_task: specifies a fixed task to use for all episodes. Set to None
      to use tasks sampled from the task distribution.
    num_parallel_environments: (int) Number of parallel environments.
  Returns:
    tf_env: the environment, build from a dynamics and task distribution. This
      environment is an instance of TFPyEnvironment.
    task_distribution: the task distribution used for the environment.
  """

  def env_load_fn(return_task_distribution=False):
    py_env, task_distribution = get_py_env(
        env_name,
        max_episode_steps=max_episode_steps,
        constant_task=constant_task)
    if return_task_distribution:
      return (py_env, task_distribution)
    else:
      return py_env

  py_env, task_distribution = env_load_fn(return_task_distribution=True)
  if num_parallel_environments > 1:
    del py_env
    py_env = parallel_py_environment.ParallelPyEnvironment(
        [env_load_fn] * num_parallel_environments)
  tf_env = tf_py_environment.TFPyEnvironment(py_env)
  return tf_env, task_distribution


class DataCollector(object):

  """Class for collecting data and adding trajectories to the replay buffer."""

  def __init__(
      self,
      tf_env,
      policy,
      replay_buffer,
      max_episode_steps=None,
      observers=(),
  ):
    """Initialize the DataCollector.

    Args:
      tf_env: The environment.
      policy: The policy to use for data collection.
      replay_buffer: The replay buffer for storing experience.
      max_episode_steps: Maximum number of steps per episode.
      observers: List of functions to apply to each transition.
    """

    self._tf_env = tf_env
    self._policy = policy
    self._replay_buffer = replay_buffer
    self._traj_vec = []
    self._ts = tf_env.reset()
    self._max_episode_steps = max_episode_steps
    self._observers = observers

  def step(self, policy=None):
    """Takes one step in the environment.

    Args:
      policy: (optional) If specified, uses a particular policy for choosing
        actions. Otherwise, uses the policy provided when instantiated.
    """
    if policy is None:
      policy = self._policy
    action = policy.action(self._ts)
    next_ts = self._tf_env.step(action)
    traj = trajectory.from_transition(self._ts, action, next_ts)
    self._traj_vec.append(traj)
    if next_ts.is_last() or len(self._traj_vec) == self._max_episode_steps:
      self._replay_buffer.add_batch(self._traj_vec)
      self._traj_vec = []
      self._ts = self._tf_env.reset()
    else:
      self._ts = next_ts

    # When collecting data for training, we sometimes artificially
    # start a new episode before the old one has finished. Since
    # we don't want to set done = True (which would interfere with
    # training), the observers don't know that we've done this and
    # assume that the previous episode is continuing. We therefore
    # artificially set done = True in the trajectory passed to
    # the observers whenever we are about to start a new episode.
    for observer in self._observers:
      if self._traj_vec:  # i.e., if len(self._traj_vec) == 0:
        observer(
            traj.replace(next_step_type=tf.constant([time_step.StepType.LAST])))
      else:
        observer(traj)

  def reset(self):
    self._ts = self._tf_env.reset()


class AverageSuccessMetric(tf_metrics.AverageEpisodeLengthMetric):
  """Metric to compute the average success rate.

  Modified to say that we are  successful if the episode length is less than the
  maximum.
  """

  def __init__(self, max_episode_steps, name="AverageSuccess", **kwargs):
    self._max_episode_steps = max_episode_steps
    super(AverageSuccessMetric, self).__init__(name=name, **kwargs)

  def result(self):
    return tf.reduce_mean(
        tf.cast(self._buffer.data < self._max_episode_steps, self._dtype))


class FixedTask(object):
  """A scope for fixing the task used in each episode."""

  def __init__(self, env, task):
    """Initialize the FixedTask scope.

    Args:
      env: an instace of multitask.Environment.
      task: a vector indicating the task to use.
    """
    assert len(env.envs) == 1
    self._env = env.envs[0].gym.env
    assert isinstance(self._env, multitask.Environment)
    self._task = task
    self._orig_task = None

  def __enter__(self):
    if self._env._constant_task is not None:
      self._orig_task = self._env._constant_task.copy()
    self._env.set_constant_task(self._task)

  def __exit__(self, type_, value, traceback):
    del type_, value, traceback
    self._env.set_constant_task(self._orig_task)
    self._orig_task = None
