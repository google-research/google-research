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

"""TF-agents suite for loading Adversarial environments.

Adds two new functions: reset_agent, and step_adversary in addition to usual
RL env functions. Therefore we have the following environment functions:
  env.reset(): completely resets the environment and removes anything the
    adversary has built.
  env.reset_agent(): resets the position of the agent, but does not
    remove the obstacles the adversary has created when building the env.
  env.step(): steps the agent as before in the environment. i.e. if the agent
    passes action 'left' it will move left.
  env.step_adversary(): processes an adversary action, which involves choosing
    the location of the agent, goal, or an obstacle.

Adds additional functions for logging metrics related to the generated
environments, like the shortest path length to the goal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import gym
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import batched_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.utils import nest_utils


@gin.configurable
def load(environment_name,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None,
         gym_kwargs=None,
         auto_reset=True):
  """Loads the selected environment and wraps it with the specified wrappers.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    environment_name: Name for the environment to load.
    discount: Discount to use for the environment.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no max_episode_steps set in the environment's spec.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    env_wrappers: Iterable with references to wrapper classes to use on the
      gym_wrapped environment.
    spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
      default dtype for the tensors. An easy way to configure a custom
      mapping through Gin is to define a gin-configurable function that returns
      desired mapping and call it in your Gin config file, for example:
      `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
    gym_kwargs: The kwargs to pass to the Gym environment class.
    auto_reset: If True (default), reset the environment automatically after a
      terminal state is reached.

  Returns:
    A PyEnvironment instance.
  """
  gym_kwargs = gym_kwargs if gym_kwargs else {}
  gym_spec = gym.spec(environment_name)
  gym_env = gym_spec.make(**gym_kwargs)

  if max_episode_steps is None and gym_spec.max_episode_steps is not None:
    max_episode_steps = gym_spec.max_episode_steps

  for wrapper in gym_env_wrappers:
    gym_env = wrapper(gym_env)

  env = AdversarialGymWrapper(
      gym_env,
      discount=discount,
      spec_dtype_map=spec_dtype_map,
      auto_reset=auto_reset,
  )

  if max_episode_steps is not None and max_episode_steps > 0:
    env = wrappers.TimeLimit(env, max_episode_steps)

  for wrapper in env_wrappers:
    env = wrapper(env)

  return env


class AdversarialGymWrapper(gym_wrapper.GymWrapper):
  """Wrapper implementing PyEnvironment interface for adversarial environments.

  Implements special reset_agent and step_adversary functions that are not
  present in a normal Gym environment.
  """

  def __init__(self,
               gym_env,
               discount=1.0,
               spec_dtype_map=None,
               match_obs_space_dtype=True,
               auto_reset=False,
               simplify_box_bounds=True):

    super(AdversarialGymWrapper, self).__init__(
        gym_env, discount, spec_dtype_map, match_obs_space_dtype, auto_reset,
        simplify_box_bounds)

    self.adversary_observation_spec = gym_wrapper.spec_from_gym_space(
        self._gym_env.adversary_observation_space, name='observation')
    self.adversary_action_spec = gym_wrapper.spec_from_gym_space(
        self._gym_env.adversary_action_space, name='action')
    self.adversary_time_step_spec = ts_lib.time_step_spec(
        self.adversary_observation_spec, self.reward_spec())
    self.adversary_flat_obs_spec = tf.nest.flatten(
        self.adversary_observation_spec)

  def _reset(self):
    observation = self._gym_env.reset()
    self._info = None
    self._done = False

    if self._match_obs_space_dtype:
      observation = self._adversary_to_obs_space_dtype(observation)
    reset_step = ts_lib.restart(observation, reward_spec=self.reward_spec())
    return reset_step

  def reset_random(self):
    observation = self._gym_env.reset_random()
    self._info = None
    self._done = False

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)
    self._current_time_step = ts_lib.restart(
        observation, reward_spec=self.reward_spec())
    return self._current_time_step

  def reset_agent(self):
    observation = self._gym_env.reset_agent()
    self._info = None
    self._done = False

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)
    self._current_time_step = ts_lib.restart(
        observation, reward_spec=self.reward_spec())
    return self._current_time_step

  def _adversary_to_obs_space_dtype(self, observation):
    # Make sure we handle cases where observations are provided as a list.
    flat_obs = nest_utils.flatten_up_to(
        self.adversary_observation_spec, observation)

    matched_observations = []
    for spec, obs in zip(self.adversary_flat_obs_spec, flat_obs):
      matched_observations.append(np.asarray(obs, dtype=spec.dtype))
    return tf.nest.pack_sequence_as(self.adversary_observation_spec,
                                    matched_observations)

  def _step(self, action):
    # Automatically reset the environments on step if they need to be reset.
    if self._handle_auto_reset and self._done:
      return self.reset_agent()

    action = action.item() if self._action_is_discrete else action

    observation, reward, self._done, self._info = self._gym_env.step(action)

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)

    reward = np.asarray(reward, dtype=self.reward_spec().dtype)
    outer_dims = nest_utils.get_outer_array_shape(reward, self.reward_spec())

    if self._done:
      return ts_lib.termination(observation, reward, outer_dims=outer_dims)
    else:
      return ts_lib.transition(observation, reward, self._discount,
                               outer_dims=outer_dims)

  def step_adversary(self, action):
    action = action.item() if self._action_is_discrete else action

    observation, reward, self._done, self._info = self._gym_env.step_adversary(
        action)

    if self._match_obs_space_dtype:
      observation = self._adversary_to_obs_space_dtype(observation)

    reward = np.asarray(reward, dtype=self.reward_spec().dtype)
    outer_dims = nest_utils.get_outer_array_shape(reward, self.reward_spec())

    if self._done:
      return ts_lib.termination(observation, reward, outer_dims=outer_dims)
    else:
      return ts_lib.transition(observation, reward, self._discount,
                               outer_dims=outer_dims)


@gin.configurable
class AdversarialBatchedPyEnvironment(
    batched_py_environment.BatchedPyEnvironment):
  """Batch together multiple adversarial py environments acting as single batch.

  The environments should only access shared python variables using
  shared mutex locks (from the threading module).
  """

  def __init__(self, envs, multithreading=True):
    super(AdversarialBatchedPyEnvironment, self).__init__(
        envs, multithreading=multithreading)

    self.adversary_action_spec = self._envs[0].adversary_action_spec
    self.adversary_observation_spec = self._envs[0].adversary_observation_spec
    self.adversary_time_step_spec = self._envs[0].adversary_time_step_spec

  def get_num_blocks(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].n_clutter_placed, tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.n_clutter_placed, tf.float32), self._envs)

  def get_distance_to_goal(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].distance_to_goal, tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.distance_to_goal, tf.float32), self._envs)

  def get_deliberate_placement(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].deliberate_agent_placement, tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.deliberate_agent_placement, tf.float32),
          self._envs)

  def get_goal_x(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].get_goal_x(), tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.get_goal_x(), tf.float32),
          self._envs)

  def get_goal_y(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].get_goal_y(), tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.get_goal_y(), tf.float32),
          self._envs)

  def get_passable(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].passable, tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.passable, tf.float32),
          self._envs)

  def get_shortest_path_length(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_tensors(
          tf.cast(self._envs[0].shortest_path_length, tf.float32))
    else:
      return tf.stack(
          lambda env: tf.cast(env.shortest_path_length, tf.float32),
          self._envs)

  def reset_agent(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(self._envs[0].reset_agent())
    else:
      time_steps = self._execute(lambda env: env.reset_agent(), self._envs)
      return nest_utils.stack_nested_arrays(time_steps)

  def reset_random(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(self._envs[0].reset_random())
    else:
      time_steps = self._execute(lambda env: env.reset_random(), self._envs)
      return nest_utils.stack_nested_arrays(time_steps)

  def step_adversary(self, actions):
    if self._num_envs == 1:
      actions = nest_utils.unbatch_nested_array(actions)
      time_steps = self._envs[0].step_adversary(actions)
      return nest_utils.batch_nested_array(time_steps)
    else:
      unstacked_actions = batched_py_environment.unstack_actions(actions)
      if len(unstacked_actions) != self.batch_size:
        raise ValueError(
            'Primary dimension of action items does not match '
            'batch size: %d vs. %d' % (len(unstacked_actions), self.batch_size))
      time_steps = self._execute(
          lambda env_action: env_action[0].step_adversary(env_action[1]),
          zip(self._envs, unstacked_actions))
      return nest_utils.stack_nested_arrays(time_steps)


class AdversarialTFPyEnvironment(tf_py_environment.TFPyEnvironment):
  """Override TFPyEnvironment to add support for additional adversary functions.

  Note that the 'step' function resets the agent, but 'reset' resets the whole
  environment. Therefore use 'reset_agent' to reset just the agent to its
  initial location, without resetting the environment the adversary has created.

  The time_step_spec and other specs relate to the agent's observations, and
  there are additional specs for the adversarial policy that alters the
  environment.

  The adversary's specs should match the output of reset(), step_adversary(),
  _current_time_step(), and self.time_step, while the agent's specs should
  match reset_agent(), step(), _current_agent_time_step(), and
  self._agent_time_step.
  """

  def __init__(self, environment, check_dims=False, isolation=False):
    """Calls parent constructors and initializes adversary specs.

    Args:
      environment: A tf-agents PyEnvironment, or a `callable` that returns
        an environment of this form.
      check_dims: Whether the batch dimensions should be checked in the 'step'
        function.
      isolation: If True, create a dedicated thread for interactions with the
        environment. If Falso, interactions with the environment occur within
        whichever thread calls a method belonging to this class. See tf-agents
        parent class documentation for more details.
    """
    # Prevent parent class from using its own batched environment
    super(AdversarialTFPyEnvironment, self).__init__(
        environment, check_dims=check_dims, isolation=isolation)

    if not environment.batched:
      self._env = AdversarialBatchedPyEnvironment(
          [environment], multithreading=not self._pool)

    self._agent_time_step = None

    self.adversary_action_spec = tensor_spec.from_spec(
        self._env.adversary_action_spec)
    self.adversary_time_step_spec = tensor_spec.from_spec(
        self._env.adversary_time_step_spec)
    self.adversary_observation_spec = tensor_spec.from_spec(
        self._env.adversary_observation_spec)

    self._adversary_time_step_dtypes = [
        s.dtype for s in tf.nest.flatten(self.adversary_time_step_spec)
    ]

  # Make sure this is called without conversion from tf.function.
  @tf.autograph.experimental.do_not_convert()
  def reset_agent(self):
    def _reset_py():
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        self._agent_time_step = self._env.reset_agent()

    def _isolated_reset_py():
      return self._execute(_reset_py)

    with tf.name_scope('reset_agent'):
      reset_op = tf.numpy_function(
          _isolated_reset_py,
          [],  # No inputs.
          [],
          name='reset_py_func')
      with tf.control_dependencies([reset_op]):
        return self._current_agent_time_step()

  @tf.autograph.experimental.do_not_convert()
  def _current_time_step(self):
    def _current_time_step_py():
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        if self._time_step is None:
          self._time_step = self._env.reset()
        return tf.nest.flatten(self._time_step)

    def _isolated_current_time_step_py():
      return self._execute(_current_time_step_py)

    with tf.name_scope('current_time_step'):
      outputs = tf.numpy_function(
          _isolated_current_time_step_py,
          [],  # No inputs.
          self._time_step_dtypes,
          name='current_time_step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]
      return self._set_names_and_shapes(
          self.adversary_time_step_spec, self.adversary_observation_spec,
          step_type, reward, discount, *flat_observations)

  @tf.autograph.experimental.do_not_convert()
  def _current_agent_time_step(self):
    def _current_agent_time_step_py():
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        if self._agent_time_step is None:
          self._agent_time_step = self._env.reset_agent()
        return tf.nest.flatten(self._agent_time_step)

    def _isolated_current_agent_time_step_py():
      return self._execute(_current_agent_time_step_py)

    with tf.name_scope('current_agent_time_step'):
      outputs = tf.numpy_function(
          _isolated_current_agent_time_step_py,
          [],  # No inputs.
          self._time_step_dtypes,
          name='current_agent_time_step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]
      return self._set_names_and_shapes(
          self.time_step_spec(), self.observation_spec(),
          step_type, reward, discount, *flat_observations)

  @tf.autograph.experimental.do_not_convert()
  def reset_random(self):
    def _reset_py():
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        self._time_step = self._env.reset_random()

    def _isolated_reset_py():
      return self._execute(_reset_py)

    with tf.name_scope('reset_random'):
      reset_op = tf.numpy_function(
          _isolated_reset_py,
          [],  # No inputs.
          [],
          name='reset_py_func')
      with tf.control_dependencies([reset_op]):
        return self._current_random_time_step()

  @tf.autograph.experimental.do_not_convert()
  def _current_random_time_step(self):
    def _current_random_time_step_py():
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        if self._time_step is None:
          self._time_step = self._env.reset_random()
        return tf.nest.flatten(self._time_step)

    def _isolated_current_random_time_step_py():
      return self._execute(_current_random_time_step_py)

    with tf.name_scope('current_random_time_step'):
      outputs = tf.numpy_function(
          _isolated_current_random_time_step_py,
          [],  # No inputs.
          self._time_step_dtypes,
          name='current_random_time_step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]
      return self._set_names_and_shapes(
          self.time_step_spec(), self.observation_spec(),
          step_type, reward, discount, *flat_observations)

  def _set_names_and_shapes(
      self, ts_spec, obs_spec, step_type, reward, discount, *flat_observations):
    """Returns a `TimeStep` namedtuple."""
    step_type = tf.identity(step_type, name='step_type')
    reward = tf.identity(reward, name='reward')
    discount = tf.identity(discount, name='discount')
    batch_shape = () if not self.batched else (self.batch_size,)
    batch_shape = tf.TensorShape(batch_shape)
    if not tf.executing_eagerly():
      # Shapes are not required in eager mode.
      reward.set_shape(batch_shape.concatenate(ts_spec.reward.shape))
      step_type.set_shape(batch_shape)
      discount.set_shape(batch_shape)
    # Give each tensor a meaningful name and set the static shape.
    named_observations = []
    for obs, spec in zip(flat_observations, tf.nest.flatten(obs_spec)):
      named_observation = tf.identity(obs, name=spec.name)
      if not tf.executing_eagerly():
        named_observation.set_shape(batch_shape.concatenate(spec.shape))
      named_observations.append(named_observation)

    observations = tf.nest.pack_sequence_as(obs_spec, named_observations)

    return ts_lib.TimeStep(step_type, reward, discount, observations)

  # Make sure this is called without conversion from tf.function.
  @tf.autograph.experimental.do_not_convert()
  def _step(self, actions):
    def _step_py(*flattened_actions):
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        packed = tf.nest.pack_sequence_as(
            structure=self.action_spec(), flat_sequence=flattened_actions)
        self._agent_time_step = self._env.step(packed)
        return tf.nest.flatten(self._agent_time_step)

    def _isolated_step_py(*flattened_actions):
      return self._execute(_step_py, *flattened_actions)

    with tf.name_scope('step'):
      flat_actions = [tf.identity(x) for x in tf.nest.flatten(actions)]
      if self._check_dims:
        for action in flat_actions:
          dim_value = tf.compat.dimension_value(action.shape[0])
          if (action.shape.rank == 0 or
              (dim_value is not None and dim_value != self.batch_size)):
            raise ValueError(
                'Expected actions whose major dimension is batch_size (%d), '
                'but saw action with shape %s:\n   %s' %
                (self.batch_size, action.shape, action))
      outputs = tf.numpy_function(
          _isolated_step_py,
          flat_actions,
          self._time_step_dtypes,
          name='step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]

      return self._set_names_and_shapes(
          self.time_step_spec(), self.observation_spec(),
          step_type, reward, discount, *flat_observations)

  # Make sure this is called without conversion from tf.function.
  @tf.autograph.experimental.do_not_convert()
  def step_adversary(self, actions):
    def _step_adversary_py(*flattened_actions):
      with tf_py_environment._check_not_called_concurrently(self._lock):  # pylint:disable=protected-access
        packed = tf.nest.pack_sequence_as(
            structure=self.adversary_action_spec,
            flat_sequence=flattened_actions)
        self._time_step = self._env.step_adversary(packed)
        return tf.nest.flatten(self._time_step)

    def _isolated_step_adversary_py(*flattened_actions):
      return self._execute(_step_adversary_py, *flattened_actions)

    with tf.name_scope('step_adversary'):
      flat_actions = [tf.identity(x) for x in tf.nest.flatten(actions)]
      if self._check_dims:
        for action in flat_actions:
          dim_value = tf.compat.dimension_value(action.shape[0])
          if (action.shape.rank == 0 or
              (dim_value is not None and dim_value != self.batch_size)):
            raise ValueError(
                'Expected adversary actions whose major dimension is batch_size '
                '(%d), but saw action with shape %s:\n   %s' %
                (self.batch_size, action.shape, action))
      outputs = tf.numpy_function(
          _isolated_step_adversary_py,
          flat_actions,
          self._adversary_time_step_dtypes,
          name='step_adversary_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]

      return self._set_names_and_shapes(
          self.adversary_time_step_spec, self.adversary_observation_spec,
          step_type, reward, discount, *flat_observations)
