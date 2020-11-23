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

"""Helper functions for C-learning."""

import gin
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common


def truncated_geometric(horizon, gamma):
  """Generates sampled from a truncated geometric distribution.

  Args:
    horizon: A 1-d tensor of horizon lengths for each element in the batch.
      The returned samples will be less than the corresponding horizon.
    gamma: The discount factor. Importantly, we sample from a Geom(1 - gamma)
      distribution.
  Returns:
    indices: A 1-d tensor of integers, one for each element of the batch.
  """
  max_horizon = tf.reduce_max(horizon)
  batch_size = tf.shape(horizon)[0]
  indices = tf.tile(
      tf.range(max_horizon, dtype=tf.float32)[None], (batch_size, 1))
  probs = tf.where(indices < horizon[:, None], gamma**indices,
                   tf.zeros_like(indices))
  probs = probs / tf.reduce_sum(probs, axis=1)[:, None]
  indices = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
  return indices[:, 0]  # Remove the extra dimension.


def get_future_goals(observation, discount, gamma):
  """Samples future goals according to a geometric distribution."""
  num_obs = observation.shape[0]
  traj_len = observation.shape[1]
  first_terminal_or_zero = tf.argmax(
      discount == 0, axis=1, output_type=tf.int32)
  any_terminal = tf.reduce_any(discount == 0, axis=1)
  first_terminal = tf.where(any_terminal, first_terminal_or_zero, traj_len)
  first_terminal = tf.cast(first_terminal, tf.float32)
  if num_obs == 0:
    # The truncated_geometric function breaks if called on an empty list.
    # In that case, we manually create an empty list of future goals.
    indices = tf.zeros((0,), dtype=tf.int32)
  else:
    indices = truncated_geometric(first_terminal, gamma)
  stacked_indices = tf.stack([tf.range(num_obs), indices], axis=1)
  return tf.gather_nd(observation, stacked_indices)


def get_last_goals(observation, discount):
  """Extracts that final observation before termination.

  Args:
    observation: a B x T x D tensor storing the next T time steps. These time
      steps may be part of a new trajectory. This function will only consider
      observations that occur before the first terminal.
    discount: a B x T tensor indicating whether the episode has terminated.
  Returns:
    last_obs: a B x D tensor storing the last observation in each trajectory
      that occurs before the first terminal.
  """
  num_obs = observation.shape[0]
  traj_len = observation.shape[1]
  first_terminal_or_zero = tf.argmax(
      discount == 0, axis=1, output_type=tf.int32)
  any_terminal = tf.reduce_any(discount == 0, axis=1)
  first_terminal = tf.where(any_terminal, first_terminal_or_zero, traj_len)
  # If the first state is terminal then first_terminal - 1 = -1. In this case we
  # use the state itself as the goal.
  last_nonterminal = tf.clip_by_value(first_terminal - 1, 0, traj_len)
  stacked_indices = tf.stack([tf.range(num_obs), last_nonterminal], axis=1)
  last_obs = tf.gather_nd(observation, stacked_indices)
  return last_obs


@gin.configurable
def obs_to_goal(obs, start_index=0, end_index=None):
  if end_index is None:
    return obs[:, start_index:]
  else:
    return obs[:, start_index:end_index]


@gin.configurable
def goal_fn(experience,
            buffer_info,
            relabel_orig_prob=0.0,
            relabel_next_prob=0.5,
            relabel_future_prob=0.0,
            relabel_last_prob=0.0,
            batch_size=None,
            obs_dim=None,
            gamma=None):
  """Given experience, sample goals in three ways.

  The three ways are using the next state, an arbitrary future state, or a
  random state. For the future state relabeling, care must be taken to ensure
  that we don't sample experience across the episode boundary. We automatically
  set relabel_random_prob = (1 - relabel_next_prob - relabel_future_prob).

  Args:
    experience: The experience that we aim to relabel.
    buffer_info: Information about the replay buffer. We will not change this.
    relabel_orig_prob: (float) Fraction of experience to not relabel.
    relabel_next_prob: (float) Fraction of experience to relabel with the next
      state.
    relabel_future_prob: (float) Fraction of experience to relabel with a future
      state.
    relabel_last_prob: (float) Fraction of experience to relabel with the
      final state.
    batch_size: (int) The size of the batch.
    obs_dim: (int) The dimension of the observation.
    gamma: (float) The discount factor. Future states are sampled according to
      a Geom(1 - gamma) distribution.
  Returns:
    experience: A modified version of the input experience where the goals
      have been changed and the rewards and terminal flags are recomputed.
    buffer_info: Information about the replay buffer.

  """
  assert batch_size is not None
  assert obs_dim is not None
  assert gamma is not None
  relabel_orig_num = int(relabel_orig_prob * batch_size)
  relabel_next_num = int(relabel_next_prob * batch_size)
  relabel_future_num = int(relabel_future_prob * batch_size)
  relabel_last_num = int(relabel_last_prob * batch_size)
  relabel_random_num = batch_size - (
      relabel_orig_num + relabel_next_num + relabel_future_num +
      relabel_last_num)
  assert relabel_random_num >= 0

  orig_goals = experience.observation[:relabel_orig_num, 0, obs_dim:]

  index = relabel_orig_num
  next_goals = experience.observation[index:index + relabel_next_num,
                                      1, :obs_dim]

  index = relabel_orig_num + relabel_next_num
  future_goals = get_future_goals(
      experience.observation[index:index + relabel_future_num, :, :obs_dim],
      experience.discount[index:index + relabel_future_num], gamma)

  index = relabel_orig_num + relabel_next_num + relabel_future_num
  last_goals = get_last_goals(
      experience.observation[index:index + relabel_last_num, :, :obs_dim],
      experience.discount[index:index + relabel_last_num])

  # For random goals we take other states from the same batch.
  random_goals = tf.random.shuffle(experience.observation[:relabel_random_num,
                                                          0, :obs_dim])
  new_goals = obs_to_goal(tf.concat([next_goals, future_goals,
                                     last_goals, random_goals], axis=0))
  goals = tf.concat([orig_goals, new_goals], axis=0)

  obs = experience.observation[:, :2, :obs_dim]
  reward = tf.reduce_all(obs_to_goal(obs[:, 1]) == goals, axis=-1)
  reward = tf.cast(reward, tf.float32)
  reward = tf.tile(reward[:, None], [1, 2])
  new_obs = tf.concat([obs, tf.tile(goals[:, None, :], [1, 2, 1])], axis=2)
  experience = experience.replace(
      observation=new_obs,  # [B x 2 x 2 * obs_dim]
      action=experience.action[:, :2],
      step_type=experience.step_type[:, :2],
      next_step_type=experience.next_step_type[:, :2],
      discount=experience.discount[:, :2],
      reward=reward,
  )
  return experience, buffer_info


@gin.configurable
class ClassifierCriticNetwork(critic_network.CriticNetwork):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_fc_layer_params=None,
               action_fc_layer_params=None,
               joint_fc_layer_params=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               name='ClassifierCriticNetwork'):
    super(ClassifierCriticNetwork, self).__init__(
        input_tensor_spec,
        observation_fc_layer_params=observation_fc_layer_params,
        action_fc_layer_params=action_fc_layer_params,
        joint_fc_layer_params=joint_fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=last_kernel_initializer,
        name=name,
    )

    last_layers = [
        tf.keras.layers.Dense(
            1,
            activation=tf.math.sigmoid,
            kernel_initializer=last_kernel_initializer,
            name='value')
    ]
    self._joint_layers = self._joint_layers[:-1] + last_layers


class BaseDistanceMetric(tf_metric.TFStepMetric):
  """Computes the initial distance to the goal."""

  def __init__(self,
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10,
               obs_dim=None,
               start_index=0,
               end_index=None,
               name=None):
    assert obs_dim is not None
    self._start_index = start_index
    self._end_index = end_index
    self._obs_dim = obs_dim
    name = self.NAME if name is None else name
    super(BaseDistanceMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dist_buffer = tf_metrics.TFDeque(
        1000, dtype)  # Episodes should have length less than 1k
    self.dtype = dtype

  @common.function(autograph=True)
  def call(self, trajectory):
    obs = trajectory.observation
    s = obs[:, :self._obs_dim]
    g = obs[:, self._obs_dim:]
    dist_to_goal = tf.norm(
        obs_to_goal(obs_to_goal(s), self._start_index, self._end_index) -
        obs_to_goal(g, self._start_index, self._end_index),
        axis=1)
    tf.assert_equal(tf.shape(obs)[0], 1)
    if trajectory.is_mid():
      self._dist_buffer.extend(dist_to_goal)
    if trajectory.is_last()[0] and self._dist_buffer.length > 0:
      self._update_buffer()
      self._dist_buffer.clear()
    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()

  def _update_buffer(self):
    raise NotImplementedError


class InitialDistance(BaseDistanceMetric):
  """Computes the initial distance to the goal."""
  NAME = 'InitialDistance'

  def _update_buffer(self):
    initial_dist = self._dist_buffer.data[0]
    self._buffer.add(initial_dist)


class FinalDistance(BaseDistanceMetric):
  """Computes the final distance to the goal."""
  NAME = 'FinalDistance'

  def _update_buffer(self):
    final_dist = self._dist_buffer.data[-1]
    self._buffer.add(final_dist)


class AverageDistance(BaseDistanceMetric):
  """Computes the average distance to the goal."""
  NAME = 'AverageDistance'

  def _update_buffer(self):
    avg_dist = self._dist_buffer.mean()
    self._buffer.add(avg_dist)


class MinimumDistance(BaseDistanceMetric):
  """Computes the minimum distance to the goal."""
  NAME = 'MinimumDistance'

  def _update_buffer(self):
    min_dist = self._dist_buffer.min()
    tf.Assert(
        tf.math.is_finite(min_dist), [
            min_dist, self._dist_buffer.length, self._dist_buffer._head,  # pylint: disable=protected-access
            self._dist_buffer.data
        ],
        summarize=1000)
    self._buffer.add(min_dist)


class DeltaDistance(BaseDistanceMetric):
  """Computes the net distance traveled towards the goal. Positive is good."""
  NAME = 'DeltaDistance'

  def _update_buffer(self):
    delta_dist = self._dist_buffer.data[0] - self._dist_buffer.data[-1]
    self._buffer.add(delta_dist)
