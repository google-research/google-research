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

"""Replay buffer that performs relabeling."""

import gin
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


@gin.configurable
class RelabellingReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
  """A replay buffer that relabels experience."""

  def __init__(self, *args, **kwargs):
    """Initialize the replay buffer.

    Args:
      *args: Arguments.
      **kwargs: Keyword arguments.

    Additional arguments:
      task_distribution: an instance of multitask.TaskDistribution.
      sample_batch_size: (int) the batch size.
      num_parallel_calls: (int) number of parallel calls for sampling.
      num_future_states: (int) number of future states to consider for
        future state relabeling.
      actor: the actor network.
      critic: the critic network.
      gamma: (float) the discount factor.
      relabel_type: (str) indicator of the relabeling strategy.
      candidate_task_type: (str) within each back, should we use the states,
        next_states, or originally commanded tasks as possible tasks when
        relabeling.
      relabel_prob: (float) fraction of experience to relabel when sampling.
      keep_current_goal: (bool) for ``last'' and ``final'' relabeling,
        should we add both the originally commanded task and the relabeled
        task when inserting new experience into the replay buffer.
      normalize_cols: (bool) Normalizing the columns has the effect of
        including the partition function.
    """
    self._task_distribution = kwargs.pop("task_distribution")
    self._sample_batch_size = kwargs.pop("sample_batch_size")
    self._num_parallel_calls = kwargs.pop("num_parallel_calls")
    self._num_future_states = kwargs.pop("num_future_states", 4)
    self._actor = kwargs.pop("actor")
    self._critic = kwargs.pop("critic")
    self._gamma = kwargs.pop("gamma")
    self._relabel_type = kwargs.pop("relabel_type", None)
    assert self._relabel_type in [None, "last", "future", "soft", "random"]
    self._candidate_task_type = kwargs.pop("candidate_task_type", "states")
    assert self._candidate_task_type in ["states", "next_states", "tasks"]
    self._relabel_prob = kwargs.pop("relabel_prob", 1.0)
    self._keep_current_goal = kwargs.pop("keep_current_goal", False)

    self._normalize_cols = kwargs.pop("normalize_cols", True)

    self._iterator = None
    super(RelabellingReplayBuffer, self).__init__(*args, **kwargs)

  def get_batch(self):
    if self._iterator is None:
      dataset = self.as_dataset(
          sample_batch_size=self._sample_batch_size,
          num_parallel_calls=self._num_parallel_calls,
          num_steps=2,
      ).prefetch(3)
      self._iterator = iter(dataset)
    experience, unused_info = next(self._iterator)
    if self._relabel_type in ["soft", "random"]:
      experience = self._soft_relabel(experience)
    elif self._relabel_type in ["last", "future"]:
      # Reassign the next_states to have the same goal as the current states
      _, tasks = self._task_distribution.split(experience.observation[:, 0])
      next_states, _ = self._task_distribution.split(experience.observation[:,
                                                                            1])
      next_states_and_tasks = self._task_distribution.combine(
          next_states, tasks)
      new_observation = tf.concat(
          [
              experience.observation[:, 0][:, None], next_states_and_tasks[:,
                                                                           None]
          ],
          axis=1,
      )
      assert new_observation.shape == experience.observation.shape
      experience = experience.replace(observation=new_observation)
    if self._relabel_type is not None:
      # Recompute rewards and done flags
      states, tasks = self._task_distribution.split(experience.observation[:,
                                                                           0])
      next_states, next_tasks = self._task_distribution.split(
          experience.observation[:, 1])
      rewards, dones = self._task_distribution.evaluate(states,
                                                        experience.action[:, 0],
                                                        tasks)
      # Strictly speaking, we don't need to relabel the next rewards and next
      # dones because they end up being thrown away. Only the current rewards
      # and dones end up being important.
      next_rewards, next_dones = self._task_distribution.evaluate(
          next_states, experience.action[:, 1], next_tasks)

      new_rewards = tf.concat([rewards[:, None], next_rewards[:, None]], axis=1)
      new_dones = tf.concat([dones[:, None], next_dones[:, None]], axis=1)
      # 0 if episode is done, 1 if episode is continuing
      new_discount = 1.0 - tf.cast(new_dones, tf.float32)
      assert new_rewards.shape == experience.reward.shape
      assert new_discount.shape == experience.discount.shape
      experience = experience.replace(reward=new_rewards, discount=new_discount)
    return experience

  def _soft_relabel(self, experience):
    """Reassigns tasks to each state and next state.

    Does not recompute the rewards or done flags.

    Args:
      experience: The experience that we want to relabel with inverse RL.
    Returns:
      relabeled_experience: The relabeled experience.
    """
    raise NotImplementedError

  def _add_batch(self, items):
    """Adds a trajectory to the replay buffer."""
    assert items[0].is_first()
    for item in items:
      # The items are batched already, so we remove the first dimension.
      assert item.observation.shape[1:] == self.data_spec.observation.shape
      super(RelabellingReplayBuffer, self)._add_batch(item)


class GoalRelabellingReplayBuffer(RelabellingReplayBuffer):
  """Implements a replay buffer for relabeling goals."""

  def _add_batch(self, items):
    """Adds a trajectory to the replay buffer."""
    batch_size = len(items)
    if self._relabel_type in ["future", "last"]:
      relabelled_items = []
      for i in range(batch_size):
        if self._relabel_type == "future":
          relabel_indices = np.random.randint(
              i, batch_size, size=self._num_future_states)
        else:
          relabel_indices = [batch_size - 1]
        if self._keep_current_goal:
          relabelled_items.append(items[i])
        for j in relabel_indices:
          state, _ = self._task_distribution.split(items[i].observation)
          next_state, _ = self._task_distribution.split(items[j].observation)
          task = self._task_distribution.state_to_task(next_state)
          state_and_task = self._task_distribution.combine(state, task)
          new_item = items[i].replace(observation=state_and_task)
          relabelled_items.append(new_item)
      items = relabelled_items
    super(GoalRelabellingReplayBuffer, self)._add_batch(items)

  @tf.function
  def _soft_relabel(self, experience):
    # experience.observation.shape = [B x T=2 x obs_dim+state_dim]
    states, orig_tasks = self._task_distribution.split(
        experience.observation[:, 0])
    if self._task_distribution.tasks is None:
      tasks = orig_tasks
    else:
      tasks = tf.constant(self._task_distribution.tasks, dtype=tf.float32)
    next_states, _ = self._task_distribution.split(experience.observation[:, 1])
    if self._candidate_task_type == "states":
      candidate_tasks = self._task_distribution.state_to_task(states)
    elif self._candidate_task_type == "next_states":
      candidate_tasks = self._task_distribution.state_to_task(next_states)
    else:
      assert self._candidate_task_type == "tasks"
      candidate_tasks = tasks

    actions = experience.action[:, 0]
    num_tasks = tasks.shape[0]
    batch_size = states.shape[0]
    task_dim = tasks.shape[1]
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]
    action_spec = self._actor.output_tensor_spec

    states_tiled = tf.tile(states[:, None], [1, num_tasks, 1])  # B x B x D
    states_tiled = tf.reshape(states_tiled,
                              [batch_size * num_tasks, obs_dim])  # B*B x D
    actions_tiled = tf.tile(actions[:, None], [1, num_tasks, 1])  # B x B x D
    actions_tiled = tf.reshape(actions_tiled,
                               [batch_size * num_tasks, action_dim])  # B*B x D
    tasks_tiled = tf.tile(tasks[None], [batch_size, 1, 1])  # B x B x D
    tasks_tiled = tf.reshape(tasks_tiled,
                             [batch_size * num_tasks, task_dim])  # B*B x D

    next_states_tiled = tf.tile(next_states[:, None], [1, num_tasks, 1])
    next_states_tiled = tf.reshape(next_states_tiled,
                                   [batch_size * num_tasks, obs_dim])  # B*B x D
    next_relabelled_obs = self._task_distribution.combine(
        next_states_tiled, tasks_tiled)

    sampled_actions_tiled = self._actor(
        next_relabelled_obs, step_type=(), network_state=())[0].sample()
    critic_input = (next_relabelled_obs, sampled_actions_tiled)
    q_vals, _ = self._critic(critic_input, training=False)
    q_vals_vec = tf.reshape(q_vals, (batch_size, num_tasks))

    rewards, dones = self._task_distribution.evaluate(states_tiled,
                                                      actions_tiled,
                                                      tasks_tiled)
    dones = tf.cast(dones, tf.float32)
    rewards_vec = tf.reshape(rewards, (batch_size, num_tasks))
    dones_vec = tf.reshape(dones, (batch_size, num_tasks))

    relabelled_obs = self._task_distribution.combine(states_tiled, tasks_tiled)
    action_distribution = self._actor(
        relabelled_obs, step_type=(), network_state=())[0]
    log_pi = common.log_probability(action_distribution, actions_tiled,
                                    action_spec)
    log_pi_vec = tf.reshape(log_pi, (batch_size, num_tasks))

    logits_vec = (
        rewards_vec - log_pi_vec + self._gamma * (1.0 - dones_vec) * q_vals_vec)
    if self._relabel_type == "random":
      logits_vec = tf.ones_like(logits_vec)  # Hack to make sampling random

    ## End new version
    if self._normalize_cols:
      logits_vec = logits_vec - tf.math.reduce_logsumexp(
          logits_vec, axis=0)[None]
    relabel_indices = tf.random.categorical(logits=logits_vec, num_samples=1)

    ### Metrics
    global_step = tf.compat.v1.train.get_or_create_global_step()
    orig_indices = tf.range(
        self._sample_batch_size, dtype=relabel_indices.dtype)
    with tf.name_scope("relabelling"):
      # How often are the originally commanded goals most optimal?
      opt_indices = tf.argmax(logits_vec, axis=1)
      orig_is_opt = opt_indices == orig_indices
      orig_opt_frac = tf.reduce_mean(tf.cast(orig_is_opt, tf.float32))
      tf.compat.v2.summary.scalar(
          name="orig_task_optimal", data=orig_opt_frac, step=global_step)

      # How often is the relabelled goal optimal?
      # The relabel_indices are [B, 1], so we need to remove the extra dim.
      relabel_is_opt = tf.squeeze(relabel_indices) == orig_indices
      relabel_opt_frac = tf.reduce_mean(tf.cast(relabel_is_opt, tf.float32))
      tf.compat.v2.summary.scalar(
          name="relabel_task_optimal", data=relabel_opt_frac, step=global_step)

      # What are the average Q values of the original tasks?
      if batch_size == num_tasks:
        indices = tf.transpose(tf.stack([orig_indices, orig_indices], axis=0))
        orig_q_vals = tf.gather_nd(logits_vec, indices)
        tf.compat.v2.summary.scalar(
            name="orig_q_vals",
            data=tf.reduce_mean(orig_q_vals),
            step=global_step,
        )

      # What are the average Q values of the relabelled tasks?
      indices = tf.transpose(
          tf.stack([orig_indices, tf.squeeze(relabel_indices)], axis=0))
      relabel_q_vals = tf.gather_nd(logits_vec, indices)
      tf.compat.v2.summary.scalar(
          name="relabel_q_vals",
          data=tf.reduce_mean(relabel_q_vals),
          step=global_step,
      )

      max_q = tf.reduce_max(logits_vec, axis=1)
      tf.compat.v2.summary.scalar(
          name="max_q", data=tf.reduce_mean(max_q), step=global_step)

    ### End metrics

    # For both state-centric and goal-centric relabelling, the implementation of
    # mixing is the same: we randomly replace some of the indices with the
    # diagonal.
    relabelled_tasks = tf.gather(candidate_tasks, tf.squeeze(relabel_indices))

    if self._relabel_prob == 0:
      relabelled_tasks = orig_tasks
    elif 0 < self._relabel_prob < 1:
      logits = tf.log([1.0 - self._relabel_prob, self._relabel_prob])
      mask = tf.squeeze(
          tf.random.categorical(
              logits[None], num_samples=self._sample_batch_size))
      mask = tf.cast(mask, tf.float32)[:, None]
      relabelled_tasks = mask * orig_tasks + (1 - mask) * relabelled_tasks

    states_and_tasks = self._task_distribution.combine(states, relabelled_tasks)
    next_states_and_tasks = self._task_distribution.combine(
        next_states, relabelled_tasks)
    new_observation = tf.concat(
        [states_and_tasks[:, None], next_states_and_tasks[:, None]], axis=1)
    assert new_observation.shape == experience.observation.shape
    experience = experience.replace(observation=new_observation)
    return experience
