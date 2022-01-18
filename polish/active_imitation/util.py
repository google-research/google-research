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

"""Various utility functions and classes for active imitation learning.

The classes are an agent class for learning a behavior policy and
a discriminator class used to compute where to collect expert data.

The utils are for collecting rollout trajectories and adaptive sampling for
expert data.
"""

import collections
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


class Agent(object):
  """A class to represent an agent."""

  def __init__(self, in_dim, out_dim, action_type, action_bound=0, hidden=256):
    self.action_type = action_type
    self.action_bound = action_bound
    self.discriminator = None

    self.policy = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

    self.policy.add(tf.keras.layers.Dense(
        hidden,
        input_shape=in_dim,
        kernel_initializer=kernel_init))
    self.policy.add(tf.keras.layers.BatchNormalization())
    self.policy.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    self.policy.add(tf.keras.layers.Dense(hidden))
    self.policy.add(tf.keras.layers.BatchNormalization())
    self.policy.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    # Differentiate action types of discrete actions and continuous ones.
    if action_type == 'discrete':
      self.policy.add(tf.keras.layers.Dense(out_dim))
      self.policy.add(tf.keras.layers.BatchNormalization())
      self.policy.add(tf.keras.layers.Activation(
          tf.keras.activations.softmax))
    else:
      self.policy.add(tf.keras.layers.Dense(out_dim))
      self.policy.add(tf.keras.layers.BatchNormalization())
      self.policy.add(tf.keras.layers.Activation(tf.keras.activations.tanh))
      self.policy.add(
          tf.keras.layers.Lambda(lambda x: x * action_bound))

    self.policy.compile(optimizer='adam',
                        loss='mean_squared_error')

  def action(self, obs):
    """Compute an action given an observation."""

    if self.action_type == 'discrete':
      out = self.policy.predict(tf.expand_dims(obs, 0))
      action = np.argmax(out)
    else:
      out = self.policy.predict(tf.expand_dims(obs, 0))
      action = self.clip_action(out, self.action_bound)

    return action[0]

  def clip_action(self, action, bound):
    """Clip a continous action to a valid bound."""

    return np.clip(action, -bound, bound)


class Discriminator(object):
  """Implementation of a discriminator network."""

  def __init__(self, input_dim, hidden=256):
    """Initializes a discriminator.

    Args:
       input_dim: size of the input space.
       hidden: the number of hidden units.
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Dense(
        units=hidden,
        input_shape=(input_dim,),
        kernel_initializer=kernel_init))
    self.model.add(tf.keras.layers.BatchNormalization())
    self.model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    self.model.add(tf.keras.layers.Dense(
        units=hidden,
        kernel_initializer=kernel_init))
    self.model.add(tf.keras.layers.BatchNormalization())
    self.model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    self.model.add(tf.keras.layers.Dense(
        units=1,
        kernel_initializer=kernel_init))

  def train(self, agent_data, expert_data, epoch, batch_size=1024):
    """Train the discriminator with data from the current agent and the expert.

    Args:
      agent_data: a list of state-action pairs from agent's trajectories.
      expert_data: a list of state-action pairs from expert feedback.
      epoch: the number of epoches to train.
      batch_size: the number of samples to sample for each batch.

    Returns:
      validation accuracy and validation loss.
    """
    self.model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
    early_stopping = EarlyStopping(patience=5)
    agent_s_a = []
    expert_s_a = []
    labels = []
    # sample a batch from both agent_data and expert_data.
    agent_batches = random.sample(range(len(agent_data)),
                                  k=min(batch_size, len(agent_data)))
    expert_batches = random.sample(range(len(expert_data)),
                                   k=min(batch_size, len(expert_data)))

    for i, j in zip(agent_batches, expert_batches):
      agent_s_a.append(np.concatenate((agent_data[i][0], agent_data[i][1])))
      labels.append(0)
      expert_s_a.append(np.concatenate((expert_data[j][0], expert_data[j][1])))
      labels.append(1)

    shuffle_order = tf.random.shuffle(range(len(agent_s_a) * 2))
    s_a = agent_s_a + expert_s_a
    s_a = tf.gather(s_a, shuffle_order)
    labels = tf.gather(labels, shuffle_order)

    history = tf.keras.callbacks.History()
    self.model.fit(tf.convert_to_tensor(s_a),
                   tf.convert_to_tensor(labels),
                   epochs=epoch,
                   validation_split=0.2,
                   callbacks=[early_stopping,
                              history])

    return (history.history['val_binary_accuracy'][-1],
            history.history['val_loss'][-1])


def policy_rollout(agent, env, num_traj, horizon):
  """Rollout an agent to collect trajectories.

  Args:
    agent: an agent to rollout.
    env: an environment to perform rollouts.
    num_traj: the number of trajectories to collect.
    horizon: the maximal number of steps for each trajectory.

  Returns:
    states, actions, rewards and observations from rollout trajectories.
  """
  traj_states = []
  traj_actions = []
  traj_rewards = []
  traj_observations = []

  for _ in range(num_traj):
    time_step = env.reset()
    states = []
    rewards = []
    actions = []
    observations = []

    for _ in range(horizon):
      # MuJoCo specific operations.
      states.append(env._gym_env.get_state())  # pylint: disable=protected-access
      observations.append(time_step)

      action = agent.action(time_step.observation)
      actions.append(action)
      time_step = env.step(action)
      rewards.append(float(time_step.reward))

      if time_step.is_last():
        break

    traj_states.append(states)
    traj_actions.append(actions)
    traj_rewards.append(rewards)
    traj_observations.append(observations)

  return traj_states, traj_actions, traj_rewards, traj_observations


def query_expert(obs, expert):
  """Query the expert policy for actions at selected observations.

  Args:
    obs: a list of observations.
    expert: an expert policy.

  Returns:
    observations and actions from the expert policy.
  """
  expert_obs = []
  expert_actions = []

  for ob in obs:
    action = expert.action(ob).action
    expert_obs.append(ob.observation)
    expert_actions.append(action)

  return expert_obs, expert_actions


def bucket_select(scores, num_bucket, num_sample, reverse=False):
  """Select a subset of scores by dividing them into buckets and pick the highest/lowest scores within each bucket.

  Args:
    scores: a list of scores.
    num_bucket: the number of bucket to divide into.
    num_sample: the number of samples to return.
    reverse: if True, return the lowest scores, otherwise, return the highest
      ones.

  Returns:
    selected indices.
  """
  selected = []
  num_bucket = min(num_bucket, len(scores))

  # split the complete range into buckets.
  bucket_range = np.array_split(range(len(scores)),
                                min(len(scores), num_bucket))
  # sample buckets with replacement.
  selected_bucket = random.choices(range(num_bucket), k=num_sample)

  # compute how many times each bucket is selected.
  num_selected = collections.defaultdict(lambda: 0)

  for bucket in selected_bucket:
    num_selected[bucket] += 1

  # within each bucket, select the items.
  for idx, bucket in enumerate(bucket_range):
    sub_scores = scores[bucket[0] : bucket[-1] + 1]
    argsort = np.argsort(sub_scores)
    for i in range(min(num_selected[idx], len(sub_scores))):
      if reverse:
        selected.append(argsort[i] + bucket[0])
      else:
        selected.append(argsort[-i-1] + bucket[0])

  return selected


def behavior_cloning(agent, expert_data):
  """Train the model with expert data via behavior cloning."""

  states = []
  actions = []

  for i in range(len(expert_data)):
    state, action = expert_data[i]
    states.append(state)
    actions.append(action)

  states = tf.convert_to_tensor(states)
  actions = tf.convert_to_tensor(actions)

  agent.policy.compile(optimizer='adam',
                       loss='mean_squared_error')
  early_stopping = EarlyStopping(patience=5)
  history = tf.keras.callbacks.History()

  agent.policy.fit(states, actions,
                   batch_size=256,
                   validation_split=0.2,
                   epochs=1000,
                   callbacks=[early_stopping,
                              history])
  return history.history['val_loss'][-1]


def top_discriminator(discriminator, traj_obs, traj_actions, expert, num_bucket,
                      num_sample):
  """Select states to collect according to lowest discriminator scores.

  Args:
    discriminator: a discriminator.
    traj_obs: observations from a list of trajectories.
    traj_actions: corresponding actions for observations.
    expert: an expert policy.
    num_bucket: the number of buckets.
    num_sample: the number of samples to collect.

  Returns:
    new expert data and the agent data from current trajectories.
  """
  expert_data = []
  agent_data = []
  for i in range(len(traj_obs)):
    obs = traj_obs[i]
    acts = traj_actions[i]
    inputs = []
    for ob, act in zip(obs, acts):
      inputs.append(np.concatenate((ob.observation, act)))
      agent_data.append((ob.observation, act))

    # compute the discriminator scores for state-action pairs from a single
    # trajectory.
    dis_scores = tf.squeeze(
        discriminator.model.predict(tf.convert_to_tensor(inputs)))
    selected = bucket_select(dis_scores,
                             num_bucket,
                             num_sample,
                             reverse=True)
    # query expert actions on the selected states.
    for j in selected:
      expert_action = expert.action(obs[j]).action
      expert_data.append((obs[j].observation, expert_action))

  return expert_data, agent_data


def random_samples(traj_obs, expert, num_sample):
  """Randomly sample a subset of states to collect expert feedback.

  Args:
    traj_obs: observations from a list of trajectories.
    expert: an expert policy.
    num_sample: the number of samples to collect.

  Returns:
    new expert data.
  """

  expert_data = []
  for i in range(len(traj_obs)):
    obs = traj_obs[i]
    random.shuffle(obs)
    new_expert_data = []
    chosen = np.random.choice(range(len(obs)),
                              size=min(num_sample, len(obs)),
                              replace=False)
    for ch in chosen:
      state = obs[ch].observation
      action_step = expert.action(obs[ch])
      action = action_step.action
      new_expert_data.append((state, action))

    expert_data.extend(new_expert_data)

  return expert_data
