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

"""Policies used by various agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tf_agents.specs import tensor_spec


class DeterministicSoftPolicy(tf.Module):
  """Returns mode of policy distribution."""

  def __init__(self, a_network):
    super(DeterministicSoftPolicy, self).__init__()
    self._a_network = a_network

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network(observation)[0]
    return action, state


class RandomSoftPolicy(tf.Module):
  """Returns sample from policy distribution."""

  def __init__(self, a_network):
    super(RandomSoftPolicy, self).__init__()
    self._a_network = a_network

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network(observation)[1]
    return action, state


class MaxQSoftPolicy(tf.Module):
  """Samples a few actions from policy, returns the one with highest Q-value."""

  def __init__(self, a_network, q_network, n=10):
    super(MaxQSoftPolicy, self).__init__()
    self._a_network = a_network
    self._q_network = q_network
    self._n = n

  @tf.function
  def __call__(self, observation, state=()):
    batch_size = observation.shape[0]
    actions = self._a_network.sample_n(observation, self._n)[1]
    actions_ = tf.reshape(actions, [self._n * batch_size, -1])
    states_ = tf.tile(observation[None], (self._n, 1, 1))
    states_ = tf.reshape(states_, [self._n * batch_size, -1])
    qvals = self._q_network(states_, actions_)
    qvals = tf.reshape(qvals, [self._n, batch_size])
    a_indices = tf.argmax(qvals, axis=0)
    gather_indices = tf.stack(
        [a_indices, tf.range(batch_size, dtype=tf.int64)], axis=-1)
    action = tf.gather_nd(actions, gather_indices)
    return action, state


class ContinuousRandomPolicy(tf.Module):
  """Samples actions uniformly at random."""

  def __init__(self, action_spec):
    super(ContinuousRandomPolicy, self).__init__()
    self._action_spec = action_spec

  def __call__(self, observation, state=()):
    action = tensor_spec.sample_bounded_spec(
        self._action_spec, outer_dims=[observation.shape[0]])
    return action, state


class EpsilonGreedyRandomSoftPolicy(tf.Module):
  """Switches between samples from actor network and uniformly random action."""

  def __init__(self, a_network, epsilon):
    super(EpsilonGreedyRandomSoftPolicy, self).__init__()
    self._a_network = a_network
    self._epsilon = epsilon

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network(observation)[1]
    rand_action = tensor_spec.sample_bounded_spec(
        self._a_network.action_spec, outer_dims=[observation.shape[0]])
    seed = tf.random.uniform([observation.shape[0]])
    is_random = tf.less(seed, self._epsilon)
    action = tf.compat.v2.where(is_random, rand_action, action)
    return action, state


class GaussianRandomSoftPolicy(tf.Module):
  """Adds Gaussian noise to actor's action."""

  def __init__(self, a_network, std=0.1, clip_eps=1e-3):
    super(GaussianRandomSoftPolicy, self).__init__()
    self._a_network = a_network
    self._std = std
    self._clip_eps = clip_eps

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network(observation)[1]
    noise = tf.random_normal(shape=action.shape, stddev=self._std)
    action = action + noise
    spec = self._a_network.action_spec
    action = tf.clip_by_value(action, spec.minimum + self._clip_eps,
                              spec.maximum - self._clip_eps)
    return action, state


class GaussianEpsilonGreedySoftPolicy(tf.Module):
  """Switches between Gaussian-perturbed and uniform random action."""

  def __init__(self, a_network, std=0.1, clip_eps=1e-3, eps=0.1):
    super(GaussianEpsilonGreedySoftPolicy, self).__init__()
    self._a_network = a_network
    self._std = std
    self._clip_eps = clip_eps
    self._eps = eps

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network(observation)[1]
    noise = tf.random_normal(shape=action.shape, stddev=self._std)
    action = action + noise
    spec = self._a_network.action_spec
    action = tf.clip_by_value(action, spec.minimum + self._clip_eps,
                              spec.maximum - self._clip_eps)
    rand_action = tensor_spec.sample_bounded_spec(
        self._a_network.action_spec, outer_dims=[observation.shape[0]])
    seed = tf.random.uniform([observation.shape[0]])
    is_random = tf.less(seed, self._eps)
    action = tf.compat.v2.where(is_random, rand_action, action)
    return action, state


class BCQPolicy(tf.Module):
  """Policy used by BCQ."""

  def __init__(self, a_network, q_network, b_network, n=10):
    super(BCQPolicy, self).__init__()
    self._a_network = a_network
    self._q_network = q_network
    self._b_network = b_network
    self._n = n

  @tf.function
  def __call__(self, observation, state=()):
    batch_size = observation.shape[0]
    s_dup = tf.tile(observation, [self._n, 1])
    sampled_actions = self._b_network.sample(s_dup)
    actions = self._a_network(s_dup, sampled_actions)
    qvals = self._q_network(s_dup, actions)
    qvals = tf.reshape(qvals, [self._n, batch_size])
    a_indices = tf.argmax(qvals, axis=0)
    gather_indices = tf.stack(
        [a_indices, tf.range(batch_size, dtype=tf.int64)], axis=-1)
    actions = tf.reshape(actions, [self._n, batch_size, -1])
    action = tf.gather_nd(actions, gather_indices)
    return action, state


class VAEPolicy(tf.Module):
  """Policy based on VAE."""

  def __init__(self, a_network):
    super(VAEPolicy, self).__init__()
    self._a_network = a_network

  @tf.function
  def __call__(self, observation, state=()):
    action = self._a_network.sample(observation)
    return action, state
