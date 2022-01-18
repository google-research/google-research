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
"""Policies for interacting with environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import gin
from gym import spaces
import numpy as np
import six
import tensorflow.compat.v1 as tf
from dql_grasping import cross_entropy


class Policy(six.with_metaclass(abc.ABCMeta, object)):
  """Base policy abstraction.

  Subclasses should implement `reset` and `sample_action` methods to ensure
  compatibility with the train_collect_eval function.
  """

  @abc.abstractmethod
  def reset(self):
    """Clear episode-specific state variables.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def restore(self, checkpoint):
    """Restore policy parameters from saved checkpoint.

    Args:
      checkpoint: String pointing to location of saved model variables.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def sample_action(self, obs, explore_prob):
    """Compute action given an observation.

    Args:
      obs: np.float32 array of shape (height, width, 3), corresponding to an
        RGB image.
      explore_prob: Probability (float) that policy executes
        exploration action rather than greedy action.
    Returns:
      np.float32 array of shape (action_size,) corresponding to commanded
      action, where action_size depends on environment.
    """
    raise NotImplementedError


class TFDQNPolicy(Policy):
  """Base class for Tensorflow-based DQN.

  Subclasses are expected to implement the `sample_action`, `build_tf_graph`
  methods.
  """

  def __init__(self,
               q_func,
               state_shape,
               use_gpu=True,
               checkpoint=None):
    """Constructor.

    Args:
      q_func: Python function with state and action arguments (both numpy
        arrays with dtype=np.float32), which constructs TF graph for
        computing state-action value Q(s,a). Calling q_func returns a possibly
        minibatched tf.Tensor of dtype=tf.float32.
      state_shape: Tuple of integers describing unbatched shape of state, *not*
        including timestep.
      use_gpu: Boolean of whether to use GPU at inference time.
      checkpoint: If provided, restores Q function/weights from this filepath.
    """
    super(TFDQNPolicy, self).__init__()
    self.q_func = q_func
    self._state_shape = state_shape
    self.build_tf_graph()
    config = tf.ConfigProto(device_count={'GPU': 1 if use_gpu else 0})
    self._sess = tf.Session(config=config)
    self._saver = tf.train.Saver()
    if checkpoint:
      self.restore(checkpoint)
    else:
      init_op = tf.global_variables_initializer()
      self._sess.run(init_op)

  def get_q_func(self, is_training=False, reuse=False, scope='q_func'):
    """Return Q function graph functor for building training/inference graph.

    Args:
      is_training: Whether this graph is for training or inference.
      reuse: Re-use existing variables/scope.
      scope: The variable scope to use for the function variables.
    Returns:
      Python function for constructing the Q function graph.
    """
    return functools.partial(self.q_func,
                             scope=scope,
                             reuse=reuse,
                             is_training=is_training)

  def reset(self):
    """See base class (Policy) for description.

    Subclasses are expected to feed self._timestep to their q_func graphs and
    increment self._timestep in `sample_action`.
    """
    self._timestep = np.array([0])

  def restore(self, checkpoint):
    self._saver.restore(self._sess, checkpoint)

  def build_tf_graph(self):
    """Constructs the TF computation graph for the critic.
    """
    raise NotImplementedError


@gin.configurable
class RandomGraspingPolicyD4(Policy):
  """Random policy for continuous grasping environment with action_space=4."""

  def __init__(self, height_hack_prob=0.9):
    super(RandomGraspingPolicyD4, self).__init__()
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(4,))

  def reset(self):
    pass

  def restore(self, checkpoint):
    pass

  def sample_action(self, obs, explore_prob):
    del explore_prob
    dx, dy, dz, da = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da], None


@gin.configurable
class PerStepSwitchPolicy(Policy):
  """Interpolates between an exploration policy and a greedy policy.

  A typical use case would be a scripted policy used to get some reasonable
  amount of random successes, and a greedy policy that is learned.

  Each of the exploration and greedy policies can still perform their own
  exploration actions after being selected by the PerStepSwitchPolicy.
  """

  def __init__(self, explore_policy_class, greedy_policy_class):
    super(PerStepSwitchPolicy, self).__init__()
    self._explore_policy = explore_policy_class()
    self._greedy_policy = greedy_policy_class()

  def reset(self):
    self._explore_policy.reset()
    self._greedy_policy.reset()

  def get_q_func(self, is_training=False, reuse=False, scope='q_func'):
    return self._greedy_policy.get_q_func(is_training, reuse, scope=scope)

  def get_a_func(self, is_training=False, reuse=False):
    return self._greedy_policy.get_a_func(is_training, reuse)

  def restore(self, checkpoint):
    self._explore_policy.restore(checkpoint)
    self._greedy_policy.restore(checkpoint)

  def sample_action(self, obs, explore_prob):
    if np.random.random() < explore_prob:
      return self._explore_policy.sample_action(obs, explore_prob)
    else:
      return self._greedy_policy.sample_action(obs, explore_prob)


@gin.configurable
class CEMActorPolicy(TFDQNPolicy):
  """Learned policy for grasping (continuous). Uses CEM for selecting actions.
  """

  def __init__(self,
               q_func,
               state_shape,
               action_size,
               use_gpu=True,
               batch_size=64,
               build_target=False,
               include_timestep=True,
               checkpoint=None):
    """Initializes the policy.

    Args:
      q_func: Python function that takes in state, action, scope as input
        and returns Q(state, action) and intermediate endpoints dictionary.
      state_shape: Tuple of ints describing shape of the state observation.
      action_size: (int) Size of the vector-encoded action.
      use_gpu: If True, use GPU for inference.
      batch_size: Size of CEM population. Also used as size of minibatch for
        inference.
      build_target: If True, construct a separate Q network.
      include_timestep: If True, include timestep of policy along with state.
      checkpoint: Restore Q function (and possibly target Q function) from a
        checkpoint.
    """
    self._action_ph = tf.placeholder(tf.float32, (batch_size, action_size))
    self._batch_size = batch_size
    self._action_size = action_size
    self._build_target = build_target
    self._action_space = spaces.Box(low=-1, high=1, shape=(action_size,))
    self._include_timestep = include_timestep
    super(CEMActorPolicy, self).__init__(
        q_func, state_shape, use_gpu=use_gpu, checkpoint=checkpoint)

  def build_tf_graph(self, reuse=False):
    """Constructs the TF computation graph for the critic.

    Args:
      reuse: If True, re-uses pre-defined variables for Q network and target Q
      network.
    """
    self._state_ph = tf.placeholder(tf.float32, self._state_shape)
    if self._include_timestep:
      self._step_ph = tf.placeholder(tf.int32, (1,))
      state = (self._state_ph, self._step_ph)
    else:
      state = self._state_ph
    self._qs, self._end_points_t = self.q_func(
        state, self._action_ph, scope='q_func', reuse=reuse, is_training=False)
    if self._build_target:
      self._target_qs, self._end_points_tp1 = self.q_func(
          state, self._action_ph, scope='target_q_func', reuse=reuse,
          is_training=False)

  def sample_action(self, obs, explore_prob):
    """Compute action given an observation.

    This policy does not implement its own exploration strategy. Use
    PerStepSwitchPolicy instead to perform exploration.

    Args:
      obs: np.float32 array of shape (height, width, 3), corresponding to an
        RGB image.
      explore_prob: Probability (float) that policy executes
        exploration action rather than greedy action.
    Returns:
      Tuple (action, debug) where action is a np.float32 array of shape
      (action_size,) corresponding to commanded action, where action_size
      depends on environment. Debug contains intermediate values that may be of
      interest to the training loop.
    """
    del explore_prob
    img = np.expand_dims(obs, 0)

    def objective_fn(samples):
      feed_dict = {self._state_ph: img, self._action_ph: samples}
      if self._include_timestep:
        feed_dict[self._step_ph] = self._timestep
      q_values = self._sess.run(self._qs, feed_dict)
      return q_values

    def sample_fn(mean, stddev):
      return mean + stddev * np.random.randn(
          self._batch_size, self._action_size)

    def update_fn(params, elite_samples):
      del params
      return {
          'mean': np.mean(elite_samples, axis=0),
          'stddev': np.std(elite_samples, axis=0, ddof=1),
      }

    mu = np.zeros(self._action_size)
    mu[2] = -1  # Downward bias in Z axis.
    initial_params = {'mean': mu, 'stddev': .5 * np.ones(self._action_size)}
    samples, values, final_params = cross_entropy.CrossEntropyMethod(
        sample_fn,
        objective_fn,
        update_fn,
        initial_params,
        num_elites=10,
        num_iterations=3)

    idx = np.argmax(values)
    best_continuous_action, best_continuous_value = samples[idx], values[idx]
    debug = {'q': best_continuous_value, 'final_params': final_params}
    self._timestep += 1
    return best_continuous_action, debug


@gin.configurable
class DDPGPolicy(TFDQNPolicy):
  """Actor-Critic DDPG for Continuous Control https://arxiv.org/abs/1509.02971.
  """

  def __init__(self,
               a_func,
               q_func,
               state_shape,
               action_size,
               use_gpu=True,
               build_target=False,
               include_timestep=True,
               checkpoint=None):
    """Initializes the policy.

    Args:
      a_func: Python function that takes in state, scope as input
        and returns action and intermediate endpoints dictionary.
      q_func: Python function that takes in state, action, scope as input
        and returns Q(state, action) and intermediate endpoints dictionary.
      state_shape: Tuple of ints describing shape of the state observation.
      action_size: (int) Size of the vector-encoded action.
      use_gpu: If True, use GPU for inference.
      build_target: If True, construct a separate Q network.
      include_timestep: If True, include timestep of policy along with state.
      checkpoint: Restore Q function (and possibly target Q function) from a
        checkpoint.
    """
    self.a_func = a_func
    self._action_size = action_size
    self._build_target = build_target
    self._include_timestep = include_timestep
    super(DDPGPolicy, self).__init__(
        q_func, state_shape, use_gpu=use_gpu, checkpoint=checkpoint)

  def build_tf_graph(self, reuse=False):
    """Constructs the TF computation graph for the critic.

    Args:
      reuse: If True, re-uses pre-defined variables for Q network and target Q
      network.
    """
    self._state_ph = tf.placeholder(tf.float32, self._state_shape)
    if self._include_timestep:
      self._step_ph = tf.placeholder(tf.int32, (1,))
      state = (self._state_ph, self._step_ph)
    else:
      state = self._state_ph
    self._action, _ = self.a_func(
        state, self._action_size, scope='a_func', is_training=False)
    self._qs, self._end_points_t = self.q_func(
        state, self._action, scope='q_func', reuse=reuse, is_training=False)
    if self._build_target:
      self._target_qs, self._end_points_tp1 = self.q_func(
          state, self._action, scope='target_q_func', reuse=reuse,
          is_training=False)

  def get_a_func(self, is_training=False, reuse=False):
    """Return Actor graph functor for building training/inference graph.

    Args:
      is_training: Whether this graph is for training or inference.
      reuse: Re-use existing variables/scope.
    Returns:
      Python function for constructing the Q function graph.
    """
    return functools.partial(self.a_func,
                             num_actions=self._action_size,
                             scope='a_func',
                             reuse=reuse,
                             is_training=is_training)

  def sample_action(self, obs, explore_prob):
    """Compute action given an observation.

    This policy does not implement its own exploration strategy. Use
    PerStepSwitchPolicy instead to perform exploration.

    Args:
      obs: np.float32 array of shape (height, width, 3), corresponding to an
        RGB image.
      explore_prob: Probability (float) that policy executes
        exploration action rather than greedy action.
    Returns:
      Tuple (action, debug) where action is a np.float32 array of shape
      (action_size,) corresponding to commanded action, where action_size
      depends on environment. Debug contains intermediate values that may be of
      interest to the training loop.
    """
    del explore_prob
    img = np.expand_dims(obs, 0)

    feed_dict = {self._state_ph: img}
    if self._include_timestep:
      feed_dict[self._step_ph] = self._timestep

    action, q_values = self._sess.run([self._action, self._qs],
                                      feed_dict)
    self._timestep += 1
    return action[0], {'q': q_values}
