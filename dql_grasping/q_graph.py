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

"""Continuous Q-Learning via random sampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import enum
import gin
import tensorflow.compat.v1 as tf


class DQNTarget(enum.Enum):
  """Enum constants for DQN target network variants.

  Attributes:
    notarget: No target network used. Next-step action-value computed using
      using online Q network.
    normal: Target network used to select action and evaluate next-step
      action-value.
    doubleq: Double-Q Learning as proposed by https://arxiv.org/abs/1509.06461.
      Action is selected by online Q network but evaluated using target network.
  """
  notarget = 'notarget'
  normal = 'normal'
  doubleq = 'doubleq'


gin.constant('DQNTarget.notarget', DQNTarget.notarget)
gin.constant('DQNTarget.normal', DQNTarget.normal)
gin.constant('DQNTarget.doubleq', DQNTarget.doubleq)


@gin.configurable
def discrete_q_graph(q_func,
                     transition,
                     target_network_type=DQNTarget.normal,
                     gamma=1.0,
                     loss_fn=tf.losses.huber_loss,
                     extra_callback=None):
  """Construct loss/summary graph for discrete Q-Learning (DQN).

  This Q-function loss implementation is derived from OpenAI baselines.
  This function supports dynamic batch sizes.

  Args:
    q_func: Python function that takes in state, scope as input
      and returns a tensor Q(a_0...a_N) for each action a_0...a_N, and
      intermediate endpoints dictionary.
    transition: SARSTransition namedtuple.
    target_network_type: Option to use Q Learning without target network, Q
      Learning with a target network (default), or Double-Q Learning with a
      target network.
    gamma: Discount factor.
    loss_fn: Function that computes the td_loss tensor. Takes as arguments
      (target value tensor, predicted value tensor).
    extra_callback: Optional function that takes in (transition, end_points_t,
      end_points_tp1) and adds additional TF graph elements.

  Returns:
    A tuple (loss, summaries) where loss is a scalar loss tensor to minimize,
    summaries are TensorFlow summaries.
  """
  state = transition.state
  action = transition.action
  state_p1 = transition.state_p1
  reward = transition.reward
  done = transition.done

  q_t, end_points_t = q_func(state, scope='q_func')
  num_actions = q_t.get_shape().as_list()[1]
  q_t_selected = tf.reduce_sum(q_t * tf.one_hot(action, num_actions), 1)
  if gamma != 0:
    if target_network_type == DQNTarget.notarget:
      # Evaluate target values using the current net only.
      q_tp1_using_online_net, end_points_tp1 = q_func(
          state_p1, scope='q_func', reuse=True)
      q_tp1_best = tf.reduce_max(q_tp1_using_online_net, 1)
    elif target_network_type == DQNTarget.normal:
      # Target network Q values at t+1.
      q_tp1_target, end_points_tp1 = q_func(state_p1, scope='target_q_func')
      q_tp1_best = tf.reduce_max(q_tp1_target, 1)
    elif target_network_type == DQNTarget.doubleq:
      q_tp1_target, end_points_tp1 = q_func(state_p1, scope='target_q_func')
      # Q values at t+1.
      q_tp1_using_online_net, _ = q_func(state_p1, scope='q_func', reuse=True)
      # Q values for action we select at t+1.
      q_tp1_best_using_online_net = tf.one_hot(
          tf.argmax(q_tp1_using_online_net, 1), num_actions)
      # Q value of target network at t+1, but for action selected by online net.
      q_tp1_best = tf.reduce_sum(q_tp1_target * q_tp1_best_using_online_net, 1)
    else:
      logging.error('Invalid target_network_mode %s', target_network_type)
    q_tp1_best_masked = (1.0 - done) * q_tp1_best
    q_t_selected_target = tf.stop_gradient(reward + gamma * q_tp1_best_masked)
  else:
    q_t_selected_target = tf.stop_gradient(reward)

  td_error = q_t_selected - q_t_selected_target

  if extra_callback is not None:
    extra_callback(transition, end_points_t, end_points_tp1)

  tf.summary.histogram('td_error', td_error)
  tf.summary.histogram('q_t_selected', q_t_selected)
  tf.summary.histogram('q_t_selected_target', q_t_selected_target)
  tf.summary.scalar('mean_q_t_selected', tf.reduce_mean(q_t_selected))

  td_loss = loss_fn(q_t_selected_target, q_t_selected)
  tf.summary.scalar('td_loss', td_loss)
  reg_loss = tf.losses.get_regularization_loss()
  tf.summary.scalar('reg_loss', reg_loss)
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('total_loss', loss)

  summaries = tf.summary.merge_all()

  return loss, summaries


@gin.configurable
def random_sample_box(batch_size,
                      action_size,
                      num_samples,
                      minval=-1.,
                      maxval=1.):
  """Samples actions for each batch element uniformly from a hyperrectangle.

  Args:
    batch_size: tf.Tensor (dtype=tf.int32) or int representing the minibatch
      size of the state tensors.
    action_size: (int) Size of continuous actio space.
    num_samples: (int) Number of action samples for each minibatch element.
    minval: (float) Minimum value for each action dimension.
    maxval: (float) Maximum value for each action dimension.

  Returns:
    Tensor (dtype=tf.float32) of shape (batch_size * num_samples, action_size).
  """
  return tf.random_uniform(
      (batch_size * num_samples, action_size), minval=minval, maxval=maxval)


def _q_tp1_notarget(q_func, state_p1, batch_size, num_samples, random_actions):
  """Evaluate target values at t+1 using online Q function (no target network).

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    state_p1: Tensor (potentially any dtype) representing next .
    batch_size: tf.Tensor (dtype=tf.int32) or int representing the minibatch
      size of the state tensors.
    num_samples: (int) Number of action samples for each minibatch element.
    random_actions: tf.Tensor (dtype=tf.float32) of candidate actions.

  Returns:
    Tuple (q_tp1_best, end_points_tp1). See _get_q_tp1 docs for description.
  """
  # Evaluate target values using the current net only.
  q_tp1_using_online_net, end_points_tp1 = q_func(
      state_p1, random_actions, scope='q_func', reuse=True)
  q_tp1_using_online_net_2d = tf.reshape(
      q_tp1_using_online_net, (batch_size, num_samples))
  q_tp1_best = tf.reduce_max(q_tp1_using_online_net_2d, 1)
  return q_tp1_best, end_points_tp1


def _q_tp1_normal(q_func, state_p1, batch_size, num_samples, random_actions):
  """Evaluate target values at t+1 using separate target network network.

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    state_p1: Tensor (potentially any dtype) representing next .
    batch_size: tf.Tensor (dtype=tf.int32) or int representing the minibatch
      size of the state tensors.
    num_samples: (int) Number of action samples for each minibatch element.
    random_actions: tf.Tensor (dtype=tf.float32) of candidate actions.

  Returns:
    Tuple (q_tp1_best, end_points_tp1). See _get_q_tp1 docs for description.
  """
  q_tp1_target, end_points_tp1 = q_func(
      state_p1, random_actions, scope='target_q_func')
  q_tp1_target_2d = tf.reshape(q_tp1_target, (batch_size, num_samples))
  q_tp1_best = tf.reduce_max(q_tp1_target_2d, 1)
  return q_tp1_best, end_points_tp1


def _q_tp1_doubleq(q_func,
                   state_p1,
                   batch_size,
                   action_size,
                   num_samples,
                   random_actions):
  """Q(s_p1, a_p1) via Double Q Learning with stochastic sampling.

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    state_p1: Tensor (potentially any dtype) representing next .
    batch_size: tf.Tensor (dtype=tf.int32) or int representing the minibatch
      size of the state tensors.
    action_size: (int) Size of continuous actio space.
    num_samples: (int) Number of action samples for each minibatch element.
    random_actions: tf.Tensor (dtype=tf.float32) of candidate actions.

  Returns:
    Tuple (q_tp1_best, end_points_tp1). See _get_q_tp1 docs for description.
  """
  # Target Q values at t+1, for action selected by online net.
  q_tp1_using_online_net, end_points_tp1 = q_func(
      state_p1, random_actions, scope='q_func', reuse=True)
  q_tp1_using_online_net_2d = tf.reshape(
      q_tp1_using_online_net, (batch_size, num_samples))
  q_tp1_indices_using_online_net = tf.argmax(q_tp1_using_online_net_2d, 1)
  random_actions = tf.reshape(
      random_actions, (batch_size, num_samples, action_size))
  batch_indices = tf.cast(tf.range(batch_size), tf.int64)
  indices = tf.stack([batch_indices, q_tp1_indices_using_online_net], axis=1)
  # For each batch item, slice into the num_samples,
  # action subarray using the corresponding to yield the chosen action.
  q_tp1_best_action = tf.gather_nd(random_actions, indices)
  q_tp1_best, end_points_tp1 = q_func(
      state_p1, q_tp1_best_action, scope='target_q_func')
  return q_tp1_best, end_points_tp1


def _get_q_tp1(q_func,
               state_p1,
               batch_size,
               action_size,
               num_samples,
               random_sample_fn,
               target_network_type):
  """Computes non-discounted Bellman target Q(s_p1, a_p1).

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    state_p1: Tensor (potentially any dtype) representing next .
    batch_size: tf.Tensor (dtype=tf.int32) or int representing the minibatch
      size of the state tensors.
    action_size: (int) Size of continuous action space.
    num_samples: (int) Number of action samples for each minibatch element.
    random_sample_fn: See random_continuous_q_graph.
    target_network_type: See random_continuous_q_graph.

  Returns:
    Tuple (q_tp1_best, end_points_tp1) where q_tp1_best is a tensor of best
    next-actions as computed by a greedy stochastic policy for each minibatch
    element in state_p1. end_points_tp1 is any auxiliary ouputs computed via
    q_func.
  """
  random_actions = random_sample_fn(batch_size, action_size, num_samples)
  if target_network_type == DQNTarget.notarget:
    q_tp1_best, end_points_tp1 = _q_tp1_notarget(
        q_func, state_p1, batch_size, num_samples, random_actions)
  elif target_network_type == DQNTarget.normal:
    q_tp1_best, end_points_tp1 = _q_tp1_normal(
        q_func, state_p1, batch_size, num_samples, random_actions)
  elif target_network_type == DQNTarget.doubleq:
    q_tp1_best, end_points_tp1 = _q_tp1_doubleq(
        q_func, state_p1, batch_size, action_size, num_samples, random_actions)
  else:
    logging.error('Invalid target_network_mode %s', target_network_type)
  return q_tp1_best, end_points_tp1


@gin.configurable
def random_continuous_q_graph(q_func,
                              transition,
                              random_sample_fn=random_sample_box,
                              num_samples=10,
                              target_network_type=DQNTarget.normal,
                              gamma=1.0,
                              loss_fn=tf.losses.huber_loss,
                              extra_callback=None,
                              log_input_image=True):
  """Construct loss/summary graph for continuous Q-Learning via sampling.

  This Q-function loss implementation is derived from OpenAI baselines, extended
  to work in the continuous case. This function supports batch sizes whose value
  is only known at runtime.

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    transition: SARSTransition namedtuple.
    random_sample_fn: Function that samples actions for Bellman Target
      maximization.
    num_samples: For each state, how many actions to randomly sample in order
      to compute approximate max over Q values.
    target_network_type: Option to use Q Learning without target network, Q
      Learning with a target network (default), or Double-Q Learning with a
      target network.
    gamma: Discount factor.
    loss_fn: Function that computes the td_loss tensor. Takes as arguments
      (target value tensor, predicted value tensor).
    extra_callback: Optional function that takes in (transition, end_points_t,
      end_points_tp1) and adds additional TF graph elements.
    log_input_image: If True, creates an image summary of the first element of
      the state tuple (assumed to be an image tensor).

  Returns:
    A tuple (loss, summaries) where loss is a scalar loss tensor to minimize,
    summaries are TensorFlow summaries.
  """
  state = transition.state
  action = transition.action
  state_p1 = transition.state_p1
  reward = transition.reward
  done = transition.done

  q_t_selected, end_points_t = q_func(state, action, scope='q_func')

  if log_input_image:
    tf.summary.image('input_image', state[0])

  if gamma != 0:
    action_size = action.get_shape().as_list()[1]
    batch_size = tf.shape(done)[0]
    q_tp1_best, end_points_tp1 = _get_q_tp1(
        q_func, state_p1, batch_size, action_size, num_samples,
        random_sample_fn, target_network_type)
    # Bellman eq is Q(s,a) = r + max_{a_p1} Q(s_p1, a_p1)
    # Q(s_T, a_T) is regressed to r, and the max_{a_p1} Q(s_p1, a_p1)
    # term is masked to zero.
    q_tp1_best_masked = (1.0 - done) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(reward + gamma * q_tp1_best_masked)
  else:
    # Supervised Target.
    end_points_tp1 = None
    q_t_selected_target = reward

  td_error = q_t_selected - q_t_selected_target

  if extra_callback is not None:
    extra_callback(transition, end_points_t, end_points_tp1)

  tf.summary.histogram('td_error', td_error)
  tf.summary.histogram('q_t_selected', q_t_selected)
  tf.summary.histogram('q_t_selected_target', q_t_selected_target)
  tf.summary.scalar('mean_q_t_selected', tf.reduce_mean(q_t_selected))

  td_loss = loss_fn(q_t_selected_target, q_t_selected)
  tf.summary.scalar('td_loss', td_loss)
  reg_loss = tf.losses.get_regularization_loss()
  tf.summary.scalar('reg_loss', reg_loss)
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('total_loss', loss)

  summaries = tf.summary.merge_all()

  return loss, summaries


def _get_tau_var(tau, tau_curriculum_steps):
  """Variable which increases linearly from 0 to tau over so many steps."""
  if tau_curriculum_steps > 0:
    tau_var = tf.get_variable('tau', [],
                              initializer=tf.constant_initializer(0.0),
                              trainable=False)
    tau_var = tau_var.assign(
        tf.minimum(float(tau), tau_var + float(tau) / tau_curriculum_steps))
  else:
    tau_var = tf.get_variable('tau', [],
                              initializer=tf.constant_initializer(float(tau)),
                              trainable=False)
  return tau_var


def _get_pcl_values(q_func, not_pad, state, tstep, action,
                    random_sample_fn, num_samples, target_network_type):
  """Computes Q- and V-values for batch of episodes."""
  # get dimensions of input
  batch_size = tf.shape(state)[0]
  episode_length = tf.shape(state)[1]
  img_height = state.get_shape().as_list()[2]
  img_width = state.get_shape().as_list()[3]
  img_channels = state.get_shape().as_list()[4]
  action_size = action.get_shape().as_list()[2]

  # flatten input so each row corresponds to single transition
  flattened_state = tf.reshape(state, [batch_size * episode_length,
                                       img_height, img_width, img_channels])
  flattened_tstep = tf.reshape(tstep, [batch_size * episode_length])
  flattened_action = tf.reshape(action,
                                [batch_size * episode_length, action_size])

  flattened_q_t, end_points_q_t = q_func(
      (flattened_state, flattened_tstep), flattened_action, scope='q_func')
  flattened_v_t, end_points_v_t = _get_q_tp1(
      q_func, (flattened_state, flattened_tstep),
      batch_size * episode_length, action_size, num_samples,
      random_sample_fn, target_network_type)

  # reshape to correspond to original input
  q_t = not_pad * tf.reshape(flattened_q_t, [batch_size, episode_length])
  v_t = not_pad * tf.reshape(flattened_v_t, [batch_size, episode_length])
  v_t = tf.stop_gradient(v_t)

  return q_t, v_t, end_points_q_t, end_points_v_t


@gin.configurable
def random_continuous_pcl_graph(q_func,
                                transition,
                                random_sample_fn=random_sample_box,
                                num_samples=10,
                                target_network_type=None,
                                gamma=1.0,
                                rollout=20,
                                loss_fn=tf.losses.huber_loss,
                                tau=1.0,
                                tau_curriculum_steps=0,
                                stop_gradient_on_adv=False,
                                extra_callback=None):
  """Construct loss/summary graph for continuous PCL via sampling.

  This is an implementation of "Corrected MC", a specific variant of PCL.
  See https://arxiv.org/abs/1802.10264

  Args:
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    transition: SARSTransition namedtuple containing a batch of episodes.
    random_sample_fn: Function that samples actions for Bellman Target
      maximization.
    num_samples: For each state, how many actions to randomly sample in order
      to compute approximate max over Q values.
    target_network_type: Option to use Q Learning without target network, Q
      Learning with a target network (default), or Double-Q Learning with a
      target network.
    gamma: Float discount factor.
    rollout: Integer rollout parameter.  When rollout = 1 we recover Q-learning.
    loss_fn: Function that computes the td_loss tensor. Takes as arguments
      (target value tensor, predicted value tensor).
    tau: Coefficient on correction terms (i.e. on advantages).
    tau_curriculum_steps: Increase tau linearly from 0 over this many steps.
    stop_gradient_on_adv: Whether to allow training of q-values to targets in
      the past.
    extra_callback: Optional function that takes in (transition, end_points_t,
      end_points_tp1) and adds additional TF graph elements.

  Returns:
    A tuple (loss, summaries) where loss is a scalar loss tensor to minimize,
    summaries are TensorFlow summaries.
  """
  if target_network_type is None:
    target_network_type = DQNTarget.normal
  tau_var = _get_tau_var(tau, tau_curriculum_steps)

  state, tstep = transition.state
  action = transition.action
  reward = transition.reward
  done = transition.done

  not_pad = tf.to_float(tf.equal(tf.cumsum(done, axis=1, exclusive=True), 0.0))
  reward *= not_pad

  q_t, v_t, end_points_q_t, end_points_v_t = _get_pcl_values(
      q_func, not_pad, state, tstep, action,
      random_sample_fn, num_samples, target_network_type)

  discounted_sum_rewards = discounted_future_sum(reward, gamma, rollout)

  advantage = q_t - v_t  # equivalent to tau * log_pi in PCL
  if stop_gradient_on_adv:
    advantage = tf.stop_gradient(advantage)
  discounted_sum_adv = discounted_future_sum(
      shift_values(advantage, gamma, 1), gamma, rollout - 1)

  last_v = shift_values(v_t, gamma, rollout)

  # values we regress on
  pcl_values = q_t
  # targets we regress to
  pcl_targets = -tau_var * discounted_sum_adv + discounted_sum_rewards + last_v
  # error is discrepancy between values and targets
  pcl_error = pcl_values - pcl_targets

  if extra_callback:
    extra_callback(transition, end_points_q_t, end_points_v_t)

  tf.summary.histogram('pcl_error', pcl_error)
  tf.summary.histogram('q_t', q_t)
  tf.summary.histogram('v_t', v_t)
  tf.summary.scalar('mean_q_t', tf.reduce_mean(q_t))

  pcl_loss = loss_fn(pcl_values, pcl_targets, weights=not_pad)
  tf.summary.scalar('pcl_loss', pcl_loss)
  reg_loss = tf.losses.get_regularization_loss()
  tf.summary.scalar('reg_loss', reg_loss)
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('total_loss', loss)

  summaries = tf.summary.merge_all()

  return loss, summaries


def shift_values(values, discount, rollout):
  """Shift values up by some amount of time.

  Args:
    values: Tensor of shape [batch_size, time].
    discount: Scalar (float) representing discount factor.
    rollout: Amount (int) to shift values in time by.

  Returns:
    Tensor of shape [batch_size, time] with values shifted.

  """
  final_values = tf.zeros_like(values[:, 0])
  roll_range = tf.cumsum(tf.ones_like(values[:, :rollout]), 0,
                         exclusive=True, reverse=True)
  final_pad = tf.expand_dims(final_values, 1) * discount ** roll_range
  return tf.concat([discount ** rollout * values[:, rollout:],
                    final_pad], 1)


def discounted_future_sum(values, discount, rollout):
  """Discounted future sum of values.

  Args:
    values: A tensor of shape [batch_size, episode_length].
    discount: Scalar discount factor.
    rollout: Number of steps to compute sum.

  Returns:
    Tensor of same shape as values.
  """
  if not rollout:
    return tf.zeros_like(values)

  discount_filter = tf.reshape(
      discount ** tf.range(float(rollout)), [-1, 1, 1])
  expanded_values = tf.concat(
      [values, tf.zeros([tf.shape(values)[0], rollout - 1])], 1)

  conv_values = tf.squeeze(tf.nn.conv1d(
      tf.expand_dims(expanded_values, -1), discount_filter,
      stride=1, padding='VALID'), -1)

  return conv_values
