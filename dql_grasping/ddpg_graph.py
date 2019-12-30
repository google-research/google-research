# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Standard DDPG Objective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin
import tensorflow.compat.v1 as tf
from dql_grasping.q_graph import DQNTarget


@gin.configurable
def ddpg_graph(a_func,
               q_func,
               transition,
               target_network_type=DQNTarget.normal,
               gamma=1.0,
               dqda_clipping=0.0,
               loss_fn=tf.losses.huber_loss,
               extra_callback=None):
  """DDPG. https://arxiv.org/abs/1509.02971.

  Args:
    a_func: Python function that takes in state, scope as input
      and returns action and intermediate endpoints dictionary.
    q_func: Python function that takes in state, action, scope as input
      and returns Q(state, action) and intermediate endpoints dictionary.
    transition: SARSTransition namedtuple.
    target_network_type: Option to use Q Learning without target network, Q
      Learning with a target network (default), or Double-Q Learning with a
      target network.
    gamma: Discount factor.
    dqda_clipping: (float) clips the gradient dqda element-wise between
        [-dqda_clipping, dqda_clipping]. Does not perform clipping if
        dqda_clipping == 0.
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

  q_t_selected, end_points_t = q_func(state, action, scope='q_func')

  if gamma != 0:
    action_p1, _ = a_func(state_p1, scope='a_func')
    if target_network_type == DQNTarget.notarget:
      # Evaluate target values using the current net only.
      q_tp1_best, end_points_tp1 = q_func(
          state_p1, action_p1, scope='q_func', reuse=True)
    elif target_network_type == DQNTarget.normal:
      # Target network Q values at t+1.
      q_tp1_best, end_points_tp1 = q_func(
          state_p1, action_p1, scope='target_q_func')
    else:
      logging.error('Invalid target_network_mode %s', target_network_type)
    q_tp1_best_masked = (1.0 - done) * q_tp1_best
    q_t_selected_target = tf.stop_gradient(reward + gamma * q_tp1_best_masked)
  else:
    # Supervised Target.
    q_t_selected_target = tf.stop_gradient(reward)

  # Critic Loss
  td_error = q_t_selected - q_t_selected_target
  critic_loss = loss_fn(q_t_selected_target, q_t_selected)

  # Actor Loss (maximize E[Q(a_t|s_t)] via policy grdient)
  policy_action, _ = a_func(state, scope='a_func', reuse=True)
  q_t, _ = q_func(state, policy_action, scope='q_func', reuse=True)

  dqda = tf.gradients(q_t, policy_action)[0]
  if dqda_clipping > 0:
    dqda = tf.clip_by_value(dqda, -dqda_clipping, dqda_clipping)
  actor_loss = tf.losses.mean_squared_error(
      tf.stop_gradient(dqda + policy_action), policy_action)
  loss = tf.losses.get_total_loss()

  if extra_callback is not None:
    extra_callback(transition, end_points_t, end_points_tp1)

  tf.summary.histogram('td_error', td_error)
  tf.summary.histogram('q_t_selected', q_t_selected)
  tf.summary.histogram('q_t_selected_target', q_t_selected_target)
  tf.summary.scalar('mean_q_t_selected', tf.reduce_mean(q_t_selected))
  tf.summary.scalar('critic_loss', critic_loss)
  tf.summary.scalar('actor_loss', actor_loss)
  tf.summary.scalar('actor_mean_q', tf.reduce_mean(q_t, 0))
  tf.summary.scalar('total_loss', loss)

  all_summaries = tf.summary.merge_all()

  # Make this a named tuple.
  return actor_loss, critic_loss, all_summaries
