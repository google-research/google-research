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

"""Policy and value losses for Proximal Policy Optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import tensorflow.compat.v1 as tf


def ppo_entropy_loss(dist_new):
  """Entropy loss calculated on the current policy.

  Args:
    dist_new: current policy distribution.

  Returns:
    entropy loss.
  """
  entropy_tf = dist_new.entropy()
  entropy_loss = tf.reduce_mean(entropy_tf)
  return entropy_loss


def l2_norm_policy_loss(policy_mean, policy_logstd, mcts_mean, mcts_logstd):
  """Computes the l2 norm between mean/standard deviations of two distributions.

  Args:
    policy_mean: Mean value of policy distribution (Multivariate Normal
      Distribution).
    policy_logstd: Standard deviation value of policy distribution (Multivariate
      Normal Distribution).
    mcts_mean: Mean value of MCTS policy. We assume that MCTS samples
      are coming from a Multivariate Normal Distribution.
    mcts_logstd: Standard deviation value of MCTS policy. We assume that MCTS
      samples are coming from a Multivariate Normal Distribution.

  Returns:
    mean of MSE for means and standard deviations between policy and MCTS.
  """
  duplicated_logstd = tf.tile(policy_logstd, [tf.shape(mcts_logstd)[0], 1])
  mean_mse = tf.losses.mean_squared_error(
      labels=mcts_mean, predictions=policy_mean)
  logstd_mse = tf.losses.mean_squared_error(
      labels=mcts_logstd, predictions=duplicated_logstd)
  return tf.reduce_mean(mean_mse), tf.reduce_mean(logstd_mse)


@gin.configurable
def ppo_policy_loss(neg_logprobs_old,
                    actions,
                    advantages,
                    dist_new,
                    policy_gradient_enable=False,
                    mcts_sampling=False,
                    clipping_coeff=0.2,
                    mcts_clipping_coeff=0.9,
                    tanh_action_clipping=False):
  """Use the formula in PPO baseline for calculating policy loss.

  paper: https://arxiv.org/abs/1707.06347

  Args:
    neg_logprobs_old: old negative log of probability.
    actions: actions from old policy.
    advantages: advantages from old policy.
    dist_new: the latest trained policy distribution.
    policy_gradient_enable: if True, vanilla policy gradient with advantage
      is used.
    mcts_sampling: If True, the data samples are generated with MCTS sampling.
    clipping_coeff: the coefficient used to clip the probability ratio.
    mcts_clipping_coeff: the coefficient used to clip the probability ration,
      when the data are sampled using MCTS.
    tanh_action_clipping: if True, performs tanh action clipping. Enabling tanh
      action clipping bound the actions to [-1, 1].
      Paper --> https://arxiv.org/pdf/1801.01290.pdf

  Returns:
    policy_loss: policy loss.
  """
  neg_logprobs_new = dist_new.negative_log_prob(actions)

  current_clipping_coeff = tf.cond(
      tf.equal(mcts_sampling, True), lambda: tf.constant(mcts_clipping_coeff),
      lambda: tf.constant(clipping_coeff))

  # Calculate correction for logprob if tanh clipping is enabled
  # A mechanism for clipping the actions between [-1., 1.]
  # paper: https://arxiv.org/pdf/1801.01290.pdf
  if tanh_action_clipping:
    logprobs_correction = tf.reduce_sum(
        tf.log(1 - tf.tanh(actions)**2 + 1e-6), axis=1)
    neg_logprobs_new = neg_logprobs_new + logprobs_correction

  p_ratio = tf.exp(neg_logprobs_old - neg_logprobs_new, name='ratio')

  if policy_gradient_enable:
    pg_losses = advantages * neg_logprobs_new
    pg_loss = tf.reduce_mean(pg_losses, name='policy_loss')
  else:  # using PPO formulat to calculate policy loss
    # Defining Loss = - J is equivalent to max J
    pg_losses = -advantages * p_ratio
    pg_losses2 = -advantages * tf.clip_by_value(
        p_ratio, 1. - current_clipping_coeff, 1. + current_clipping_coeff)
    pg_loss = tf.reduce_mean(
        tf.maximum(pg_losses, pg_losses2), name='policy_loss')
  # KL between new and old policy
  approxkl = .5 * tf.reduce_mean(tf.square(neg_logprobs_new - neg_logprobs_old))
  # Which fraction of policy ratios get clipped
  clipfrac = tf.reduce_mean(
      tf.to_float(tf.greater(tf.abs(p_ratio - 1.), current_clipping_coeff)))

  return pg_loss, approxkl, clipfrac, p_ratio


def ppo1_value_loss(pred_value, returns):
  """Use the formula in PPO1 for calculating value loss.

  https://github.com/openai/baselines/tree/master/baselines/ppo1
  paper: https://arxiv.org/abs/1707.06347

  Args:
    pred_value: predicted state-value from value network.
    returns: calculated returns from old policy.

  Returns:
    value loss, measuring the difference between the predicted state-values from
    the neural model and the calculated returns.
  """
  return tf.reduce_mean(tf.square(pred_value - returns))


@gin.configurable
def ppo2_value_loss(value_old, pred_value, returns, clipping_coeff=0.2):
  """Use the formula in PPO2 for calculating value loss.

  This is a newer policy loss for PPO. That is, it also performs clipping
  on the state-values.

  https://github.com/openai/baselines/tree/master/baselines/ppo2

  Args:
    value_old: state-value from old value network.
    pred_value: state-value from current value network.
    returns: calculated returns from old policy.
    clipping_coeff: the coefficient used to clipping the value loss.

  Returns:
    value loss, measuring the difference between the predicted state-values from
    the neural model and the calculated returns. The value loss here is a newer
    version compared to what proposed in PPO paper at
    https://arxiv.org/pdf/1707.06347.pdf), suggested in the OpenAI baselines
    repository at https://github.com/openai/baselines/tree/master/baselines/ppo2
  """
  vpredclipped = value_old + tf.clip_by_value(pred_value - value_old,
                                              -clipping_coeff, clipping_coeff)
  # Unclipped value
  vf_losses1 = tf.square(pred_value - returns)
  # Clipped value
  vf_losses2 = tf.square(vpredclipped - returns)
  value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
  return value_loss
