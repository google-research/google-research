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

"""Forward-backward algorithm for integrating out discrete states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from snlds import utils

namedtuple = collections.namedtuple


def forward_pass(log_a, log_b, logprob_s0):
  """Computing the forward pass of Baum-Welch Algorithm.

  By employing log-exp-sum trick, values are computed in log space, including
  the output. Notation is adopted from https://arxiv.org/abs/1910.09588.
  `log_a` is the likelihood of discrete states, `log p(s[t] | s[t-1], x[t-1])`,
  `log_b` is the likelihood of observations, `log p(x[t], z[t] | s[t])`,
  and `logprob_s0` is the likelihood of initial discrete states, `log p(s[0])`.
  Forward pass calculates the filtering likelihood of `log p(s_t | x_1:t)`.

  Args:
    log_a: a float `Tensor` of size [batch, num_steps, num_categ, num_categ]
      stores time dependent transition matrices, `log p(s[t] | s[t-1], x[t-1])`.
      `A[i, j]` is the transition probability from `s[t-1]=j` to `s[t]=i`.
    log_b: a float `Tensor` of size [batch, num_steps, num_categ] stores time
      dependent emission matrices, 'log p(x[t](, z[t])| s[t])`.
    logprob_s0: a float `Tensor` of size [num_categ], initial discrete states
      probability, `log p(s[0])`.

  Returns:
    forward_pass: a float 3D `Tensor` of size [batch, num_steps, num_categ]
      stores the forward pass probability of `log p(s_t | x_1:t)`, which is
      normalized.
    normalizer: a float 2D `Tensor` of size [batch, num_steps] stores the
      normalizing probability, `log p(x_t | x_1:t-1)`.
  """
  num_steps = log_a.get_shape().with_rank_at_least(3).dims[1].value

  tas = [tf.TensorArray(tf.float32, num_steps, name=n)
         for n in ["forward_prob", "normalizer"]]

  # The function will return normalized forward probability and
  # normalizing constant as a list, [forward_logprob, normalizer].
  init_updates = utils.normalize_logprob(
      logprob_s0[tf.newaxis, :] + log_b[:, 0, :], axis=-1)

  tas = utils.write_updates_to_tas(tas, 0, init_updates)

  prev_prob = init_updates[0]
  init_state = (1,
                prev_prob,
                tas)

  def _cond(t, *unused_args):
    return t < num_steps

  def _steps(t, prev_prob, fwd_tas):
    """One step forward in iterations."""
    bi_t = log_b[:, t, :]  # log p(x[t+1] | s[t+1])
    aij_t = log_a[:, t, :, :]  # log p(s[t+1] | s[t], x[t])

    current_updates = tf.math.reduce_logsumexp(
        bi_t[:, :, tf.newaxis] + aij_t + prev_prob[:, tf.newaxis, :],
        axis=-1)
    current_updates = utils.normalize_logprob(current_updates, axis=-1)

    prev_prob = current_updates[0]
    fwd_tas = utils.write_updates_to_tas(fwd_tas, t, current_updates)

    return (t+1, prev_prob, fwd_tas)

  _, _, tas_final = tf.while_loop(
      _cond,
      _steps,
      init_state
  )

  # transpose to [batch, step, state]
  forward_prob = tf.transpose(tas_final[0].stack(), [1, 0, 2])
  normalizer = tf.transpose(tf.squeeze(tas_final[1].stack(), axis=[-1]), [1, 0])
  return forward_prob, normalizer


def backward_pass(log_a, log_b, logprob_s0):
  """Computing the backward pass of Baum-Welch Algorithm.

  Args:
    log_a: a float `Tensor` of size [batch, num_steps, num_categ, num_categ]
      stores time dependent transition matrices, `log p(s[t] | s[t-1], x[t-1])`.
      `A[:, t, i, j]` is the transition probability from `s[t-1]=j` to `s[t]=i`.
      Since `A[:, t, :, :]` is using the information from `t-1`, `A[:, 0, :, :]`
      is meaningless, i.e. set to zeros.
    log_b: a float `Tensor` of size [batch, num_steps, num_categ] stores time
      dependent emission matrices, `log p(x[t](, z[t])| s[t])`
    logprob_s0: a float `Tensor` of size [num_categ], initial discrete states
      probability, `log p(s[0])`.

  Returns:
    backward_pass: a float `Tensor` of size [batch_size, num_steps, num_categ]
      stores the backward-pass  probability log p(s_t | x_t+1:T(, z_t+1:T)).
    normalizer: a float `Tensor` of size [batch, num_steps, num_categ] stores
      the normalizing probability, log p(x_t | x_t:T).
  """
  batch_size = tf.shape(log_b)[0]
  num_steps = tf.shape(log_b)[1]
  num_categ = logprob_s0.shape[0]

  tas = [tf.TensorArray(tf.float32, num_steps, name=n)
         for n in ["backward_prob", "normalizer"]]

  init_updates = [tf.zeros([batch_size, num_categ]), tf.zeros([batch_size, 1])]

  tas = utils.write_updates_to_tas(tas, num_steps-1, init_updates)

  next_prob = init_updates[0]
  init_state = (num_steps-2,
                next_prob,
                tas)

  def _cond(t, *unused_args):
    return t > -1

  def _steps(t, next_prob, bwd_tas):
    """One step backward."""
    bi_tp1 = log_b[:, t+1, :]  # log p(x[t+1] | s[t+1])
    aij_tp1 = log_a[:, t+1, :, :]  # log p(s[t+1] | s[t], x[t])
    current_updates = tf.math.reduce_logsumexp(
        next_prob[:, :, tf.newaxis] + aij_tp1 + bi_tp1[:, :, tf.newaxis],
        axis=-2)

    current_updates = utils.normalize_logprob(current_updates, axis=-1)

    next_prob = current_updates[0]
    bwd_tas = utils.write_updates_to_tas(bwd_tas, t, current_updates)

    return (t-1, next_prob, bwd_tas)

  _, _, tas_final = tf.while_loop(
      _cond,
      _steps,
      init_state
  )

  backward_prob = tf.transpose(tas_final[0].stack(), [1, 0, 2])
  normalizer = tf.transpose(tf.squeeze(tas_final[1].stack(), axis=[-1]), [1, 0])

  return backward_prob, normalizer


def forward_backward(log_a, log_b, log_init):
  """Forward backward algorithm."""
  fwd, _ = forward_pass(log_a, log_b, log_init)
  bwd, _ = backward_pass(log_a, log_b, log_init)

  m_fwd = fwd[:, 0:-1, tf.newaxis, :]
  m_bwd = bwd[:, 1:, :, tf.newaxis]
  m_a = log_a[:, 1:, :, :]
  m_b = log_b[:, 1:, :, tf.newaxis]

  # posterior score
  posterior = fwd + bwd
  gamma_ij = m_fwd + m_a + m_bwd + m_b

  # normalize the probability matrices
  posterior, _ = utils.normalize_logprob(posterior, axis=-1)
  gamma_ij, _ = utils.normalize_logprob(gamma_ij, axis=[-2, -1])

  # padding the matrix to the same shape of inputs
  gamma_ij = tf.concat([tf.zeros([tf.shape(log_a)[0], 1,
                                  tf.shape(log_a)[2], tf.shape(log_a)[3]]),
                        gamma_ij], axis=1, name="concat_f_b")

  return fwd, bwd, posterior, gamma_ij
