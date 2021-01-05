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

"""Divergences for BRAC agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import utils

EPS = 1e-8  # Epsilon for avoiding numerical issues.
CLIP_EPS = 1e-3  # Epsilon for clipping actions.


@gin.configurable
def gradient_penalty(s, a_p, a_b, c_fn, gamma=5.0):
  """Calculates interpolated gradient penalty."""
  batch_size = s.shape[0]
  alpha = tf.random.uniform([batch_size])
  a_intpl = a_p + alpha[:, None] * (a_b - a_p)
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a_intpl)
    c_intpl = c_fn(s, a_intpl)
  grad = tape.gradient(c_intpl, a_intpl)
  slope = tf.sqrt(EPS + tf.reduce_sum(tf.square(grad), axis=-1))
  grad_penalty = tf.reduce_mean(tf.square(tf.maximum(slope - 1.0, 0.0)))
  return grad_penalty * gamma


class Divergence(object):
  """Basic interface for divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    raise NotImplementedError

  def dual_critic_loss(self, s, a_p, a_b, c_fn):
    return (- tf.reduce_mean(self.dual_estimate(s, a_p, a_b, c_fn))
            + gradient_penalty(s, a_p, a_b, c_fn))

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    raise NotImplementedError


class FDivergence(Divergence):
  """Interface for f-divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    logits_p = c_fn(s, a_p)
    logits_b = c_fn(s, a_b)
    return self._dual_estimate_with_logits(logits_p, logits_b)

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    raise NotImplementedError

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    _, abn, abn_logb = b_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, utils.clip_by_eps(apn, action_spec, CLIP_EPS))
    abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    return self._primal_estimate_with_densities(
        apn_logp, apn_logb, abn_logp, abn_logb)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    raise NotImplementedError


class KL(FDivergence):
  """KL divergence."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return (- utils.soft_relu(logits_b)
            + tf.log(utils.soft_relu(logits_p) + EPS) + 1.0)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    return tf.reduce_mean(apn_logp - apn_logb, axis=0)


class W(FDivergence):
  """Wasserstein distance."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return logits_p - logits_b


@gin.configurable
def laplacian_kernel(x1, x2, sigma=20.0):
  d12 = tf.reduce_sum(
      tf.abs(x1[None] - x2[:, None]), axis=-1)
  k12 = tf.exp(- d12 / sigma)
  return k12


@gin.configurable
def mmd(x1, x2, kernel, use_sqrt=False):
  k11 = tf.reduce_mean(kernel(x1, x1), axis=[0, 1])
  k12 = tf.reduce_mean(kernel(x1, x2), axis=[0, 1])
  k22 = tf.reduce_mean(kernel(x2, x2), axis=[0, 1])
  if use_sqrt:
    return tf.sqrt(k11 + k22 - 2 * k12 + EPS)
  else:
    return k11 + k22 - 2 * k12


class MMD(Divergence):
  """MMD."""

  def primal_estimate(
      self, s, p_fn, b_fn, n_samples,
      kernel=laplacian_kernel, action_spec=None):
    apn = p_fn.sample_n(s, n_samples)[1]
    abn = b_fn.sample_n(s, n_samples)[1]
    return mmd(apn, abn, kernel)


CLS_DICT = dict(
    kl=KL,
    w=W,
    mmd=MMD,
    )


def get_divergence(name):
  return CLS_DICT[name]()
