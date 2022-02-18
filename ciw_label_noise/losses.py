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

r"""Loss functions.

Implements the Constrained Instance and Class reWeighted losses (CIW/CICW)
proposed in https://arxiv.org/abs/2111.05428.
"""

import numpy as np
import cvxpy as cp
import tensorflow as tf

from ciw_label_noise import utils


def ce_loss(labels, preds, from_logits=True):
  """Cross entropy loss."""
  num_classes = preds.shape[1]
  labels_oh = utils.maybe_one_hot(labels, depth=num_classes)
  cce = tf.keras.backend.categorical_crossentropy(
      labels_oh, preds, from_logits=from_logits)
  return tf.reduce_mean(cce)


def get_inner_loss_cvx(logits, labels, div_type_cls, gamma):
  """Compute class-reweighted loss using CVX."""
  num_classes = logits.shape[1]
  ce_loss_all = -1 * (
      logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True))
  ce_loss_all_numpy = ce_loss_all.numpy()
  num_examples = logits.shape[0]
  dim = num_classes
  v = cp.Variable((num_examples, dim))
  v.value = v.project(labels)
  constraints = [v >= 0, cp.sum(v, axis=1) == 1]
  if div_type_cls == 'l2':
    constraints.append(cp.sum(cp.square(v - labels), axis=1) <= gamma)
  elif div_type_cls == 'l1':
    constraints.append(cp.sum(cp.abs(v - labels), axis=1) <= gamma)
  elif div_type_cls == 'kl':
    constraints.append(
        cp.sum(
            cp.multiply(labels,
                        cp.log(labels + 1e-6) - cp.log(v + 1e-6)),
            axis=1) <= gamma)
  obj = cp.Minimize(cp.sum(cp.multiply(v, ce_loss_all_numpy)))
  prob = cp.Problem(obj, constraints)
  try:
    prob.solve(warm_start=True)
  except cp.error.SolverError:
    v.value = v.project(labels)
    prob.solve(solver='SCS', warm_start=True)
  inner_loss = tf.reduce_sum(tf.multiply(v.value, ce_loss_all), axis=1)
  return inner_loss


def get_inner_loss(logits, labels, div_type_cls, gamma):
  """Compute class-reweighted loss."""
  num_classes = logits.shape[1]
  ce_loss_all = -1 * (
      logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True))
  ce_loss_all_numpy = ce_loss_all.numpy()
  n = ce_loss_all.shape[0]
  ce_loss_cls = tf.gather_nd(ce_loss_all,
                             tf.stack([np.arange(n), labels], axis=1))
  ce_loss_cls_numpy = ce_loss_cls.numpy()
  if div_type_cls == 'l1':
    ce_loss_min = tf.reduce_min(ce_loss_all, axis=1)
    inner_loss = (1 - gamma) * ce_loss_cls + gamma * ce_loss_min
  elif div_type_cls == 'l2':
    less_eq_cls = tf.cast(
        tf.less_equal(ce_loss_all_numpy, ce_loss_cls_numpy[:, None]),
        tf.float32)
    means = tf.reduce_sum(
        ce_loss_all_numpy * less_eq_cls, axis=1) / tf.reduce_sum(
            less_eq_cls, axis=1)
    active_ind = tf.logical_or(
        tf.less(ce_loss_all_numpy, means[:, None]),
        tf.cast(utils.maybe_one_hot(labels, depth=num_classes), tf.bool))
    inactive_ind = tf.logical_not(active_ind)
    mu_neg = tf.reduce_sum(
        ce_loss_all_numpy * tf.cast(active_ind, tf.float32),
        axis=1) / tf.reduce_sum(
            tf.cast(active_ind, tf.float32), axis=1)
    nu = tf.cast(inactive_ind, tf.float32) * (
        ce_loss_all_numpy - mu_neg[:, None])
    v = ce_loss_all_numpy - nu - mu_neg[:, None]
    lambda_sq = tf.reduce_sum(tf.square(v), axis=1) / gamma
    lambda_sq_clamped = tf.where(tf.equal(lambda_sq, 0), tf.ones(n), lambda_sq)
    # exact sol in most cases of interest
    v = tf.maximum(
        -1 / tf.sqrt(lambda_sq_clamped[:, None]) * v +
        utils.maybe_one_hot(labels, depth=num_classes), 0.)
    inner_loss_weighted = tf.reduce_sum(tf.multiply(ce_loss_all, v), axis=1)
    inner_loss = tf.where(
        tf.greater(lambda_sq, 0), inner_loss_weighted, ce_loss_cls)
  else:
    raise ValueError('Unknown divergence type {}'.format(div_type_cls))

  return inner_loss


def get_loss_weights(losses, div_type, alpha, lambda_hyp, w_type, iteration,
                     burnin):
  """Compute weights for reweighing instance losses."""
  if iteration <= burnin or div_type == 'none':
    weights = np.ones_like(losses)
  elif div_type == 'alpha':
    if np.abs(alpha - 1.) < 1e-3:
      weights = tf.exp(-1 * losses / lambda_hyp)
      weights = weights.numpy()
    else:
      weights = np.power(
          np.maximum((1. - alpha) * losses + lambda_hyp, 0.0),
          1. / (alpha - 1.))
  else:
    raise NotImplementedError(
        'Divergence {} is not implemented'.format(div_type))

  if w_type == 'normalized':
    weights = weights / np.sum(weights)  #  * len(labels)
  return weights


def div_constrained_loss(labels,
                         preds,
                         div_type,
                         alpha,
                         lambda_hyp,
                         w_type,
                         div_type_cls,
                         gamma,
                         iteration,
                         burnin,
                         from_logits=True,
                         mixup_type='none',
                         mixup_alpha=32.,
                         m_pow=1,
                         dcl_w_mixup=0,
                         model=None,
                         images=None,
                         is_train=True):
  """Divergence Constrained Instance and Class reWeighted loss."""
  num_classes = preds.shape[1]
  labels_oh = utils.maybe_one_hot(labels, depth=num_classes)
  cce = tf.keras.backend.categorical_crossentropy(
      labels_oh, preds, from_logits=from_logits)

  if div_type_cls == 'none' or gamma == 0 or iteration <= burnin:
    inner_losses = cce
  else:
    inner_losses = get_inner_loss_cvx(preds, labels_oh, div_type_cls, gamma)
    # inner_losses = get_inner_loss(preds, labels, div_type_cls, gamma)
  inner_losses_numpy = inner_losses.numpy()
  weights = get_loss_weights(inner_losses_numpy, div_type, alpha, lambda_hyp,
                             w_type, iteration, burnin)
  if mixup_type == 'simple' or mixup_type == 'none' or iteration <= burnin:
    final_loss = tf.reduce_sum(tf.multiply(weights, inner_losses))
  else:
    mixup_weights = weights**m_pow
    mixup_weights = mixup_weights / np.sum(mixup_weights)
    if mixup_type == 'sample_w':
      images_mix, labels_mix = utils.mixup(
          images,
          labels,
          num_classes,
          mixup_alpha,
          mixing_weights=None,
          mixing_probs=mixup_weights)
    elif mixup_type == 'weight_w':
      images_mix, labels_mix = utils.mixup(
          images,
          labels,
          num_classes,
          mixup_alpha,
          mixing_weights=mixup_weights,
          mixing_probs=None)
    elif mixup_type == 'sample_weight_w':
      images_mix, labels_mix = utils.mixup(
          images,
          labels,
          num_classes,
          mixup_alpha,
          mixing_weights=mixup_weights,
          mixing_probs=mixup_weights)
    elif mixup_type == 'label_smoothing_w':
      w_min = np.maximum(np.min(mixup_weights) - 0.01, 0.0)
      w_max = np.minimum(np.max(mixup_weights) + 0.01, 1.0)
      smoothing_weights = (mixup_weights - w_min) / (w_max - w_min)
      preds = tf.one_hot(tf.argmax(preds, -1), depth=num_classes)
      smoothed_labels = utils.get_smoothed_labels(labels_oh, preds,
                                                  smoothing_weights)
      smoothed_labels = tf.stop_gradient(smoothed_labels)
      images_mix, labels_mix = utils.mixup(
          images,
          tf.constant(smoothed_labels, dtype=tf.float32),
          num_classes,
          mixup_alpha,
          mixing_weights=None,
          mixing_probs=None)
    else:
      raise ValueError('Unknown mixup_type: {}'.format(mixup_type))
    preds_mix = model([images_mix, is_train])
    cce_mix = tf.keras.backend.categorical_crossentropy(
        labels_mix, preds_mix, from_logits=from_logits)
    if dcl_w_mixup:
      if div_type_cls == 'none' or gamma == 0 or iteration <= burnin:
        inner_losses = cce_mix
      else:
        inner_losses = get_inner_loss_cvx(preds_mix, labels_mix, div_type_cls,
                                          gamma)
        # inner_losses = get_inner_loss(preds_mix, labels_mix, div_type_cls,
        #                               gamma)
      inner_losses_numpy = inner_losses.numpy()
      weights = get_loss_weights(inner_losses_numpy, div_type, alpha,
                                 lambda_hyp, w_type, iteration, burnin)
      final_loss = tf.reduce_sum(tf.multiply(weights, inner_losses))
    else:
      final_loss = tf.reduce_mean(cce_mix)

  return final_loss
