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

"""A JAX implementation of Robust Bi-tempered loss.

Source: https://bit.ly/3jSol8T
"""

import functools

import jax
from jax.lax import cond
from jax.lax import while_loop
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jax.jit
def _cross_entropy_loss(logits,
                        labels):
  log_preds = jax.nn.log_softmax(logits)
  return jnp.sum(labels * (jnp.log(labels + 1e-15) - log_preds), axis=-1)


@jax.jit
def log_t(u, t):
  """Compute log_t for `u`."""

  def _internal_log_t(u, t):
    return (jnp.power(u, (1.0 - t)) - 1.0) / (1.0 - t)

  return cond(
      jnp.abs(t - 1.0) < 1e-15, jnp.log,
      functools.partial(_internal_log_t, t=t), u)


@jax.jit
def exp_t(u, t):
  """Compute exp_t for `u`."""

  def _internal_exp_t(u, t):
    return jnp.power(jnp.maximum(1.0 + (1.0 - t) * u, 0.0), 1.0 / (1.0 - t))

  return cond(
      jnp.abs(t - 1.0) < 1e-15, jnp.exp,
      functools.partial(_internal_exp_t, t=t), u)


@functools.partial(jax.jit, static_argnums=2)
def compute_normalization_fixed_point(activations,
                                      t,
                                      num_iters = 5):
  """Returns the normalization value for each example (t > 1.0).

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
  Return: An array of same rank as activation with the last dimension being 1.
  """

  mu = jnp.max(activations, -1, keepdims=True)
  normalized_activations_step_0 = activations - mu

  def cond_fun(carry):
    _, iters = carry
    return iters < num_iters

  def body_fun(carry):
    normalized_activations, iters = carry
    logt_partition = jnp.sum(
        exp_t(normalized_activations, t), -1, keepdims=True)
    normalized_activations_t = normalized_activations_step_0 * jnp.power(
        logt_partition, 1.0 - t)
    return normalized_activations_t, iters + 1

  normalized_activations_t, _ = while_loop(cond_fun, body_fun,
                                           (normalized_activations_step_0, 0))
  logt_partition = jnp.sum(
      exp_t(normalized_activations_t, t), -1, keepdims=True)
  return -log_t(1.0 / logt_partition, t) + mu


@functools.partial(jax.jit, static_argnums=2)
def compute_normalization_binary_search(activations,
                                        t,
                                        num_iters = 10):
  """Returns the normalization value for each example (t < 1.0).

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support).
    num_iters: Number of iterations to run the method.
  Return: An array of same rank as activation with the last dimension being 1.
  """
  mu = jnp.max(activations, -1, keepdims=True)
  normalized_activations = activations - mu
  shape_activations = activations.shape
  effective_dim = jnp.float32(
      jnp.sum(
          jnp.int32(normalized_activations > -1.0 / (1.0 - t)),
          -1,
          keepdims=True))
  shape_partition = list(shape_activations[:-1]) + [1]
  lower = jnp.zeros(shape_partition)
  upper = -log_t(1.0 / effective_dim, t) * jnp.ones(shape_partition)

  def cond_fun(carry):
    _, _, iters = carry
    return iters < num_iters

  def body_fun(carry):
    lower, upper, iters = carry
    logt_partition = (upper + lower) / 2.0
    sum_probs = jnp.sum(
        exp_t(normalized_activations - logt_partition, t), -1, keepdims=True)
    update = jnp.float32(sum_probs < 1.0)
    lower = jnp.reshape(lower * update + (1.0 - update) * logt_partition,
                        shape_partition)
    upper = jnp.reshape(upper * (1.0 - update) + update * logt_partition,
                        shape_partition)
    return lower, upper, iters + 1

  lower = jnp.zeros(shape_partition)
  upper = -log_t(1.0 / effective_dim, t) * jnp.ones(shape_partition)
  lower, upper, _ = while_loop(cond_fun, body_fun, (lower, upper, 0))

  logt_partition = (upper + lower) / 2.0
  return logt_partition + mu


@functools.partial(jax.jit, static_argnums=2)
def compute_tempered_normalization(activations,
                                   t,
                                   num_iters = 5):
  return cond(
      t < 1.0,
      functools.partial(
          compute_normalization_binary_search, t=t, num_iters=num_iters),
      functools.partial(
          compute_normalization_fixed_point, t=t, num_iters=num_iters),
      activations)


@functools.partial(jax.jit, static_argnums=2)
def compute_normalization(activations,
                          t,
                          num_iters = 5):
  """Returns the normalization value for each example.

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
  Return: An array of same rank as activation with the last dimension being 1.
  """
  return cond(
      jnp.abs(t - 1.0) < 1e-15,
      functools.partial(logsumexp, axis=-1, keepdims=True),
      functools.partial(
          compute_tempered_normalization, t=t, num_iters=num_iters),
      activations)


@functools.partial(jax.jit, static_argnums=2)
def tempered_sigmoid(activations, t, num_iters=5):
  """Tempered sigmoid function.

  Args:
    activations: Activations for the positive class for binary classification.
    t: Temperature array > 0.0.
    num_iters: Number of iterations to run the method.

  Returns:
    A probabilities array.
  """
  input_shape = activations.shape
  activations_2d = jnp.reshape(activations, [-1, 1])
  internal_activations = jnp.concatenate(
      [jnp.zeros_like(activations_2d), activations_2d], 1)
  internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
  one_class_probabilities = internal_probabilities[:, 1]
  return jnp.reshape(one_class_probabilities, input_shape)


@functools.partial(jax.jit, static_argnums=2)
def tempered_softmax(activations, t, num_iters=5):
  """Tempered softmax function.

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    t: Temperature array > 0.0.
    num_iters: Number of iterations to run the method.

  Returns:
    A probabilities array.
  """
  normalization_constants = compute_normalization(activations, t, num_iters)
  return exp_t(activations - normalization_constants, t)


def _internal_bi_tempered_logistic_loss(activations, labels, t1, t2):
  """Computes the Bi-Tempered logistic loss.

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    labels: batch_size
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness).

  Returns:
    A loss array for robust loss.
  """
  normalization_constants = compute_normalization(activations, t2, num_iters=5)
  if t2 == 1.0:
    if t1 == 1.0:
      return normalization_constants + jnp.sum(
          jnp.multiply(labels,
                       jnp.log(labels + 1e-10) - activations), -1)
    else:
      shifted_activations = jnp.exp(activations - normalization_constants)
      one_minus_t1 = (1.0 - t1)
      one_minus_t2 = 1.0
  else:
    one_minus_t1 = (1.0 - t1)
    one_minus_t2 = (1.0 - t2)
    shifted_activations = jnp.maximum(
        1.0 + one_minus_t2 * (activations - normalization_constants), 0.0)

  if t1 == 1.0:
    return jnp.sum(
        jnp.multiply(
            jnp.log(labels + 1e-10) -
            jnp.log(jnp.power(shifted_activations, 1.0 / one_minus_t2)),
            labels), -1)
  else:
    beta = 1.0 + one_minus_t1
    logt_probs = (jnp.power(shifted_activations, one_minus_t1 / one_minus_t2) -
                  1.0) / one_minus_t1
    return jnp.sum(
        jnp.multiply(log_t(labels, t1) - logt_probs, labels) - 1.0 / beta *
        (jnp.power(labels, beta) -
         jnp.power(shifted_activations, beta / one_minus_t2)), -1)


@jax.custom_vjp
def bi_tempered_logistic_loss(activations,
                              labels,
                              t1,
                              t2,
                              label_smoothing=0.0,
                              num_iters=5):
  """Bi-Tempered Logistic Loss with custom gradient.

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    labels: An array with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.

  Returns:
    A loss array.
  """
  loss_values, _ = bi_tempered_logistic_loss_fwd(activations, labels, t1, t2,
                                                 label_smoothing, num_iters)
  return loss_values


@functools.partial(jax.jit, static_argnums=5)
def bi_tempered_logistic_loss_fwd(activations,
                                  labels,
                                  t1,
                                  t2,
                                  label_smoothing=0.0,
                                  num_iters=5):
  """Forward pass function for bi-tempered logistic loss.

  Args:
    activations: A multi-dimensional array with last dimension `num_classes`.
    labels: An array with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.

  Returns:
    A loss array, residuals.
  """
  num_classes = jnp.int32(labels.shape[-1])
  labels = cond(
      label_smoothing > 0.0,
      lambda u:  # pylint: disable=g-long-lambda
      (1 - num_classes /
       (num_classes - 1) * label_smoothing) * u + label_smoothing /
      (num_classes - 1),
      lambda u: u,
      labels)
  probabilities = tempered_softmax(activations, t2, num_iters)

  def _tempred_cross_entropy_loss(unused_activations):
    loss_values = jnp.multiply(
        labels,
        log_t(labels + 1e-10, t1) -
        log_t(probabilities, t1)) - 1.0 / (2.0 - t1) * (
            jnp.power(labels, 2.0 - t1) - jnp.power(probabilities, 2.0 - t1))
    loss_values = jnp.sum(loss_values, -1)
    return loss_values

  loss_values = cond(
      jnp.logical_and(
          jnp.less(jnp.abs(t1 - 1.0), 1e-15),
          jnp.less(jnp.abs(t2 - 1.0), 1e-15)),
      functools.partial(_cross_entropy_loss, labels=labels),
      _tempred_cross_entropy_loss,
      activations)
  return loss_values, (labels, t1, t2, probabilities)


@jax.jit
def bi_tempered_logistic_loss_bwd(res, d_loss):
  """Backward pass function for bi-tempered logistic loss.

  Args:
    res: Residuals.
    d_loss: Differential.

  Returns:
    Derivatives.
  """
  labels, t1, t2, probabilities = res
  delta_probs = probabilities - labels
  forget_factor = jnp.power(probabilities, t2 - t1)
  delta_probs_times_forget_factor = jnp.multiply(delta_probs, forget_factor)
  delta_forget_sum = jnp.sum(
      delta_probs_times_forget_factor, -1, keepdims=True)
  escorts = jnp.power(probabilities, t2)
  escorts = escorts / jnp.sum(escorts, -1, keepdims=True)
  derivative = delta_probs_times_forget_factor - jnp.multiply(
      escorts, delta_forget_sum)
  return (jnp.multiply(d_loss, derivative), None, None, None, None, None)


bi_tempered_logistic_loss.defvjp(
    bi_tempered_logistic_loss_fwd, bi_tempered_logistic_loss_bwd)


def bi_tempered_binary_logistic_loss(activations,
                                     labels,
                                     t1,
                                     t2,
                                     label_smoothing=0.0,
                                     num_iters=5):
  """Bi-Tempered binary logistic loss.

  Args:
    activations: An array containing activations for class 1.
    labels: An array with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing
    num_iters: Number of iterations to run the method.

  Returns:
    A loss array.
  """
  out_shape = labels.shape
  labels_2d = jnp.reshape(labels, [-1, 1])
  activations_2d = jnp.reshape(activations, [-1, 1])
  internal_labels = jnp.concatenate([1.0 - labels_2d, labels_2d], 1)
  internal_logits = jnp.concatenate(
      [jnp.zeros_like(activations_2d), activations_2d], 1)
  losses = bi_tempered_logistic_loss(internal_logits, internal_labels, t1, t2,
                                     label_smoothing, num_iters)
  return jnp.reshape(losses, out_shape)
