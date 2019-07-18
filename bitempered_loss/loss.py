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

"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://arxiv.org/pdf/1906.03361.pdf
"""

import tensorflow as tf


def log_t(u, t):
  """Compute log_t for `u`."""
  if t == 1:
    return tf.log(u)
  else:
    return (u**(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
  """Compute exp_t for `u`."""
  if t == 1:
    return tf.exp(u)
  else:
    return tf.nn.relu(1.0 + (1.0 - t) * u)**(1.0 / (1.0 - t))


def compute_normalization(activations, t, num_iters=5):
  """Returns the normalization value for each example.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
  Return: A tensor of same rank as activation with the last dimension being 1.
  """

  mu = tf.reduce_max(activations, -1, keep_dims=True)
  normalized_activations_step_0 = activations - mu

  def iter_condition(i, unused_normalized_activations):
    return i < num_iters

  def iter_body(i, normalized_activations):
    logt_partition = tf.reduce_sum(
        exp_t(normalized_activations, t), -1, keep_dims=True)
    normalized_activations_t = normalized_activations_step_0 * tf.pow(
        logt_partition, 1 - t)
    return [i + 1, normalized_activations_t]

  _, normalized_activations_t = tf.while_loop(
      iter_condition,
      iter_body, [0, normalized_activations_step_0],
      maximum_iterations=num_iters)
  logt_partition = tf.reduce_sum(
      exp_t(normalized_activations_t, t), -1, keep_dims=True)
  return -log_t(1.0 / logt_partition, t) + mu


def _internal_bi_tempered_logistic_loss(activations, labels, t1, t2):
  """Computes the Bi-Tempered logistic loss.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: batch_size
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness).

  Returns:
    A loss tensor for robust loss.
  """
  if t2 == 1.0:
    normalization_constants = tf.log(
        tf.reduce_sum(tf.exp(activations), -1, keep_dims=True))
    if t1 == 1.0:
      return normalization_constants + tf.reduce_sum(
          tf.multiply(labels, tf.log(labels + 1e-10) - activations), -1,
          keep_dims=True)
    else:
      shifted_activations = tf.exp(activations - normalization_constants)
      one_minus_t2 = 1.0
  else:
    one_minus_t1 = (1.0 - t1)
    one_minus_t2 = (1.0 - t2)
    normalization_constants = compute_normalization(
        activations, t2, num_iters=5)
    shifted_activations = tf.nn.relu(1.0 + one_minus_t2 *
                                     (activations - normalization_constants))

  if t1 == 1.0:
    return tf.reduce_sum(
        tf.multiply(
            tf.log(labels + 1e-10) -
            tf.log(tf.pow(shifted_activations, 1.0 / one_minus_t2)), labels),
        -1,
        keep_dims=True)
  else:
    beta = 1.0 + one_minus_t1
    logt_probs = (tf.pow(shifted_activations, one_minus_t1 / one_minus_t2) -
                  1.0) / one_minus_t1
    return tf.reduce_sum(
        tf.multiply(log_t(labels, t1) - logt_probs, labels) - 1.0 / beta *
        (tf.pow(labels, beta) -
         tf.pow(shifted_activations, beta / one_minus_t2)), -1)


def tempered_softmax(activations, t, num_iters=5):
  """Tempered softmax function.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature > 1.0.
    num_iters: Number of iterations to run the method.

  Returns:
    A probabilities tensor.
  """
  if t == 1.0:
    normalization_constants = tf.log(
        tf.reduce_sum(tf.exp(activations), -1, keep_dims=True))
  else:
    normalization_constants = compute_normalization(activations, t, num_iters)
  return exp_t(activations - normalization_constants, t)


def bi_tempered_binary_logistic_loss(activations,
                                     labels,
                                     t1,
                                     t2,
                                     label_smoothing=0.0,
                                     num_iters=5):
  """Bi-Tempered binary logistic loss.

  Args:
    activations: A tensor containing activations for class 1.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness).
    label_smoothing: Label smoothing
    num_iters: Number of iterations to run the method.

  Returns:
    A loss tensor.
  """
  with tf.name_scope('binary_bitempered_logistic'):
    out_shape = tf.shape(labels)
    labels_2d = tf.reshape(labels, [-1, 1])
    activations_2d = tf.reshape(activations, [-1, 1])
    internal_labels = tf.concat([1.0 - labels_2d, labels_2d], 1)
    internal_logits = tf.concat([tf.zeros_like(activations_2d), activations_2d],
                                1)
    losses = bi_tempered_logistic_loss(internal_logits, internal_labels, t1, t2,
                                       label_smoothing, num_iters)
    return tf.reshape(losses, out_shape)


def bi_tempered_logistic_loss(activations,
                              labels,
                              t1,
                              t2,
                              label_smoothing=0.0,
                              num_iters=5):
  """Bi-Tempered Logistic Loss with custom gradient.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.

  Returns:
    A loss tensor.
  """
  with tf.name_scope('bitempered_logistic'):
    if label_smoothing > 0.0:
      num_classes = tf.cast(tf.shape(labels)[-1], tf.float32)
      labels = (
          1 - num_classes /
          (num_classes - 1) * label_smoothing) * labels + label_smoothing / (
              num_classes - 1)

    @tf.custom_gradient
    def _custom_gradient_bi_tempered_logistic_loss(activations):
      """Bi-Tempered Logistic Loss with custom gradient.

      Args:
        activations: A multi-dimensional tensor with last dim `num_classes`.

      Returns:
        A loss tensor, grad.
      """
      with tf.name_scope('gradient_bitempered_logistic'):
        probabilities = tempered_softmax(activations, t2, num_iters)
        loss_values = tf.multiply(
            labels,
            log_t(labels + 1e-10, t1) -
            log_t(probabilities, t1)) - 1.0 / (2.0 - t1) * (
                tf.pow(labels, 2.0 - t1) - tf.pow(probabilities, 2.0 - t1))

        def grad(d_loss):
          delta_probs = probabilities - labels
          forget_factor = tf.pow(probabilities, t2 - t1)
          delta_forget_sum = tf.reduce_sum(
              tf.multiply(delta_probs, forget_factor), -1, keep_dims=True)
          escorts = tf.pow(probabilities, t2)
          escorts = escorts / tf.reduce_sum(escorts, -1, keep_dims=True)
          derivative = tf.multiply(delta_probs, forget_factor) - tf.multiply(
              escorts, delta_forget_sum)
          return tf.multiply(d_loss, derivative)

        return loss_values, grad

    loss_values = _custom_gradient_bi_tempered_logistic_loss(activations)
    return tf.reduce_sum(loss_values, -1)
