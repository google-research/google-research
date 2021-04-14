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

"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
"""

import functools
import tensorflow as tf


def for_loop(num_iters, body, initial_args):
  """Runs a simple for-loop with given body and initial_args.

  Args:
    num_iters: Maximum number of iterations.
    body: Body of the for-loop.
    initial_args: Args to the body for the first iteration.

  Returns:
    Output of the final iteration.
  """
  for i in range(num_iters):
    if i == 0:
      outputs = body(*initial_args)
    else:
      outputs = body(*outputs)
  return outputs


def log_t(u, t):
  """Compute log_t for `u`."""

  def _internal_log_t(u, t):
    return (u**(1.0 - t) - 1.0) / (1.0 - t)

  return tf.cond(
      tf.equal(t, 1.0), lambda: tf.math.log(u),
      functools.partial(_internal_log_t, u, t))


def exp_t(u, t):
  """Compute exp_t for `u`."""

  def _internal_exp_t(u, t):
    return tf.nn.relu(1.0 + (1.0 - t) * u)**(1.0 / (1.0 - t))

  return tf.cond(
      tf.equal(t, 1.0), lambda: tf.math.exp(u),
      functools.partial(_internal_exp_t, u, t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
  """Returns the normalization value for each example (t > 1.0).

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
  Return: A tensor of same rank as activation with the last dimension being 1.
  """

  mu = tf.reduce_max(activations, -1, True)
  normalized_activations_step_0 = activations - mu
  shape_normalized_activations = tf.shape(normalized_activations_step_0)

  def iter_body(i, normalized_activations):
    logt_partition = tf.reduce_sum(
        exp_t(normalized_activations, t), -1, True)
    normalized_activations_t = tf.reshape(
        normalized_activations_step_0 * tf.pow(logt_partition, 1.0 - t),
        shape_normalized_activations)
    return [i + 1, normalized_activations_t]

  _, normalized_activations_t = for_loop(num_iters, iter_body,
                                         [0, normalized_activations_step_0])
  logt_partition = tf.reduce_sum(
      exp_t(normalized_activations_t, t), -1, True)
  return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization_binary_search(activations, t, num_iters=10):
  """Returns the normalization value for each example (t < 1.0).

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support).
    num_iters: Number of iterations to run the method.
  Return: A tensor of same rank as activation with the last dimension being 1.
  """
  mu = tf.reduce_max(activations, -1, True)
  normalized_activations = activations - mu
  shape_activations = tf.shape(activations)
  effective_dim = tf.cast(
      tf.reduce_sum(
          tf.cast(
              tf.greater(normalized_activations, -1.0 / (1.0 - t)), tf.int32),
          -1,
          True), tf.float32)
  shape_partition = tf.concat([shape_activations[:-1], [1]], 0)
  lower = tf.zeros(shape_partition)
  upper = -log_t(1.0 / effective_dim, t) * tf.ones(shape_partition)

  def iter_body(i, lower, upper):
    logt_partition = (upper + lower)/2.0
    sum_probs = tf.reduce_sum(exp_t(
        normalized_activations - logt_partition, t), -1, True)
    update = tf.cast(tf.less(sum_probs, 1.0), tf.float32)
    lower = tf.reshape(lower * update + (1.0 - update) * logt_partition,
                       shape_partition)
    upper = tf.reshape(upper * (1.0 - update) + update * logt_partition,
                       shape_partition)
    return [i + 1, lower, upper]

  _, lower, upper = for_loop(num_iters, iter_body, [0, lower, upper])
  logt_partition = (upper + lower)/2.0
  return logt_partition + mu


def compute_normalization(activations, t, num_iters=5):
  """Returns the normalization value for each example.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
  Return: A tensor of same rank as activation with the last dimension being 1.
  """
  return tf.cond(
      tf.less(t, 1.0),
      functools.partial(compute_normalization_binary_search, activations, t,
                        num_iters),
      functools.partial(compute_normalization_fixed_point, activations, t,
                        num_iters))


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
    normalization_constants = tf.math.log(
        tf.reduce_sum(tf.math.exp(activations), -1, True))
    if t1 == 1.0:
      return normalization_constants + tf.reduce_sum(
          tf.multiply(labels, tf.math.log(labels + 1e-10) - activations), -1)
    else:
      shifted_activations = tf.math.exp(activations - normalization_constants)
      one_minus_t1 = (1.0 - t1)
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
            tf.math.log(labels + 1e-10) -
            tf.math.log(tf.pow(shifted_activations, 1.0 / one_minus_t2)),
            labels), -1)
  else:
    beta = 1.0 + one_minus_t1
    logt_probs = (tf.pow(shifted_activations, one_minus_t1 / one_minus_t2) -
                  1.0) / one_minus_t1
    return tf.reduce_sum(
        tf.multiply(log_t(labels, t1) - logt_probs, labels) - 1.0 / beta *
        (tf.pow(labels, beta) -
         tf.pow(shifted_activations, beta / one_minus_t2)), -1)


def tempered_sigmoid(activations, t, num_iters=5):
  """Tempered sigmoid function.

  Args:
    activations: Activations for the positive class for binary classification.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.

  Returns:
    A probabilities tensor.
  """
  t = tf.convert_to_tensor(t)
  input_shape = tf.shape(activations)
  activations_2d = tf.reshape(activations, [-1, 1])
  internal_activations = tf.concat(
      [tf.zeros_like(activations_2d), activations_2d], 1)
  normalization_constants = tf.cond(
      # pylint: disable=g-long-lambda
      tf.equal(t, 1.0),
      lambda: tf.math.log(
          tf.reduce_sum(tf.math.exp(internal_activations), -1, True)),
      functools.partial(compute_normalization, internal_activations, t,
                        num_iters))
  internal_probabilities = exp_t(internal_activations - normalization_constants,
                                 t)
  one_class_probabilities = tf.split(internal_probabilities, 2, axis=1)[1]
  return tf.reshape(one_class_probabilities, input_shape)


def tempered_softmax(activations, t, num_iters=5):
  """Tempered softmax function.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.

  Returns:
    A probabilities tensor.
  """
  t = tf.convert_to_tensor(t)
  normalization_constants = tf.cond(
      tf.equal(t, 1.0),
      lambda: tf.math.log(tf.reduce_sum(tf.math.exp(activations), -1, True)),
      functools.partial(compute_normalization, activations, t, num_iters))
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
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing
    num_iters: Number of iterations to run the method.

  Returns:
    A loss tensor.
  """
  with tf.name_scope('binary_bitempered_logistic'):
    t1 = tf.convert_to_tensor(t1)
    t2 = tf.convert_to_tensor(t2)
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
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.

  Returns:
    A loss tensor.
  """
  with tf.name_scope('bitempered_logistic'):
    t1 = tf.convert_to_tensor(t1)
    t2 = tf.convert_to_tensor(t2)
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
          """Explicit gradient calculation.

          Args:
            d_loss: Infinitesimal change in the loss value.
          Returns:
            Loss gradient.
          """
          delta_probs = probabilities - labels
          forget_factor = tf.pow(probabilities, t2 - t1)
          delta_probs_times_forget_factor = tf.multiply(delta_probs,
                                                        forget_factor)
          delta_forget_sum = tf.reduce_sum(
              delta_probs_times_forget_factor, -1, True)
          escorts = tf.pow(probabilities, t2)
          escorts = escorts / tf.reduce_sum(escorts, -1, True)
          derivative = delta_probs_times_forget_factor - tf.multiply(
              escorts, delta_forget_sum)
          return tf.multiply(d_loss, derivative)

        return loss_values, grad

    def reduced_loss(activations):
      return tf.reduce_sum(
          _custom_gradient_bi_tempered_logistic_loss(activations), -1)

    loss_values = tf.cond(
        tf.math.logical_and(tf.equal(t1, 1.0), tf.equal(t2, 1.0)),
        functools.partial(
            tf.nn.softmax_cross_entropy_with_logits,
            labels=labels,
            logits=activations), functools.partial(reduced_loss, activations))
    return loss_values


def sparse_bi_tempered_logistic_loss(activations, labels, t1, t2, num_iters=5):
  """Sparse Bi-Tempered Logistic Loss with custom gradient.

  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with dtype of int32.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    num_iters: Number of iterations to run the method.

  Returns:
    A loss tensor.
  """
  with tf.name_scope('sparse_bitempered_logistic'):
    t1 = tf.convert_to_tensor(t1)
    t2 = tf.convert_to_tensor(t2)
    num_classes = tf.shape(activations)[-1]

    @tf.custom_gradient
    def _custom_gradient_sparse_bi_tempered_logistic_loss(activations):
      """Sparse Bi-Tempered Logistic Loss with custom gradient.

      Args:
        activations: A multi-dimensional tensor with last dim `num_classes`.

      Returns:
        A loss tensor, grad.
      """
      with tf.name_scope('gradient_sparse_bitempered_logistic'):
        probabilities = tempered_softmax(activations, t2, num_iters)
        # TODO(eamid): Replace one hot with gather.
        loss_values = -log_t(
            tf.reshape(
                tf.gather_nd(probabilities,
                             tf.where(tf.one_hot(labels, num_classes))),
                tf.shape(activations)[:-1]), t1) - 1.0 / (2.0 - t1) * (
                    1.0 - tf.reduce_sum(tf.pow(probabilities, 2.0 - t1), -1))

        def grad(d_loss):
          """Explicit gradient calculation.

          Args:
            d_loss: Infinitesimal change in the loss value.
          Returns:
            Loss gradient.
          """
          delta_probs = probabilities - tf.one_hot(labels, num_classes)
          forget_factor = tf.pow(probabilities, t2 - t1)
          delta_probs_times_forget_factor = tf.multiply(delta_probs,
                                                        forget_factor)
          delta_forget_sum = tf.reduce_sum(
              delta_probs_times_forget_factor, -1, True)
          escorts = tf.pow(probabilities, t2)
          escorts = escorts / tf.reduce_sum(escorts, -1, True)
          derivative = delta_probs_times_forget_factor - tf.multiply(
              escorts, delta_forget_sum)
          return tf.multiply(d_loss, derivative)

        return loss_values, grad

    loss_values = tf.cond(
        tf.math.logical_and(tf.equal(t1, 1.0), tf.equal(t2, 1.0)),
        functools.partial(tf.nn.sparse_softmax_cross_entropy_with_logits,
                          labels=labels, logits=activations),
        functools.partial(_custom_gradient_sparse_bi_tempered_logistic_loss,
                          activations))
    return loss_values
