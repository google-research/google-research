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

"""Loss utility functions."""

import math

import tensorflow as tf


# Define measure types.
TYPE_MEASURE_GAN = 'GAN'  # Vanilla GAN.
TYPE_MEASURE_JSD = 'JSD'  # Jensen-Shannon divergence.
TYPE_MEASURE_KL = 'KL'    # KL-divergence.
TYPE_MEASURE_RKL = 'RKL'  # Reverse KL-divergence.
TYPE_MEASURE_H2 = 'H2'    # Squared Hellinger.
TYPE_MEASURE_W1 = 'W1'    # Wasserstein distance (1-Lipschitz).

# Define generator loss types.
TYPE_GENERATOR_LOSS_MM = 'MM'  # Minimax.
TYPE_GENERATOR_LOSS_NS = 'NS'  # Non-saturating.


def compute_positive_expectation(samples, measure, reduce_mean=False):
  """Computes the positive part of a divergence or difference.

  Args:
    samples: A tensor for the positive samples.
    measure: A string for measure to compute. See TYPE_MEASURE_* for supported
      measure types.
    reduce_mean: A boolean indicating whether to reduce results

  Returns:
    A tensor (has the same shape as samples) or scalar (if reduced) for the
      positive expectation of the inputs.
  """
  if measure == TYPE_MEASURE_GAN:
    expectation = -tf.math.softplus(-samples)
  elif measure == TYPE_MEASURE_JSD:
    expectation = math.log(2.) - tf.math.softplus(-samples)
  elif measure == TYPE_MEASURE_KL:
    expectation = samples
  elif measure == TYPE_MEASURE_RKL:
    expectation = -tf.math.exp(-samples)
  elif measure == TYPE_MEASURE_H2:
    expectation = 1. - tf.math.exp(-samples)
  elif measure == TYPE_MEASURE_W1:
    expectation = samples
  else:
    raise ValueError('Measure {} not supported'.format(measure))

  if reduce_mean:
    return tf.math.reduce_mean(expectation)
  else:
    return expectation


def compute_negative_expectation(samples, measure, reduce_mean=False):
  """Computes the negative part of a divergence or difference.

  Args:
    samples: A tensor for the negative samples.
    measure: A string for measure to compute. See TYPE_MEASURE_* for supported
      measure types.
    reduce_mean: A boolean indicating whether to reduce results

  Returns:
    A tensor (has the same shape as samples) or scalar (if reduced) for the
      negative expectation of the inputs.
  """
  if measure == TYPE_MEASURE_GAN:
    expectation = tf.math.softplus(-samples) + samples
  elif measure == TYPE_MEASURE_JSD:
    expectation = tf.math.softplus(-samples) + samples - math.log(2.)
  elif measure == TYPE_MEASURE_KL:
    expectation = tf.math.exp(samples - 1.)
  elif measure == TYPE_MEASURE_RKL:
    expectation = samples - 1.
  elif measure == TYPE_MEASURE_H2:
    expectation = tf.math.exp(samples) - 1.
  elif measure == TYPE_MEASURE_W1:
    expectation = samples
  else:
    raise ValueError('Measure {} not supported'.format(measure))

  if reduce_mean:
    return tf.math.reduce_mean(expectation)
  else:
    return expectation


def compute_fenchel_dual_loss(local_features,
                              global_features,
                              measure,
                              positive_indicator_matrix=None):
  """Computes the f-divergence loss.

  It is the distance between positive and negative joint distributions.
  Divergences (measures) supported are Jensen-Shannon (JSD), GAN (equivalent
  to JSD), Squared Hellinger (H2), KL and reverse KL (RKL).

  Reference:
    Hjelm et al. Learning deep representations by mutual information estimation
    and maximization. https://arxiv.org/pdf/1808.06670.pdf.

  Args:
    local_features: A tensor for local features. Shape = [batch_size,
      num_locals, feature_dim].
    global_features: A tensor for local features. Shape = [batch_size,
      num_globals, feature_dim].
    measure: A string for f-divergence measure.
    positive_indicator_matrix: A tensor for indicating positive sample pairs.
      1.0 means positive and otherwise 0.0. It should be symmetric with 1.0 at
      diagonal. Shape = [batch_size, batch_size].

  Returns:
    A scalar for the computed loss.
  """
  batch_size, num_locals, feature_dim = local_features.shape
  num_globals = global_features.shape[-2]

  # Make the input tensors the right shape.
  local_features = tf.reshape(local_features, (-1, feature_dim))
  global_features = tf.reshape(global_features, (-1, feature_dim))

  # Compute outer product, we want a [batch_size, num_locals, batch_size,
  # num_globals] tensor.
  product = tf.linalg.matmul(local_features, global_features, transpose_b=True)
  product = tf.reshape(
      product, (batch_size, num_locals, batch_size, num_globals))

  # Compute the indicator_matrix for positive and negative samples.
  if positive_indicator_matrix is None:
    positive_indicator_matrix = tf.eye(batch_size, dtype=tf.dtypes.float32)
  negative_indicator_matrix = 1. - positive_indicator_matrix

  # Compute the positive and negative scores, and average the spatial locations.
  positive_expectation = compute_positive_expectation(
      product, measure, reduce_mean=False)
  negative_expectation = compute_negative_expectation(
      product, measure, reduce_mean=False)

  positive_expectation = tf.math.reduce_mean(positive_expectation, axis=[1, 3])
  negative_expectation = tf.math.reduce_mean(negative_expectation, axis=[1, 3])

  # Mask positive and negative terms.
  positive_expectation = tf.math.reduce_sum(
      positive_expectation * positive_indicator_matrix) / tf.math.maximum(
          tf.math.reduce_sum(positive_indicator_matrix), 1e-12)
  negative_expectation = tf.math.reduce_sum(
      negative_expectation * negative_indicator_matrix) / tf.math.maximum(
          tf.math.reduce_sum(negative_indicator_matrix), 1e-12)

  return negative_expectation - positive_expectation


def compute_info_nce_loss(local_features,
                          global_features,
                          positive_indicator_matrix=None,
                          temperature=1.0):
  """Computes the InfoNCE (CPC) loss.

  Reference:
    Oord et al. Representation Learning with Contrastive Predictive Coding.
    https://arxiv.org/pdf/1807.03748.pdf.

  Args:
    local_features: A tensor for local features. Shape = [batch_size,
      num_locals, feature_dim].
    global_features: A tensor for local features. Shape = [batch_size,
      num_globals, feature_dim].
    positive_indicator_matrix: A tensor for indicating positive sample pairs.
      1.0 means positive and otherwise 0.0. It should be symmetric with 1.0 at
      diagonal. Shape = [batch_size, batch_size].
    temperature: A float for temperature hyperparameter.

  Returns:
    A scalar for the computed loss.
  """
  batch_size, num_locals, feature_dim = local_features.shape
  num_globals = global_features.shape[-2]

  # Make the input tensors the right shape.
  # -> Shape = [batch_size * num_locals, feature_dim].
  local_features_reshaped = tf.reshape(local_features, (-1, feature_dim))
  # -> Shape = [batch_size * num_globals, feature_dim].
  global_features_reshaped = tf.reshape(global_features, (-1, feature_dim))

  # Inner product for positive samples.
  # -> Shape = [batch_size, num_locals, num_globals]
  positive_expectation = tf.linalg.matmul(
      local_features, tf.transpose(global_features, (0, 2, 1)))
  if temperature != 1.0:
    positive_expectation /= temperature
  # -> Shape = [batch_size, num_locals, 1, num_globals]
  positive_expectation = tf.expand_dims(positive_expectation, axis=2)

  # Outer product for negative. We want a [batch_size, batch_size, num_locals,
  # num_globals] tensor.
  # -> Shape = [batch_size * num_globals, batch_size * num_locals].
  product = tf.linalg.matmul(
      global_features_reshaped, local_features_reshaped, transpose_b=True)
  if temperature != 1.0:
    product /= temperature
  product = tf.reshape(product,
                       (batch_size, num_globals, batch_size, num_locals))
  # -> Shape = [batch_size, batch_size, num_locals, num_globals].
  product = tf.transpose(product, (0, 2, 3, 1))

  # Mask positive part of the negative tensor.
  if positive_indicator_matrix is None:
    positive_indicator_matrix = tf.eye(batch_size, dtype=tf.dtypes.float32)
  # -> Shape = [batch_size, batch_size, 1, 1].
  positive_indicator_matrix = positive_indicator_matrix[:, :, tf.newaxis,
                                                        tf.newaxis]
  # -> Shape = [batch_size, batch_size, 1, 1].
  negative_indicator_matrix = 1. - positive_indicator_matrix

  # Masking is done by shifting the diagonal before exp.
  negative_expectation = (
      negative_indicator_matrix * product - 1e12 * positive_indicator_matrix)
  negative_expectation = tf.reshape(
      negative_expectation, (batch_size, batch_size * num_locals, num_globals))
  # -> Shape = [batch_size, 1, batch_size * num_locals, num_globals]
  negative_expectation = tf.expand_dims(negative_expectation, axis=1)
  # -> Shape = [batch_size, num_locals, batch_size * num_locals, num_globals]
  negative_expectation = tf.tile(negative_expectation,
                                 tf.constant((1, num_locals, 1, 1)))

  # -> Shape = [batch_size, num_locals, 1 + batch_size * num_locals,
  #             num_globals].
  logits = tf.concat([positive_expectation, negative_expectation], axis=2)
  # -> Shape = [batch_size, num_locals, num_globals].
  loss = tf.nn.log_softmax(logits, axis=2)

  # The positive score is the first element of the log softmax.
  # -> Shape = [batch_size, num_locals].
  loss = -loss[:, :, 0]
  # -> Shape = [].
  return tf.reduce_mean(loss)


def compute_log_likelihood(x_mean, x_logvar, y):
  """Computes the log-likelihood of y|x.

  Args:
    x_mean: A tensor for mean of y estimated from x. Shape = [batch_size,
      feature_dim].
    x_logvar: A tensor for log variance of y estimated from x. Shape = [
      batch_size, feature_dim].
    y: A tensor for value of y. Shape = [batch_size, feature_dim].

  Returns:
    A scalar for the computed log-likelihood of y|x.
  """
  likelihood = -(x_mean - y) ** 2 / tf.math.exp(x_logvar) / 2. - x_logvar / 2.
  return tf.math.reduce_mean(likelihood)


def compute_contrastive_log_ratio(x_mean,
                                  x_logvar,
                                  y,
                                  positive_indicator_matrix=None):
  """Computes the contrastive log-ratio of y|x.

  The result can be used as a variational upper-bound estimation of the mutual
  information I(x, y).

  Reference:
    Cheng et al. CLUB: A Contrastive Log-ratio Upper Bound of Mutual
    Information. https://arxiv.org/pdf/2006.12013.pdf.

  Args:
    x_mean: A tensor for mean of y estimated from x. Shape = [batch_size,
      feature_dim].
    x_logvar: A tensor for log variance of y estimated from x. Shape = [
      batch_size, feature_dim].
    y: A tensor for value of y. Shape = [batch_size, feature_dim].
    positive_indicator_matrix: A tensor for indicating positive sample pairs.
      1.0 means positive and otherwise 0.0. It should be symmetric with 1.0 at
      diagonal. Shape = [batch_size, batch_size].

  Returns:
    A scalar for the contrastive log-ratio of y|x.
  """
  batch_size = tf.shape(x_logvar)[0]

  # Compute the indicator_matrix for positive and negative samples.
  if positive_indicator_matrix is None:
    positive_indicator_matrix = tf.eye(batch_size, dtype=tf.dtypes.float32)
  negative_indicator_matrix = 1. - positive_indicator_matrix

  # Compute the log-likelihood of y|x samples.
  y = tf.expand_dims(y, axis=0)
  x_mean = tf.expand_dims(x_mean, axis=1)
  x_logvar = tf.expand_dims(x_logvar, axis=1)
  log_likelihood = -(x_mean - y)**2 / tf.math.exp(x_logvar) / 2. - x_logvar / 2.
  log_likelihood = tf.math.reduce_mean(log_likelihood, axis=-1)

  # Compute the positive and negative scores.
  positive_expectation = tf.math.reduce_sum(
      log_likelihood * positive_indicator_matrix) / tf.math.maximum(
          tf.math.reduce_sum(positive_indicator_matrix), 1e-12)
  negative_expectation = tf.math.reduce_sum(
      log_likelihood * negative_indicator_matrix) / tf.math.maximum(
          tf.math.reduce_sum(negative_indicator_matrix), 1e-12)

  # Clip the loss to be non-negative since mutual information should always be
  # a non-negative scalar.
  return tf.math.maximum(0., positive_expectation - negative_expectation)


def compute_gradient_penalty(discriminator, inputs, penalty_weight=1.0):
  """Computes the gradient penalty.

  Reference:
    Mescheder et al. Which Training Methods for GANs do actually Converge?
    https://arxiv.org/pdf/1801.04406.pdf.

  Args:
    discriminator: Network to apply penalty through.
    inputs: An input tensor. Shape = [batch_size, ...].
    penalty_weight: A float for the weight of penalty.

  Returns:
    penalty: A scalar for the gradient penalty loss.
    outputs: A tensor for the network outputs.
  """
  batch_size = tf.shape(inputs)[0]

  with tf.GradientTape() as tape:
    tape.watch(inputs)
    outputs = discriminator(inputs, training=True)

  gradients = tape.gradient(outputs, inputs)
  gradients = tf.reshape(gradients, (batch_size, -1))
  penalty = tf.reduce_sum(tf.square(gradients), axis=-1)
  penalty = tf.reduce_mean(penalty) * penalty_weight
  return penalty, outputs


def compute_discriminator_loss(discriminator, real_inputs, fake_inputs):
  """Computes the discriminator loss.

  Args:
    discriminator: The discriminator network.
    real_inputs: A tensor for the real inputs. Shape = [batch_size, ...].
    fake_inputs: A tensor for the fake inputs. Shape = [batch_size, ...].

  Returns:
    loss: A scalar for the discriminator loss.
    real_outputs: A tensor for the real outputs.
    fake_outputs: A tensor for the fake outputs.
  """
  real_gradient_penalty, real_outputs = compute_gradient_penalty(
      discriminator, real_inputs)
  fake_gradient_penalty, fake_outputs = compute_gradient_penalty(
      discriminator, fake_inputs)
  gradient_penalty_loss = 0.5 * (real_gradient_penalty + fake_gradient_penalty)

  positive_expectation = compute_positive_expectation(
      real_outputs, TYPE_MEASURE_GAN, reduce_mean=True)
  negative_expectation = compute_negative_expectation(
      fake_outputs, TYPE_MEASURE_GAN, reduce_mean=True)
  expectation = 0.5 * (positive_expectation - negative_expectation)
  loss = -expectation + gradient_penalty_loss

  return loss, real_outputs, fake_outputs


def compute_generator_loss(fake_samples, loss_type):
  """Computes the generator loss.

  Args:
    fake_samples: A tensor for the fake samples. Shape = [batch_size, ...].
    loss_type: A string for the type of loss. See TYPE_GENERATOR_LOSS_* for
      supported loss types.

  Returns:
    A scalar for the generator loss.
  """
  if loss_type == TYPE_GENERATOR_LOSS_MM:
    return compute_negative_expectation(
        fake_samples, TYPE_MEASURE_GAN, reduce_mean=True)
  elif loss_type == TYPE_GENERATOR_LOSS_NS:
    return -compute_positive_expectation(
        fake_samples, TYPE_MEASURE_GAN, reduce_mean=True)
  else:
    raise ValueError('Generator loss {} not supported'.format(loss_type))
