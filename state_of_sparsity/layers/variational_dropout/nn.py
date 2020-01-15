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

"""Defines standard networks layers that train using variational dropout."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from state_of_sparsity.layers.utils import layer_utils
from state_of_sparsity.layers.variational_dropout import common


def _verify_variational_params(variational_params):
  """Verifies that the format of the input `variational_params`.

  Checks that the input parameters is a 2-tuple of tensors of equal shape.

  Args:
    variational_params: The parameters to check.

  Raises:
    RuntimeError: If the input is not a 2-tuple of tensors with equal shape.

  Returns:
    The input `variational_parameters`.
  """
  if len(variational_params) != 2:
    raise RuntimeError("Incorrect number of variational parameters.")
  if variational_params[0].shape != variational_params[1].shape:
    raise RuntimeError("Variational parameters must be the same shape.")
  return variational_params


def matmul_train(
    x,
    variational_params,
    transpose_a=False,
    transpose_b=False,
    clip_alpha=None,
    eps=common.EPSILON):
  R"""Training computation for a variation matmul.

  In variational dropout we train a Bayesian neural network where we assume a
  fully-factorized Gaussian posterior and log uniform prior over the weights.

  During training, we need to sample weights from this distribution. Rather
  than sample weights for each sample in the input batch, we can calculate the
  parameters of the distribution over the pre-activations analytically (this
  step is called the local reparameterization trick). This function calculates
  the mean and standard deviation of the distribution over the pre-activations,
  and then draws a single sample for each element in the input batch and passes
  them as output.

  Args:
    x: 2D Tensor representing the input batch.
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    transpose_a: If True, a is transposed before multiplication.
    transpose_b: If True, b is transposed before multiplication.
    clip_alpha: Int or None. If integer, we clip the log \alpha values to
      [-clip_alpha, clip_alpha]. If None, don't clip the values.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the matmul operation.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """
  # We expect a 2D input tensor, as in standard in fully-connected layers
  x.get_shape().assert_has_rank(2)

  theta, log_sigma2 = _verify_variational_params(
      variational_params)

  if clip_alpha is not None:
    # Compute the log_alphas and then compute the
    # log_sigma2 again so that we can clip on the
    # log alpha magnitudes
    log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, clip_alpha)
    log_sigma2 = common.compute_log_sigma2(log_alpha, theta, eps)

  # Compute the mean and standard deviation of the distributions over the
  # activations
  mu_activation = tf.matmul(
      x,
      theta,
      transpose_a=transpose_a,
      transpose_b=transpose_b)
  std_activation = tf.sqrt(tf.matmul(
      tf.square(x),
      tf.exp(log_sigma2),
      transpose_a=transpose_a,
      transpose_b=transpose_b) + eps)

  output_shape = tf.shape(std_activation)
  return mu_activation + std_activation * tf.random_normal(output_shape)


def matmul_eval(
    x,
    variational_params,
    transpose_a=False,
    transpose_b=False,
    threshold=3.0,
    eps=common.EPSILON):
  R"""Evaluation computation for a variation matmul.

  In variational dropout we train a Bayesian neural network where we assume a
  fully-factorized Gaussian posterior and log uniform prior over the weights.

  The parameters of the posterior are learned during training, and at eval
  time we use the learned mean as the weight values.

  This method also supports the pruning of weights based on their log \alpha
  values. All weights with log \alpha >= `threshold` are set to zero.

  Args:
    x: 2D Tensor representing the input batch.
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    transpose_a: If True, a is transposed before multiplication.
    transpose_b: If True, b is transposed before multiplication.
    threshold: Weights with a log \alpha_{ij} value greater than this will be
      set to zero.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the variational matmul operation.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """
  # We expect a 2D input tensor, as is standard in fully-connected layers
  x.get_shape().assert_has_rank(2)

  theta, log_sigma2 = _verify_variational_params(
      variational_params)

  # Compute the weight mask by thresholding on
  # the log-space alpha values
  log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, value_limit=None)
  weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

  return tf.matmul(
      x,
      theta * weight_mask,
      transpose_a=transpose_a,
      transpose_b=transpose_b)


def broadcast_matmul_train(
    x,
    variational_params,
    clip_alpha=None,
    eps=common.EPSILON):
  R"""Training computation for VD matrix multiplication with N input matrices.

  Multiplies a 3D tensor `x` with a set of 2D parameters. Each 2D matrix
  `x[i, :, :]` in the input tensor is multiplied indendently with the
  parameters, resulting in a 3D output tensor with shape
  `x.shape[:2] + weight_parameters[0].shape[1]`.

  Args:
    x: 3D Tensor representing the input batch.
    variational_params: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    clip_alpha: Int or None. If integer, we clip the log \alpha values to
      [-clip_alpha, clip_alpha]. If None, don't clip the values.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the batched matmul operation.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """
  theta, log_sigma2 = _verify_variational_params(
      variational_params)
  theta.get_shape().assert_has_rank(2)
  log_sigma2.get_shape().assert_has_rank(2)

  # The input data must have be rank 2 or greater
  assert x.get_shape().ndims >= 2
  input_rank = x.get_shape().ndims

  if clip_alpha is not None:
    # Compute the log_alphas and then compute the
    # log_sigma2 again so that we can clip on the
    # log alpha magnitudes
    log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, clip_alpha)
    log_sigma2 = common.compute_log_sigma2(log_alpha, theta, eps)

  # Compute the mean and standard deviation of the distributions over the
  # activations
  mu_activation = tf.tensordot(x, theta, [[input_rank-1], [0]])

  var_activation = tf.tensordot(
      tf.square(x),
      tf.exp(log_sigma2),
      [[input_rank-1], [0]])
  std_activation = tf.sqrt(var_activation + eps)

  # Reshape the output back to the rank of the input
  input_shape = x.get_shape().as_list()
  weight_shape = theta.get_shape().as_list()
  output_shape = input_shape[:-1] + [weight_shape[1]]
  mu_activation.set_shape(output_shape)
  std_activation.set_shape(output_shape)

  # NOTE: We sample noise for each weight in theta, which will be shared by
  # each matrix product that was done. This is equivalent to sampling the same
  # set of weights for all matrix products done by this op in an iteration.
  # The element-wise multiply below broadcasts.
  num_pad_dims = len(output_shape) - 2
  padding = [tf.constant(1, dtype=tf.int32) for _ in range(num_pad_dims)]

  # NOTE: On GPU, the first dim may not be defined w/ the Transformer. Create
  # a tf.Tensor from the list shape and TF should match the first dim
  # appropriately
  batch_size = tf.shape(x)[0]
  data_dim = tf.shape(theta)[-1]
  noise_shape = tf.stack([batch_size] + padding + [data_dim], axis=0)

  output = mu_activation + std_activation * tf.random_normal(noise_shape)
  return output


def broadcast_matmul_eval(
    x,
    variational_params,
    threshold=3.0,
    eps=common.EPSILON):
  R"""Evaluation computation for VD matrix multiplication with N input matrices.

  Multiplies a 3D tensor `x` with a set of 2D parameters. Each 2D matrix
  `x[i, :, :]` in the input tensor is multiplied indendently with the
  parameters, resulting in a 3D output tensor with shape
  `x.shape[:2] + weight_parameters[0].shape[1]`.

  Args:
    x: 3D Tensor representing the input batch.
    variational_params: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    threshold: Weights with a log \alpha_{ij} value greater than this will be
      set to zero.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the batched matmul operation.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """
  theta, log_sigma2 = _verify_variational_params(
      variational_params)
  theta.get_shape().assert_has_rank(2)
  log_sigma2.get_shape().assert_has_rank(2)

  # The input data must have be rank 2 or greater
  assert x.get_shape().ndims >= 2
  input_rank = x.get_shape().ndims

  # Compute the weights mask by thresholding on the log-space alpha values
  log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, value_limit=None)
  weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

  output = tf.tensordot(x, theta * weight_mask, [[input_rank-1], [0]])

  # Reshape the output back to the rank of the input
  input_shape = x.get_shape().as_list()
  weight_shape = theta.get_shape().as_list()
  output_shape = input_shape[:-1] + [weight_shape[1]]
  output.set_shape(output_shape)
  return output


def conv2d_train(x,
                 variational_params,
                 strides,
                 padding,
                 data_format="NHWC",
                 clip_alpha=None,
                 eps=common.EPSILON):
  R"""Training computation for a variational conv2d.

  In variational dropout we train a Bayesian neural network where we assume a
  fully-factorized Gaussian posterior and log uniform prior over the weights.

  During training, we need to sample weights from this distribution. Rather
  than sample weights for each sample in the input batch, we can calculate the
  parameters of the distribution over the pre-activations analytically (this
  step is called the local reparameterization trick). This function calculates
  the mean and standard deviation of the distribution over the pre-activations,
  and then draws a single sample for each element in the input batch and passes
  them as output.

  Args:
    x: NHWC tf.Tensor representing the input batch of features.
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    strides: The stride of the sliding window for each dimension of `x`.
      Identical to standard strides argument for tf.conv2d.
    padding: String. One of "SAME", or "VALID". Identical to standard padding
      argument for tf.conv2d.
    data_format: 'NHWC' or 'NCHW' ordering of 4-D input Tensor.
    clip_alpha: Int or None. If integer, we clip the log \alpha values to
      [-clip_alpha, clip_alpha]. If None, don't clip the values.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the conv2d operation.

  Raises:
    RuntimeError: If the variational_params argument
    is not a 2-tuple.
  """
  theta, log_sigma2 = _verify_variational_params(variational_params)

  if clip_alpha:
    # Compute the log_alphas and then compute the
    # log_sigma2 again so that we can clip on the
    # log alpha magnitudes
    log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, clip_alpha)
    log_sigma2 = common.compute_log_sigma2(log_alpha, theta, eps)

  # Compute the mean and standard deviation of the distribution over the
  # convolution outputs
  mu_activation = tf.nn.conv2d(
      x, theta, strides, padding, data_format=data_format)
  std_activation = tf.sqrt(
      tf.nn.conv2d(
          tf.square(x),
          tf.exp(log_sigma2),
          strides,
          padding,
          data_format=data_format) + eps)

  output_shape = tf.shape(std_activation)
  return mu_activation + std_activation * tf.random_normal(output_shape)


def conv2d_eval(x,
                variational_params,
                strides,
                padding,
                data_format="NHWC",
                threshold=3.0,
                eps=common.EPSILON):
  R"""Evaluation computation for a variation conv2d.

  In variational dropout we train a Bayesian neural network where we assume a
  fully-factorized Gaussian posterior and log uniform prior over the weights.

  The parameters of the posterior are learned during training, and at eval
  time we use the learned mean as the weight values.

  This method also supports the pruning of weights based on their log \alpha
  values. All weights with log \alpha >= `threshold` are set to zero.

  Args:
    x: Tensor representing the input batch.
    variational_params: 2-tuple of Tensors, where the first tensor is the
      \theta values and the second contains the log of the \sigma^2 values.
    strides: The stride of the sliding window for each dimension of `x`.
      Identical to standard strides argument for tf.conv2d.
    padding: String. One of "SAME", or "VALID". Identical to standard
     padding argument for tf.conv2d.
    data_format: 'NHWC' or 'NCHW' ordering of 4-D input Tensor.
    threshold: Weights with a log \alpha_{ij} value greater than this will
      be set to zero.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the conv2d operation.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """
  theta, log_sigma2 = _verify_variational_params(
      variational_params)

  # Compute the weight mask by thresholding on
  # the log-space alpha values
  log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, value_limit=None)
  weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

  return tf.nn.conv2d(
      x, theta * weight_mask, strides, padding, data_format=data_format)


# NOTE: This implementation of variational dropout on an embedding samples
# new noise for each embedding vectors at all timesteps in the batch
# and across sequences in the batch. An alternative implementation would
# be to sample a noise vector for each token in the vocabulary, so that
# all instances of an embedding vector for a given token would be the
# same within a batch. Another alternative implementation would be to
# sample a noise vector for each token in the vocabulary for each element
# in the batch so that, within a sequence, all instances of an embedding
# vector for a given token would be the same, but across different elements
# in the batch they could be different.
#
# The first alternative implementation would add another embedding lookup
# to the implementation. We'd generate a noise tensor with shape
# [vocab_size, embedding_size], and for each token id in the batch we'd
# do an embedding lookup to get the appropriate noise vector. We'd then
# do two more embedding lookups, one to get the mean vector and one to
# get the log variance vector for the token. These 3 tensors with shape
# [batch_size, seq_length, embedding_size] would then be combined the
# same way they are in this implementation.
#
# This last implementation may not be practical, because we would have to
# sample `vocab_size * embedding_size * batch_size` random values per
# iteration. We'd also have unique noise embeddings for each element in
# the batch, meaning we'd have to do `batch_size` + 2 embedding lookups.
#
# This implementation is the most efficient in terms of embedding lookups
# and noise sampling.
def embedding_lookup_train(
    variational_params,
    ids,
    name=None,
    clip_alpha=None,
    eps=common.EPSILON):
  R"""Embedding trained with variational dropout.

  In a standard embedding lookup, `ids` are looked-up in a list of embedding
  tensors. In an embedding trained with variational dropout, we lookup the
  parameters of the fully-factorized Gaussian posterior over the embedding
  tensor for each index in `ids` and draw a sample from this distribution
  that is returned.

  The `ids` argument is analogous to those in the standard tf.embedding_lookup.

  Args:
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    ids: A Tensor with type int32 or int64 containing the ids to be looked up
      in params.
    name: String. Name of the operator.
    clip_alpha: Int or None. If integer, we clip the log \alpha values
      to [-clip_alpha, clip_alpha]. If None, don't clip the values.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    The output Tensor result of the embedding lookup.

  Raises:
    RuntimeError: If the input variational_params is not a 2-tuple of Tensors
      that have the same shape.
  """
  theta, log_sigma2 = _verify_variational_params(
      variational_params)

  # Before we do anything, lookup the mean and log variances of the embedding
  # vectors we are going to output and do all our operations in this lower
  # dimensional space
  embedding_theta = layer_utils.gather(theta, ids)
  embedding_log_sigma2 = layer_utils.gather(log_sigma2, ids)

  if clip_alpha:
    # Compute the log_alphas and then compute the
    # log_sigma2 again so that we can clip on the
    # log alpha magnitudes
    embedding_log_alpha = common.compute_log_alpha(
        embedding_log_sigma2, embedding_theta, eps, clip_alpha)
    embedding_log_sigma2 = common.compute_log_sigma2(
        embedding_log_alpha, embedding_theta, eps)

  # Calculate the standard deviation from the log variance
  embedding_std = tf.sqrt(tf.exp(embedding_log_sigma2) + eps)

  # Output samples from the distribution over the embedding vectors
  output_shape = tf.shape(embedding_std)
  embedding = embedding_theta + embedding_std * tf.random_normal(output_shape)
  return tf.identity(embedding, name=name)


def embedding_lookup_eval(
    variational_params,
    ids,
    name=None,
    threshold=3.0,
    eps=common.EPSILON):
  R"""Evaluation mode embedding trained with variational dropout.

  In a standard embedding lookup, `ids` are looked-up in a list of embedding
  tensors. In an embedding trained with variational dropout, we lookup the
  parameters of the fully-factorized Gaussian posterior over the embedding
  tensor for each index in `ids` and draw a sample from this distribution
  that is returned. At evaluation time, we use the mean of the posterior
  over each embedding tensor instead of sampling.

  The `ids` and `partition_strategy` arguments are analogous to those in the
  standard tf.embedding_lookup.

  Args:
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    ids: A Tensor with type int32 or int64 containing the ids to be looked up
      in params.
    name: String. Name of the operator.
    threshold: Weights with a log \alpha_{ij} value greater than this will be
      set to zero.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    The output Tensor result of the embedding lookup.

  Raises:
    RuntimeError: If the input variational_params is not a 2-tuple of Tensors
      that have the same shape.
  """
  theta, log_sigma2 = _verify_variational_params(
      variational_params)

  # Rather than mask the whole embedding every iteration, we can do a second
  # embedding lookup on the log \sigma2 values, compute the log \alpha values
  # for each output embedding vector, and then mask the much lower dimensional
  # output embedding vectors
  embedding_theta = layer_utils.gather(theta, ids)
  embedding_log_sigma2 = layer_utils.gather(log_sigma2, ids)

  # Compute the weight mask by thresholding on the log-space alpha values
  embedding_log_alpha = common.compute_log_alpha(
      embedding_log_sigma2, embedding_theta, eps, value_limit=None)
  embedding_mask = tf.cast(tf.less(embedding_log_alpha, threshold), tf.float32)

  # Return the masked embedding vectors
  return tf.identity(embedding_theta * embedding_mask, name=name)


def negative_dkl(variational_params=None,
                 clip_alpha=None,
                 eps=common.EPSILON,
                 log_alpha=None):
  R"""Compute the negative kl-divergence loss term.

  Computes the negative kl-divergence between the log-uniform prior over the
  weights and the variational posterior over the weights for each element
  in the set of variational parameters. Each contribution is summed and the
  sum is returned as a scalar Tensor.

  The true kl-divergence is intractable, so we compute the tight approximation
  from https://arxiv.org/abs/1701.05369.

  Args:
    variational_params: 2-tuple of Tensors, where the first tensor is the \theta
      values and the second contains the log of the \sigma^2 values.
    clip_alpha: Int or None. If integer, we clip the log \alpha values to
      [-clip_alpha, clip_alpha]. If None, don't clip the values.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.
    log_alpha: float32 tensor of log alpha values.
  Returns:
    Output scalar Tensor containing the sum of all negative kl-divergence
    contributions for each element in the input variational_params.

  Raises:
    RuntimeError: If the variational_params argument is not a 2-tuple.
  """

  if variational_params is not None:
    theta, log_sigma2 = _verify_variational_params(variational_params)

  if log_alpha is None:
    log_alpha = common.compute_log_alpha(log_sigma2, theta, eps, clip_alpha)

  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1

  # Compute each term of the KL and combine
  term_1 = k1 * tf.nn.sigmoid(k2 + k3*log_alpha)
  term_2 = -0.5 * tf.log1p(tf.exp(tf.negative(log_alpha)))
  eltwise_dkl = term_1 + term_2 + c
  return -tf.reduce_sum(eltwise_dkl)
