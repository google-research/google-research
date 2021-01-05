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

"""Defines standard network layers that train using l0 regularization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from state_of_sparsity.layers.l0_regularization import common
from state_of_sparsity.layers.utils import layer_utils


def _verify_weight_parameters(weight_parameters):
  """Verifies that the format of the input `weight_parameters`.

  Checks that the input parameters is a 2-tuple of tensors of equal shape.

  Args:
    weight_parameters: The parameters to check.

  Raises:
    RuntimeError: If the input is not a 2-tuple of tensors with equal shape.

  Returns:
    The input `weight_parameters`.
  """
  if len(weight_parameters) != 2:
    raise RuntimeError("Incorrect number of weight parameters. Expected "
                       "2 tensors, got {}".format(len(weight_parameters)))
  if weight_parameters[0].shape != weight_parameters[1].shape:
    raise RuntimeError("Expected theta and log alpha parameter tensor "
                       "to be same shape. Got shapes {} and {}"
                       .format(weight_parameters[0].get_shape().as_list(),
                               weight_parameters[1].get_shape().as_list()))
  return weight_parameters


def matmul_train(
    x,
    weight_parameters,
    transpose_a=False,
    transpose_b=False,
    beta=common.BETA,
    gamma=common.GAMMA,
    zeta=common.ZETA,
    eps=common.EPSILON):
  """Training computation for a l0-regularized matmul.

  Args:
    x: 2D Tensor representing the input batch.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    transpose_a: If True, a is transposed before multiplication.
    transpose_b: If True, b is transposed before multiplication.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the matmul operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  x.get_shape().assert_has_rank(2)
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Sample the z values from the hard-concrete distribution
  weight_noise = common.hard_concrete_sample(
      log_alpha,
      beta,
      gamma,
      zeta,
      eps)
  weights = theta * weight_noise
  return tf.matmul(x, weights, transpose_a=transpose_a, transpose_b=transpose_b)


def matmul_eval(
    x,
    weight_parameters,
    transpose_a=False,
    transpose_b=False,
    gamma=common.GAMMA,
    zeta=common.ZETA):
  """Evaluation computation for a l0-regularized matmul.

  Args:
    x: 2D Tensor representing the input batch.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    transpose_a: If True, a is transposed before multiplication.
    transpose_b: If True, b is transposed before multiplication.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    Output Tensor of the matmul operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  x.get_shape().assert_has_rank(2)
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Use the mean of the learned hard-concrete distribution as the
  # deterministic weight noise at evaluation time
  weight_noise = common.hard_concrete_mean(log_alpha, gamma, zeta)
  weights = theta * weight_noise
  return tf.matmul(x, weights, transpose_a=transpose_a, transpose_b=transpose_b)


def broadcast_matmul_train(
    x,
    weight_parameters,
    beta=common.BETA,
    gamma=common.GAMMA,
    zeta=common.ZETA,
    eps=common.EPSILON):
  """Training computation for l0 matrix multiplication with N input matrices.

  Multiplies a 3D tensor `x` with a set of 2D parameters. Each 2D matrix
  `x[i, :, :]` in the input tensor is multiplied independently with the
  parameters, resulting in a 3D output tensor with shape
  `x.shape[:2] + weight_parameters[0].shape[1]`.

  Args:
    x: 3D Tensor representing the input batch.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the batched matmul operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)
  theta.get_shape().assert_has_rank(2)

  # The input data must have be rank 2 or greater
  assert x.get_shape().ndims >= 2
  input_rank = x.get_shape().ndims

  # Sample the z values from the hard-concrete distribution
  weight_noise = common.hard_concrete_sample(
      log_alpha,
      beta,
      gamma,
      zeta,
      eps)
  weights = theta * weight_noise

  # Compute the batch of matmuls
  return tf.tensordot(x, weights, [[input_rank-1], [0]])


def broadcast_matmul_eval(
    x,
    weight_parameters,
    gamma=common.GAMMA,
    zeta=common.ZETA):
  """Evaluation computation for l0 matrix multiplication with N input matrices.

  Multiplies a 3D tensor `x` with a set of 2D parameters. Each 2D matrix
  `x[i, :, :]` in the input tensor is multiplied independently with the
  parameters, resulting in a 3D output tensor with shape
  `x.shape[:2] + weight_parameters[0].shape[1]`.

  Args:
    x: 3D Tensor representing the input batch.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    Output Tensor of the batched matmul operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)
  theta.get_shape().assert_has_rank(2)

  # The input data must have be rank 2 or greater
  assert x.get_shape().ndims >= 2
  input_rank = x.get_shape().ndims

  # Use the mean of the learned hard-concrete distribution as the
  # deterministic weight noise at evaluation time
  weight_noise = common.hard_concrete_mean(log_alpha, gamma, zeta)
  weights = theta * weight_noise

  # Compute the batch of matmuls
  return tf.tensordot(x, weights, [[input_rank-1], [0]])


def conv2d_train(
    x,
    weight_parameters,
    strides,
    padding,
    data_format="NHWC",
    beta=common.BETA,
    gamma=common.GAMMA,
    zeta=common.ZETA,
    eps=common.EPSILON):
  """Training computation for a l0-regularized conv2d.

  Args:
    x: NHWC tf.Tensor representing the input batch of features.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    strides: The stride of the sliding window for each dimension of 'x'.
      Identical to standard strides argument for tf.conv2d.
    padding: String. One of "SAME", or "VALID". Identical to standard
      padding argument for tf.conv2d.
    data_format: 'NHWC' or 'NCHW' ordering of 4-D input Tensor.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the conv2d operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Sample the z values from the hard-concreate distribution
  weight_noise = common.hard_concrete_sample(
      log_alpha,
      beta,
      gamma,
      zeta,
      eps)
  weights = theta * weight_noise
  return tf.nn.conv2d(x, weights, strides, padding, data_format=data_format)


def conv2d_eval(
    x,
    weight_parameters,
    strides,
    padding,
    data_format="NHWC",
    gamma=common.GAMMA,
    zeta=common.ZETA):
  """Evaluation computation for a l0-regularized conv2d.

  Args:
    x: NHWC tf.Tensor representing the input batch of features.
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    strides: The stride of the sliding window for each dimension of 'x'.
      Identical to standard strides argument for tf.conv2d.
    padding: String. One of "SAME", or "VALID". Identical to standard
      padding argument for tf.conv2d.
    data_format: 'NHWC' or 'NCHW' ordering of 4-D input Tensor.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    Output Tensor of the conv2d operation.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Use the mean of the learned hard-concrete distribution as the
  # deterministic weight noise at evaluation time
  weight_noise = common.hard_concrete_mean(log_alpha, gamma, zeta)
  weights = theta * weight_noise
  return tf.nn.conv2d(x, weights, strides, padding, data_format=data_format)


def embedding_lookup_train(
    weight_parameters,
    ids,
    name=None,
    beta=common.BETA,
    gamma=common.GAMMA,
    zeta=common.ZETA,
    eps=common.EPSILON):
  """Training computation for a l0-regularized embedding lookup.

  Args:
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    ids: A Tensor with type int32 or int64 containing the ids to be looked up
      in params.
    name: String. Name of the operator.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    Output Tensor of the embedding lookup.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Before we do anything, lookup the theta values and log_alpha
  # values so that we can do our sampling and weight scaling in
  # the lower dimensional output batch
  embedding_theta = layer_utils.gather(theta, ids)
  embedding_log_alpha = layer_utils.gather(log_alpha, ids)

  # Sample the z values for the output batch from the hard-concrete
  embedding_noise = common.hard_concrete_sample(
      embedding_log_alpha,
      beta,
      gamma,
      zeta,
      eps)
  return tf.identity(embedding_theta * embedding_noise, name=name)


def embedding_lookup_eval(
    weight_parameters,
    ids,
    name=None,
    gamma=common.GAMMA,
    zeta=common.ZETA):
  """Evaluation computation for a l0-regularized embedding lookup.

  Args:
    weight_parameters: 2-tuple of Tensors, where the first tensor is the
      unscaled weight values and the second is the log of the alpha values
      for the hard concrete distribution.
    ids: A Tensor with type int32 or int64 containing the ids to be looked up
      in params.
    name: String. Name of the operator.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    Output Tensor of the embedding lookup.

  Raises:
    RuntimeError: If the weight_parameters argument is not a 2-tuple.
  """
  theta, log_alpha = _verify_weight_parameters(weight_parameters)

  # Before we do anything, lookup the theta values and log_alpha
  # values so that we can do our sampling and weight scaling in
  # the lower dimensional output batch
  embedding_theta = layer_utils.gather(theta, ids)
  embedding_log_alpha = layer_utils.gather(log_alpha, ids)

  # Calculate the mean of the learned hard-concrete distribution
  # and scale the output embedding vectors
  embedding_noise = common.hard_concrete_mean(
      embedding_log_alpha,
      gamma,
      zeta)
  return tf.identity(embedding_theta * embedding_noise, name=name)


def l0_norm(
    log_alpha,
    beta=common.BETA,
    gamma=common.GAMMA,
    zeta=common.ZETA):
  """Calculate the l0-regularization contribution to the loss.

  Args:
    log_alpha: Tensor of the log alpha parameters for the hard concrete
      distribution.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    Scalar tensor containing the unweighted l0-regularization term contribution
    to the loss.
  """
  # Value of the CDF of the hard-concrete distribution evaluated at 0
  reg_per_weight = tf.sigmoid(log_alpha - beta * tf.log(-gamma / zeta))
  return tf.reduce_sum(reg_per_weight)
