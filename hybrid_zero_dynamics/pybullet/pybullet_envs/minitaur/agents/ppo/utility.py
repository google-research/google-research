# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import re

import tensorflow as tf
from tensorflow.python.client import device_lib


def create_nested_vars(tensors):
  """Create variables matching a nested tuple of tensors.

  Args:
    tensors: Nested tuple of list of tensors.

  Returns:
    Nested tuple or list of variables.
  """
  if isinstance(tensors, (tuple, list)):
    return type(tensors)(create_nested_vars(tensor) for tensor in tensors)
  return tf.Variable(tensors, False)


def reinit_nested_vars(variables, indices=None):
  """Reset all variables in a nested tuple to zeros.

  Args:
    variables: Nested tuple or list of variaables.
    indices: Indices along the first dimension to reset, defaults to all.

  Returns:
    Operation.
  """
  if isinstance(variables, (tuple, list)):
    return tf.group(*[
        reinit_nested_vars(variable, indices) for variable in variables])
  if indices is None:
    return variables.assign(tf.zeros_like(variables))
  else:
    zeros = tf.zeros([tf.shape(indices)[0]] + variables.shape[1:].as_list())
    return tf.scatter_update(variables, indices, zeros)


def assign_nested_vars(variables, tensors):
  """Assign tensors to matching nested tuple of variables.

  Args:
    variables: Nested tuple or list of variables to update.
    tensors: Nested tuple or list of tensors to assign.

  Returns:
    Operation.
  """
  if isinstance(variables, (tuple, list)):
    return tf.group(*[
        assign_nested_vars(variable, tensor)
        for variable, tensor in zip(variables, tensors)])
  return variables.assign(tensors)


def discounted_return(reward, length, discount):
  """Discounted Monte-Carlo returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * reward, [1]), [1, 0]),
      tf.zeros_like(reward[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')


def fixed_step_return(reward, value, length, discount, window):
  """N-step discounted return."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.zeros_like(reward)
  for _ in range(window):
    return_ += reward
    reward = discount * tf.concat(
        [reward[:, 1:], tf.zeros_like(reward[:, -1:])], 1)
  return_ += discount ** window * tf.concat(
      [value[:, window:], tf.zeros_like(value[:, -window:]), 1])
  return tf.check_numerics(tf.stop_gradient(mask * return_), 'return')


def lambda_return(reward, value, length, discount, lambda_):
  """TD-lambda returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  sequence = mask * reward + discount * value * (1 - lambda_)
  discount = mask * discount * lambda_
  sequence = tf.stack([sequence, discount], 2)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * agg,
      tf.transpose(tf.reverse(sequence, [1]), [1, 2, 0]),
      tf.zeros_like(value[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')


def lambda_advantage(reward, value, length, discount):
  """Generalized Advantage Estimation."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  next_value = tf.concat([value[:, 1:], tf.zeros_like(value[:, -1:])], 1)
  delta = reward + discount * next_value - value
  advantage = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * delta, [1]), [1, 0]),
      tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(advantage), 'advantage')


def diag_normal_kl(mean0, logstd0, mean1, logstd1):
  """Epirical KL divergence of two normals with diagonal covariance."""
  logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
  return 0.5 * (
      tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
      tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
      tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
      mean0.shape[-1].value)


def diag_normal_logpdf(mean, logstd, loc):
  """Log density of a normal with diagonal covariance."""
  constant = -0.5 * (math.log(2 * math.pi) + logstd)
  value = -0.5 * ((loc - mean) / tf.exp(logstd)) ** 2
  return tf.reduce_sum(constant + value, -1)


def diag_normal_entropy(mean, logstd):
  """Empirical entropy of a normal with diagonal covariance."""
  constant = mean.shape[-1].value * math.log(2 * math.pi * math.e)
  return (constant + tf.reduce_sum(2 * logstd, 1)) / 2


def available_gpus():
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def gradient_summaries(grad_vars, groups=None, scope='gradients'):
  """Create histogram summaries of the gradient.

  Summaries can be grouped via regexes matching variables names.

  Args:
    grad_vars: List of (gradient, variable) tuples as returned by optimizers.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for grad, var in grad_vars:
    if grad is None:
      continue
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(grad)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  for name, grads in grouped.items():
    grads = [tf.reshape(grad, [-1]) for grad in grads]
    grads = tf.concat(grads, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, grads))
  return tf.summary.merge(summaries)


def variable_summaries(vars_, groups=None, scope='weights'):
  """Create histogram summaries for the provided variables.

  Summaries can be grouped via regexes matching variables names.

  Args:
    vars_: List of variables to summarize.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for var in vars_:
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(var)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  for name, vars_ in grouped.items():
    vars_ = [tf.reshape(var, [-1]) for var in vars_]
    vars_ = tf.concat(vars_, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, vars_))
  return tf.summary.merge(summaries)
