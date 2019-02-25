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

"""Defines common utilities for l0-regularization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Small constant value to add when taking logs or sqrts to avoid NaNs
EPSILON = 1e-8

# The default hard-concrete distribution parameters
BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1


def hard_concrete_sample(
    log_alpha,
    beta=BETA,
    gamma=GAMMA,
    zeta=ZETA,
    eps=EPSILON):
  """Sample values from the hard concrete distribution.

  The hard concrete distribution is described in
  https://arxiv.org/abs/1712.01312.

  Args:
    log_alpha: The log alpha parameters that control the "location" of the
      distribution.
    beta: The beta parameter, which controls the "temperature" of
      the distribution. Defaults to 2/3 from the above paper.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.
    eps: A small constant value to add to logs and sqrts to avoid NaNs.

  Returns:
    A tf.Tensor representing the output of the sampling operation.
  """
  random_noise = tf.random_uniform(
      tf.shape(log_alpha),
      minval=0.0,
      maxval=1.0)

  # NOTE: We add a small constant value to the noise before taking the
  # log to avoid NaNs if a noise value is exactly zero. We sample values
  # in the range [0, 1), so the right log is not at risk of NaNs.
  gate_inputs = tf.log(random_noise + eps) - tf.log(1.0 - random_noise)
  gate_inputs = tf.sigmoid((gate_inputs + log_alpha) / beta)
  stretched_values = gate_inputs * (zeta - gamma) + gamma

  return tf.clip_by_value(
      stretched_values,
      clip_value_max=1.0,
      clip_value_min=0.0)


def hard_concrete_mean(log_alpha, gamma=GAMMA, zeta=ZETA):
  """Calculate the mean of the hard concrete distribution.

  The hard concrete distribution is described in
  https://arxiv.org/abs/1712.01312.

  Args:
    log_alpha: The log alpha parameters that control the "location" of the
      distribution.
    gamma: The gamma parameter, which controls the lower bound of the
      stretched distribution. Defaults to -0.1 from the above paper.
    zeta: The zeta parameters, which controls the upper bound of the
      stretched distribution. Defaults to 1.1 from the above paper.

  Returns:
    A tf.Tensor representing the calculated means.
  """
  stretched_values = tf.sigmoid(log_alpha) * (zeta - gamma) + gamma
  return tf.clip_by_value(
      stretched_values,
      clip_value_max=1.0,
      clip_value_min=0.0)
