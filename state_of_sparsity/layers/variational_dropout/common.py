# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Defines common utilties for variational dropout layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


EPSILON = 1e-8


def compute_log_alpha(log_sigma2, theta, eps=EPSILON, value_limit=8.):
  R"""Compute the log \alpha values from \theta and log \sigma^2.

  The relationship between \sigma^2, \theta, and \alpha as defined in the
  paper https://arxiv.org/abs/1701.05369 is

  \sigma^2 = \alpha * \theta^2

  This method calculates the log \alpha values based on this relation.

  Args:
    log_sigma2: tf.Variable. The log variance for each weight.
    theta: tf.Variable. The mean for each weight.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.
    value_limit: If not None, the log_alpha values will be clipped to the
     range [-value_limit, value_limit]. This is consistent with the
     implementation provided with the publication.

  Returns:
    A tf.Tensor representing the calculated log \alpha values.
  """
  log_alpha = log_sigma2 - tf.log(tf.square(theta) + eps)

  if value_limit is not None:
    # If a limit is specified, clip the alpha values
    return tf.clip_by_value(log_alpha, -value_limit, value_limit)
  return log_alpha


def compute_log_sigma2(log_alpha, theta, eps=EPSILON):
  R"""Compute the log \sigma^2 values from log \alpha and \theta.

  The relationship between \sigma^2, \theta, and \alpha as defined in the
  paper https://arxiv.org/abs/1701.05369 is

  \sigma^2 = \alpha * \theta^2

  This method calculates the log \sigma^2 values based on this relation.

  Args:
    log_alpha: tf.Tensor. The log alpha values for each weight.
    theta: tf.Variable. The mean for each weight.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.

  Returns:
    A tf.Tensor representing the calculated log \sigma^2 values.
  """
  return log_alpha + tf.log(tf.square(theta) + eps)
