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

"""Common custom tf ops for Munchausen RL agents.

A library of various functions used by the agents.

It defines more numerically stable log_softmax and softmax operations. TF
already implements the logSumExp trick for this kind of compute, but we have to
implement it by hand to take into account a temperature.
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def stable_scaled_log_softmax(x, tau, axis=-1):
  """Scaled log_softmax operation.

  Args:
    x: tensor of floats, inputs of the softmax (logits).
    tau: float, softmax temperature.
    axis: int, axis to perform the softmax operation.
  Returns:
    tau * tf.log_softmax(x/tau, axis=axis)
  """
  max_x = tf.reduce_max(x, axis=axis, keepdims=True)
  y = x - max_x
  tau_lse = max_x + tau * tf.math.log(
      tf.reduce_sum(tf.math.exp(y / tau), axis=axis, keepdims=True))
  return x - tau_lse


def stable_softmax(x, tau, axis=-1):
  """Stable softmax operation.

  Args:
    x: tensor of floats, inputs of the softmax (logits).
    tau: float, softmax temperature.
    axis: int, axis to perform the softmax operation.
  Returns:
    softmax(x/tau, axis=axis)
  """
  max_x = tf.reduce_max(x, axis=axis, keepdims=True)
  y = x - max_x
  return tf.nn.softmax(y/tau, axis=axis)
