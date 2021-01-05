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

"""Utils for tensorflow/keras."""
import tensorflow.compat.v2 as tf


def my_reset_states(metric):
  """Resets metric states.

  Args:
    metric: A keras metric to reset states for.
  """
  for var in metric.variables:
    var.assign(0)


def orthogonal_regularization(model, reg_coef=1e-4):
  """Orthogonal regularization v2.

  See equation (3) in https://arxiv.org/abs/1809.11096.

  Args:
    model: A keras model to apply regualization for.
    reg_coef: Orthogonal regularization coefficient.

  Returns:
    A regularization loss term.
  """

  reg = 0
  for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
      prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
      reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
  return reg * reg_coef
