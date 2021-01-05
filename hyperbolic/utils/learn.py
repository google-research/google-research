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

# Lint as: python3
"""Loss helper functions."""

import tensorflow.compat.v2 as tf


def softmax_cross_entropy(pos, neg):
  """softmax cross entropy loss.

  Let d_p = pos, d_n = neg.
  we minimize:
  log(1+exp(d_p)) + log(1+exp(-d_n))
  for stability, is it equivalent to
  d_p + log(1+exp(-d_p)) + log(1+exp(-d_n))

  Args:
    pos: Tensor.
    neg: Tensor of the same shape of pos.

  Returns:
    Tensor holding pointwise loss of the same shape as pos.
  """
  log_exp_pos = tf.math.log1p(tf.math.exp(-pos))
  log_exp_neg = tf.math.log1p(tf.math.exp(-neg))
  return pos + log_exp_pos + log_exp_neg
