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

"""Script contains losses implementations for NCE, linear critic, etc.
"""

import tensorflow as tf


def categorical_kl(probs1, probs2=None):
  if probs2 is None:
    probs2 = tf.ones_like(probs1) * tf.reduce_sum(probs1) / tf.reduce_sum(
        tf.ones_like(probs1))

  kl = tf.reduce_sum(
      probs1 * (-tf.math.log(1e-8 + probs2) + tf.math.log(1e-8 + probs1)), -1)
  return kl




