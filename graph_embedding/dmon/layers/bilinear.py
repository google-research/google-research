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
"""TODO(tsitsulin): add headers, tests, and improve style."""
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Layer


class Bilinear(Layer):  # pylint: disable=missing-class-docstring

  def __init__(self, input_dim, output_dim):
    super(Bilinear, self).__init__()
    self.w = self.add_weight(
        shape=(input_dim, output_dim),
        initializer='random_normal',
        trainable=True)

  def call(self, inputs):
    first, second = inputs
    output = tf.matmul(first, tf.matmul(self.w, second))
    return output
