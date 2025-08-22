# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Feature Selection Using Batch-Wise Attenuation and Feature Mask Normalization.

Yiwen Liao; Raphael Latty; Bin Yang

https://ieeexplore.ieee.org/document/9533531
"""

import tensorflow as tf


class LiaoLattyYangMask(tf.Module):
  """A feature selection algorithm based on attention mechanism."""

  def __init__(self, num_inputs, name='liao_latty_yang_mask', **kwargs):
    super(LiaoLattyYangMask, self).__init__(name=name, **kwargs)

    mlp_sequence = [
        tf.keras.layers.Dense(
            dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        for dim in [128, 64, num_inputs]
    ]
    self.mlp_model = tf.keras.Sequential(mlp_sequence)

  def __call__(self, inputs):
    nonlinear = self.mlp_model(inputs)
    batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
    logits = tf.reduce_sum(nonlinear, axis=0) / batch_size
    return tf.nn.softmax(logits)
