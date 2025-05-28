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

"""Feature selection with Liao-Latty-Yang."""

from sequential_attention.benchmarks.liao_latty_yang import LiaoLattyYangMask
from sequential_attention.experiments.models.mlp import MLPModel
import tensorflow as tf


class LiaoLattyYangModel(MLPModel):
  """MLP with Liao-Latty-Yang."""

  def __init__(self, num_inputs, **kwargs):
    """Initialize the model."""

    super(LiaoLattyYangModel, self).__init__(**kwargs)

    self.lly = LiaoLattyYangMask(num_inputs=num_inputs)

  def call(self, inputs, training=False, return_attention=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    attention_weights = self.lly(inputs)
    if return_attention:
      return attention_weights
    inputs = tf.multiply(inputs, attention_weights)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction
