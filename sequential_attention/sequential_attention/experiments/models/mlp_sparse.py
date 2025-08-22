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

"""Sparse MLP Model."""

from sequential_attention.experiments.models.mlp import MLPModel
import tensorflow as tf


class SparseModel(MLPModel):
  """MLP with Sequential Attention."""

  def __init__(self, selected_features, **kwargs):
    """Initialize the model."""

    super(SparseModel, self).__init__(**kwargs)
    self.selected_features = selected_features

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    inputs = tf.multiply(inputs, self.selected_features)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction
