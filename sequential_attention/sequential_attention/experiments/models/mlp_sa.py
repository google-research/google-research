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

"""Feature selection with Sequential Attention."""

from sequential_attention.experiments.models.mlp import MLPModel
from sequential_attention.sequential_attention import SequentialAttention
import tensorflow as tf


class SequentialAttentionModel(MLPModel):
  """MLP with Sequential Attention."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_train_steps,
      num_inputs_to_select_per_step=1,
      **kwargs,
  ):
    """Initialize the model."""

    super(SequentialAttentionModel, self).__init__(**kwargs)
    self.seqatt = SequentialAttention(
        num_candidates=num_inputs,
        num_candidates_to_select=num_inputs_to_select,
        num_candidates_to_select_per_step=num_inputs_to_select_per_step,
    )
    self.num_train_steps = num_train_steps

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    training_percentage = self.optimizer.iterations / self.num_train_steps
    feature_weights = self.seqatt(training_percentage)
    inputs = tf.multiply(inputs, feature_weights)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction
