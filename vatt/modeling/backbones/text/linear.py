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
"""Main factory for building vatt Models."""

import tensorflow as tf


class LinearLM(tf.keras.layers.Layer):
  """Linear language model."""

  def __init__(self,
               d_model=2048,
               name="linear_lm",
               **kwargs):

    super(LinearLM, self).__init__(name=name)
    self._d_model = d_model
    self.conv1d = tf.keras.layers.Conv1D(
        self._d_model, 1, name="text_conv1")

  def call(self,
           inputs,
           inputs_embeddings,
           attention_mask=None,
           training=True):
    """Connects graph to sentence representation."""

    del inputs

    # extract features
    features = self.conv1d(inputs_embeddings)
    features = tf.nn.relu(features)

    outputs = {
        "hidden_states": [features]
    }

    return outputs
