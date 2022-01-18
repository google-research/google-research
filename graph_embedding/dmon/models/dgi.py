# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
from typing import List
import tensorflow.compat.v2 as tf
from graph_embedding.dmon.layers.bilinear import Bilinear


def deep_graph_infomax(inputs,  # pylint: disable=missing-function-docstring
                       encoder):
  features, features_corrupted, graph = inputs

  representations_clean = features
  representations_corrupted = features_corrupted

  for layer in encoder:
    representations_clean = layer([representations_clean, graph])
    representations_corrupted = layer([representations_corrupted, graph])

  representation_summary = tf.math.reduce_mean(representations_clean, axis=0)
  representation_summary = tf.nn.sigmoid(representation_summary)
  representation_summary = tf.reshape(representation_summary, [-1, 1])

  transform = Bilinear(representations_clean.shape[-1],
                       representations_clean.shape[-1])

  discriminator_clean = transform(
      [representations_clean, representation_summary])
  discriminator_corrupted = transform(
      [representations_corrupted, representation_summary])

  features_output = tf.concat([discriminator_clean, discriminator_corrupted], 0)

  return tf.keras.Model(
      inputs=[features, features_corrupted, graph],
      outputs=[representations_clean, features_output])
