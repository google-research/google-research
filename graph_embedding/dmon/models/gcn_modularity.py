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

"""TODO(tsitsulin): add headers, tests, and improve style."""
from typing import List
import tensorflow.compat.v2 as tf
from graph_embedding.dmon.layers.gcn import GCN
from graph_embedding.dmon.layers.modularity import ModularityPooling


def gcn_modularity(inputs,  # TODO(tsitsulin): improve signature and documentation pylint: disable=dangerous-default-value,missing-function-docstring
                   channel_sizes,
                   orthogonality_regularization = 0.3,
                   cluster_size_regularization = 1.0,
                   dropout_rate = 0.75,
                   pooling_mlp_sizes = []):
  features, graph, adjacency = inputs
  output = features
  for n_channels in channel_sizes[:-1]:
    output = GCN(n_channels)([output, graph])
  pool, pool_assignment = ModularityPooling(
      channel_sizes[-1],
      do_unpool=False,
      orthogonality_regularization=orthogonality_regularization,
      cluster_size_regularization=cluster_size_regularization,
      dropout_rate=dropout_rate,
      mlp_sizes=pooling_mlp_sizes)([output, adjacency])
  return tf.keras.Model(
      inputs=[features, graph, adjacency], outputs=[pool, pool_assignment])
