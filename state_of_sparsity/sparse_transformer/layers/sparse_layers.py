# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Common layers used in the sparse transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensor2tensor.layers import common_layers
import tensorflow as tf

from state_of_sparsity.sparse_transformer.layers import common_sparse


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     sparsity_technique=None,
                     threshold=3.0,
                     clip_alpha=None,
                     training=True,
                     name=None,
                     initial_sparsity=None):
  """Hidden layer with RELU activation followed by linear projection."""
  layer_fn = common_layers.dense
  if sparsity_technique:
    layer_fn = functools.partial(
        common_sparse.dense,
        sparsity_technique=sparsity_technique,
        threshold=threshold,
        training=training,
        clip_alpha=clip_alpha,
        initial_sparsity=initial_sparsity)

  layer_name = "%s_{}" % name if name else "{}"
  h = layer_fn(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = common_layers.dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = layer_fn(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      name=layer_name.format("conv2"))
  return o
