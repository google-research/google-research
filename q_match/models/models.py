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

"""Generic models for building blocks.

Can be used either for the main prediction task or as a building blocks
for pretext and main models.

All batch norm attributes (e.g., self.batch_norm_layers = ...) should contain
'batch_norm' in the object name, because this string is matched for weight
decay purposes (batch norm parameters are excluded.)
"""

from typing import Sequence

from flax import linen as nn
import jax


class BaseMLP(nn.Module):
  """Base MLP with BatchNorm on inputs.

  Uses BatchNorm with running average before features are passes through
  the model.

  Relu after every layer except the last.
  """

  layer_sizes: Sequence[int]
  training: bool

  def setup(self):
    self.layers = [nn.Dense(layer_size) for layer_size in self.layer_sizes]
    self.initial_batch_norm_layer = nn.BatchNorm(use_running_average=True)
    self.batch_norm_layers = [
        nn.BatchNorm(use_running_average=(not self.training))
        for _ in self.layer_sizes
    ]

  def __call__(self, inputs):
    x = inputs
    x = self.initial_batch_norm_layer(x)
    for i, layer in enumerate(self.layers):
      x = self.batch_norm_layers[i](x)
      x = layer(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x


class ImixMLP(nn.Module):
  """Base MLP used in iMix.

  A replica of the iMix mode architecture.

  See page 17 here: https://arxiv.org/pdf/2010.08887.pdf

  'The output dimensions of layers are (2048-2048-4096-4096-8192),
  where all layers have batch normalization followed by ReLU except
  for the last layer. The last layer activation is maxout (Goodfellow et al.,
  2013) with 4 sets, such that the output dimension is 2048.
  On top of this five-layer MLP, we attach two-layer
  MLP (2048-128)  as a projection head.'

  See the code here: https://github.com/kibok90/imix/blob/main/models/mlp.py
  """

  training: bool

  def setup(self):
    self.dense_layers = [nn.Dense(2048, use_bias=False),
                         nn.Dense(2048, use_bias=False),
                         nn.Dense(4096, use_bias=False),
                         nn.Dense(4096, use_bias=False),
                         nn.Dense(8192, use_bias=True)]
    self.initial_batch_norm_layer = nn.BatchNorm(use_running_average=True)
    self.batch_norm_layers = [
        nn.BatchNorm(use_running_average=(not self.training))
        for _ in self.dense_layers
    ]

    self.projection_head = nn.Dense(128)

  def __call__(self, inputs):
    x = inputs
    x = self.initial_batch_norm_layer(x)
    for i, layer in enumerate(self.dense_layers):
      x = layer(x)
      if i != len(self.dense_layers) - 1:
        x = self.batch_norm_layers[i](x)
        x = nn.relu(x)
      else:
        x = x.reshape(x.shape[:-1] + (x.shape[-1]//4, 4))
        x = jax.numpy.max(x, axis=-1)
    x = self.projection_head(x)
    return x


class MLP(nn.Module):
  """Base MLP with BatchNorm.

  Relu after every layer except the last.
  """

  layer_sizes: Sequence[int]
  training: bool

  def setup(self):
    self.layers = [nn.Dense(layer_size) for layer_size in self.layer_sizes]
    self.batch_norm_layers = [
        nn.BatchNorm(use_running_average=(not self.training))
        for _ in self.layer_sizes
    ]

  def __call__(self, inputs):
    x = inputs
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i != len(self.layers) - 1:
        x = self.batch_norm_layers[i](x)
        x = nn.relu(x)
    return x


class Resnet(nn.Module):
  """Resnet architecure as defined in Revisiting DL Models for Tabular Data.

  https://arxiv.org/pdf/2106.11959.pdf
  """

  training: bool
  num_blocks: int = 12
  factor: float = 3.402
  dropout_rate_1: float = 0.3612
  dropout_rate_2: float = 0.0
  layer_size: int = 235

  @nn.compact
  def __call__(self, inputs):
    first_projection = nn.Dense(self.layer_size)
    hidden_size = int(self.layer_size * self.factor)
    batch_norm_layers = [
        nn.BatchNorm(use_running_average=(not self.training))
        for _ in range(self.num_blocks)
    ]
    linear_first = [
        nn.Dense(hidden_size, use_bias=False)
        for _ in range(self.num_blocks)
    ]
    linear_second = [nn.Dense(self.layer_size) for _ in range(self.num_blocks)]

    dropout_first = [
        nn.Dropout(self.dropout_rate_1, deterministic=(not self.training))
        for _ in range(self.num_blocks)
    ]

    dropout_second = [
        nn.Dropout(self.dropout_rate_2, deterministic=(not self.training))
        for _ in range(self.num_blocks)
    ]

    x = inputs
    x = first_projection(x)
    for i in range(self.num_blocks):
      x += dropout_second[i](
          linear_second[i](nn.relu(dropout_first[i](
              linear_first[i](batch_norm_layers[i](x))))))

    return x
