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

"""Implementation of feature voxel modulated neural fields.

Architecture has been used in the following papers:
1. Block-NeRF: https://arxiv.org/abs/2202.05263
2. Convolutional Occupancy Networks https://arxiv.org/abs/2003.04618
"""

from typing import Sequence, Type, Union

import flax.linen as nn
import jax
import ml_collections

from nf_diffusion.models.layers.feat_grid import FeatureGrid
from nf_diffusion.models.layers.pos_enc import PositionalEncoding


class ID(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

FeatureType = Union[Type[PositionalEncoding], Type[FeatureGrid], Type[ID]]


class SimpleMLP(nn.Module):
  """Neural fields MLP with different types of features."""
  inp_dim: int
  hid_dim: Sequence[int]
  out_dim: int

  @nn.compact
  def __call__(self, feat):
    for nlayer, hdim in enumerate(self.hid_dim):
      feat = nn.Dense(hdim, name="dense{}".format(nlayer))(feat)
      feat = jax.nn.leaky_relu(feat, negative_slope=0.01)
    feat = nn.Dense(self.out_dim)(feat) # The name will be Dense_0
    return feat


class NeuralFieldMLP(nn.Module):
  """Neural fields MLP with different types of features."""
  inp_dim: int
  hid_dim: Sequence[int]
  out_dim: int

  # Positional encoding configurations
  feat_type: str
  feat: ml_collections.ConfigDict

  # Layers
  # layers: Sequence[nn.Module]
  # feat_layer: FeatureType

  def setup(self):
    if self.feat_type == "pos_enc":
      self.feat_layer = PositionalEncoding(**self.feat)
    elif self.feat_type == "feat_grid":
      self.feat_layer = FeatureGrid(**self.feat)
    else:
      self.feat_layer = ID()

    # self.layers = [
    #     nn.Dense(features=hdim, name=f'dense_{n}')
    #     for n, hdim in enumerate(self.hid_dim)
    # ]
    # self.out_layer = nn.Dense(features=self.out_dim, name='dense_out')
    self.mlp = SimpleMLP(
        inp_dim=self.inp_dim, hid_dim=self.hid_dim, out_dim=self.out_dim)

  def __call__(self, x):
    feat, _ = self.feat_layer(x)  # second argument is x_rel
    # for layer in self.layers:
    #   feat = layer(feat)
    #   # TODO(guandao) allow choice of nonlinearity
    #   feat = jax.nn.leaky_relu(feat, negative_slope=0.01)
    # feat = self.out_layer(feat)
    feat = self.mlp(feat)
    return feat
