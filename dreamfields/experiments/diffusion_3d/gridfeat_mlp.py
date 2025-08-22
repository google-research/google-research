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

"""MLP architecture based on tiny-cuda-nn."""

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
# pylint: disable=g-import-not-at-top
try:
  import tinycudann as tcnn
except ImportError as e:
  print("WARN: Unable to import tinycudann. Error:", e)

# pylint: disable=g-bad-import-order
import neural_fields
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top


class GridFeatureMLP(nn.Module):
  """MLP architecture."""

  def __init__(
      self,
      *,
      activation="ReLU",
      parameterization = "VolSDF",
      base_radius=1.,
      features_density,
      layers_density,
      features_color,
      layers_color,
      emb_density_config,
      emb_color_config,
      seed=1337,
      backend_density="CutlassMLP",
      backend_color="FullyFusedMLP"):
    super().__init__()
    self.base_radius = base_radius
    self.parameterization = parameterization

    # Canonical density or distance network.
    # TODO(jainajay): Fuse tcnn encoding and MLP.
    self.emb_density_config = emb_density_config
    self.emb_density = tcnn.Encoding(3, dict(emb_density_config))
    emb_density_dim = (
        emb_density_config["n_levels"] *
        emb_density_config["n_features_per_level"])
    assert layers_density >= 2
    trunk_density = neural_fields.WrappedNetwork(
        emb_density_dim,
        features_density,
        {
            "otype": backend_density,
            "activation": activation,
            "output_activation": "ReLU",
            "n_neurons": features_density,
            # Subtract 3 since we construct and init an extra layer below,
            # and there are input and output layers in the wrapped network.
            "n_hidden_layers": layers_density - 3,
        },
        seed=seed + 1,
        computation_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    self.layers_density = nn.Sequential(
        trunk_density, nn.Linear(features_density, features_color + 1))

    # Canonical color network with appearance code conditioning.
    self.emb_color_config = emb_color_config
    self.emb_color = tcnn.Encoding(3, dict(emb_color_config))
    emb_color_dim = (
        emb_color_config["n_levels"] * emb_color_config["n_features_per_level"])
    assert layers_color >= 2
    self.layers_color = neural_fields.WrappedNetwork(
        emb_color_dim + features_color,
        3,
        {
            "otype": backend_color,
            "activation": activation,
            "output_activation": "None",
            "n_neurons": features_color,
            # Subtract two due to input and output layers.
            "n_hidden_layers": layers_color - 2,
        },
        seed=seed + 1,
        computation_dtype=None,
        output_dtype=torch.float32,
    )

    self.reset_parameters()

  def reset_parameters(self):
    for layer in self.children():
      if isinstance(layer, nn.Linear):
        torch.nn.init.zeros_(layer.bias)

    # Initialize last layer for sphere displacement near zero.
    if self.parameterization == "volsdf":
      if self.base_radius is not None:
        torch.nn.init.uniform_(
            self.layers_density[-1].weight[0], a=-1e-1, b=1e-1)

  def forward(self,
              mean,
              cov=None,
              *,
              decayscales,
              render_color=True):
    """Run MLP. mean is [D, *batch, 3] and cov is [D, *batch, 3, 3]."""
    # NOTE(jainajay): cov is ignored as there IPE for grid embeddings
    # is not implemented.
    batch_shape = mean.shape[:-1]
    num_points = np.prod(batch_shape)
    assert mean.shape[-1] == 3

    x = mean.reshape(num_points, 3)

    # Density network.
    emb_density = neural_fields.mask_grid_features(
        self.emb_density(x), decayscales["density"],
        self.emb_density_config["n_features_per_level"])
    features = self.layers_density(emb_density)
    d, features = features[Ellipsis, :1], features[Ellipsis, 1:]
    assert d.dtype == torch.float32

    # Parameterize canonical frame VolSDF distance as offset from sphere.
    if self.parameterization == "volsdf" and self.base_radius is not None:
      d = d + torch.norm(x, dim=-1, keepdim=True) - self.base_radius

    # Appearance network.
    if render_color:
      emb_color = neural_fields.mask_grid_features(
          self.emb_color(x), decayscales["color"],
          self.emb_color_config["n_features_per_level"])
      features = torch.cat([emb_color, features], dim=-1)
      raw_rgb = self.layers_color(features)
      raw_output = torch.cat([raw_rgb, d], dim=-1)
      return raw_output.view(*batch_shape, 4)

    return d.view(*batch_shape)
