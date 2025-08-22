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

"""Feed-forward MLP architecture supporting Fourier Feature encodings."""

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

# pylint: disable=g-bad-import-order
import neural_fields
# pylint: enable=g-bad-import-order


class FourierFeatureMLP(nn.Module):
  """MLP architecture."""

  def __init__(self,
               *,
               activation="ReLU",
               parameterization = "VolSDF",
               base_radius=1.,
               features_density,
               layers_density,
               features_color,
               layers_color,
               fourfeat_config,
               seed=1337,
               backend_density="CutlassMLP",
               backend_color="FullyFusedMLP"):
    super().__init__()
    self.base_radius = base_radius
    self.parameterization = parameterization
    self.backend_density = backend_density

    # Canonical density or distance network.
    self.emb = neural_fields.FourierFeatureIPE(
        num_dims_to_encode=3,
        dtype=torch.float32,
        **fourfeat_config,
    )
    self.layers_density = neural_fields.WrappedNetwork(
        self.emb.output_dim,
        features_density + 1,
        {
            "otype": backend_density,
            "activation": activation,
            "output_activation": "None",
            "n_neurons": features_density,
            "n_hidden_layers": layers_density - 2
        },
        seed=seed + 1,
        computation_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    # Canonical color network with appearance code conditioning.
    self.layers_color = neural_fields.WrappedNetwork(
        features_density,
        3,
        {
            "otype": backend_color,
            "activation": activation,
            "output_activation": "None",
            "n_neurons": features_color,
            "n_hidden_layers": layers_color - 2
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
        assert self.backend_density == "torch", (
            "Need torch backend to init last layer")
        torch.nn.init.uniform_(
            self.layers_density.layers[-1].weight[0], a=-1e-1, b=1e-1)

  def forward(self, mean, cov=None, *, decayscales, render_color=True):
    """Run MLP. mean is [D, *batch, 3] and cov is [D, *batch, 3, 3]."""
    batch_shape = mean.shape[:-1]
    num_points = np.prod(batch_shape)
    assert mean.shape[-1] == 3

    x = mean.reshape(num_points, 3)

    # Canonical network. If enabled, use IPE.
    assert cov is not None
    if not isinstance(cov, float):
      cov = cov.reshape(num_points, 3, 3)
    emb = self.emb(x, cov, decayscales["density"])
    features = self.layers_density(emb)
    d, features = features[Ellipsis, :1], features[Ellipsis, 1:]
    assert d.dtype == torch.float32

    # Parameterize canonical frame VolSDF distance as offset from sphere.
    if self.parameterization == "volsdf" and self.base_radius is not None:
      d = d + torch.norm(x, dim=-1, keepdim=True) - self.base_radius

    # Appearance network.
    if render_color:
      raw_rgb = self.layers_color(features)
      raw_output = torch.cat([raw_rgb, d], dim=-1)
      return raw_output.view(*batch_shape, 4)

    return d.view(*batch_shape)
