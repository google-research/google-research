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

"""MLP architecture based on tiny-cuda-nn."""

from typing import Any, Tuple, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# pylint: disable=g-import-not-at-top
try:
  import tinycudann as tcnn
except ImportError as e:
  print("WARN: Unable to import tinycudann. Error:", e)

# pylint: disable=g-bad-import-order
import neural_fields
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top


class TCNNMLP(nn.Module):
  """MLP architecture."""

  def __init__(
      self,
      *,
      activation="ReLU",
      parameterization = "VolSDF",
      # SDF parameters.
      base_radius=1.,
      # Deformation architecture.
      deformation_code_dim,
      features_deformation,
      layers_deformation,
      rotation_deformation = True,
      emb_deformation_config,
      # Canonical density architecture.
      features_density,
      layers_density,
      emb_density_config,
      # Canonical color architecture.
      features_color,
      layers_color,
      emb_color_config,
      seed=1337,
      # Backends.
      backend_deformation="FullyFusedMLP",
      backend_density="CutlassMLP",
      backend_color="FullyFusedMLP"):
    super().__init__()
    self.base_radius = base_radius
    self.parameterization = parameterization

    # Deformation network.
    self.rotation_deformation = rotation_deformation
    self.emb_deformation_config = emb_deformation_config
    self.embedding_deformation = tcnn.Encoding(3, dict(emb_deformation_config))
    embedding_deformation_dim = (
        emb_deformation_config["n_levels"] *
        emb_deformation_config["n_features_per_level"])
    project_deformation_codes_dim = features_deformation
    self.project_deformation_codes = nn.Linear(deformation_code_dim,
                                               project_deformation_codes_dim)
    trunk_deformation = neural_fields.WrappedNetwork(
        embedding_deformation_dim + project_deformation_codes_dim,
        features_deformation,
        {
            "otype": backend_deformation,
            "activation": activation,
            "output_activation": activation,
            "n_neurons": features_deformation,
            "n_hidden_layers": layers_deformation
        },
        seed=seed,
        computation_dtype=None,
        output_dtype=torch.float32,
    )
    self.layers_deformation = nn.Sequential(
        trunk_deformation,
        nn.Linear(features_deformation, 6),
    )

    # Canonical density or distance network.
    # TODO(jainajay): Fuse tcnn encoding and MLP.
    self.emb_density_config = emb_density_config
    self.emb_density = tcnn.Encoding(3, dict(emb_density_config))
    emb_density_dim = (
        emb_density_config["n_levels"] *
        emb_density_config["n_features_per_level"])
    trunk_density = neural_fields.WrappedNetwork(
        emb_density_dim,
        features_density,
        {
            "otype": backend_density,
            "activation": activation,
            "output_activation": "ReLU",
            "n_neurons": features_density,
            "n_hidden_layers": layers_density - 1  # TODO(jainajay): this is 0.
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
    self.layers_color = neural_fields.WrappedNetwork(
        emb_color_dim + features_color,
        3,
        {
            "otype": backend_color,
            "activation": activation,
            "output_activation": "None",
            "n_neurons": features_color,
            "n_hidden_layers": layers_color
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

    # Initialize last layer of deformations near zero (Sec 3.2, nerfies).
    torch.nn.init.uniform_(self.layers_deformation[-1].weight, a=-1e-6, b=1e-6)

    # Initialize last layer for sphere displacement near zero.
    if self.parameterization == "volsdf":
      if self.base_radius is not None:
        torch.nn.init.uniform_(
            self.layers_density[-1].weight[0], a=-1e-1, b=1e-1)

  def deformation(self, x, deformation_codes,
                  decayscale):
    """Per-point deformation network.

    Does not use IPE, but can use decayscale.

    Args:
      x: Ray coordinates, in observation frame,
      deformation_codes: Latent variables describing deformations.
      decayscale: Coarse-to-fine coefficient.

    Returns:
      x_transformed: Transformed ray coordinates, in canonical frame.
      aux: Dictionary of metrics and losses.
    """
    batch_shape = x.shape[:-1]
    new_dims = (1,) * (len(batch_shape) - 1)

    # Project deformation codes and expand embedding to match points x.
    dc_norm = F.normalize(deformation_codes, dim=-1)
    dc_proj = self.project_deformation_codes(dc_norm)
    dc_proj = dc_proj.view(dc_proj.size(0), *new_dims,
                           -1).expand(*batch_shape, -1)

    # Flatten to get a batch size multiple of 128 for tcnn MLPs.
    dc_proj = dc_proj.reshape(-1, dc_proj.size(-1))
    x_flat = x.view(-1, x.size(-1))
    deformation_emb = self.embedding_deformation(x_flat)
    n_features_per_level = (self.emb_deformation_config["n_features_per_level"])
    deformation_emb = mask_grid_features(
        deformation_emb, decayscale, n_features_per_level=n_features_per_level)

    # Predict deformation as a screw axis.
    deformation_input = torch.cat([deformation_emb, dc_proj], dim=-1)
    s_per_point = self.layers_deformation(deformation_input)  # [*batch, 6].
    s_per_point = s_per_point.view(*batch_shape, 6)  # Reshape back.
    assert s_per_point.dtype == torch.float32

    # Compute rotation matrix and translation.
    # er is [*batch, 3, 3], p is [*batch, 3].
    er, p = neural_fields.get_deformation(
        s_per_point, with_rotation=self.rotation_deformation)

    # Apply deformation.
    x_transformed = torch.matmul(
        er, x.unsqueeze(-1)).squeeze(-1) + p  # [*batch, 3].

    # Compute losses to regularize deformation.
    spp_sq = torch.square(s_per_point)
    aux = {}
    if self.rotation_deformation:
      aux["per_point_deformation_loss"] = torch.mean(spp_sq)
    else:
      aux["per_point_deformation_loss"] = torch.mean(spp_sq[Ellipsis, 3:])

    # Compute metrics about deformation.
    with torch.no_grad():
      aux["per_point_deformation_rotate"] = torch.mean(
          spp_sq[Ellipsis, :3].sum(dim=-1).sqrt() * (180. / np.pi))
      aux["per_point_deformation_translate"] = torch.mean(spp_sq[Ellipsis, 3:])

    return x_transformed, aux

  def forward(self,
              mean,
              deformation_codes,
              *,
              cov=None,
              decayscales,
              render_deformation=True,
              render_color=True):
    """Run MLP. mean is [D, *batch, 3] and cov is [D, *batch, 3, 3]."""
    batch_shape = mean.shape[:-1]
    num_points = np.prod(batch_shape)
    assert mean.shape[-1] == 3

    if render_deformation:
      x, aux_def = self.deformation(mean, deformation_codes,
                                    decayscales["deformation"])
    else:
      x, aux_def = mean, {}
    x = x.reshape(num_points, 3)

    # Density positional embedding.
    emb_density = mask_grid_features(
        self.emb_density(x), decayscales["density"],
        self.emb_density_config["n_features_per_level"])

    # Canonical density network.
    features = self.layers_density(emb_density)
    d, features = features[Ellipsis, :1], features[Ellipsis, 1:]
    assert d.dtype == torch.float32

    # Parameterize canonical frame VolSDF distance as offset from sphere.
    if self.parameterization == "volsdf" and self.base_radius is not None:
      d = d + torch.norm(x, dim=-1, keepdim=True) - self.base_radius

    # Appearance positional embedding.
    emb_color = mask_grid_features(
        self.emb_color(x), decayscales["color"],
        self.emb_color_config["n_features_per_level"])

    # Canonical appearance network.
    if render_color:
      # TODO(jainajay): Concatenate density embedding?
      raw_rgb = self.layers_color(torch.cat([emb_color, features], dim=-1))
      raw_output = torch.cat([raw_rgb, d], dim=-1)
      return raw_output.view(*batch_shape, 4), aux_def

    return d.view(*batch_shape), aux_def


def mask_grid_features(features, num_levels_to_mask, n_features_per_level):
  """For coarse-to-fine, zero out higher resolution features."""
  num_levels_to_mask = int(num_levels_to_mask)
  if num_levels_to_mask == 0:
    return features

  original_shape = features.shape
  batch_shape = original_shape[:-1]

  # Assuming features are stored as L0, L0, L1, L1, ...
  features = features.view(*batch_shape, -1, n_features_per_level)
  features = F.pad(
      features[Ellipsis, :-num_levels_to_mask, :], (0, 0, 0, num_levels_to_mask),
      mode="constant",
      value=0.)

  features = features.view(*batch_shape, -1)
  assert features.shape == original_shape
  return features
