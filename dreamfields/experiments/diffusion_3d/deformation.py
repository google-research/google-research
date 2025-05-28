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

"""Predict and apply deformations, mapping from an obs frame to canonical."""

from typing import Any, Mapping, Tuple

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


def get_deformation(
    screw_axis,
    # Rotation params.
    with_rotation = True,
    fix_axis_vertical = False,
    # Scaling params.
    with_isotropic_scaling = False,
    min_scale = 0.5,
    max_scale = 1.5,
):
  """Get screw axis encoding of per-point rigid transformation.

  Args:
    screw_axis: [*batch, 6]. Per point, contains 3-dim rotation $r in so(3)$ and
      3 dimensions related to the translation, $v$.
    with_rotation:
    fix_axis_vertical:
    with_isotropic_scaling:
    min_scale:
    max_scale: float. Maximum scale. The scale is activated with a sigmoid
      between 0 and max_scale. Note that this affects the scale at a 0 logit.
      With max_scale=2, the zero-logit scaling is 1.

  Returns:
    er: [*batch, 3, 3]. Rotation matrix $e^r in SO(3)$ where r is the
    unnormalized axis of rotation.
    p: [*batch, 3]. Translation component of screw motion.
    with_isotropic_scaling (bool): Whether to scale points within rotation
    matrix.
  """
  identity = torch.eye(3, dtype=torch.float, device=screw_axis.device)

  if with_rotation:
    # Compute angle of rotation.
    norm = torch.linalg.norm(
        screw_axis[Ellipsis, :3], dim=-1, keepdim=True)  # [*batch, 1].
    theta = norm.unsqueeze(-1)  # [*batch, 1, 1].
    thetasq = torch.square(theta)
    thetacubed = thetasq * theta

    # Get axis of rotation and compute $[r]_x$ and $[r]_x^2$.
    axis = screw_axis[Ellipsis, :3] / norm
    tx1, tx2, tx3 = axis[Ellipsis, 0], axis[Ellipsis, 1], axis[Ellipsis, 2]  # [*batch].

    if fix_axis_vertical:
      tx1 = torch.zeros_like(tx1)
      tx3 = torch.zeros_like(tx3)

    rx = neural_fields.cross_product_matrix(tx1, tx2, tx3)  # [*batch, 3, 3].
    rxsq = torch.matmul(rx, rx)

    # Get exponentiated axis of rotation.
    # Nerfies equation 9, Rodrigues' formula.
    er = (
        identity + torch.sin(theta) / theta * rx +
        (1 - torch.cos(theta)) / thetasq * rxsq)

    # Get translation vector.
    # Nerfies equation 3 or 11.
    g = (
        identity + (1 - torch.cos(theta)) / thetasq * rx +
        (theta - torch.sin(theta)) / thetacubed * rxsq)
    p = torch.matmul(g, screw_axis[Ellipsis, 3:6, None]).squeeze(-1)
  else:
    er = identity
    p = screw_axis[Ellipsis, 3:6]

  if with_isotropic_scaling:
    scale = (
        torch.sigmoid(screw_axis[Ellipsis, 6]) * (max_scale - min_scale) + min_scale)
    new_dims = (1,) * scale.ndim
    scale_matrix = (
        identity.view(*new_dims, 3, 3) * scale.view(*scale.shape, 1, 1))
    er = torch.matmul(er, scale_matrix)

  return er, p


class Deformation(nn.Module):
  """Implements a simplified version of the Nerfies deformation network."""

  def __init__(
      self,
      *,
      activation="ReLU",
      deformation_code_dim,
      n_features,
      n_layers,
      rotation = True,
      embedding_type = "grid_feature",  # grid_feature|fourier_feature_ipe.
      grid_config,
      fourfeat_config,
      backend_deformation="FullyFusedMLP",
      seed=1337,
  ):
    super().__init__()
    self.rotation = rotation
    self.embedding_type = embedding_type

    # Positional embedding.
    if embedding_type == "grid_feature":
      self.grid_n_features_per_level = grid_config["n_features_per_level"]
      self.emb = tcnn.Encoding(3, dict(grid_config))
      embedding_dim = grid_config["n_levels"] * self.grid_n_features_per_level
    elif embedding_type == "fourier_feature_ipe":
      self.emb = neural_fields.FourierFeatureIPE(
          num_dims_to_encode=3,
          **fourfeat_config,
          dtype=torch.float32,
      )
      embedding_dim = self.emb.output_dim
    else:
      raise ValueError

    # Project deformation codes.
    self.project_deformation_codes = nn.Linear(deformation_code_dim, n_features)

    # Network mapping codes + embedding to a screw axis.
    if n_layers > 0:
      trunk_deformation = neural_fields.WrappedNetwork(
          embedding_dim + n_features,
          n_features,
          {
              "otype": backend_deformation,
              "activation": activation,
              "output_activation": activation,
              "n_neurons": n_features,
              "n_hidden_layers": n_layers
          },
          seed=seed,
          computation_dtype=None,
          output_dtype=torch.float32,
      )
      self.layers_deformation = nn.Sequential(
          trunk_deformation,
          nn.Linear(n_features, 6),
      )
    else:
      self.layers_deformation = nn.Sequential(
          nn.ReLU(),
          nn.Linear(embedding_dim + n_features, 6),
      )

    self.reset_parameters()

  def reset_parameters(self):
    # Initialize last layer of deformations near zero (Sec 3.2, nerfies).
    torch.nn.init.uniform_(self.layers_deformation[-1].weight, a=-1e-6, b=1e-6)

  def forward(self,
              x,
              deformation_codes,
              decayscale,
              enabled = True):
    """Per-point deformation network.

    Does not use IPE, but can use decayscale.

    Args:
      x: Ray coordinates, in observation frame,
      deformation_codes: Latent variables describing deformations.
      decayscale: Coarse-to-fine coefficient.
      enabled: Transform the points if True, identity if False.

    Returns:
      x_transformed: Transformed ray coordinates, in canonical frame.
      aux: Dictionary of metrics and losses.
    """
    if not enabled:
      return x, {}

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

    # Positional encoding, updated with projected deformation codes.
    if self.embedding_type == "grid_feature":
      deformation_emb = self.emb(x_flat)
      deformation_emb = neural_fields.mask_grid_features(
          deformation_emb,
          decayscale,
          n_features_per_level=self.grid_n_features_per_level)
    elif self.embedding_type == "fourier_feature_ipe":
      assert isinstance(self.emb, neural_fields.FourierFeatureIPE)
      deformation_emb = self.emb(x_flat, 1e-4, decayscale)  # [*batch, emb_dim].
    else:
      raise ValueError

    # Predict deformation as a screw axis.
    deformation_input = torch.cat([deformation_emb, dc_proj], dim=-1)
    s_per_point = self.layers_deformation(deformation_input)  # [*batch, 6].
    s_per_point = s_per_point.view(*batch_shape, 6)  # Reshape back.
    assert s_per_point.dtype == torch.float32

    # Compute rotation matrix and translation.
    # er is [*batch, 3, 3], p is [*batch, 3].
    er, p = get_deformation(s_per_point, with_rotation=self.rotation)

    # Apply deformation.
    x_transformed = torch.matmul(
        er, x.unsqueeze(-1)).squeeze(-1) + p  # [*batch, 3].

    # Compute losses to regularize deformation.
    spp_sq = torch.square(s_per_point)
    aux = {}
    if self.rotation:
      aux["per_point_deformation_loss"] = torch.mean(spp_sq)
    else:
      aux["per_point_deformation_loss"] = torch.mean(spp_sq[Ellipsis, 3:])

    # Compute metrics about deformation.
    with torch.no_grad():
      aux["per_point_deformation_rotate"] = torch.mean(
          spp_sq[Ellipsis, :3].sum(dim=-1).sqrt() * (180. / np.pi))
      aux["per_point_deformation_translate"] = torch.mean(spp_sq[Ellipsis, 3:])

    return x_transformed, aux
