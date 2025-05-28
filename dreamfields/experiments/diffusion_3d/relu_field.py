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

"""Neural field that uses a ReLU after trilinearly interpolating features."""

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# pylint: disable=g-import-not-at-top
try:
  import tinycudann as tcnn
except ImportError as e:
  print("WARN: Unable to import tinycudann. Error:", e)
# pylint: enable=g-import-not-at-top


class ReLUField(nn.Module):
  """MLP architecture."""

  def __init__(
      self,
      *,
      parameterization = "VolSDF",
      base_radius=1.,  # SDF parameter.
      emb_config,
      seed=1337,
      backend_deformation="FullyFusedMLP",
    ):
    super().__init__()
    self.base_radius = base_radius
    self.parameterization = parameterization
    self.emb = tcnn.Encoding(3, dict(emb_config))
    self.reset_parameters()

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

    x = mean.reshape(num_points, 3)

    # Positional embedding. NOTE(jainajay): decayscales is unused.
    emb = self.emb(x).type(torch.float32)
    emb_density, emb_color = emb[Ellipsis, 3:], emb[Ellipsis, :3]

    # Parameterize canonical frame VolSDF distance as offset from sphere.
    if self.parameterization == "volsdf" and self.base_radius is not None:
      # TODO(jainajay): explicitly redistance?
      d = emb_density + torch.norm(x, dim=-1, keepdim=True) - self.base_radius
    else:
      d = F.relu(emb_density)
    assert d.dtype == torch.float32

    # Canonical appearance network.
    if render_color:
      # TODO(jainajay): Concatenate density embedding?
      raw_rgb = F.relu(emb_color)
      raw_output = torch.cat([raw_rgb, d], dim=-1)
      return raw_output.view(*batch_shape, 4)

    return d.view(*batch_shape)
