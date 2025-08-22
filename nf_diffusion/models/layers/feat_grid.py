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

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from nf_diffusion.models.utils import resample


class FeatureGrid(nn.Module):

  """Feature grid (dense) for coordinate embedding."""

  dim: int  # Feature dimension of the grid.
  res: Sequence[int] = (32, 32)
  # TODO(guandao) double-check the initialization schema
  features_init_std: float = 1.0
  edge_behavior: str = "CONSTANT_OUTSIDE"
  coordinate_order: str = "xy"

  @nn.compact
  def __call__(self, x):
    """Extract feature from the grid at corresponding location.

    Args:
      x: is (..., len(self.res)) shape tensor, range from [-0.5, 0.5], the
        location where the feature should be extracted.

    Returns:
      [out] is a (..., self.dim) tensor for the locatoin [x]
      [rel_x] is a (..., len(self.res)) tensor, range from [-0.5, 0.5], being
        the relative coordinate (to the local feature cell) of location [x]
    """
    features = self.param(
        "features",
        # Initialization function
        nn.initializers.normal(self.features_init_std),
        (1, *self.res, self.dim),
    )  # shape info.

    # Since the resampler functions assumed locations are in integers.
    resolution = jnp.broadcast_to(jnp.array(self.res), x.shape)
    locations = (x + 0.5) * (resolution - 1)

    if len(self.res) == 2:
      out = resample.resample_2d(
          features,
          locations,
          edge_behavior=self.edge_behavior,
          coordinate_order=self.coordinate_order,
      )
    elif len(self.res) == 3:
      out = resample.resample_3d(
          features,
          locations,
          edge_behavior=self.edge_behavior,
          coordinate_order=self.coordinate_order,
      )
    else:
      raise ValueError(f"Invalid dimension: {self.res}")

    # TODO(guandao) return the relative coordinates
    rel_x = None

    return out, rel_x
