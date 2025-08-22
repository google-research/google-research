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

# pylint: skip-file
"""NeRF and its MLPs, with helper functions for construction and rendering."""

from collections import namedtuple
import dataclasses
import functools
import operator
import time
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
from google_research.yobo.internal import image
from google_research.yobo.internal import render
from google_research.yobo.internal import utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


@gin.configurable
class VolumeIntegrator(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""

  config: Any = None  # A Config class, must be set upon construction.
  bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.

  def setup(self):
    pass

  @nn.compact
  def __call__(
      self,
      rng,
      shader_results,
      train_frac = 1.0,
      train = True,
      percentiles = (5, 50, 95),
      linear_rgb = False,
      compute_extras = False,
      **kwargs
  ):
    # Define or sample the background color for each ray.
    random_background = False

    if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
      # If the min and max of the range are equal, just take it.
      bg_rgbs = self.bg_intensity_range[0]
    elif rng is None:
      # If rendering is deterministic, use the midpoint of the range.
      bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
    else:
      random_background = True

      # Sample RGB values from the range for each ray.
      key, rng = utils.random_split(rng)
      # bg_rgbs = random.uniform(
      #     key,
      #     shape=shader_results['weights'].shape[:-1] + (3,),
      #     minval=self.bg_intensity_range[0],
      #     maxval=self.bg_intensity_range[1],
      # )
      bg_rgbs = random.normal(
          key,
          shape=shader_results['weights'].shape[:-1] + (3,),
      ) * (self.bg_intensity_range[1] - self.bg_intensity_range[0])

    # Render each ray.
    extras_to_render = [
        # Lighting
        'lighting_irradiance',
        'lighting_emission',
        'integrated_multiplier_specular',
        'integrated_multiplier_diffuse',
        'integrated_multiplier_irradiance',
        'integrated_energy',
        'predicted_energy',
        # Material
        'material_total_albedo',
        'material_residual_albedo',
        'material_albedo',
        'material_roughness',
        'material_F_0',
        'material_metalness',
        'material_diffuseness',
        # Geometry
        'normals',
        'normals_pred',
        'point_offset',
        'point_offset_contract',
    ]

    # print(shader_results['weights'].shape)
    # shader_results['weights'] = jnp.ones_like(
    #     shader_results['weights']
    # ) / shader_results['weights'].shape[-1]

    rendering = render.volumetric_rendering(
        shader_results['rgb'],
        shader_results['weights'],
        shader_results['tdist'],
        bg_rgbs,
        compute_extras,
        extras={
            k: v for k, v in shader_results.items() if k in extras_to_render
        },
        percentiles=percentiles,
    )

    if random_background:
      rendering['bg_noise'] = (
          1.0 - shader_results['weights'].sum(axis=-1, keepdims=True)
      ) * bg_rgbs
      rendering['rgb'] = rendering['rgb'] - rendering['bg_noise']
    else:
      rendering['bg_noise'] = (
          1.0 - shader_results['weights'].sum(axis=-1, keepdims=True)
      ) * 0.0

    # Linear to srgb
    if not linear_rgb and (
        self.config.linear_to_srgb and rendering['rgb'] is not None
    ):
      rendering['rgb'] = jnp.clip(
          image.linear_to_srgb(rendering['rgb']),
          0.0,
          float('inf'),
          # 1.0
      )

    return rendering


@gin.configurable
class GeometryVolumeIntegrator(VolumeIntegrator):
  """A mip-Nerf360 model containing all MLPs."""

  config: Any = None  # A Config class, must be set upon construction.
  bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.

  def setup(self):
    pass

  @nn.compact
  def __call__(
      self,
      rng,
      sampler_results,
      train_frac = 1.0,
      train = True,
      **kwargs
  ):
    # Geometry buffers
    extras_to_render = [
        'normals_to_use',
        'normals',
        'normals_pred',
        'feature',
        'means',
        'covs',
    ]

    # Reshape covariance
    sampler_results['covs'] = sampler_results['covs'].reshape(
        sampler_results['covs'].shape[:-2] + (9,)
    )

    rendering = render.volumetric_rendering(
        sampler_results['means'],
        sampler_results['weights'],
        sampler_results['tdist'],
        0.0,
        True,
        extras={
            k: v for k, v in sampler_results.items() if k in extras_to_render
        },
        normalize_weights_for_extras=False,
    )

    del rendering['rgb']

    # Reshape covariance again
    sampler_results['covs'] = sampler_results['covs'].reshape(
        sampler_results['covs'].shape[:-1] + (3, 3)
    )
    rendering['covs'] = rendering['covs'].reshape(
        rendering['covs'].shape[:-1] + (3, 3)
    )

    # Reshape all
    rendering = jax.tree_util.tree_map(lambda x: x[Ellipsis, None, :], rendering)

    return rendering
