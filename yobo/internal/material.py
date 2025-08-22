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
from google_research.yobo.internal import coord
from google_research.yobo.internal import geopoly
from google_research.yobo.internal import grid_utils
from google_research.yobo.internal import math
from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import shading
from google_research.yobo.internal import utils
from google_research.yobo.internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


gin.config.external_configurable(math.abs, module='math')
gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(math.power_3, module='math')
gin.config.external_configurable(math.laplace_cdf, module='math')
gin.config.external_configurable(math.scaled_softplus, module='math')
gin.config.external_configurable(math.power_ladder, module='math')
gin.config.external_configurable(math.inv_power_ladder, module='math')
gin.config.external_configurable(coord.contract, module='coord')
gin.config.external_configurable(coord.contract_constant, module='coord')
gin.config.external_configurable(coord.contract_constant_squash, module='coord')
gin.config.external_configurable(
    coord.contract_constant_squash_small, module='coord'
)
gin.config.external_configurable(coord.contract_cube, module='coord')
gin.config.external_configurable(coord.contract_cube, module='coord')
gin.config.external_configurable(coord.contract_projective, module='coord')


@gin.configurable
class MaterialMLP(shading.BaseShader):
  """A PosEnc MLP."""

  config: Any = None

  random_generator_2d: Any = render_utils.RandomGenerator2D(1, 1, False)
  num_secondary_samples: int = 8
  render_num_secondary_samples: int = 8
  importance_sampler_configs: Any = (
      ('microfacet', 1),
      ('cosine', 1),
  )
  render_importance_sampler_configs: Any = (
      ('microfacet', 1),
      ('cosine', 1),
  )
  use_mis: bool = True
  material_type: str = 'microfacet'
  stratified_sampling: bool = False

  bias_albedo: float = -1.0  # The shift added to raw colors pre-activation.
  bias_roughness: float = -1.0  # The shift added to raw colors pre-activation.

  brdf_bias: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({
      'albedo': 0.0,
      'specular_albedo': 2.0,
      'roughness': -1.0,
      'F_0': 1.0,
      'metalness': -1.0,
      'diffuseness': 0.0,
  })
  brdf_activation: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({
          'albedo': jax.nn.sigmoid,
          'specular_albedo': jax.nn.sigmoid,
          'roughness': jax.nn.sigmoid,
          'F_0': jax.nn.sigmoid,
          'metalness': jax.nn.sigmoid,
          'diffuseness': jax.nn.sigmoid,
      })
  )

  rgb_emission_activation: Callable[Ellipsis, Any] = (
      nn.sigmoid
  )  # The RGB activation.
  rgb_bias_emission: float = (
      -1.0
  )  # The shift added to raw colors pre-activation.

  rgb_residual_albedo_activation: Callable[Ellipsis, Any] = (
      nn.sigmoid
  )  # The RGB activation.
  rgb_bias_residual_albedo: float = (
      -1.0
  )  # The shift added to raw colors pre-activation.

  use_brdf_correction: bool = (
      True  # Use brdf to weight secondary ray contribution
  )
  anisotropic_brdf_correction: bool = (
      False  # Use brdf to weight secondary ray contribution
  )

  use_diffuse_emission: bool = False  # If True, use diffuse emission
  emission_window_frac: float = 0.1  # If True, use diffuse emission
  emission_variate_weight_start: float = 1.0  # If True, use diffuse emission
  emission_variate_weight_end: float = 1.0  # If True, use diffuse emission

  net_width_brdf: int = 64  # Learned BRDF layer width
  net_depth_brdf: int = 2  #  Learned BRDF layer depth
  deg_brdf: int = 2  # Degree of encoding for mlp BRDF output
  deg_brdf_anisotropic: int = 2  # Degree of encoding for mlp BRDF output

  stopgrad_secondary_rgb: bool = (
      True  # Stop gradients from flowing through cache sampling
  )
  stopgrad_secondary_rgb_weight: float = (
      1.0  # Stop gradients from flowing through cache sampling
  )

  filter_backfacing: bool = True
  use_mesh_points: bool = True
  use_mesh_points_for_prediction: bool = True
  use_mesh_normals: bool = True

  use_corrected_normals: bool = True
  enable_normals_offset: bool = False  # If True compute predicted normals.
  normals_target: str = 'normals_to_use'

  stopgrad_normals: bool = False
  stopgrad_normals_weight: float = (
      1.0  # Stop gradients from flowing through cache sampling
  )

  stopgrad_samples: bool = True
  stopgrad_material: bool = True
  stopgrad_light: bool = True

  near_min: float = 1e-2
  near_max: float = 1e-1

  def setup(self):
    self.dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    self.layers = [
        self.dense_layer(self.net_width) for i in range(self.net_depth)
    ]

    self.bottleneck_layer = self.dense_layer(self.bottleneck_width)

    # Emission
    if self.use_diffuse_emission:
      self.rgb_diffuse_emission_layer = self.dense_layer(self.num_rgb_channels)

      self.rgb_residual_albedo_layer = self.dense_layer(self.num_rgb_channels)

    # BRDF
    if self.material_type == 'microfacet':
      self.pred_brdf_layer = self.dense_layer(10)  # microfacet
    elif self.material_type == 'phong':
      self.pred_brdf_layer = self.dense_layer(7)  # phong
    elif self.material_type == 'lambertian':
      self.pred_brdf_layer = self.dense_layer(3)  # lambertian

    if self.use_brdf_correction:

      def brdf_enc_fn(direction):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_brdf, append_identity=True
        )

      self.brdf_enc_fn = brdf_enc_fn

      def brdf_enc_fn_anisotropic(direction):
        return coord.pos_enc(
            direction,
            min_deg=0,
            max_deg=self.deg_brdf_anisotropic,
            append_identity=True,
        )

      self.brdf_enc_fn_anisotropic = brdf_enc_fn_anisotropic

      self.brdf_correction_layers = [
          self.dense_layer(self.net_width_brdf)
          for i in range(self.net_depth_brdf)
      ]
      self.output_brdf_correction_layer = self.dense_layer(2)

    # Grid
    if self.use_grid:
      self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
          self.grid_representation.lower()
      ](name='grid', **self.grid_params)
    else:
      self.grid = None

    # Predicted normals
    self.zeros_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, 'zeros')
    )
    self.pred_normals_layer = self.zeros_layer(3)

    # Light sampling
    self.importance_samplers = []

    for conf in self.importance_sampler_configs:
      self.importance_samplers = self.importance_samplers + (
          (render_utils.IMPORTANCE_SAMPLER_BY_NAME[conf[0]](), conf[1]),
      )

    self.render_importance_samplers = []

    for conf in self.render_importance_sampler_configs:
      self.render_importance_samplers = self.render_importance_samplers + (
          (render_utils.IMPORTANCE_SAMPLER_BY_NAME[conf[0]](), conf[1]),
      )

  def get_brdf_correction(self, x_input, ref_samples, train):
    num_secondary_samples = self.get_num_secondary_samples(train)

    # Direction dependent inputs
    brdf_input = jnp.concatenate(
        [
            jnp.broadcast_to(
                ref_samples['local_viewdirs'][Ellipsis, 2:3],
                ref_samples['local_lightdirs'].shape[:-1] + (1,),
            ),
            ref_samples['local_lightdirs'][Ellipsis, 2:3],
        ],
        axis=-1,
    )
    brdf_input = jnp.concatenate(
        [
            jnp.sort(brdf_input, axis=-1),
            math.dot(
                ref_samples['local_viewdirs'],
                ref_samples['local_lightdirs'],
            ),
        ],
        axis=-1,
    )

    # Encode inputs
    brdf_input = self.brdf_enc_fn(brdf_input)

    # Anisotropic correction
    if self.anisotropic_brdf_correction:
      brdf_input_anisotropic = jnp.concatenate(
          [
              (
                  ref_samples['global_viewdirs']
                  + ref_samples['global_lightdirs']
              ),
              jnp.abs(
                  ref_samples['global_viewdirs']
                  - ref_samples['global_lightdirs']
              ),
          ],
          axis=-1,
      )
      brdf_input_anisotropic = self.brdf_enc_fn_anisotropic(
          brdf_input_anisotropic
      )
      brdf_input = jnp.concatenate(
          [brdf_input, brdf_input_anisotropic], axis=-1
      )

    # Position dependent inputs
    brdf_input = jnp.concatenate(
        [
            brdf_input,
            jnp.repeat(
                x_input.reshape(-1, 1, x_input.shape[-1]),
                num_secondary_samples,
                axis=-2,
            ),
        ],
        axis=-1,
    )

    # Run network
    x = brdf_input

    for i in range(self.net_depth_brdf):
      x = self.brdf_correction_layers[i](x)
      x = self.net_activation(x)

    x = self.output_brdf_correction_layer(x)

    # Output
    brdf_correction = jnp.concatenate(
        [
            nn.sigmoid(x[Ellipsis, 0:1] - jnp.log(3.0)),
            nn.sigmoid(x[Ellipsis, 1:2] + jnp.log(3.0)),
        ],
        axis=-1,
    )

    return brdf_correction

  def get_importance_samplers(self, train):
    if train:
      return self.importance_samplers
    else:
      return self.render_importance_samplers

  def get_num_secondary_samples(self, train):
    if train:
      return self.num_secondary_samples
    else:
      return min(self.render_num_secondary_samples, self.num_secondary_samples)

  def get_material(self, brdf_params):
    if self.material_type == 'microfacet':
      material = {
          'albedo': self.brdf_activation['albedo'](
              brdf_params[Ellipsis, 0:3] + self.brdf_bias['albedo']
          ),
          'specular_albedo': self.brdf_activation['specular_albedo'](
              brdf_params[Ellipsis, 3:6] + self.brdf_bias['specular_albedo']
          ),
          'roughness': self.brdf_activation['roughness'](
              brdf_params[Ellipsis, 6:7] + self.brdf_bias['roughness']
          ),
          'F_0': self.brdf_activation['F_0'](
              brdf_params[Ellipsis, 7:8] + self.brdf_bias['F_0']
          ),
          'metalness': self.brdf_activation['metalness'](
              brdf_params[Ellipsis, 8:9] + self.brdf_bias['metalness']
          ),
          'diffuseness': self.brdf_activation['diffuseness'](
              brdf_params[Ellipsis, 9:10] + self.brdf_bias['diffuseness']
          ),
      }  # microfacet
    elif self.material_type == 'phong':
      material = {
          'albedo': jax.nn.sigmoid(brdf_params[Ellipsis, 0:3]),
          'specular_albedo': jax.nn.sigmoid(brdf_params[Ellipsis, 3:6]),
          'specular_exponent': math.safe_exp(brdf_params[Ellipsis, 6:7] - 0.5),
      }  # phong
    elif self.material_type == 'lambertian':
      material = {
          'albedo': jax.nn.sigmoid(brdf_params[Ellipsis, 0:3]),
      }  # lambertian
    else:
      material = {}

    return material

  def correct_geometry(
      self,
      rays,
      feature,
      sampler_results,
  ):
    # Points
    if 'mesh_points' in sampler_results and self.use_mesh_points_for_prediction:
      sampler_results = jax.tree_util.tree_map(lambda x: x, sampler_results)
      sampler_results['means'] = sampler_results['mesh_points']
      sampler_results['points'] = sampler_results['mesh_points']

    # Predict normals
    normals = sampler_results[self.normals_target]

    if self.enable_normals_offset:
      normals_offset = self.pred_normals_layer(feature)

      normals = normals + normals_offset
      normals = ref_utils.l2_normalize(normals)

    if self.use_corrected_normals:
      normals = jnp.where(
          math.dot(normals, rays.viewdirs[Ellipsis, None, :]) < 0, normals, -normals
      )

    if 'mesh_normals' in sampler_results and self.use_mesh_normals:
      normals = sampler_results['mesh_normals']

    sampler_results[self.normals_target] = normals
    return sampler_results

  def get_outgoing_radiance(
      self,
      rng,
      rays,
      feature,
      sampler_results,
      material,
      train_frac=1.0,
      train = True,
      mesh=None,
      radiance_cache=None,
      light_rotation=None,
      light_sampler_results=None,
      **kwargs,
  ):
    # Get reflected rays
    samplers = self.get_importance_samplers(train)
    num_secondary_samples = self.get_num_secondary_samples(train)

    # Stop gradient
    normals = sampler_results[self.normals_target]

    if self.stopgrad_normals:
      normals = jax.lax.stop_gradient(normals)
    else:
      normals = normals * self.stopgrad_normals_weight + jax.lax.stop_gradient(
          normals
      ) * (1.0 - self.stopgrad_normals_weight)

    if self.stopgrad_material:
      material_for_secondary = jax.lax.stop_gradient(material)
    else:
      material_for_secondary = material

    if self.stopgrad_light:
      light_for_secondary = jax.lax.stop_gradient(light_sampler_results)
    else:
      light_for_secondary = light_sampler_results

    key, rng = utils.random_split(rng)
    ref_rays, ref_samples = render_utils.get_secondary_rays(
        key,
        rays,
        sampler_results['points'],
        rays.viewdirs,
        normals,
        material_for_secondary,
        refdir_eps=self.near_min,
        random_generator_2d=self.random_generator_2d,
        stratified_sampling=self.stratified_sampling,
        use_mis=self.use_mis,
        samplers=samplers,
        num_secondary_samples=num_secondary_samples,
        light_rotation=light_rotation,
        light_sampler_results=light_for_secondary,
        offset_origins=True,
    )

    if self.stopgrad_samples:
      ref_samples = jax.lax.stop_gradient(ref_samples)

    # Query radiance cache
    key, rng = utils.random_split(rng)
    ref_ray_outputs = radiance_cache(
        key,
        ref_rays,
        train_frac=train_frac,
        train=train,
        compute_extras=False,
        zero_glo=(
            'glo_vec' not in sampler_results
            or sampler_results['glo_vec'] is None
        ),
        mesh=mesh,
        stopgrad_proposal=True,
        stopgrad_weights=False,
        zero_backfacing=True,
        linear_rgb=True,
    )

    rgb = ref_ray_outputs['render']['rgb']

    if self.stopgrad_secondary_rgb:
      rgb = jax.lax.stop_gradient(rgb)
    else:
      rgb = rgb * self.stopgrad_secondary_rgb_weight + jax.lax.stop_gradient(
          rgb
      ) * (1.0 - self.stopgrad_secondary_rgb_weight)

    # Integrate multiple reflected rays
    rgb = rgb.reshape(-1, num_secondary_samples, 3)

    ref_samples = jax.tree_util.tree_map(
        lambda x: x.reshape(rgb.shape[0], -1, x.shape[-1]), ref_samples
    )

    # BRDF correction
    brdf_correction = jnp.concatenate(
        [
            jnp.ones_like(rgb[Ellipsis, 0:1]),
            jnp.zeros_like(rgb[Ellipsis, 0:1]),
        ],
        axis=-1,
    )

    if self.use_brdf_correction:
      brdf_correction = self.get_brdf_correction(feature, ref_samples, train)

    # Integrate
    ref_samples['radiance_in'] = rgb
    ref_samples['brdf_correction'] = brdf_correction

    integrated_outputs = render_utils.integrate_reflect_rays(
        self.material_type,
        self.use_brdf_correction,
        material,
        ref_samples,
    )

    sh = sampler_results['points'].shape
    app_mask = jnp.ones_like(sampler_results['points'][Ellipsis, :1])

    if 't_to_nearest' in sampler_results and self.filter_backfacing:
      app_mask = sampler_results['t_to_nearest'] > -0.1

    integrated_outputs = jax.tree_util.tree_map(
        lambda x: x.reshape(sh[:-1] + (x.shape[-1],)) * app_mask,
        integrated_outputs,
    )

    return integrated_outputs, ref_rays, ref_samples

  def predict_bottleneck_feature(
      self,
      rng,
      rays,
      sampler_results,
      train = True,
  ):
    # Appearance feature
    key, rng = utils.random_split(rng)
    predict_appearance_kwargs = self.get_predict_appearance_kwargs(
        key,
        rays,
        sampler_results,
    )

    feature = self.predict_appearance_feature(
        sampler_results,
        train=train,
        **predict_appearance_kwargs,
    )

    if self.bottleneck_width > 0:
      key, rng = utils.random_split(rng)
      feature = self.get_bottleneck_feature(key, feature)

    return feature

  def predict_appearance(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      **kwargs,
  ):
    outputs = {}

    # Bottleneck feature
    key, rng = utils.random_split(rng)
    feature = self.predict_bottleneck_feature(
        rng=key,
        rays=rays,
        sampler_results=sampler_results,
        train=train,
    )

    # Correct geometry
    sampler_results = self.correct_geometry(
        rays=rays, feature=feature, sampler_results=sampler_results
    )

    # BRDF
    brdf_params = self.pred_brdf_layer(feature)
    material = self.get_material(brdf_params)

    # Integrate
    key, rng = utils.random_split(rng)
    (integrated_outputs, ref_rays, ref_samples) = self.get_outgoing_radiance(
        rng=key,
        rays=rays,
        feature=feature,
        sampler_results=sampler_results,
        material=material,
        train_frac=train_frac,
        train=train,
        **kwargs,
    )
    rgb = integrated_outputs['radiance_out']

    # Reshape material
    material = jax.tree_util.tree_map(
        lambda x: x.reshape(rgb.shape[:-1] + (x.shape[-1],)), material
    )

    # Combine diffuse and specular
    emission = jnp.zeros_like(rgb)
    residual_albedo = jnp.zeros_like(rgb)

    # Emission
    if self.use_diffuse_emission:
      # Residual albedo & emission
      residual_albedo = self.rgb_residual_albedo_activation(
          self.rgb_premultiplier * self.rgb_residual_albedo_layer(feature)
          + self.rgb_bias_residual_albedo
      )

      emission = self.rgb_emission_activation(
          self.rgb_premultiplier * self.rgb_diffuse_emission_layer(feature)
          + self.rgb_bias_emission
      )

      if True:
        if self.emission_window_frac > 0.0:
          w = jnp.clip(train_frac / self.emission_window_frac, 0.0, 1.0)
        else:
          w = 1.0

        emission_variate_weight = (
            (1.0 - w) * self.emission_variate_weight_start
            + w * self.emission_variate_weight_end
        )

        rgb = rgb + (
            emission * emission_variate_weight
            + jax.lax.stop_gradient(emission) * (1.0 - emission_variate_weight)
        )
      else:
        rgb = rgb + residual_albedo * integrated_outputs['irradiance']

    # Secondary ray outputs
    if 'light_sampling' in self.config.extra_losses:
      outputs['ref_rays'] = ref_rays
      outputs['ref_samples'] = ref_samples

    # Material outputs
    outputs['material_residual_albedo'] = residual_albedo

    if not train:
      outputs['material_total_albedo'] = (
          material['albedo']
          * integrated_outputs['integrated_multiplier'][Ellipsis, 1:2]
          * material['diffuseness']
          + residual_albedo
      )

      for k in material.keys():
        outputs['material_' + k] = material[k] * jnp.ones_like(
            material['albedo']
        )

    # Emission outputs
    outputs['lighting_emission'] = emission
    outputs['lighting_irradiance'] = integrated_outputs['irradiance']
    (outputs['integrated_multiplier_irradiance']) = integrated_outputs[
        'integrated_multiplier_irradiance'
    ] * jnp.ones_like(rgb)

    if not train:
      (outputs['integrated_multiplier_specular']) = integrated_outputs[
          'integrated_multiplier'
      ][Ellipsis, 0:1] * jnp.ones_like(rgb)

      (outputs['integrated_multiplier_diffuse']) = integrated_outputs[
          'integrated_multiplier'
      ][Ellipsis, 1:2] * jnp.ones_like(rgb)

    # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
    rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
    outputs['rgb'] = rgb

    # Return
    return outputs
