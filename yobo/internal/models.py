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
from google_research.yobo.internal import configs
from google_research.yobo.internal import integration
from google_research.yobo.internal import light_sampler
from google_research.yobo.internal import material
from google_research.yobo.internal import math
from google_research.yobo.internal import sampling
from google_research.yobo.internal import shading
from google_research.yobo.internal import utils
from google_research.yobo.internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


@gin.configurable
class Model(nn.Module):
  config: Any = None  # A Config class, must be set upon construction.

  # Random generator
  random_generator_2d: Any = render_utils.RandomGenerator2D(1, 1, False)

  # Importance samplers
  uniform_importance_samplers: Any = (
      (render_utils.UniformHemisphereSampler(), 1.0),
  )
  light_importance_samplers: Any = (
      # ( render_utils.UniformHemisphereSampler(), 0.5 ),
      # ( render_utils.LightSampler(), 0.5 ),
      (render_utils.UniformHemisphereSampler(), 1.0),
  )
  distance_importance_samplers: Any = (
      (render_utils.UniformHemisphereSampler(), 1.0),
  )

  def filter_sampler_results(
      self,
      rng,
      sampler_results,
      num_resample,
      uniform=False,
  ):
    if rng is None:
      rng = random.PRNGKey(1)

    num_samples = sampler_results['weights'].shape[-1]

    key, rng = utils.random_split(rng)
    logits = jnp.repeat(
        math.safe_log(sampler_results['weights'][Ellipsis, None]), 1, axis=-1
    )
    inds = jax.random.categorical(
        key,
        logits=logits if not uniform else jnp.ones_like(logits),
        axis=-2,
        shape=(sampler_results['points'].shape[:-2] + (num_resample,)),
    )

    def take_multiple(x, cur_inds=None):
      if cur_inds is None:
        cur_inds = inds

      if isinstance(x, jnp.ndarray):
        if len(x.shape) < len(sampler_results['points'].shape):
          return jnp.take_along_axis(x, cur_inds, axis=-1)
        elif len(x.shape) > len(sampler_results['points'].shape):
          return jnp.take_along_axis(x, cur_inds[Ellipsis, :, None, None], axis=-3)
        else:
          return jnp.take_along_axis(x, cur_inds[Ellipsis, :, None], axis=-2)

      return x

    # Filtered sampler results
    filtered_sampler_results = jax.tree_util.tree_map(
        take_multiple,
        sampler_results,
    )

    if uniform:
      filtered_sampler_results['weights'] = (
          filtered_sampler_results['weights']
      ) * num_samples
    else:
      filtered_sampler_results['weights'] = jnp.ones_like(
          filtered_sampler_results['weights']
      )

    filtered_sampler_results['tdist'] = jnp.concatenate(
        [
            take_multiple(sampler_results['tdist'], inds),
            take_multiple(sampler_results['tdist'], inds + 1),
        ],
        axis=-1,
    )
    filtered_sampler_results['sdist'] = jnp.concatenate(
        [
            take_multiple(sampler_results['sdist'], inds),
            take_multiple(sampler_results['sdist'], inds + 1),
        ],
        axis=-1,
    )
    return filtered_sampler_results, inds

  def scatter_to(self, results, resampled_results, inds, keys):
    for key in results.keys():
      if key not in keys:
        continue

      all_inds = jnp.arange(results[key].shape[-2])
      all_inds = all_inds.reshape(
          tuple(1 for _ in results[key].shape[:-2]) + (-1, 1)
      )

      results[key] = jnp.where(
          (
              jnp.repeat(inds[Ellipsis, None], results[key].shape[-1], axis=-1)
              == all_inds
          ),
          resampled_results[key],
          results[key],
      )

    return results


@gin.configurable
class NeRFModel(Model):
  sampler_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  shader_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  integrator_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  extra_model_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  def setup(self):
    self.sampler = sampling.ProposalVolumeSampler(
        config=self.config,
        **self.sampler_params,
        **self.extra_model_params,
        name='Sampler',
    )

    self.shader = shading.NeRFMLP(
        config=self.config,
        **self.shader_params,
        name='Shader',
    )

    self.integrator = integration.VolumeIntegrator(
        config=self.config,
        **self.integrator_params,
        name='Integrator',
    )

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac = 1.0,
      train = True,
      resample = False,
      num_resample = 1,
      **render_kwargs
  ):
    key, rng = utils.random_split(rng)
    sampler_results = self.sampler(
        rng=key,
        rays=rays,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    if resample:
      key, rng = utils.random_split(rng)
      last_sampler_results, inds = self.filter_sampler_results(
          key,
          sampler_results[-1],
          num_resample,
      )
    else:
      last_sampler_results = sampler_results[-1]

    key, rng = utils.random_split(rng)
    shader_results = self.shader(
        rng=key,
        rays=rays,
        sampler_results=last_sampler_results,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    key, rng = utils.random_split(rng)
    integrator_results = self.integrator(
        rng=key,
        rays=rays,
        shader_results=shader_results,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    return {
        'main': {
            'loss_weight': 1.0,
            'sampler': sampler_results,
            'shader': shader_results,
            'integrator': integrator_results,
        },
        'render': integrator_results,
    }


@gin.configurable
class DeferredNeRFModel(Model):
  sampler_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  shader_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  integrator_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  aux_shader_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  aux_integrator_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  extra_model_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  def setup(self):
    self.sampler = sampling.ProposalVolumeSampler(
        config=self.config,
        **self.sampler_params,
        **self.extra_model_params,
        name='Sampler',
    )

    self.geometry_integrator = integration.GeometryVolumeIntegrator(
        config=self.config,
        **self.integrator_params,
        name='Integrator',
    )

    self.shader = shading.NeRFMLP(
        config=self.config,
        **self.shader_params,
        name='Shader',
    )

    self.aux_shader = shading.NeRFMLP(
        config=self.config,
        **self.aux_shader_params,
        name='AuxShader',
    )

    self.aux_integrator = integration.VolumeIntegrator(
        config=self.config,
        **self.aux_integrator_params,
        name='AuxIntegrator',
    )

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac = 1.0,
      train = True,
      **render_kwargs
  ):
    key, rng = utils.random_split(rng)
    sampler_results = self.sampler(
        rng=key,
        rays=rays,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    key, rng = utils.random_split(rng)
    geometry_integrator_results = self.geometry_integrator(
        rng=key,
        rays=rays,
        sampler_results=sampler_results[-1],
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    key, rng = utils.random_split(rng)
    shader_results = self.shader(
        rng=key,
        rays=rays,
        sampler_results=geometry_integrator_results,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    # Combine
    integrator_results = jax.tree_util.tree_map(
        lambda x: x[Ellipsis, 0, :], shader_results
    )

    # Auxiliary
    key, rng = utils.random_split(rng)
    aux_shader_results = self.aux_shader(
        rng=key,
        rays=rays,
        sampler_results=sampler_results[-1],
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    key, rng = utils.random_split(rng)
    aux_integrator_results = self.aux_integrator(
        rng=key,
        rays=rays,
        shader_results=aux_shader_results,
        train_frac=train_frac,
        train=train,
        **render_kwargs,
    )

    return {
        'main': {
            'loss_weight': 1.0,
            'sampler': sampler_results,
            'shader': shader_results,
            'integrator': integrator_results,
        },
        'aux': {
            'loss_weight': 1.0,
            'sampler': None,
            'shader': aux_shader_results,
            'integrator': aux_integrator_results,
        },
        'render': integrator_results,
    }


@gin.configurable
class MaterialModel(Model):
  use_material: bool = True
  use_light_sampler: bool = True

  stopgrad_samples: bool = False
  resample_material: bool = False
  num_resample_material: int = 1

  render_variate: bool = False

  aux_loss_weight: float = 1.0
  aux_loss: str = 'charb'
  aux_linear_to_srgb: bool = True

  aux_loss_weight_material: float = 1.0
  aux_loss_material: str = 'rawnerf_unbiased'
  aux_linear_to_srgb_material: bool = False

  stopgrad_rgb_weight: float = 0.0
  stopgrad_weights_weight: float = 1.0

  loss_weight: float = 1.0
  loss: str = 'rawnerf_unbiased'
  linear_to_srgb: bool = False

  ## Cache
  cache_model_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  ## Light sampler
  light_sampler_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  ## Material model
  sampler_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  shader_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  integrator_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  ## Extra
  extra_model_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  def setup(self):
    ## Cache
    self.cache = NeRFModel(
        config=self.config,
        **self.cache_model_params,
        **self.extra_model_params,
        name='Cache',
    )

    ## Light sampler
    self.light_sampler = light_sampler.LightMLP(
        config=self.config,
        **self.light_sampler_params,
        name='LightSampler',
    )

    ## Material
    self.shader = material.MaterialMLP(
        config=self.config,
        **self.shader_params,
        name='MaterialShader',
    )

    self.integrator = integration.VolumeIntegrator(
        config=self.config,
        **self.integrator_params,
        name='MaterialIntegrator',
    )

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac = 1.0,
      train = True,
      mesh=None,
      passes=('cache', 'light', 'material'),
      **render_kwargs
  ):
    ## Passes
    use_material = ('material' in passes) and self.use_material
    use_light = ('light' in passes) and (self.use_light_sampler)

    ## Outputs
    outputs = {}

    ## Cache model
    key, rng = utils.random_split(rng)
    cache_results = self.cache(
        rng=key,
        rays=rays,
        train_frac=train_frac,
        train=train,
        mesh=mesh,
        **render_kwargs,
    )

    cache_outputs = {
        'loss_weight': self.aux_loss_weight,
        'loss_type': self.aux_loss,
        'linear_to_srgb': self.aux_linear_to_srgb,
        'sampler': cache_results['main']['sampler'],
        'shader': cache_results['main']['shader'],
        'integrator': cache_results['main']['integrator'],
    }

    outputs['main'] = cache_outputs
    outputs['render'] = cache_outputs['integrator']

    if use_material:
      outputs['cache_main'] = cache_outputs

    ## Cache results
    if self.use_material:
      key, rng = utils.random_split(rng)
      material_cache_results = self.cache(
          rng=key,
          rays=rays,
          train_frac=train_frac,
          train=train,
          mesh=mesh,
          **render_kwargs,
      )
      sampler_results = material_cache_results['main']['sampler']
      shader_results = material_cache_results['main']['shader']
    else:
      sampler_results = jax.tree_util.tree_map(
          lambda x: x,
          cache_results['main']['sampler'],
      )

      shader_results = jax.tree_util.tree_map(
          lambda x: x,
          cache_results['main']['shader'],
      )

    ## Filter
    if (not train) or self.resample_material:
      key, rng = utils.random_split(rng)
      last_sampler_results, inds = self.filter_sampler_results(
          key,
          sampler_results[-1],
          self.num_resample_material,
      )
    else:
      last_sampler_results = sampler_results[-1]
      inds = None

    ## Light importance sampler
    if use_light:
      key, rng = utils.random_split(rng)
      light_sampler_results = self.light_sampler(
          rng=key,
          rays=rays,
          sampler_results=jax.lax.stop_gradient(last_sampler_results),
          train_frac=train_frac,
          train=train,
          mesh=mesh,
          **render_kwargs,
      )
    else:
      light_sampler_results = None

    ## Material
    if use_material:
      if self.stopgrad_samples:
        last_sampler_results = jax.lax.stop_gradient(last_sampler_results)

      key, rng = utils.random_split(rng)
      material_shader_results = self.shader(
          rng=key,
          rays=rays,
          sampler_results=last_sampler_results,
          train_frac=train_frac,
          train=train,
          mesh=mesh,
          radiance_cache=self.cache,
          light_sampler_results=light_sampler_results,
          **render_kwargs,
      )

      key, rng = utils.random_split(rng)
      material_integrator_results = self.integrator(
          rng=key,
          rays=rays,
          shader_results=material_shader_results,
          train_frac=train_frac,
          train=train,
          **render_kwargs,
      )

      # Volume render *with* cache
      if self.render_variate:
        # Sampler results
        if self.shader.enable_normals_offset:
          sampler_results[-1] = self.scatter_to(
              sampler_results[-1],
              material_shader_results,
              inds,
              ['normals', 'normals_pred', 'normals_to_use'],
          )

        # Aux loss
        outputs['material'] = {
            'loss_weight': self.aux_loss_weight_material,
            'loss_type': self.aux_loss_material,
            'linear_to_srgb': self.aux_linear_to_srgb_material,
            'sampler': None,
            'shader': material_shader_results,
            'integrator': material_integrator_results,
        }

        # Shader results
        rgb_weight = self.stopgrad_rgb_weight
        shader_results['rgb'] = (
            jax.lax.stop_gradient(shader_results['rgb']) * (1.0 - rgb_weight)
            + shader_results['rgb'] * rgb_weight
        )

        weights_weight = self.stopgrad_weights_weight
        shader_results['weights'] = (
            jax.lax.stop_gradient(shader_results['weights'])
            * (1.0 - weights_weight)
            + shader_results['weights'] * weights_weight
        )

        shader_results = self.scatter_to(
            shader_results,
            material_shader_results,
            inds,
            ['normals', 'normals_pred', 'normals_to_use', 'rgb'],
        )

        # Integrator results
        key, rng = utils.random_split(rng)
        resampled_integrator_results = self.integrator(
            rng=key,
            rays=rays,
            shader_results=shader_results,
            train_frac=train_frac,
            train=train,
            **render_kwargs,
        )

        material_integrator_results = jax.tree_util.tree_map(
            lambda x: x,
            material_integrator_results,
        )

        material_integrator_results['rgb'] = resampled_integrator_results['rgb']

        if 'normals' in resampled_integrator_results:
          material_integrator_results['normals'] = resampled_integrator_results[
              'normals'
          ]

        if 'normals_pred' in resampled_integrator_results:
          material_integrator_results['normals_pred'] = (
              resampled_integrator_results['normals_pred']
          )

      # Material outputs
      material_outputs = {
          'loss_weight': self.loss_weight,
          'loss_type': self.loss,
          'linear_to_srgb': self.linear_to_srgb,
          'sampler': sampler_results,
          'shader': material_shader_results,
          'integrator': material_integrator_results,
      }

      outputs['main'] = material_outputs
      outputs['render'] = material_integrator_results

      # Extra material outputs
      if self.render_variate:
        outputs['render']['material_rgb'] = outputs['material']['integrator'][
            'rgb'
        ]
      else:
        outputs['render']['material_rgb'] = outputs['render']['rgb']

    # Extra cache outputs
    outputs['render']['cache_rgb'] = cache_results['render']['rgb']

    # Extra light outputs
    if light_sampler_results is not None:
      outputs['main']['light_sampler'] = light_sampler_results

      outputs['render'] = dict(
          **outputs['render'],
          **light_sampler_results,
      )

    ## Return
    return outputs


def construct_model(rng, rays, config, dataset=None):
  """Construct a mip-NeRF 360 model.

  Args:
    rng: jnp.ndarray. Random number generator.
    rays: an example of input Rays.
    config: A Config class.
    dataset: Dataset, used to set max_exposure.

  Returns:
    model: initialized nn.Module, a NeRF model with parameters.
    init_variables: flax.Module.state, initialized NeRF model parameters.
  """
  # Grab just 10 rays, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10], rays
  )

  extra_model_params = {}

  if dataset is not None and dataset.max_exposure is not None:
    extra_model_params['max_exposure'] = dataset.max_exposure

  if config.model_type == configs.ModelType.DEFAULT:
    model = NeRFModel(
        config=config,
        extra_model_params=extra_model_params,
    )
  elif config.model_type == configs.ModelType.DEFERRED:
    model = DeferredNeRFModel(
        config=config,
        extra_model_params=extra_model_params,
    )
  elif config.model_type == configs.ModelType.MATERIAL:
    model = MaterialModel(
        config=config,
        extra_model_params=extra_model_params,
    )
  else:
    model = NeRFModel(
        config=config,
        extra_model_params=extra_model_params,
    )

  init_variables = model.init(
      rng,  # The RNG used by flax to initialize random weights.
      rng=None,  # The RNG used by sampling within the model.
      rays=ray,
      train_frac=1.0,
      mesh=dataset.mesh,
  )

  return model, init_variables
