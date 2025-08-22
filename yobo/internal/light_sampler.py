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
class LightMLP(shading.BaseShader):
  """A PosEnc MLP."""

  config: Any = None

  num_components: int = 64  # Learned BRDF layer width
  vmf_scale: float = 5.0  #  Learned BRDF layer depth
  random_seed: int = 1

  vmf_bias: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({
      'vmf_means': 0.0,
      'vmf_kappas': 1.0,
      'vmf_weights': 1.0,
  })
  vmf_activation: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({
          'vmf_means': lambda x: x,
          'vmf_kappas': jax.nn.softplus,
          'vmf_weights': jax.nn.softplus,
      })
  )

  normals_target: str = 'normals_to_use'

  def setup(self):
    self.dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # VMF prediction
    self.layers = [
        self.dense_layer(self.net_width) for i in range(self.net_depth)
    ]

    self.output_layer = self.dense_layer(self.num_components * 5)

    # Grid
    if self.use_grid:
      self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
          self.grid_representation.lower()
      ](name='grid', **self.grid_params)
    else:
      self.grid = None

  def get_vmfs(self, vmf_params):
    # rng = random.PRNGKey(self.random_seed)

    # means_key, rng = utils.random_split(rng)
    # kappas_key, rng = utils.random_split(rng)
    # weights_key, rng = utils.random_split(rng)

    # means_random = jax.random.normal(
    #     means_key, shape=vmf_params.shape[:-1] + (3,)
    # ) * self.vmf_scale / 2.0

    vmfs = {
        'vmf_means': self.vmf_activation['vmf_means'](
            vmf_params[Ellipsis, 0:3]
            + self.vmf_bias['vmf_means']
            # + means_random
        ),
        'vmf_kappas': self.vmf_activation['vmf_kappas'](
            vmf_params[Ellipsis, 3:4] + self.vmf_bias['vmf_kappas']
        ),
        'vmf_weights': self.vmf_activation['vmf_weights'](
            vmf_params[Ellipsis, 4:5] + self.vmf_bias['vmf_weights']
        ),
    }

    return vmfs

  def predict_lighting(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      zero_glo = False,
      **kwargs,
  ):
    outputs = {}

    means, covs = sampler_results['means'], sampler_results['covs']
    viewdirs = rays.viewdirs

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

    # Predict VMFs
    vmf_params = self.output_layer(feature)
    vmf_params = vmf_params.reshape(means.shape[:-1] + (self.num_components, 5))
    vmfs = self.get_vmfs(vmf_params)

    vmfs['vmf_origins'] = means[Ellipsis, None, :]
    vmfs['vmf_normals'] = sampler_results[self.normals_target][Ellipsis, None, :]

    return vmfs

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      zero_backfacing = None,
      **kwargs,
  ):
    # Appearance model
    return self.predict_lighting(
        rng=rng,
        rays=rays,
        sampler_results=sampler_results,
        train_frac=train_frac,
        train=train,
        **kwargs,
    )
