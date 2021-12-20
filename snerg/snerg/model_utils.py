# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Functions to extract and query individual MLPs from a trained SNeRG model."""

import functools
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from snerg.nerf import model_utils


def extract_snerg_mlps(optimized_model, scene_params):
  """Creates the two SNeRG MLPs and extracts their parameters.

  Specifically, this function creates the per-sample MLP, as well as the per-ray
  view-dependent MLP, and also extracts their parameters from a trained SNeRG
  model.

  Args:
    optimized_model: A trained SNeRG model (from state.optimizer.target)
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    mlp_model: A nerf.model_utils.MLP that predicts per-sample color, density,
      and the SNeRG feature vector.
    mlp_params: A dict containing the MLP parameters for the per-sample MLP.
    viewdir_mlp_model: A nerf.model_utils.MLP that predicts the per-ray
      view-dependent residual color.
    viewdir_mlp_params: A dict containing the MLP parameters for the per-sample
      view-dependence MLP.
  """
  mlp_params = {}
  mlp_params['params'] = optimized_model['params']['MLP_2']

  viewdir_mlp_params = {}
  viewdir_mlp_params['params'] = optimized_model['params']['MLP_3']

  net_activation = nn.relu

  mlp_model = model_utils.MLP(
      net_depth=scene_params['_net_depth'],
      net_width=scene_params['_net_width'],
      skip_layer=scene_params['_skip_layer'],
      num_rgb_channels=scene_params['_channels'],
      num_sigma_channels=scene_params['_num_sigma_channels'],
      net_activation=net_activation)

  mlp_params = {}
  mlp_params['params'] = optimized_model['params']['MLP_2']

  viewdir_mlp_model = model_utils.MLP(
      net_depth=scene_params['_viewdir_net_depth'],
      net_width=scene_params['_viewdir_net_width'],
      skip_layer=scene_params['_skip_layer'],
      num_rgb_channels=scene_params['_num_rgb_channels'],
      num_sigma_channels=scene_params['_num_sigma_channels'],
      net_activation=net_activation)

  return mlp_model, mlp_params, viewdir_mlp_model, viewdir_mlp_params


def pmap_model_fn(mlp_model, mlp_params, samples, scene_params):
  """Calls the per-sample MLP to extract colors+densities+features (in a pmap).

  Args:
    mlp_model: A nerf.model_utils.MLP that predicts per-sample color, density,
      and the SNeRG feature vector.
    mlp_params: A dict containing the MLP parameters for the per-sample MLP.
    samples: A [num_local_devices, 1, N, 3] JAX tensor of 3D positions.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    rgb_features: A [num_total_devices, num_total_devices, 1, N, 7] JAX tensor
      containing the color and computed feature vector for each sample.
    sigma: A [num_total_devices, num_total_devices, 1, N, 1] JAX tensor
      containing the densiy for each sample.
  """
  rgb_activation = nn.sigmoid
  sigma_activation = nn.relu
  @functools.partial(jax.pmap, in_axes=(None, 0), axis_name='batch')
  def inner_pmap(params, samples):
    """We need an inner function as only JAX types can be passed to a pmap."""
    samples_enc = model_utils.posenc(samples, scene_params['_min_deg_point'],
                                     scene_params['_max_deg_point'],
                                     scene_params['_legacy_posenc_order'])
    raw_rgb_features, raw_sigma = mlp_model.apply(params, samples_enc)
    rgb_features = rgb_activation(raw_rgb_features)
    sigma = sigma_activation(raw_sigma)
    return lax.all_gather((rgb_features, sigma), axis_name='batch')
  return inner_pmap(mlp_params, samples)


def viewdir_fn(viewdir_mlp_model, viewdir_mlp_params, rgb_features, viewdirs,
               scene_params):
  """Calls the per-ray view-dependence MLP to compute color residuals.

  Args:
    viewdir_mlp_model: A nerf.model_utils.MLP that predicts the per-ray
      view-dependent residual color.
    viewdir_mlp_params: A dict containing the MLP parameters for the per-ray
      view-dependence MLP.
    rgb_features:  A [H, W, 7] JAX tensor containing the composited color and
      computed feature vector for each ray.
    viewdirs: A [H, W, 3] JAX tensor of ray directions.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    A [H, W, 3] JAX tensor with the view-dependent RGB residual for each ray.
  """
  rgb_activation = nn.sigmoid
  viewdirs_enc = model_utils.posenc(viewdirs, 0, scene_params['_deg_view'],
                                    scene_params['_legacy_posenc_order'])
  viewdirs_enc_features = jnp.concatenate([viewdirs_enc, rgb_features], axis=-1)
  viewdirs_enc_features = jnp.expand_dims(viewdirs_enc_features, -2)
  raw_rgb, _ = viewdir_mlp_model.apply(viewdir_mlp_params,
                                       viewdirs_enc_features)
  return rgb_activation(raw_rgb)
