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

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from typing import Any, Callable
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from snerg.nerf import model_utils
from snerg.nerf import utils


def get_model(key, example_batch, args):
  """A helper function that wraps around a 'model zoo'."""
  model_dict = {
      "nerf": construct_nerf,
  }
  return model_dict[args.model](key, example_batch, args)


class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  num_coarse_samples: int  # The number of samples for the coarse nerf.
  num_fine_samples: int  # The number of samples for the fine nerf.
  use_viewdirs: bool  # If True, use viewdirs as an input.
  near: float  # The distance to the near plane
  far: float  # The distance to the far plane
  noise_std: float  # The std dev of noise added to raw sigma.
  net_depth: int  # The depth of the first part of MLP.
  net_width: int  # The width of the first part of MLP.
  num_viewdir_channels: int  # The number of extra channels for view-dependence.
  viewdir_net_depth: int  # The depth of the view-dependence MLP.
  viewdir_net_width: int  # The width of the view-dependence MLP.
  net_activation: Callable[Ellipsis, Any]  # MLP activation
  skip_layer: int  # How often to add skip connections.
  num_rgb_channels: int  # The number of RGB channels.
  num_sigma_channels: int  # The number of density channels.
  white_bkgd: bool  # If True, use a white background.
  min_deg_point: int  # The minimum degree of positional encoding for positions.
  max_deg_point: int  # The maximum degree of positional encoding for positions.
  deg_view: int  # The degree of positional encoding for viewdirs.
  lindisp: bool  # If True, sample linearly in disparity rather than in depth.
  rgb_activation: Callable[Ellipsis, Any]  # Output RGB activation.
  sigma_activation: Callable[Ellipsis, Any]  # Output sigma activation.
  legacy_posenc_order: bool  # Keep the same ordering as the original tf code.

  @nn.compact
  def __call__(self, rng_0, rng_1, rays, randomized):
    """Nerf Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.

    Returns:
      ret: list, [(rgb_coarse, disp_coarse, acc_coarse, features_coarse,
      specular_coarse), (rgb, disp, acc, features, specular)]
    """
    # Stratified sampling along rays
    key, rng_0 = random.split(rng_0)
    z_vals, coarse_samples = model_utils.sample_along_rays(
        key,
        rays.origins,
        rays.directions,
        self.num_coarse_samples,
        self.near,
        self.far,
        randomized,
        self.lindisp,
    )
    coarse_samples_enc = model_utils.posenc(
        coarse_samples,
        self.min_deg_point,
        self.max_deg_point,
        self.legacy_posenc_order,
    )

    # Construct the "coarse" MLP.
    coarse_mlp = model_utils.MLP(
        net_depth=self.net_depth,
        net_width=self.net_width,
        net_activation=self.net_activation,
        skip_layer=self.skip_layer,
        num_rgb_channels=self.num_rgb_channels + self.num_viewdir_channels,
        num_sigma_channels=self.num_sigma_channels)

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_enc = model_utils.posenc(
          rays.viewdirs,
          0,
          self.deg_view,
          self.legacy_posenc_order,
      )
      raw_features_and_rgb, raw_sigma = coarse_mlp(coarse_samples_enc)
    else:
      raw_rgb, raw_sigma = coarse_mlp(coarse_samples_enc)

    # Add noises to regularize the density predictions if needed
    key, rng_0 = random.split(rng_0)
    raw_sigma = model_utils.add_gaussian_noise(
        key,
        raw_sigma,
        self.noise_std,
        randomized,
    )
    sigma = self.sigma_activation(raw_sigma)

    if self.use_viewdirs:
      coarse_viewdir_mlp = model_utils.MLP(
          net_depth=self.viewdir_net_depth,
          net_width=self.viewdir_net_width,
          net_activation=self.net_activation,
          skip_layer=self.skip_layer,
          num_rgb_channels=self.num_rgb_channels,
          num_sigma_channels=self.num_sigma_channels)

      # Overcomposite the features to get an encoding for the features.
      comp_features, _, _, _ = model_utils.volumetric_rendering(
          raw_features_and_rgb[Ellipsis, self.num_rgb_channels:(
              self.num_rgb_channels + self.num_viewdir_channels)],
          sigma,
          z_vals,
          rays.directions,
          white_bkgd=False,
      )
      features = comp_features[Ellipsis, 0:self.num_rgb_channels]

      diffuse_rgb = self.rgb_activation(
          raw_features_and_rgb[Ellipsis, 0:self.num_rgb_channels])
      comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
          diffuse_rgb,
          sigma,
          z_vals,
          rays.directions,
          white_bkgd=self.white_bkgd,
      )

      viewdirs_enc_features = jnp.concatenate(
          [viewdirs_enc, comp_rgb, comp_features], axis=-1)
      viewdirs_enc_features = jnp.expand_dims(viewdirs_enc_features, -2)
      raw_comp_rgb_residual, _ = coarse_viewdir_mlp(viewdirs_enc_features)

      output_shape = list(comp_features.shape)
      output_shape[-1] = 3
      raw_comp_rgb_residual = raw_comp_rgb_residual.reshape(output_shape)
      rgb_residual = self.rgb_activation(raw_comp_rgb_residual)
      comp_rgb += rgb_residual
    else:
      rgb = self.rgb_activation(raw_rgb)
      # Volumetric rendering.
      comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
          rgb,
          sigma,
          z_vals,
          rays.directions,
          white_bkgd=self.white_bkgd,
      )
      features = jnp.zeros_like(comp_rgb)
      rgb_residual = jnp.zeros_like(comp_rgb)

    ret = [
        (comp_rgb, disp, acc, sigma, features, rgb_residual),
    ]
    # Hierarchical sampling based on coarse predictions
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
      key, rng_1 = random.split(rng_1)
      z_vals, fine_samples = model_utils.sample_pdf(
          key,
          z_vals_mid,
          weights[Ellipsis, 1:-1],
          rays.origins,
          rays.directions,
          z_vals,
          self.num_fine_samples,
          randomized,
      )
      fine_samples_enc = model_utils.posenc(
          fine_samples,
          self.min_deg_point,
          self.max_deg_point,
          self.legacy_posenc_order,
      )

      # Construct the "fine" MLP.
      fine_mlp = model_utils.MLP(
          net_depth=self.net_depth,
          net_width=self.net_width,
          net_activation=self.net_activation,
          skip_layer=self.skip_layer,
          num_rgb_channels=self.num_rgb_channels + self.num_viewdir_channels,
          num_sigma_channels=self.num_sigma_channels)

      if self.use_viewdirs:
        raw_features_and_rgb, raw_sigma = fine_mlp(fine_samples_enc)
      else:
        raw_rgb, raw_sigma = fine_mlp(fine_samples_enc)

      key, rng_1 = random.split(rng_1)
      raw_sigma = model_utils.add_gaussian_noise(
          key,
          raw_sigma,
          self.noise_std,
          randomized,
      )
      sigma = self.sigma_activation(raw_sigma)

      _, raw_reg_sigma = fine_mlp(coarse_samples_enc)
      reg_sigma = self.sigma_activation(raw_reg_sigma)

      if self.use_viewdirs:
        fine_viewdir_mlp = model_utils.MLP(
            net_depth=self.viewdir_net_depth,
            net_width=self.viewdir_net_width,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels)

        # Overcomposite the features to get an encoding for the features.
        features_and_rgb = self.rgb_activation(raw_features_and_rgb)
        features = features_and_rgb[Ellipsis, self.num_rgb_channels:(
            self.num_rgb_channels + self.num_viewdir_channels)]

        comp_features, _, _, _ = model_utils.volumetric_rendering(
            features,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=False,
        )
        features = comp_features[Ellipsis, 0:self.num_rgb_channels]

        diffuse_rgb = features_and_rgb[Ellipsis, 0:self.num_rgb_channels]
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            diffuse_rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )

        viewdirs_enc_features = jnp.concatenate(
            [viewdirs_enc, comp_rgb, comp_features], axis=-1)
        viewdirs_enc_features = jnp.expand_dims(viewdirs_enc_features, -2)
        raw_comp_rgb_residual, _ = fine_viewdir_mlp(viewdirs_enc_features)

        output_shape = list(comp_features.shape)
        output_shape[-1] = 3
        raw_comp_rgb_residual = raw_comp_rgb_residual.reshape(output_shape)
        rgb_residual = self.rgb_activation(raw_comp_rgb_residual)
        comp_rgb += rgb_residual
      else:
        rgb = self.rgb_activation(raw_rgb)
        # Volumetric rendering.
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )
        features = jnp.zeros_like(comp_rgb)
        rgb_residual = jnp.zeros_like(comp_rgb)

      ret.append((comp_rgb, disp, acc, reg_sigma, features, rgb_residual))
    return ret


def construct_nerf(key, example_batch, args):
  """Construct a Neural Radiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  net_activation = nn.relu
  rgb_activation = nn.sigmoid
  sigma_activation = nn.relu

  # Assert that rgb_activation always produces outputs in [0, 1], and
  # sigma_activation always produce non-negative outputs.
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)

  rgb = rgb_activation(x)
  if jnp.any(rgb < 0) or jnp.any(rgb > 1):
    raise NotImplementedError(
        "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
        .format(args.rgb_activation))

  sigma = sigma_activation(x)
  if jnp.any(sigma < 0):
    raise NotImplementedError(
        "Choice of sigma_activation `{}` produces negative densities".format(
            args.sigma_activation))

  model = NerfModel(
      min_deg_point=args.min_deg_point,
      max_deg_point=args.max_deg_point,
      deg_view=args.deg_view,
      num_coarse_samples=args.num_coarse_samples,
      num_fine_samples=args.num_fine_samples,
      use_viewdirs=args.use_viewdirs,
      near=args.near,
      far=args.far,
      noise_std=args.noise_std,
      white_bkgd=args.white_bkgd,
      net_depth=args.net_depth,
      net_width=args.net_width,
      num_viewdir_channels=args.num_viewdir_channels,
      viewdir_net_depth=args.viewdir_net_depth,
      viewdir_net_width=args.viewdir_net_width,
      skip_layer=args.skip_layer,
      num_rgb_channels=args.num_rgb_channels,
      num_sigma_channels=args.num_sigma_channels,
      lindisp=args.lindisp,
      net_activation=net_activation,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      legacy_posenc_order=args.legacy_posenc_order)
  rays = example_batch["rays"]
  key1, key2, key3 = random.split(key, num=3)

  init_variables = model.init(
      key1,
      rng_0=key2,
      rng_1=key3,
      rays=utils.namedtuple_map(lambda x: x[0], rays),
      randomized=args.randomized)

  return model, init_variables
