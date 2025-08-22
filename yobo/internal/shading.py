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
from google_research.yobo.internal import image
from google_research.yobo.internal import math
from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import render
from google_research.yobo.internal import utils
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
gin.config.external_configurable(coord.contract_projective, module='coord')


@gin.configurable
class BaseShader(nn.Module):
  """A PosEnc MLP."""

  config: Any = None  # A Config class, must be set upon construction.

  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.

  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.

  bottleneck_width: int = 256  # The width of the bottleneck vector.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.

  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 4  # Max degree of positional encoding for 3D points.

  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  use_posenc_with_grid: bool = False

  num_rgb_channels: int = 3  # The number of RGB channels.
  rgb_premultiplier: float = 1.0  # Premultiplier on RGB before activation.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.0  # The shift added to raw colors pre-activation.
  rgb_bias_diffuse: float = (
      -1.0
  )  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.

  isotropize_gaussians: bool = False  # If True, make Gaussians isotropic.
  gaussian_covariance_scale: float = 1.0  # Amount to scale covariances.
  gaussian_covariance_pad: float = 0.0  # Amount to add to covariance diagonals.

  squash_before: bool = False  # Apply squash before computing density gradient.
  warp_fn: Callable[Ellipsis, Any] = None

  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).

  unscented_mip_basis: str = 'mean'  # Which unscented transform basis to use.
  unscented_sqrt_fn: str = 'sqrtm'  # How to sqrt covariance matrices in the UT.
  unscented_scale_mult: float = 0.0  # Unscented scale, 0 == disabled.

  use_density_feature: bool = True
  affine_density_feature: bool = False
  use_grid: bool = False
  grid_representation: str = 'ngp'
  grid_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  normals_target: str = 'normals_to_use'

  backfacing_target: str = 'normals_to_use'
  backfacing_noise: float = 0.0
  backfacing_noise_rate: float = float('inf')
  backfacing_near: float = 1e-1

  def run_network(self, x):
    inputs = x

    # Evaluate network to produce the output density.
    for i in range(self.net_depth):
      x = self.layers[i](x)
      x = self.net_activation(x)

      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)

    return x

  def predict_appearance_feature(
      self,
      sampler_results,
      train_frac = 1.0,
      train = True,
      **kwargs,
  ):
    means, covs = sampler_results['means'], sampler_results['covs']

    # Note that isotropize and the gaussian scaling/padding is done *before*
    # applying warp_fn. For some applications, applying these steps afterwards
    # instead may be desirable.
    if self.isotropize_gaussians:
      # Replace each Gaussian's covariance with an isotropic version with the
      # same determinant.
      covs = coord.isotropize(covs)

    if self.gaussian_covariance_scale != 1:
      covs *= self.gaussian_covariance_scale

    if self.gaussian_covariance_pad > 0:
      covs += jnp.diag(jnp.full(covs.shape[-1], self.gaussian_covariance_pad))

    x = []

    if self.use_density_feature:
      x.append(sampler_results['feature'])

    # Encode input positions.
    if self.grid is not None:
      control_offsets = kwargs['control_offsets']
      control = means[Ellipsis, None, :] + control_offsets
      perp_mag = kwargs['perp_mag']

      # Add point offset
      if 'point_offset' in kwargs:
        control = control + kwargs['point_offset'][Ellipsis, None, :]

      # Warp
      scale = None

      if not self.squash_before and self.warp_fn is not None:
        if perp_mag is not None and self.unscented_scale_mult > 0:
          if self.warp_fn.__wrapped__ == coord.contract:
            # We can accelerate the contraction a lot by special-casing
            # on the contraction and computing the cube root of the
            # determinant of the Jacobian directly.
            s = coord.contract3_isoscale(control)
            scale = self.unscented_scale_mult * (perp_mag * s)[Ellipsis, None]
            control = self.warp_fn(control)  # pylint: disable=not-callable
          else:
            control, perp_mag = coord.track_isotropic(
                self.warp_fn, control, perp_mag
            )
            scale = self.unscented_scale_mult * perp_mag[Ellipsis, None]
        else:
          control = self.warp_fn(control)  # pylint: disable=not-callable

      x.append(
          self.grid(
              control,
              x_scale=scale,
              per_level_fn=math.average_across_multisamples,
              train=train,
          )
      )

      if self.use_posenc_with_grid:
        # Encode using the strategy used in mip-NeRF 360.
        if not self.squash_before and self.warp_fn is not None:
          means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = coord.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )

        x.append(
            coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )
        )

    x = jnp.concatenate(x, axis=-1)
    return self.run_network(x)

  def get_predict_appearance_kwargs(self, rng, rays, sampler_results, **kwargs):
    means, covs = sampler_results['means'], sampler_results['covs']
    predict_appearance_kwargs = {}

    if self.grid is not None:
      # Grid/hash structures don't give us an easy way to do closed-form
      # integration with a Gaussian, so instead we sample each Gaussian
      # according to an unscented transform (or something like it) and average
      # the sampled encodings.
      control_points_key, rng = utils.random_split(rng)

      if 'tdist' in sampler_results:
        control, perp_mag = coord.compute_control_points(
            means,
            covs,
            rays,
            sampler_results['tdist'],
            control_points_key,
            self.unscented_mip_basis,
            self.unscented_sqrt_fn,
            self.unscented_scale_mult,
        )
      else:
        control = means[Ellipsis, None, :]
        perp_mag = jnp.zeros_like(control)

      control_offsets = control - means[Ellipsis, None, :]
      predict_appearance_kwargs['control_offsets'] = control_offsets
      predict_appearance_kwargs['perp_mag'] = perp_mag

    return dict(
        **predict_appearance_kwargs,
        **kwargs,
    )

  def get_bottleneck_feature(
      self,
      rng,
      feature,
  ):
    # Output of the first part of MLP.
    bottleneck = self.bottleneck_layer(feature)

    # Add bottleneck noise.
    if (rng is not None) and (self.bottleneck_noise > 0):
      key, rng = utils.random_split(rng)
      bottleneck += self.bottleneck_noise * random.normal(key, bottleneck.shape)

    return bottleneck

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      zero_backfacing = None,
      zero_backfacing_noise = None,
      **kwargs,
  ):
    # Appearance model
    shading_results = self.predict_appearance(
        rng=rng,
        rays=rays,
        sampler_results=sampler_results,
        train_frac=train_frac,
        train=train,
        **kwargs,
    )

    # Add random value to colors
    if train and (rng is not None) and self.backfacing_noise > 0:
      # Appearance mask
      dotprod = math.dot(
          sampler_results[self.backfacing_target],
          -rays.directions[Ellipsis, None, :],
      )
      app_mask = dotprod > 0.0

      key, rng = utils.random_split(rng)
      rgb_noise = (
          random.normal(key, shading_results['rgb'].shape)
          * self.backfacing_noise
          * jnp.clip(1.0 - train_frac / self.backfacing_noise_rate, 0.0, 1.0)
      )
      rgb = jnp.maximum(
          rgb_noise + jax.lax.stop_gradient(shading_results['rgb']),
          -float('inf'),
      )

      shading_results['rgb'] = jnp.where(
          app_mask,
          shading_results['rgb'],
          rgb,
      )

    # Return
    return dict(
        **sampler_results,
        **shading_results,
    )


@gin.configurable
class NeRFMLP(BaseShader):
  """A PosEnc MLP."""

  config: Any = None  # A Config class, must be set upon construction.

  net_depth_viewdirs: int = 1  # The depth of the second part of ML.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
  deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
  fp16_view_dense: bool = False  # Use fp16 for view-dependency dense layers.

  use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
  use_directional_enc: bool = False  # If True, use IDE to encode directions.

  # If False and if use_directional_enc is True, use zero roughness in IDE.
  enable_pred_roughness: bool = False
  # Roughness activation function.
  roughness_activation: Callable[Ellipsis, Any] = nn.softplus
  roughness_bias: float = -1.0  # Shift added to raw roughness pre-activation.

  use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
  use_diffuse_only: bool = False  # If True, predict diffuse & specular colors.
  use_specular_tint: bool = False  # If True, predict tint.
  use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.

  use_learned_vignette_map: bool = False
  use_exposure_at_bottleneck: bool = False
  learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).

  num_glo_features: int = 0  # GLO vector length, disabled if 0.
  num_glo_embeddings: int = 1000  # Upper bound on max number of train images.

  # GLO vectors can either be 'concatenate'd onto the `bottleneck` or used to
  # construct an 'affine' transformation on the `bottleneck``.
  glo_mode: str = 'concatenate'
  # The MLP architecture used to transform the GLO codes before they are used.
  # Setting to () is equivalent to not using an MLP.
  glo_mlp_arch: Tuple[int, Ellipsis] = tuple()
  glo_mlp_act: Callable[Ellipsis, Any] = nn.silu  # The activation for the GLO MLP.
  glo_premultiplier: float = 1.0  # Premultiplier on GLO vectors before process.

  manual_emission_value: float = 0.0  # If True, use diffuse emission
  manual_emission_dim: float = 0.50  # If True, use diffuse emission

  def setup(self):
    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
    ).T

    # Precompute and define viewdir or refdir encoding function.
    if self.use_directional_enc:
      self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
    else:

      def dir_enc_fn(direction, _):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_view, append_identity=True
        )

      self.dir_enc_fn = dir_enc_fn

    self.dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )
    self.view_dependent_dense_layer = functools.partial(
        nn.Dense,
        kernel_init=getattr(jax.nn.initializers, self.weight_init)(),
        dtype=jnp.float16 if self.fp16_view_dense else None,
    )

    # GLO
    if self.num_glo_features > 0:
      self.glo_vecs = nn.Embed(
          self.num_glo_embeddings, self.num_glo_features, name='glo_vecs'
      )

    # Exposure
    if self.learned_exposure_scaling:
      self.exposure_scaling_offsets = nn.Embed(
          self.num_glo_embeddings,
          features=3,
          embedding_init=jax.nn.initializers.zeros,
          name='exposure_scaling_offsets',
      )

    # Appearance grid
    if self.use_grid:
      self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
          self.grid_representation.lower()
      ](name='grid', **self.grid_params)
    else:
      self.grid = None

    # RGB layers
    self.rgb_diffuse_layer = self.dense_layer(self.num_rgb_channels)

    self.tint_layer = self.dense_layer(3)

    self.roughness_layer = self.dense_layer(1)

    self.layers = [
        self.dense_layer(self.net_width) for i in range(self.net_depth)
    ]

    self.bottleneck_layer = self.dense_layer(self.bottleneck_width)

    self.view_dependent_layers = [
        self.view_dependent_dense_layer(self.net_width_viewdirs)
        for i in range(self.net_depth_viewdirs)
    ]

    self.output_view_dependent_layer = self.view_dependent_dense_layer(
        self.num_rgb_channels
    )

  def get_glo_vec(self, rays, zero_glo=False):
    glo_vec = None

    if self.num_glo_features > 0:
      if not zero_glo:
        cam_idx = rays.cam_idx[Ellipsis, 0]
        glo_vec = self.glo_vecs(cam_idx)
      else:
        glo_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_glo_features,))

    return glo_vec

  def run_network(self, x):
    inputs = x

    # Evaluate network to produce the output density.
    for i in range(self.net_depth):
      x = self.layers[i](x)
      x = self.net_activation(x)

      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)

    return x

  def predict_appearance_feature(
      self,
      sampler_results,
      train_frac = 1.0,
      train = True,
      **kwargs,
  ):
    means, covs = sampler_results['means'], sampler_results['covs']

    # Note that isotropize and the gaussian scaling/padding is done *before*
    # applying warp_fn. For some applications, applying these steps afterwards
    # instead may be desirable.
    if self.isotropize_gaussians:
      # Replace each Gaussian's covariance with an isotropic version with the
      # same determinant.
      covs = coord.isotropize(covs)

    if self.gaussian_covariance_scale != 1:
      covs *= self.gaussian_covariance_scale

    if self.gaussian_covariance_pad > 0:
      covs += jnp.diag(jnp.full(covs.shape[-1], self.gaussian_covariance_pad))

    x = []

    if self.use_density_feature:
      density_feature = sampler_results['feature']

      if self.affine_density_feature:
        zeros_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, 'zeros')
        )
        affine_feature_layer = zeros_layer(sampler_results['feature'].shape[-1])
        x.append(affine_feature_layer(density_feature))
      else:
        x.append(density_feature)

    # Encode input positions.
    if self.grid is not None:
      control_offsets = kwargs['control_offsets']
      control = means[Ellipsis, None, :] + control_offsets
      perp_mag = kwargs['perp_mag']

      # Add point offset
      if 'point_offset' in kwargs:
        control = control + kwargs['point_offset'][Ellipsis, None, :]

      # Warp
      scale = None

      if not self.squash_before and self.warp_fn is not None:
        if perp_mag is not None and self.unscented_scale_mult > 0:
          if self.warp_fn.__wrapped__ == coord.contract:
            # We can accelerate the contraction a lot by special-casing
            # on the contraction and computing the cube root of the
            # determinant of the Jacobian directly.
            s = coord.contract3_isoscale(control)
            scale = self.unscented_scale_mult * (perp_mag * s)[Ellipsis, None]
            control = self.warp_fn(control)  # pylint: disable=not-callable
          else:
            control, perp_mag = coord.track_isotropic(
                self.warp_fn, control, perp_mag
            )
            scale = self.unscented_scale_mult * perp_mag[Ellipsis, None]
        else:
          control = self.warp_fn(control)  # pylint: disable=not-callable

      x.append(
          self.grid(
              control,
              x_scale=scale,
              per_level_fn=math.average_across_multisamples,
              train=train,
          )
      )

      if self.use_posenc_with_grid:
        # Encode using the strategy used in mip-NeRF 360.
        if not self.squash_before and self.warp_fn is not None:
          means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = coord.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )

        x.append(
            coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )
        )

    x = jnp.concatenate(x, axis=-1)
    return self.run_network(x)

  def predict_appearance(
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

    # Get glo vector
    glo_vec = self.get_glo_vec(rays, zero_glo=zero_glo)

    # Exposure
    exposure = rays.exposure_values

    # Predict diffuse color.
    if self.use_diffuse_color:
      raw_rgb_diffuse = self.rgb_diffuse_layer(feature)
    else:
      raw_rgb_diffuse = jnp.zeros_like(means)

    if self.use_specular_tint:
      tint = nn.sigmoid(self.tint_layer(feature))
    else:
      tint = jnp.zeros_like(means)

    if self.enable_pred_roughness:
      raw_roughness = self.roughness_layer(feature)

      roughness = self.roughness_activation(raw_roughness + self.roughness_bias)
    else:
      roughness = None

    # Output of the first part of MLP.
    if self.bottleneck_width > 0:
      key, rng = utils.random_split(rng)
      bottleneck = self.get_bottleneck_feature(key, feature)

      # Incorporate exposure in the style of HDR-NeRF, where we assume the
      # bottleneck is scaled proportional to log radiance and thus we can
      # scale it appropriately by adding log of the exposure value.
      if self.use_exposure_at_bottleneck and exposure is not None:
        bottleneck += jnp.log(exposure)[Ellipsis, None, :]

      x = [bottleneck]
    else:
      bottleneck = jnp.zeros_like(means[Ellipsis, 0:0])
      x = []

    # Encode view (or reflection) directions.
    if self.use_reflections:
      # Compute reflection directions. Note that we flip viewdirs before
      # reflecting, because they point from the camera to the point,
      # whereas ref_utils.reflect() assumes they point toward the camera.
      # Returned refdirs then point from the point to the environment.
      refdirs = ref_utils.reflect(
          -viewdirs[Ellipsis, None, :],
          sampler_results[self.normals_target],
      )

      # Encode reflection directions.
      dir_enc = self.dir_enc_fn(refdirs, roughness)
    else:
      # Encode view directions.
      dir_enc = self.dir_enc_fn(viewdirs, roughness)

      dir_enc = jnp.broadcast_to(
          dir_enc[Ellipsis, None, :],
          means.shape[:-1] + (dir_enc.shape[-1],),
      )

    # Append view (or reflection) direction encoding to bottleneck vector.
    x.append(dir_enc)

    # Append dot product between normal vectors and view directions.
    if self.use_n_dot_v:
      dotprod = jnp.sum(
          sampler_results[self.normals_target] * viewdirs[Ellipsis, None, :],
          axis=-1,
          keepdims=True,
      )
      x.append(dotprod)

    # Use the GLO vector, if it's available.
    if glo_vec is not None:
      # Pass the GLO vector (optionally) through a small MLP.
      y = glo_vec * self.glo_premultiplier

      for wi, w in enumerate(self.glo_mlp_arch):
        y = self.glo_mlp_act(nn.Dense(w, name=f'GLO_MLP_{wi}')(y))

      if self.glo_mode == 'concatenate':
        # Concatenate the transformed GLO vector onto the bottleneck.
        shape = bottleneck.shape[:-1] + y.shape[-1:]
        x.append(jnp.broadcast_to(y[Ellipsis, None, :], shape))
      elif self.glo_mode == 'affine':
        if self.bottleneck_width <= 0:
          # The user probably shouldn't use this mode if the bottleneck
          # is non-existent.
          raise ValueError('Bottleneck must have a non-zero width.')
        # Turn the transformed GLO vector into an affine transformation on
        # the bottleneck, and replace the bottleneck with that.
        y = nn.Dense(
            2 * bottleneck.shape[-1],
            name=f'GLO_MLP_{len(self.glo_mlp_arch)}',
        )(y)
        log_a, b = tuple(jnp.moveaxis(y.reshape(y.shape[:-1] + (-1, 2)), -1, 0))
        a = math.safe_exp(log_a)
        affine_bottleneck = a[Ellipsis, None, :] * bottleneck + b[Ellipsis, None, :]
        x = [affine_bottleneck] + x[1:]  # clobber the bottleneck at x[0].

    # Concatenate bottleneck, directional encoding, and GLO.
    x = jnp.concatenate(x, axis=-1)

    # Output of the second part of MLP.
    inputs = x

    for i in range(self.net_depth_viewdirs):
      x = self.view_dependent_layers[i](x)
      x = self.net_activation(x)

      if i % self.skip_layer_dir == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)

    # If using diffuse/specular colors, then `rgb` is treated as linear
    # specular color. Otherwise it's treated as the color itself.
    rgb = self.rgb_activation(
        self.rgb_premultiplier * self.output_view_dependent_layer(x)
        + self.rgb_bias
    )

    if self.use_learned_vignette_map:
      vignette_weights = self.param(
          'VignetteWeights',
          lambda x: jax.nn.initializers.zeros(x, shape=[3, 3]),
      )
      vignette = image.compute_vignette(rays.imageplane, vignette_weights)
      # Account for the extra dimensions from ray samples.
      rgb *= vignette[Ellipsis, None, :]

    # Specular
    if self.use_specular_tint:
      specular_linear = tint * rgb
    else:
      specular_linear = rgb

    # Diffuse
    if self.use_diffuse_color:
      # Initialize linear diffuse color around 0.25, so that the combined
      # linear color is initialized around 0.5.
      diffuse_linear = self.rgb_activation(
          self.rgb_premultiplier * raw_rgb_diffuse + self.rgb_bias_diffuse
      )

      # Combine specular and diffuse components and tone map to sRGB.
      if self.use_diffuse_only:
        rgb = diffuse_linear
      else:
        rgb = specular_linear + diffuse_linear

      if self.manual_emission_value > 0.0:
        rgb = jnp.where(
            (
                (jnp.abs(means[Ellipsis, 0:1]) < self.manual_emission_dim)
                & (jnp.abs(means[Ellipsis, 1:2]) < self.manual_emission_dim)
                & (jnp.abs(means[Ellipsis, 2:3] - 4.99) < 0.05)
            ),
            jnp.ones_like(rgb) * self.manual_emission_value,
            rgb,
        )

      outputs['diffuse_rgb'] = diffuse_linear
    else:
      rgb = specular_linear

    # Visualization
    outputs['specular_rgb'] = specular_linear

    # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
    rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

    outputs['rgb'] = rgb

    return outputs

  def get_predict_appearance_kwargs(self, rng, rays, sampler_results, **kwargs):
    means, covs = sampler_results['means'], sampler_results['covs']
    predict_appearance_kwargs = {}

    if self.grid is not None:
      # Grid/hash structures don't give us an easy way to do closed-form
      # integration with a Gaussian, so instead we sample each Gaussian
      # according to an unscented transform (or something like it) and average
      # the sampled encodings.
      control_points_key, rng = utils.random_split(rng)

      if 'tdist' in sampler_results:
        control, perp_mag = coord.compute_control_points(
            means,
            covs,
            rays,
            sampler_results['tdist'],
            control_points_key,
            self.unscented_mip_basis,
            self.unscented_sqrt_fn,
            self.unscented_scale_mult,
        )
      else:
        control = means[Ellipsis, None, :]
        perp_mag = jnp.zeros_like(control)

      control_offsets = control - means[Ellipsis, None, :]
      predict_appearance_kwargs['control_offsets'] = control_offsets
      predict_appearance_kwargs['perp_mag'] = perp_mag

    return dict(
        **predict_appearance_kwargs,
        **kwargs,
    )
