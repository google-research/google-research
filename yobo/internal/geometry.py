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
class DensityMLP(nn.Module):
  """A PosEnc MLP."""

  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.

  use_posenc_with_grid: bool = False
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 4  # Max degree of positional encoding for 3D points.

  density_activation: Callable[Ellipsis, Any] = nn.softplus  # Density activation.
  density_bias: float = -1.0  # Shift added to raw densities pre-activation.
  density_noise: float = (
      0.0  # Standard deviation of noise added to raw density.
  )

  enable_pred_normals: bool = False  # If True compute predicted normals.
  disable_density_normals: bool = False  # If True don't compute normals.

  isotropize_gaussians: bool = False  # If True, make Gaussians isotropic.
  gaussian_covariance_scale: float = 1.0  # Amount to scale covariances.
  gaussian_covariance_pad: float = 0.0  # Amount to add to covariance diagonals.

  warp_fn: Callable[Ellipsis, Any] = None

  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).
  unscented_mip_basis: str = 'mean'  # Which unscented transform basis to use.
  unscented_sqrt_fn: str = 'sqrtm'  # How to sqrt covariance matrices in the UT.
  unscented_scale_mult: float = 0.0  # Unscented scale, 0 == disabled.
  squash_before: bool = False  # Apply squash before computing density gradient.

  use_grid: bool = True
  grid_representation: str = 'ngp'
  grid_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  backfacing_target: str = 'normals'
  backfacing_near: float = 2e-1

  filter_backfacing: bool = False
  filter_backfacing_threshold: float = 0.5
  filter_backfacing_alpha_threshold: float = 0.25
  filter_backfacing_exponent: float = 3.0

  normals_for_filter_only: bool = False

  def setup(self):
    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
    ).T

    self.dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    self.density_dense_layer = self.dense_layer
    self.final_density_dense_layer = self.dense_layer

    # Density layers
    self.density_layers = [
        self.density_dense_layer(self.net_width) for i in range(self.net_depth)
    ]

    self.output_density_layer = self.final_density_dense_layer(1)

    self.pred_normals_layer = self.dense_layer(3)

    # Appearance grid
    if self.use_grid:
      self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
          self.grid_representation.lower()
      ](name='grid', **self.grid_params)
    else:
      self.grid = None

  def run_network(self, x, means):
    inputs = x

    # Evaluate network to produce the output density.
    for i in range(self.net_depth):
      x = self.density_layers[i](x)
      x = self.net_activation(x)

      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)

    raw_density = self.output_density_layer(x)[
        Ellipsis, 0
    ]  # Hardcoded to a single channel.

    return raw_density, x

  def predict_density(
      self, means, covs, density_key=None, train_frac=1.0, train=True, **kwargs
  ):
    """Helper function to output density."""

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

      # Gathering/scattering from a grid is impossibly slow on TPU.
      if utils.device_is_tpu():
        raise ValueError('Hash Encodings should not be used on a TPU.')

      x.append(
          self.grid(
              control,
              x_scale=scale,
              per_level_fn=math.average_across_multisamples,
              train=train,
          )
      )

    if self.grid is None or self.use_posenc_with_grid:
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

    raw_density, x = self.run_network(x, means)

    # Add noise to regularize the density predictions if needed.
    if (density_key is not None) and (self.density_noise > 0):
      raw_density += self.density_noise * random.normal(
          density_key, raw_density.shape
      )

    return raw_density, x

  def convert_raw_density(self, raw_density, means):
    # Apply bias and activation to raw density
    density = self.density_activation(raw_density + self.density_bias)

    # Zero out invalid density
    if self.grid is not None:
      warped_means = means

      if self.warp_fn is not None:
        warped_means = self.warp_fn(means)

      # Set the density of points outside the valid area to zero.
      # If there is a contraction, this mask will be true for all points. On the
      # other hand, if there isn't, invalid points that fall outside of the
      # bounding box will have their density set to zero.
      density_is_valid = jnp.all(
          (warped_means > self.grid.bbox[0])
          & (warped_means < self.grid.bbox[1]),
          axis=-1,
      )
      density = jnp.where(density_is_valid, density, 0.0)

      return density

    return density

  def get_predict_density_kwargs(
      self, rng, rays, means, covs, tdist, unscented_mip_basis, **kwargs
  ):
    predict_density_kwargs = {}

    if self.grid is not None:
      # Grid/hash structures don't give us an easy way to do closed-form
      # integration with a Gaussian, so instead we sample each Gaussian
      # according to an unscented transform (or something like it) and average
      # the sampled encodings.
      control_points_key, rng = utils.random_split(rng)
      control, perp_mag = coord.compute_control_points(
          means,
          covs,
          rays,
          tdist,
          control_points_key,
          unscented_mip_basis,
          self.unscented_sqrt_fn,
          self.unscented_scale_mult,
      )
      control_offsets = control - means[Ellipsis, None, :]
      predict_density_kwargs['control_offsets'] = control_offsets
      predict_density_kwargs['perp_mag'] = perp_mag

    return dict(
        **predict_density_kwargs,
        **kwargs,
    )

  def predict_density_normals(
      self,
      rng,
      rays,
      means,
      covs,
      tdist,
      unscented_mip_basis,
      train_frac=1.0,
      train=False,
      mesh_normals=None,
  ):
    # Additional inputs to predict density
    key, rng = utils.random_split(rng)
    predict_density_kwargs = self.get_predict_density_kwargs(
        rng=key,
        rays=rays,
        means=means,
        covs=covs,
        tdist=tdist,
        unscented_mip_basis=unscented_mip_basis,
    )

    # Predict density and analytic normal
    density_key, rng = utils.random_split(rng)
    if self.disable_density_normals or mesh_normals is not None:
      raw_density, x = self.predict_density(
          means,
          covs,
          density_key=density_key,
          train_frac=train_frac,
          train=train,
          **predict_density_kwargs,
      )
      raw_grad_density = None
      normals = None
    else:
      # Evaluate the network and its gradient on the flattened input.
      def predict_density(means, covs, **kwargs):
        return self.predict_density(
            means,
            covs,
            density_key=density_key,
            train_frac=train_frac,
            train=train,
            density_only=True,
            **kwargs,
        )

      # Flatten the input so value_and_grad can be vmap'ed.
      n_flatten = len(means.shape) - 1
      gaussians_flat, pd_kwargs_flat = jax.tree_util.tree_map(
          lambda x: x.reshape((-1,) + x.shape[n_flatten:]),
          ((means, covs), predict_density_kwargs),
      )

      # Evaluate the network and its gradient on the flattened input.
      predict_density_and_grad_fn = jax.vmap(
          jax.value_and_grad(predict_density, has_aux=True),
      )
      (raw_density_flat, x_flat), raw_grad_density_flat = (
          predict_density_and_grad_fn(*gaussians_flat, **pd_kwargs_flat)
      )

      # Unflatten the output.
      raw_density = raw_density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
      raw_grad_density = raw_grad_density_flat.reshape(means.shape)

      # Compute normal vectors as negative normalized density gradient.
      # We normalize the gradient of raw (pre-activation) density because
      # it's the same as post-activation density, but is more numerically stable
      # when the activation function has a steep or flat gradient.
      # Note: In VolSDF normals are proportional to +grad, in NeRF it's -grad.

      normals = -ref_utils.l2_normalize(raw_grad_density)

    # Convert raw density to density
    density = self.convert_raw_density(raw_density, means)

    # Predicted normals
    if self.enable_pred_normals:
      grad_pred = self.pred_normals_layer(x)

      # Normalize negative predicted gradients to get predicted normal vectors.
      normals_pred = -ref_utils.l2_normalize(grad_pred)
      normals_to_use = normals_pred
    else:
      grad_pred = None
      normals_pred = None
      normals_to_use = normals

    if mesh_normals is not None:
      normals = mesh_normals
      normals_pred = mesh_normals
      normals_to_use = mesh_normals
      raw_grad_density = mesh_normals
      density = 1e5 * jnp.ones_like(density)

    return dict(
        feature=x,
        density=density,
        raw_grad_density=raw_grad_density,
        grad_pred=grad_pred,
        normals=normals,
        normals_pred=normals_pred,
        normals_to_use=normals_to_use,
    )

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      gaussians,
      tdist=None,
      train_frac=1.0,
      train = True,
      mesh_normals=None,
      zero_backfacing = None,
      **kwargs,
  ):
    means, covs = gaussians

    sampler_results = self.predict_density_normals(
        rng=rng,
        rays=rays,
        means=means,
        covs=covs,
        tdist=tdist,
        unscented_mip_basis=self.unscented_mip_basis,
        train_frac=train_frac,
        train=train,
        mesh_normals=mesh_normals,
    )

    # Zero back-facing colors
    if (
        self.backfacing_target in sampler_results
        and sampler_results[self.backfacing_target] is not None
    ):
      dotprod = math.dot(
          sampler_results[self.backfacing_target],
          -rays.directions[Ellipsis, None, :],
      )[Ellipsis, 0]

      if zero_backfacing is not None:
        sampler_results['density'] = sampler_results['density'] * (
            (dotprod > 0.0) | (tdist[Ellipsis, :-1] > self.backfacing_near)
        )

      if self.filter_backfacing:
        _, alphas, _ = render.compute_alpha_weights(
            sampler_results['density'],
            tdist,
            rays.directions,
            opaque_background=False,
        )

        sampler_results['density_multiplier'] = jnp.where(
            (
                (dotprod > self.filter_backfacing_threshold)
                | (alphas > self.filter_backfacing_alpha_threshold)
            ),
            jnp.ones_like(dotprod),
            jnp.power(
                (dotprod + 1.0) / (self.filter_backfacing_threshold + 1.0),
                self.filter_backfacing_exponent,
            ),
        )

        sampler_results['density_multiplier'] = jax.lax.stop_gradient(
            sampler_results['density_multiplier']
        )

    if self.normals_for_filter_only:
      sampler_results['normals'] = None
      sampler_results['normals_to_use'] = None
      sampler_results['normals_pred'] = None

    return sampler_results
