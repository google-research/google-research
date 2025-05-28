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
from google_research.yobo.internal import geometry
from google_research.yobo.internal import math
from google_research.yobo.internal import render
from google_research.yobo.internal import stepfun
from google_research.yobo.internal import utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


@gin.configurable
class ProposalVolumeSampler(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""

  config: Any = None  # A Config class, must be set upon construction.
  # A list of tuples (mlp_idx, grid_idx, num_samples) for each sampling round.
  # This code defaults to what the mip-NeRF 360 codebase used, which was three
  # rounds of sampling using one "proposal" MLP and one "NeRF" MLP and no grids.

  sampling_strategy: Tuple[Tuple[int, int, int], Ellipsis] = (
      (0, None, 64),
      (0, None, 64),
      (1, None, 32),
  )

  # The specific parameters for the MLPs + grids used by this model. The length
  # of these tuples also determines how many MLPs/grids will get constructed.
  # The user must ensure that the number of MLPs/grids matches the config in
  # `sampling_strategy` or else this code will not run.
  mlp_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = (
      {},
      {},
  )

  # Grid is disabled by default.
  grid_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = ()

  anneal_slope: float = 10  # Higher = more rapid annealing.
  anneal_end: float = 1.0  # Higher = more rapid annealing.
  anneal_clip: float = 1.0  # Higher = more rapid annealing.
  sampling_anneal_rate: float = 0.025  # Higher = more rapid annealing.
  sampling_anneal_blur_start: float = 1.0  # Higher = more rapid annealing.
  sampling_anneal_blur_stop: float = 0.05  # Higher = more rapid annealing.
  stop_level_grad: bool = True  # If True, don't backprop across levels.

  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  disable_integration: bool = False  # If True, use PE instead of IPE.
  single_jitter: bool = True  # If True, jitter whole rays instead of samples.
  dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
  dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
  near_anneal_rate: Optional[float] = None  # How fast to anneal in near bound.
  near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
  resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.

  normalize_weights: bool = False  # If True, use PE instead of IPE.

  # The curve used for ray distances. Can be just a function like @jnp.log,
  # or can be of the form (fn, fn_inv, **kwargs), like
  # (@math.power_ladder, @math.inv_power_ladder, {'p': -2, 'premult': 10})
  raydist_fn: Union[Tuple[Callable[Ellipsis, Any], Ellipsis], Callable[Ellipsis, Any]] = None
  grid_representation: str = 'ngp'

  opaque_background: bool = False  # If true, make the background opaque.

  def setup(self):
    self.mlps = [
        geometry.DensityMLP(
            name=f'MLP_{i}', grid_params=self.grid_params_per_level[i], **params
        )
        for i, params in enumerate(self.mlp_params_per_level)
    ]

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac = 1.0,
      train = True,
      mesh=None,
      use_mesh=True,
      deterministic=False,
      stopgrad_proposal=False,
      stopgrad_weights=False,
      **render_kwargs,
  ):
    """The mip-NeRF Model.

    Args:
      rng: random number generator (or None for deterministic output).
      rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      percentiles: depth will be returned for these percentiles.
      train: Set to True when training.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
    # Setup mesh
    mesh_t = None
    mesh_points = None
    mesh_normals = None
    mesh_valid = None

    if mesh is not None:
      (mesh_t, mesh_points, mesh_normals, tri_normals, mesh_valid) = (
          mesh.intersect(rays.origins, rays.directions)
      )
      tri_normals = jnp.where(
          math.dot(tri_normals, rays.directions) < 0, tri_normals, -tri_normals
      )
      mesh_normals = jnp.where(
          math.dot(mesh_normals, rays.directions) < 0,
          mesh_normals,
          -mesh_normals,
      )

      if self.config.use_mesh_face_normals:
        mesh_normals = tri_normals

      mesh_normals = mesh_normals[Ellipsis, None, :]

    # Define the mapping from normalized to metric ray distance.
    if isinstance(self.raydist_fn, tuple):
      fn, fn_inv, raydist_kwargs = self.raydist_fn  # pylint: disable=unpacking-non-sequence
      t_to_s, s_to_t = coord.construct_ray_warps(
          functools.partial(fn, **raydist_kwargs),
          rays.near,
          rays.far,
          fn_inv=functools.partial(fn_inv, **raydist_kwargs),
      )
    else:
      raydist_kwargs = {}
      t_to_s, s_to_t = coord.construct_ray_warps(
          self.raydist_fn, rays.near, rays.far
      )

    # Initialize the range of (normalized) distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    # `near_anneal_rate` can be used to anneal in the near bound at the start
    # of training, eg. 0.1 anneals in the bound over the first 10% of training.
    if self.near_anneal_rate is None:
      init_s_near = 0.0
    else:
      init_s_near = jnp.clip(
          1 - train_frac / self.near_anneal_rate, 0, self.near_anneal_init
      )
    init_s_far = 1.0

    sdist = jnp.concatenate(
        [
            jnp.full_like(rays.near, init_s_near),
            jnp.full_like(rays.far, init_s_far),
        ],
        axis=-1,
    )

    resample_weights = jnp.ones_like(rays.near)
    resample_alphas = jnp.ones_like(rays.near)

    ray_history = []
    mlp_was_used = [False] * len(self.mlps)

    prod_num_samples = 1

    for i_level, (i_mlp, _, num_samples) in enumerate(self.sampling_strategy):
      if (
          use_mesh
          and mesh is not None
          and i_level < len(self.sampling_strategy) - 1
      ):
        continue

      mlp = self.mlps[i_mlp]
      mlp_was_used[i_mlp] = True

      if not use_mesh or mesh is None:
        # Dilate by some multiple of the expected span of each current interval,
        # with some bias added in.
        dilation = (
            self.dilation_bias
            + self.dilation_multiplier
            * (init_s_far - init_s_near)
            / prod_num_samples
        )

        # After the first level (where dilation would be a no-op) optionally
        # dilate the interval weights along each ray slightly so that they're
        # overestimates, which can reduce aliasing.
        use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0

        if prod_num_samples > 1 and use_dilation:
          sdist, resample_weights = stepfun.max_dilate_weights(
              sdist,
              resample_weights,
              dilation,
              domain=(init_s_near, init_s_far),
              renormalize=True,
          )
          sdist = sdist[Ellipsis, 1:-1]
          resample_weights = resample_weights[Ellipsis, 1:-1]

        # Record the product of the number of samples seen so far.
        prod_num_samples *= num_samples

        # Optionally anneal the weights as a function of training iteration.
        if self.anneal_slope > 0:
          # Schlick's bias function, see https://arxiv.org/abs/2010.09714
          bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
          anneal = jnp.clip(
              bias(train_frac / self.anneal_end, self.anneal_slope),
              0.0,
              self.anneal_clip,
          )
        else:
          anneal = self.anneal_clip

        # A slightly more stable way to compute weights**anneal. If the distance
        # between adjacent intervals is zero then its weight is fixed to 0.
        logits_resample = jnp.where(
            sdist[Ellipsis, 1:] > sdist[Ellipsis, :-1],
            anneal * math.safe_log(resample_weights + self.resample_padding),
            -jnp.inf,
        )

        # Draw sampled intervals from each ray's current weights.
        if deterministic:
          key = None
        else:
          key, rng = utils.random_split(rng)

        sdist = stepfun.sample_intervals(
            key,
            sdist,
            logits_resample,
            num_samples,
            single_jitter=self.single_jitter,
            domain=(init_s_near, init_s_far),
        )

        # Optimization will usually go nonlinear if you propagate gradients
        # through sampling.
        if self.stop_level_grad:
          sdist = jax.lax.stop_gradient(sdist)

        # Convert normalized distances to metric distances.
        tdist = s_to_t(sdist)

        # Cast our rays, by turning our distance intervals into Gaussians.
        gaussians = render.cast_rays(
            tdist,
            rays.origins,
            rays.directions,
            rays.radii,
            self.ray_shape,
            diag=False,
        )
      else:
        means = rays.origins + rays.directions * mesh_t[Ellipsis, None]
        means = means[Ellipsis, None, :]
        covs = rays.radii.ravel()[0] * jnp.eye(3)
        covs = jnp.broadcast_to(covs, means.shape + (3,))
        gaussians = (means, covs)

        tdist = jnp.concatenate(
            [
                jnp.zeros_like(means[Ellipsis, 0]) * (mesh_t[Ellipsis, None]),
                jnp.ones_like(means[Ellipsis, 0]) * (mesh_t[Ellipsis, None] + 0.1),
            ],
            axis=-1,
        )

      if self.disable_integration:
        # Setting the covariance of our Gaussian samples to 0 disables the
        # "integrated" part of integrated positional encoding.
        gaussians = (gaussians[0], jnp.zeros_like(gaussians[1]))

      # # NOTE: Debugging
      # means = rays.origins + rays.directions * mesh_t[..., None]
      # means = means[..., None, :]
      # covs = rays.radii.ravel()[0] * jnp.eye(3)
      # covs = jnp.broadcast_to(covs, means.shape + (3,))
      # gaussians = (
      #     means * jnp.ones_like(gaussians[0]),
      #     covs * jnp.ones_like(gaussians[1]),
      # )

      # Push our Gaussians through the MLP.
      key, rng = utils.random_split(rng)
      ray_results = mlp(
          rng=key,
          rays=rays,
          gaussians=gaussians,
          tdist=tdist,
          train_frac=train_frac,
          train=train,
          mesh_normals=mesh_normals if use_mesh else None,
          **render_kwargs,
          **raydist_kwargs,
      )

      # Compute "rectified" versions of all all normals, where surfaces facing
      # away from the camera have their sign flipped so that they face the
      # camera (note that flipping the sign of a normal has no effect on the
      # mirrored directions used by ref-nerf).
      rectified = {}

      for k, v in ray_results.items():
        if k.startswith('normals') and v is not None:
          p = jnp.sum(v * rays.viewdirs[Ellipsis, None, :], axis=-1, keepdims=True)
          rectified[k + '_rectified'] = v * jnp.where(p > 0, -1, 1)

      ray_results.update(rectified)

      # Get the weights used by volumetric rendering (and our other losses).
      weights, alphas, _ = render.compute_alpha_weights(
          ray_results['density'],
          tdist,
          rays.directions,
          opaque_background=self.opaque_background,
      )

      # Filter weights
      if 'density_multiplier' in ray_results:
        resample_weights, _, _ = render.compute_alpha_weights(
            ray_results['density'] * ray_results['density_multiplier'],
            tdist,
            rays.directions,
            opaque_background=self.opaque_background,
        )
        resample_alphas = alphas
      else:
        resample_weights = weights
        resample_alphas = alphas

      # Normalize weights
      if self.normalize_weights:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

      # Set weights to 1
      if mesh is not None and use_mesh:
        weights = jnp.ones_like(weights)
      elif mesh is not None:
        mesh_points = (rays.origins + rays.directions * mesh_t[Ellipsis, None])[
            Ellipsis, None, :
        ] * jnp.ones_like(gaussians[0])
        ray_results['mesh_points'] = mesh_points

        mesh_normals = mesh_normals[Ellipsis, 0:1, :] * jnp.ones_like(gaussians[0])
        ray_results['mesh_normals'] = mesh_normals

        ray_results['t_to_nearest'] = jnp.linalg.norm(
            mesh_points - rays.origins[Ellipsis, None, :],
            axis=-1,
            keepdims=True,
        ) - jnp.linalg.norm(
            gaussians[0] - rays.origins[Ellipsis, None, :],
            axis=-1,
            keepdims=True,
        )

      ray_results['points'] = gaussians[0]
      ray_results['means'] = gaussians[0]
      ray_results['covs'] = gaussians[1]
      ray_results['tdist'] = jnp.copy(tdist)
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['prod_num_samples'] = prod_num_samples
      ray_results['opaque_background'] = self.opaque_background

      if stopgrad_weights:
        ray_results['weights'] = jax.lax.stop_gradient(jnp.copy(weights))
      else:
        ray_results['weights'] = jnp.copy(weights)

      if stopgrad_proposal and (i_level < len(self.sampling_strategy) - 1):
        ray_results = jax.tree_util.tree_map(jax.lax.stop_gradient, ray_results)

      ray_history.append(ray_results)

    if not all(mlp_was_used):
      s = ', '.join([f'{i}' for i, v in enumerate(mlp_was_used) if not v])
      print(f'MLPs {s} not used by the sampling strategy.')

    return ray_history
