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

"""Helper functions for shooting and rendering rays."""

import jax.numpy as jnp


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[Ellipsis, None, :] * t_mean[Ellipsis, None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[Ellipsis, None] * d_outer_diag[Ellipsis, None, :]
    xy_cov_diag = r_var[Ellipsis, None] * null_outer_diag[Ellipsis, None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[Ellipsis, :, None] * d[Ellipsis, None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[Ellipsis, :, None] * (d / d_mag_sq)[Ellipsis, None, :]
    t_cov = t_var[Ellipsis, None, None] * d_outer[Ellipsis, None, :, :]
    xy_cov = r_var[Ellipsis, None, None] * null_outer[Ellipsis, None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def gaussianize_frustum(t0, t1):
  """Convert intervals along a conical frustum into means and variances."""
  # A more stable version of Equation 7 from https://arxiv.org/abs/2103.13415.
  s = t0 + t1
  d = t1 - t0
  eps = jnp.finfo(jnp.float32).eps ** 2
  ratio = d**2 / jnp.maximum(eps, 3 * s**2 + d**2)
  t_mean = s * (1 / 2 + ratio)
  t_var = (1 / 12) * d**2 - (1 / 15) * ratio**2 * (12 * s**2 - d**2)
  r_var = (1 / 16) * s**2 + d**2 * (5 / 48 - (1 / 15) * ratio)
  return t_mean, t_var, r_var


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag):
  """Approximate a 3D conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean, t_var, r_var = gaussianize_frustum(t0, t1)
  r_var *= base_radius**2
  mean, cov = lift_gaussian(d, t_mean, t_var, r_var, diag)
  return mean, cov


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0) ** 2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def _cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    tdist: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = tdist[Ellipsis, :-1]
  t1 = tdist[Ellipsis, 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    raise ValueError("ray_shape must be 'cone' or 'cylinder'")
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[Ellipsis, None, :]
  return means, covs


def get_sample_positions_along_ray(tdist, origins, directions, radii):
  return _cast_rays(tdist, origins, directions, radii, 'cone', diag=False)[0]


def compute_volume_rendering_weights(density, tdist, dirs):
  """Helper function for computing alpha compositing weights."""
  t_delta = tdist[Ellipsis, 1:] - tdist[Ellipsis, :-1]
  delta = t_delta * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)
  density_delta = density * delta
  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(
      -jnp.concatenate(
          [
              jnp.zeros_like(density_delta[Ellipsis, :1]),
              jnp.cumsum(density_delta[Ellipsis, :-1], axis=-1),
          ],
          axis=-1,
      )
  )
  weights = alpha * trans
  return weights
