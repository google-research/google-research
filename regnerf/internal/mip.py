# Lint as: python3
"""Helper functions for mip-NeRF."""

import functools

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from internal import math, spacing


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                   list(x.shape[:-1]) + [-1])
  four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  if append_identity:
    return jnp.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat


def expected_sin(x, x_var, compute_var=False):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = jnp.exp(-0.5 * x_var) * math.safe_sin(x)
  if compute_var:
    y_var = jnp.maximum(
        0, 0.5 * (1 - jnp.exp(-2 * x_var) * math.safe_cos(2 * x)) - y**2)
    return y, y_var
  else:
    return y


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
    mu = (t0 + t1) / 2  # The average of the two `t` values.
    hw = (t1 - t0) / 2  # The half-width of the two `t` values.
    eps = jnp.finfo(jnp.float32).eps
    t_mean = mu + (2 * mu * hw**2) / jnp.maximum(eps, 3 * mu**2 + hw**2)
    denom = jnp.maximum(eps, 3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
    r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
  else:
    # Equations 37-39 in the paper.
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = 3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  r_var *= base_radius**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


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
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    t_vals: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = t_vals[..., :-1]
  t1 = t_vals[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    assert False
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
  """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

  Args:
    x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.
    diag: bool, if true, expects input covariances to be diagonal (full
      otherwise).

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if diag:
    x, x_cov_diag = x_coord
    scales = 2**jnp.arange(min_deg, max_deg)
    shape = list(x.shape[:-1]) + [-1]
    y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = jnp.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
  else:
    x, x_cov = x_coord
    num_dims = x.shape[-1]
    basis = jnp.concatenate(
        [2**i * jnp.eye(num_dims) for i in range(min_deg, max_deg)], 1)
    y = math.matmul(x, basis)
    # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
    # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
    y_var = jnp.sum((math.matmul(x_cov, basis)) * basis, -2)

  return expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))


def compute_alpha_weights(density, t_vals, dirs):
  """Helper function for computing alpha compositing weights."""
  t_dists = t_vals[..., 1:] - t_vals[..., :-1]
  delta = t_dists * jnp.linalg.norm(dirs[..., None, :], axis=-1)
  density_delta = density * delta

  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.concatenate([
      jnp.zeros_like(density_delta[..., :1]),
      jnp.cumsum(density_delta[..., :-1], axis=-1)
  ],
                                   axis=-1))
  weights = alpha * trans
  return weights, alpha, trans, delta


def volumetric_rendering(rgbs, weights, normals, t_vals, white_background,
                         vis_num_rays, compute_extras, delta):
  """Volumetric Rendering Function.

  Args:
    rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
    normals: jnp.ndarray(float32), normals, [batch_size, num_samples, 3].
    t_vals: jnp.ndarray(float32), [batch_size, num_samples].
    white_background: bool, If True use white as the background color, black ow.
    vis_num_rays: int, the number of rays to visualize if `compute_extras`.
    compute_extras: bool, if True, compute extra quantities besides color.
    delta: jnp.ndarray(float32), delta, [batch_size, num_samples]

  Returns:
    rendering: a dict containing an rgb image of size [batch_size, 3], and other
      visualizations if compute_extras=True.
  """
  t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])

  rgb = (weights[..., None] * rgbs).sum(axis=-2)
  acc = weights.sum(axis=-1)
  if white_background:
    rgb = rgb + (1. - acc[..., None])

  rendering = {'rgb': rgb}

  if compute_extras:
    eps = jnp.finfo(jnp.float32).eps

    rendering['acc'] = acc
    normals_map = (weights[..., None] * normals).sum(axis=-2)
    if white_background:
      normals_map = normals_map + (1. - acc[..., None])
    rendering['normals'] = normals_map

    expectation = lambda x: (weights * x).sum(axis=-1) / acc
    expectation_save = lambda x: (weights * x).sum(-1) / jnp.clip(acc, eps)

    rendering['distance_mean'] = (
        jnp.clip(
            jnp.nan_to_num(expectation(t_mids), jnp.inf), t_vals[:, 0],
            t_vals[:, -1]))
    rendering['distance_mean_save'] = (
        jnp.clip(expectation_save(t_mids), t_vals[:, 0], t_vals[:, -1]))
    rendering['distance_std'] = jnp.nan_to_num(
        jnp.maximum(
            0.,
            jnp.sqrt(expectation(t_mids**2) - rendering['distance_mean']**2)),
        0.)

    # Compute several percentiles of distance, including the median.
    # We assume t_mids are sorted.
    # ps = [5, 50, 95]
    ps = [5, 25, 50, 75, 95]
    # weights_padded = jnp.clip(weights, 1e-5)
    # tmids_save = jnp.clip(t_mids, 1e-5)
    distance_percentiles = jax.vmap(
        functools.partial(math.weighted_percentile, ps=ps, assume_sorted=True),
        0)(t_mids, weights)
    for i, p in enumerate(ps):
      s = 'median' if p == 50 else 'percentile_' + str(p)
      rendering['distance_' + s] = distance_percentiles[..., i]

    # Collect some rays to visualize directly. By naming these quantities with
    # `ray_` they get treated differently downstream --- they're treated as
    # bags of rays, rather than image chunks.
    t_vals_flat = t_vals.reshape([-1, t_vals.shape[-1]])
    weights_flat = weights.reshape([-1, weights.shape[-1]])
    rgbs_flat = rgbs.reshape([-1] + list(rgbs.shape[-2:]))
    ray_entropy = -weights / delta * jnp.log(
        jnp.clip(weights / jnp.clip(delta, eps), eps, 1 - eps))
    ray_entropy = ray_entropy.reshape([-1, ray_entropy.shape[-1]])
    rendering['ray_t_vals'] = t_vals_flat[:vis_num_rays, :]
    rendering['ray_weights'] = weights_flat[:vis_num_rays, :]
    rendering['ray_weights_full'] = weights_flat
    rendering['ray_rgbs'] = rgbs_flat[:vis_num_rays, :, :]

  return rendering


def sample_along_rays(rng,
                      origins,
                      directions,
                      radii,
                      num_samples,
                      near,
                      far,
                      genspace_fn,
                      ray_shape,
                      single_jitter,
                      diag=True):
  """Stratified sampling along the rays.

  Args:
    rng: random generator. If `None`, use deterministic sampling.
    origins: [..., 3], ray origins.
    directions: [..., 3], ray directions.
    radii: [..., 3], ray radii.
    num_samples: int.
    near: [..., 1], near-plane camera distance.
    far: [..., 1], far-plane camera distance.
    genspace_fn: Callable, the curve function used when spacing t values.
    ray_shape: string, which shape ray to assume.
    single_jitter: bool, if True, apply the same offset to each sample in a ray.
    diag: bool, if True, produce diagonal covariances (full otherwise).

  Returns:
    t_vals: [..., num_samples], sampled t values,
    (means: [..., num_samples, 3], means,
     covs: [..., num_samples, 3{, 3}], covariances, shape depends on `diag`).
  """
  t_vals = spacing.genspace(near, far, num_samples + 1, fn=genspace_fn)

  sample_shape = list(origins.shape)[:-1] + [num_samples + 1]
  if rng is None:
    # Broadcast t_vals to make the returned shape consistent.
    t_vals = jnp.broadcast_to(t_vals, sample_shape)
  else:
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = jnp.concatenate([mids, t_vals[..., -1:]], axis=-1)
    lower = jnp.concatenate([t_vals[..., :1], mids], axis=-1)
    if single_jitter:
      t_rand = random.uniform(rng, sample_shape[:-1])[..., None]
    else:
      t_rand = random.uniform(rng, sample_shape)
    t_vals = lower + (upper - lower) * t_rand

  means, covs = cast_rays(
      t_vals, origins, directions, radii, ray_shape, diag=diag)
  return t_vals, (means, covs)


def resample_along_rays(rng,
                        origins,
                        directions,
                        radii,
                        t_vals,
                        weights,
                        ray_shape,
                        stop_grad,
                        resample_padding,
                        single_jitter,
                        diag=True):
  """Resampling.

  Args:
    rng: random number generator (or None for deterministic resampling).
    origins: Tensor, [..., 3], ray origins.
    directions: Tensor, [..., 3], ray directions.
    radii: Tensor, [..., 3], ray radii.
    t_vals: Tensor, [..., num_samples+1].
    weights: Tensor, weights for t_vals
    ray_shape: string, which kind of shape to assume for the ray.
    stop_grad: bool, whether or not to backprop through sampling.
    resample_padding: float, added to the weights before normalizing.
    single_jitter: bool, if True, apply the same offset to each sample in a ray.
    diag: bool, if true, produce diagonal covariances (full otherwise).

  Returns:
    t_vals: jnp.ndarray(float32), [..., num_samples+1].
    points: jnp.ndarray(float32), [..., num_samples, 3].
  """
  # Do a blurpool.
  weights_pad = (
      jnp.concatenate([
          weights[..., :1],
          weights,
          weights[..., -1:],
      ], axis=-1))
  weights_max = jnp.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
  weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

  # Add in a constant (the sampling function will renormalize the PDF).
  weights = weights_blur + resample_padding

  new_t_vals = math.sorted_piecewise_constant_pdf(
      rng,
      t_vals,
      weights,
      t_vals.shape[-1],
      single_jitter=single_jitter,
  )
  if stop_grad:
    new_t_vals = lax.stop_gradient(new_t_vals)
  means, covs = cast_rays(
      new_t_vals, origins, directions, radii, ray_shape, diag=diag)
  return new_t_vals, (means, covs)
