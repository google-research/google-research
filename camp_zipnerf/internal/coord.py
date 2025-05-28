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

"""Tools for manipulating coordinate spaces and distances along rays."""

from internal import geopoly
from internal import math
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def contract(x):
  """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
  # Clamping to 1 produces correct scale inside |x| < 1
  x_mag_sq = jnp.maximum(1, jnp.sum(x**2, axis=-1, keepdims=True))
  scale = (2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq
  z = scale * x
  return z


def inv_contract(z):
  """The inverse of contract()."""
  # Clamping to 1 produces correct scale inside |z| < 1
  z_mag_sq = jnp.maximum(1, jnp.sum(z**2, axis=-1, keepdims=True))
  inv_scale = 2 * jnp.sqrt(z_mag_sq) - z_mag_sq
  x = z / inv_scale
  return x


def track_linearize(fn, mean, cov):
  """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: A function that can be applied to `mean`.
    mean: a tensor of Gaussian means, where the last axis is the dimension.
    cov: a tensor of covariances, where the last two axes are the dimensions.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
  if (len(mean.shape) + 1) != len(cov.shape):
    raise ValueError('cov must be non-diagonal')
  fn_mean, lin_fn = jax.linearize(fn, mean)
  fn_cov = jax.vmap(lin_fn, -1, -2)(jax.vmap(lin_fn, -1, -2)(cov))
  return fn_mean, fn_cov


def track_isotropic(fn, mean, scale):
  """Apply function `fn` to a set of means and scales, ala a Kalman filter.

  This is the isotropic or scalar equivalent of track_linearize, as we're still
  linearizing a function and tracking a Gaussian through it, but the input and
  output Gaussians are all isotropic and are only represented with a single
  `scale` value (where `scale**2` is the variance of the Gaussian).

  Args:
    fn: A function that can be applied to `mean`.
    mean: a tensor of Gaussian means, where the last axis is the dimension.
    scale: a tensor of scales, with the same shape as means[..., -1].

  Returns:
    fn_mean: the transformed means.
    fn_scale: the transformed scales.
  """
  if mean.shape[:-1] != scale.shape:
    raise ValueError(
        f'mean.shape[:-1] {mean.shape}[:-1] != scale.shape {scale.shape}.'
    )
  d = mean.shape[-1]
  fn_mean, lin_fn = jax.linearize(fn, mean)

  if scale is not None:
    # Compute the Jacobian of fn function at the locations of each mean.
    jac = jax.vmap(lin_fn, in_axes=-1, out_axes=-1)(
        jnp.broadcast_to(jnp.eye(d), mean.shape + (d,))
    )

    # The cube root of the determinant of the Jacobian is the geometric mean
    # of the eigenvalues of the Jacobian, which gives us the isotropic scaling
    # implied by `fn` at each mean that `scale` should be multiplied by.
    eps = jnp.finfo(jnp.float32).tiny  # Guard against an inf gradient at 0.
    abs_det = jnp.maximum(eps, jnp.abs(jnp.linalg.det(jac)))
    # Special case d == 3 for speed's sake.
    fn_scale = scale * (jnp.cbrt(abs_det) if d == 3 else abs_det ** (1 / d))
  else:
    fn_scale = None
  return fn_mean, fn_scale


def contract3_isoscale(x):
  """A fast version of track_isotropic(contract, *)'s scaling for 3D inputs."""
  if x.shape[-1] != 3:
    raise ValueError(f'Inputs must be 3D, are {x.shape[-1]}D.')
  norm_sq = jnp.maximum(1, jnp.sum(x**2, axis=-1))
  # Equivalent to cbrt((2 * sqrt(norm_sq) - 1) ** 2) / norm_sq:
  return jnp.exp(2 / 3 * jnp.log(2 * jnp.sqrt(norm_sq) - 1) - jnp.log(norm_sq))


def construct_ray_warps(fn, t_near, t_far, *, fn_inv=None):
  """Construct a bijection between metric distances and normalized distances.

  See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
  detailed explanation.

  Args:
    fn: the function to ray distances.
    t_near: a tensor of near-plane distances.
    t_far: a tensor of far-plane distances.
    fn_inv: Optional, if not None then it's used as the inverse of fn().

  Returns:
    t_to_s: a function that maps distances to normalized distances in [0, 1].
    s_to_t: the inverse of t_to_s.
  """
  if fn is None:
    fn_fwd = lambda x: x
    fn_inv = lambda x: x
  else:
    fn_fwd = fn
    if fn_inv is None:
      # A simple mapping from some functions to their inverse.
      inv_mapping = {
          'reciprocal': jnp.reciprocal,
          'log': jnp.exp,
          'exp': jnp.log,
          'sqrt': jnp.square,
          'square': jnp.sqrt,
      }
      fn_inv = inv_mapping[fn.__name__]
  fn_t_near, fn_t_far = [fn_fwd(t) for t in (t_near, t_far)]
  # Forcibly clip t to the range of valid values, to guard against inf's.
  t_clip = lambda t: jnp.clip(t, t_near, t_far)
  t_to_s = lambda t: (fn_fwd(t_clip(t)) - fn_t_near) / (fn_t_far - fn_t_near)
  s_to_t = lambda s: t_clip(fn_inv(s * fn_t_far + (1 - s) * fn_t_near))
  return t_to_s, s_to_t


def expected_sin(mean, var):
  """Compute the mean of sin(x), x ~ N(mean, var)."""
  return jnp.exp(-0.5 * var) * math.safe_sin(mean)  # large var -> small value.


def integrated_pos_enc(mean, var, min_deg, max_deg):
  """Encode `x` with sinusoids scaled by 2^[min_deg, max_deg).

  Args:
    mean: tensor, the mean coordinates to be encoded
    var: tensor, the variance of the coordinates to be encoded.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  scales = 2.0 ** jnp.arange(min_deg, max_deg)
  shape = mean.shape[:-1] + (-1,)
  scaled_mean = jnp.reshape(mean[Ellipsis, None, :] * scales[:, None], shape)
  scaled_var = jnp.reshape(var[Ellipsis, None, :] * scales[:, None] ** 2, shape)

  return expected_sin(
      jnp.concatenate([scaled_mean, scaled_mean + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([scaled_var] * 2, axis=-1),
  )


def lift_and_diagonalize(mean, cov, basis):
  """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
  fn_mean = math.matmul(mean, basis)
  fn_cov_diag = jnp.sum(basis * math.matmul(cov, basis), axis=-2)
  return fn_mean, fn_cov_diag


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2.0 ** jnp.arange(min_deg, max_deg)
  shape = x.shape[:-1] + (-1,)
  scaled_x = x[Ellipsis, None, :] * scales[:, None]  # (..., s, c).
  scaled_x = jnp.reshape(scaled_x, shape)  # (..., s*c).
  # Note that we're not using safe_sin, unlike IPE.
  # (..., s*c + s*c).
  four_feat = jnp.sin(
      jnp.concatenate([scaled_x, scaled_x + 0.5 * jnp.pi], axis=-1)
  )
  if append_identity:
    return jnp.concatenate([x, four_feat], axis=-1)
  else:
    return four_feat


def sqrtm(mat, return_eigs=False):
  """Take the matrix square root of a PSD matrix [..., d, d]."""
  eigvec, eigval = jax.lax.linalg.eigh(
      mat, symmetrize_input=False, sort_eigenvalues=False
  )
  scaling = math.safe_sqrt(eigval)[Ellipsis, None, :]
  sqrtm_mat = math.matmul(eigvec * scaling, jnp.moveaxis(eigvec, -2, -1))
  return (sqrtm_mat, (eigvec, eigval)) if return_eigs else sqrtm_mat


def isotropize(cov, mode='accurate'):
  """Turn covariances into isotropic covariances with the same determinant."""
  d = cov.shape[-1]
  if d == 1:
    return cov
  if mode == 'fast':
    det = jnp.linalg.det(cov)
    diag_val = det ** (1 / d)
    is_invalid = (det <= jnp.finfo(jnp.float32).tiny) | ~jnp.isfinite(det)
  elif mode == 'accurate':
    log_det = jnp.linalg.slogdet(cov)[1]
    diag_val = jnp.exp(log_det / d)
    is_invalid = ~jnp.isfinite(log_det)
  else:
    raise ValueError(f'mode={mode} not implemented.')
  cov_iso = jnp.eye(d) * diag_val[Ellipsis, None, None]
  # Guard against NaN outputs when `det` is super small. Note that this does not
  # guard against NaN gradients!
  cov_iso = jnp.where(is_invalid[Ellipsis, None, None], jnp.zeros_like(cov), cov_iso)
  return cov_iso


def construct_perp_basis(directions):
  """Construct a perpendicular basis for each 3-vector in `directions`."""
  if directions.shape[-1] != 3:
    raise ValueError(f'directions must be 3D, but is {directions.shape[-1]}D')

  # To generate a vector perpendicular to `directions`, we take a cross-product
  # with an arbitrary vector [0, 0, 1].
  cross1a = jnp.cross(directions, np.array([0.0, 0.0, 1.0]))

  # In the rare case that `directions` is very close to [0, 0, 1], we compute an
  # alternate cross-product with [1, 1, 1] to use instead.
  cross1b = jnp.cross(directions, np.array([1.0, 1.0, 1.0]))
  use_b = jnp.all(jnp.abs(cross1a) < np.finfo(np.float32).eps, axis=-1)
  cross1 = jnp.where(use_b[Ellipsis, None], cross1b, cross1a)

  # Crossing `directions` with `cross1` gives us our 3rd vector.
  cross2 = jnp.cross(directions, cross1)

  # Normalize vectors before returning them.
  normalize = lambda z: z / jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True))
  return normalize(cross1), normalize(cross2)


def hexify(rng, *, origins, directions, radii, tdist):
  """Produce hexagon-shaped samples from ray segments."""
  # Construct a base set of angles, by linspacing [0, 2pi] in a specific order.
  # This is one of two orderings of angles that doesn't induce any anisotropy
  # into the sample covariance of the multisample coordinates. Any rotation and
  # mirroring along the z-axis of this ordering is also valid.
  # There exists one alternative valid ordering, which is [0, 3, 2, 5, 4, 1].
  # This seems to work less well though likely because of the strong correlation
  # between adjacent angles.
  thetas = (np.pi / 3) * np.array([0, 2, 4, 3, 5, 1])

  # Lift the angles to the size of the rays.
  sz = tdist.shape[:-1] + (tdist.shape[-1] - 1, len(thetas))
  thetas = jnp.broadcast_to(thetas, sz)

  if rng is not None:
    # Randomly reverse the order of half of the hexes.
    key, rng = random.split(rng)
    flip = random.bernoulli(key, shape=sz[:-1])
    thetas = jnp.where(flip[Ellipsis, None], thetas[Ellipsis, ::-1], thetas)

    # Rotate each hex by some random amount.
    key, rng = random.split(rng)
    thetas += (2 * jnp.pi) * random.uniform(key, shape=sz[:-1])[Ellipsis, None]
  else:
    # If we're deterministic, flip and shift every other hex by 30 degrees.
    flip = jnp.arange(thetas.shape[-2]) % 2
    thetas = jnp.where(flip[Ellipsis, None], thetas[Ellipsis, ::-1], thetas)
    thetas += (flip * jnp.pi / 6)[Ellipsis, None]

  # TODO(barron): Plumb through the dx/dy frame for the original ray in the
  # image plane, to avoid the need of this.
  perp_axis1, perp_axis2 = construct_perp_basis(directions)

  # Grab each t-interval's midpoint and half-width.
  t0, t1 = tdist[Ellipsis, :-1], tdist[Ellipsis, 1:]
  s = (t0 + t1) / 2
  d = (t1 - t0) / 2

  # Compute the length along the ray for each multisample, using mip-NeRF math.
  cz = t0[Ellipsis, None] + math.safe_div(d, (d**2 + 3 * s**2))[Ellipsis, None] * (
      (t1**2 + 2 * s**2)[Ellipsis, None]
      + (3 / np.sqrt(7))
      * (np.arange(6) * (2 / 5) - 1)
      * math.safe_sqrt(((d**2 - s**2) ** 2 + 4 * s**4))[Ellipsis, None]
  )

  # Compute the offset from the ray for each multisample.
  perp_mag = jnp.sqrt(0.5) * radii[Ellipsis, None, :] * cz

  # Go from ray coordinate to world coordinates.
  cx = perp_mag * jnp.cos(thetas)
  cy = perp_mag * jnp.sin(thetas)
  control = (
      origins[Ellipsis, None, None, :]
      + perp_axis1[Ellipsis, None, None, :] * cx[Ellipsis, None]
      + perp_axis2[Ellipsis, None, None, :] * cy[Ellipsis, None]
      + directions[Ellipsis, None, None, :] * cz[Ellipsis, None]
  )

  return control, perp_mag


def unscented_transform(mean, cov, basis, axis=0):
  """Construct "sigma points" along `axis` from each mean and covariance."""

  d = cov.shape[-1]
  mean_ex = jnp.expand_dims(mean, axis)

  if basis == 'mean':
    # This effectively disables the unscented transform.
    return mean_ex

  if basis.startswith('random_'):
    num_random = int(basis.split('_')[-1])
    # TODO(barron): use a non-fixed random seed?
    noise = random.multivariate_normal(
        random.PRNGKey(0),
        jnp.zeros_like(mean),
        cov,
        (num_random,) + mean.shape[:-1],
    )
    control = mean_ex + jnp.moveaxis(jnp.nan_to_num(noise), 0, axis)
    return control

  sqrtm_cov = sqrtm(cov)

  if any([
      basis.startswith(x) for x in ['tetrahedron', 'icosahedron', 'octahedron']
  ]):
    # Use tessellated regular polyhedra vertices (and vec(0)) as control points.
    if d != 3:
      raise ValueError(f'Input is {d}D, but polyhedra are only defined for 3D.')
    base_shape, angular_tesselation = basis.split('_')
    transform = geopoly.generate_basis(
        base_shape, int(angular_tesselation), remove_symmetries=False
    ).T
    transform1 = np.concatenate([np.zeros((d, 1)), transform], axis=-1)
    transform1 /= np.sqrt(np.mean(transform1**2, axis=1))[:, None]
    control = mean_ex + jnp.moveaxis(
        math.matmul(sqrtm_cov, transform1), -1, axis
    )
  elif basis == 'julier':
    # The most basic symmetric unscented transformation from the original paper,
    # which yields 2*d+1 control points.
    offsets = np.sqrt(d + 0.5) * jnp.moveaxis(sqrtm_cov, -1, axis)
    control = jnp.concatenate(
        [mean_ex, mean_ex + offsets, mean_ex - offsets], axis=axis
    )
  elif basis == 'menegaz':
    # A compact unscented transformation from
    # folk.ntnu.no/skoge/prost/proceedings/cdc-ecc-2011/data/papers/2263.pdf
    # which yields d+1 control points.
    if d == 3:
      # A hand-optimized version of the d==3 case.
      sqrtm_cov_sum = jnp.sum(sqrtm_cov, axis=-1, keepdims=True)
      offsets = jnp.concatenate(
          [-sqrtm_cov_sum, 2 * sqrtm_cov - sqrtm_cov_sum / 3], axis=-1
      )
      control = mean_ex + jnp.moveaxis(offsets, -1, axis)
    else:
      transform = np.sqrt(d + 1) * np.eye(d) + (1 - np.sqrt(d + 1)) / d
      #        == sqrt((d+1)) * sqrtm(eye(d) - 1/(d+1))
      transform1 = np.concatenate([-np.ones((d, 1)), transform], axis=-1)
      control = mean_ex + jnp.moveaxis(
          math.matmul(sqrtm_cov, transform1), -1, axis
      )
  else:
    raise ValueError(f'basis={basis} not implemented.')

  return control


def compute_control_points(
    means,
    covs,
    rays,
    tdist,
    rng,
    unscented_mip_basis,
    unscented_scale_mult,
):
  """Wrapper to compute unscented control points for the MLP class."""
  if unscented_mip_basis == 'hexify':
    control, perp_mag = hexify(
        rng,
        origins=rays.origins,
        directions=rays.directions,
        radii=rays.radii,
        tdist=tdist,
    )
  else:
    # Use a normal unscented transformation.
    control = unscented_transform(
        means,
        covs,
        basis=unscented_mip_basis,
        axis=-2,
    )
    if unscented_scale_mult > 0:
      if rays is None:
        raise SyntaxError(
            'Rays are required as input if unscented_scale_mult > 0.'
        )
      # Mimic the math used by hexify to produce comparable scales.
      t_recon = jnp.sum(
          (control - rays.origins[Ellipsis, None, None, :])
          * rays.directions[Ellipsis, None, None, :],
          axis=-1,
      )
      perp_mag = jnp.sqrt(0.5) * rays.radii[Ellipsis, None, :] * t_recon
    else:
      perp_mag = None
  return control, perp_mag
