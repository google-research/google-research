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

"""Unit tests for coord."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import coord
from internal import math
from internal import render
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def sample_covariance(rng, batch_size, num_dims, num_points=16):
  """Sample a random covariance matrix."""
  half_cov = jax.random.normal(rng, [batch_size, num_dims, num_points])
  cov = math.matmul(half_cov, jnp.moveaxis(half_cov, -1, -2))
  cov = (cov + jnp.moveaxis(cov, -1, -2)) / 2  # Force symmetry.
  # Rescale cov such that its determinant is 1.
  cov /= jnp.linalg.det(cov)[Ellipsis, None, None] ** (1 / num_dims)
  return cov


def stable_pos_enc(x, n):
  """A stable pos_enc for very high degrees, courtesy of Sameer Agarwal."""
  sin_x = np.sin(x)
  cos_x = np.cos(x)
  output = []
  rotmat = np.array([[cos_x, -sin_x], [sin_x, cos_x]], dtype='double')
  for _ in range(n):
    output.append(rotmat[::-1, 0, :])
    rotmat = np.einsum('ijn,jkn->ikn', rotmat, rotmat)
  return np.reshape(np.transpose(np.stack(output, 0), [2, 1, 0]), [-1, 2 * n])


def surface_stats(x):
  """Compute the sample mean and covariance along the first dimension of `x`."""
  mean = np.mean(x, axis=0)
  centered = x - mean
  cov = np.mean((centered[Ellipsis, None, :] * centered[Ellipsis, :, None]), axis=0)
  return mean, cov


def permutation_invariant_error(x, y):
  """The minimum absoute error between y and all permutations of x."""
  min_err = jnp.inf
  for perm in list(itertools.permutations(range(x.shape[0]))):
    xp = x[perm, Ellipsis]
    err = jnp.max(jnp.abs(xp - y))
    min_err = jnp.minimum(min_err, err)
  return min_err


class CoordTest(parameterized.TestCase):

  @parameterized.parameters([-4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4])
  def test_construct_ray_warps_is_finite_and_in_range(self, p):
    t_near = 0.0
    t_far = 1e8
    n = 10001
    s = jnp.linspace(0, 1, n)
    t = jnp.linspace(t_near, t_far, n)
    fn = lambda x: (x + 1) ** p - 1
    fn_inv = lambda y: (y + 1) ** (1 / p) - 1
    t_to_s, s_to_t = coord.construct_ray_warps(fn, t_near, t_far, fn_inv=fn_inv)
    t_recon = s_to_t(s)
    s_recon = t_to_s(t)
    self.assertTrue(jnp.all(jnp.isfinite(t_recon)))
    self.assertTrue(jnp.all(t_recon >= t_near))
    self.assertTrue(jnp.all(t_recon <= t_far))
    self.assertTrue(jnp.all(jnp.isfinite(s_recon)))
    self.assertTrue(jnp.all(s_recon >= 0))
    self.assertTrue(jnp.all(s_recon <= 1))

  @chex.all_variants()
  def test_construct_perp_basis(self):
    # Generate a bunch of 3-vectors.
    i = 10.0 ** np.arange(-10, 3)
    ii = np.concatenate([i[::-1], np.array([0]), i])
    dirs = np.stack(np.meshgrid(*[ii] * 3), axis=-1).reshape([-1, 3])

    # Remove rows that have very small norms.
    dirs = dirs[np.where(np.linalg.norm(dirs, axis=-1) > 1e-5)[0], :]

    # Construct a perpendicular frame.
    ax1, ax2 = self.variant(coord.construct_perp_basis)(dirs)

    # The perpendicular axes should be zero norm and should have zero dot
    # products with each other and with `directions`.
    allclose = lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5)
    allclose(np.linalg.norm(ax1, axis=-1), 1)
    allclose(np.linalg.norm(ax2, axis=-1), 1)
    allclose(np.sum(dirs * ax1, axis=-1), 0)
    allclose(np.sum(dirs * ax2, axis=-1), 0)
    allclose(np.sum(ax1 * ax2, axis=-1), 0)

  @chex.all_variants()
  @parameterized.parameters((False,), (True,))
  def test_hexify_matches_mipnerf_moments(self, randomize):
    rng = random.PRNGKey(0)

    # Generate random rays.
    key, rng = random.split(rng)
    rays = utils.generate_random_rays(
        key,
        10000,
        -1.0 + np.zeros(3),
        1.0 + np.zeros(3),
        1e-5,
        0.1,
        1,
        1,
        100,
        100,
    )

    # Apply a random scale to the ray directions so that we probe cases where
    # the directions aren't unit-norm.
    key, rng = random.split(rng)
    scales = np.exp(random.normal(key, rays.directions.shape[:-1]) / 5)  # pytype: disable=attribute-error
    rays = rays.replace(directions=rays.directions * scales[Ellipsis, None])

    # Linearly space some intervals.
    tdist = np.linspace(0, 1, 100) * (rays.far - rays.near) + rays.near

    # Cast the rays.
    means, covs = render.cast_rays(
        tdist,
        rays.origins,
        rays.directions,
        rays.radii,
        'cone',
        diag=False,
    )

    # Recover the multisample coordinates.
    rng = random.PRNGKey(0) if randomize else None
    control, _ = self.variant(coord.hexify)(
        rng,
        origins=rays.origins,
        directions=rays.directions,
        radii=rays.radii,
        tdist=tdist,
    )

    # Check that the multisample coordinates have the same means and covariances
    # as the mip-NeRF Gaussians.
    means_recon, covs_recon = surface_stats(np.moveaxis(control, -2, 0))
    np.testing.assert_allclose(means, means_recon, atol=1e-4)

    def proj_cov(c, d):
      # The variance of covariance matrix `c` along direction `d`.
      cd = jnp.matmul(c, d[Ellipsis, None, :, None])[Ellipsis, 0]
      return jnp.sum(d[Ellipsis, None, :] * cd, axis=-1)

    # Variances along the ray matches.
    np.testing.assert_allclose(
        proj_cov(covs, rays.directions),
        proj_cov(covs_recon, rays.directions),
        atol=1e-4,
    )

    # Variances with respect to radius (the sum of x-var and y-var) matches.
    normalize = lambda z: z / jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True))
    ax1 = normalize(jnp.cross(rays.directions, np.array([0, 0, 1])))
    ax2 = normalize(jnp.cross(rays.directions, ax1))
    r_var = proj_cov(covs, ax1) + proj_cov(covs, ax2)
    r_var_recon = proj_cov(covs_recon, ax1) + proj_cov(covs_recon, ax2)
    np.testing.assert_allclose(r_var, r_var_recon, atol=1e-4)

    # The whole covariance matrix is only accurate when conical frustum is
    # basically a cylinder, because the distance is must greater than the
    # radius. So here we grab some rays with very small radii and only look
    # at the last covariance matrices in each ray.
    mask = rays.radii[:, 0] < 1e-3
    assert np.sum(mask) > 100  # There better be at least 100 small rays.
    np.testing.assert_allclose(
        covs[np.where(mask)[0], -1],
        covs_recon[np.where(mask)[0], -1],
        atol=1e-4,
    )

  def test_stable_pos_enc(self):
    """Test that the stable posenc implementation works on multiples of pi/2."""
    n = 10
    x = np.linspace(-np.pi, np.pi, 5)
    z = stable_pos_enc(x, n).reshape([-1, 2, n])
    z0_true = np.zeros_like(z[:, 0, :])
    z1_true = np.ones_like(z[:, 1, :])
    z0_true[:, 0] = [0, -1, 0, 1, 0]
    z1_true[:, 0] = [-1, 0, 1, 0, -1]
    z1_true[:, 1] = [1, -1, 1, -1, 1]
    z_true = np.stack([z0_true, z1_true], axis=1)
    np.testing.assert_allclose(z, z_true, atol=1e-10)

  def test_contract_matches_special_case(self):
    """Test the math for Figure 2 of https://arxiv.org/abs/2111.12077."""
    n = 10
    _, s_to_t = coord.construct_ray_warps(jnp.reciprocal, 1, jnp.inf)
    s = jnp.linspace(0, 1 - jnp.finfo(jnp.float32).eps, n + 1)
    tc = coord.contract(s_to_t(s)[:, None])[:, 0]
    delta_tc = tc[1:] - tc[:-1]
    np.testing.assert_allclose(
        delta_tc, np.full_like(delta_tc, 1 / n), atol=1e-5, rtol=1e-5
    )

  def test_contract_is_bounded(self):
    n, d = 10000, 3
    rng = random.PRNGKey(0)
    key0, key1, rng = random.split(rng, 3)
    x = jnp.where(random.bernoulli(key0, shape=[n, d]), 1, -1) * jnp.exp(
        random.uniform(key1, [n, d], minval=-10, maxval=10)
    )
    y = coord.contract(x)
    self.assertLessEqual(jnp.max(y), 2)


  def test_contract_is_noop_when_norm_is_leq_one(self):
    n, d = 10000, 3
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    x = random.normal(key, shape=[n, d])
    xc = x / jnp.maximum(1, jnp.linalg.norm(x, axis=-1, keepdims=True))

    # Sanity check on the test itself.
    assert jnp.abs(jnp.max(jnp.linalg.norm(xc, axis=-1)) - 1) < 1e-6

    yc = coord.contract(xc)
    np.testing.assert_allclose(xc, yc, atol=1e-5, rtol=1e-5)


  def test_contract_gradients_are_finite(self):
    # Construct x such that we probe x == 0, where things are unstable.
    x = jnp.stack(jnp.meshgrid(*[jnp.linspace(-4, 4, 11)] * 2), axis=-1)
    grad = jax.grad(lambda x: jnp.sum(coord.contract(x)))(x)
    self.assertTrue(jnp.all(jnp.isfinite(grad)))

  def test_inv_contract_gradients_are_finite(self):
    z = jnp.stack(jnp.meshgrid(*[jnp.linspace(-2, 2, 21)] * 2), axis=-1)
    z = z.reshape([-1, 2])
    z = z[jnp.sum(z**2, axis=-1) < 2, :]
    grad = jax.grad(lambda z: jnp.sum(coord.inv_contract(z)))(z)
    self.assertTrue(jnp.all(jnp.isfinite(grad)))

  def test_inv_contract_inverts_contract(self):
    """Do a round-trip from metric space to contracted space and back."""
    x = jnp.stack(jnp.meshgrid(*[jnp.linspace(-4, 4, 11)] * 2), axis=-1)
    x_recon = coord.inv_contract(coord.contract(x))
    np.testing.assert_allclose(x, x_recon, atol=1e-5, rtol=1e-5)

  def test_contract3_isoscale(self):
    i = 10.0 ** jnp.arange(-16, 16, 0.5)
    ii = jnp.concatenate([-i[::-1], jnp.array([0.0]), i])
    mean = jnp.stack(np.meshgrid(*[ii] * 3, indexing='ij'), axis=-1)
    scale = jnp.ones_like(mean[Ellipsis, 0])

    # Compute our fast isoscale.
    isoscale = coord.contract3_isoscale(mean)

    # Compute the slow version that is tested elsewhere.
    _, isoscale_gt = coord.track_isotropic(coord.contract, mean, scale)

    # They should match.
    np.testing.assert_allclose(isoscale, isoscale_gt, atol=1e-6)

    # The gradient should be finite everywhere.
    grads = jax.grad(lambda z: jnp.sum(coord.contract3_isoscale(z)))(mean)
    np.testing.assert_equal(np.isfinite(grads), True)

  # TODO(barron): Make this more stable at high degrees, and see if it matters.
  @parameterized.named_parameters(
      ('05_1e-5', 5, 1e-5),
      ('10_1e-4', 10, 1e-4),
      ('15_0.005', 15, 0.005),
      ('20_0.2', 20, 0.2),  # At high degrees, our implementation is unstable.
      ('25_2', 25, 2),  # 2 is the maximum possible error.
      ('30_2', 30, 2),
  )
  def test_pos_enc(self, n, tol):
    """test pos_enc against a stable recursive implementation."""
    x = np.linspace(-np.pi, np.pi, 10001)
    z = coord.pos_enc(x[:, None], 0, n, append_identity=False)
    z_stable = stable_pos_enc(x, n)
    max_err = np.max(np.abs(z - z_stable))
    self.assertLess(max_err, tol)

  @chex.all_variants()
  @parameterized.parameters((1,), (2,))
  def test_pos_enc_matches_integrated_pos_enc_when_var_is_zero(self, dim):
    """IPE with a variance of zero must match pos_enc."""
    min_deg = -3
    max_deg = 10
    xmax = 3 * jnp.pi
    if dim == 2:
      x = np.stack(np.meshgrid(*[np.linspace(-xmax, xmax, 100)] * 2), axis=-1)
    elif dim == 1:
      x = np.linspace(-xmax, xmax, 10000)[:, None]
    z_ipe = self.variant(
        functools.partial(
            coord.integrated_pos_enc, min_deg=min_deg, max_deg=max_deg
        )
    )(x, jnp.zeros_like(x))
    z_pe = self.variant(
        functools.partial(
            coord.pos_enc,
            min_deg=min_deg,
            max_deg=max_deg,
            append_identity=False,
        )
    )(x)
    # We're using a pretty wide tolerance because IPE uses safe_sin().
    np.testing.assert_allclose(z_pe, z_ipe, atol=1e-4)

  def test_track_isotropic(self):
    rng = random.PRNGKey(0)
    batch_size = 20
    for _ in range(30):
      # Construct some random isotropic Gaussians.
      key, rng = random.split(rng)
      dims = random.randint(key, (), 1, 10)
      key, rng = random.split(rng)
      mean = jax.random.normal(key, [batch_size, dims])
      key, rng = random.split(rng)
      scale = jnp.exp(jax.random.normal(key, [batch_size]))

      # Construct a weird nonlinear function.
      def fn(x):
        return jnp.sin(x) - jnp.cos(2 * x)

      # Tracking the Gaussians through that nonlinear function.
      fn_mean, fn_scale = coord.track_isotropic(fn, mean, scale)

      # Construct full (isotropic) covariance matrices for our Gaussians.
      cov = scale[Ellipsis, None, None] ** 2 * jnp.eye(dims)
      # Track them through `fn` using the full Kalman solution.
      fn_mean_multi, fn_cov = coord.track_linearize(fn, mean, cov)
      # Isotropize the resulting covariances, grab a variance and take its sqrt.
      fn_scale_multi = jnp.sqrt(coord.isotropize(fn_cov)[Ellipsis, 0, 0])

      # The isotropized covariance solution must match the isotropic solution.
      np.testing.assert_allclose(fn_mean, fn_mean_multi, atol=1e-5)
      np.testing.assert_allclose(fn_scale, fn_scale_multi, atol=1e-5)

  def test_track_linearize(self):
    rng = random.PRNGKey(0)
    batch_size = 20
    for _ in range(30):
      # Construct some random Gaussians with dimensionalities in [1, 10].
      key, rng = random.split(rng)
      in_dims = random.randint(key, (), 1, 10)
      key, rng = random.split(rng)
      mean = jax.random.normal(key, [batch_size, in_dims])
      key, rng = random.split(rng)
      cov = sample_covariance(key, batch_size, in_dims)
      key, rng = random.split(rng)
      out_dims = random.randint(key, (), 1, 10)

      # Construct a random affine transformation.
      key, rng = random.split(rng)
      a_mat = jax.random.normal(key, [int(out_dims), int(in_dims)])
      key, rng = random.split(rng)
      b = jax.random.normal(key, [int(out_dims)])

      def fn(x):
        x_vec = x.reshape([-1, x.shape[-1]])
        y_vec = jax.vmap(lambda z: math.matmul(a_mat, z))(x_vec) + b  # pylint:disable=cell-var-from-loop
        y = y_vec.reshape(list(x.shape[:-1]) + [y_vec.shape[-1]])
        return y

      # Apply the affine function to the Gaussians.
      fn_mean_true = fn(mean)
      fn_cov_true = math.matmul(math.matmul(a_mat, cov), a_mat.T)

      # Tracking the Gaussians through a linearized function of a linear
      # operator should be the same.
      fn_mean, fn_cov = coord.track_linearize(fn, mean, cov)
      np.testing.assert_allclose(fn_mean, fn_mean_true, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(fn_cov, fn_cov_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  @parameterized.parameters(('fast',), ('accurate',))
  def test_isotropize_output_is_correct_and_finite(self, mode):
    batch_size = 20
    fn = self.variant(functools.partial(coord.isotropize, mode=mode))
    # Construct some random Gaussians with dimensionalities in [1, 5].
    for d in range(1, 5):
      cov_unscaled = sample_covariance(random.PRNGKey(0), batch_size, d)
      det_unscaled = np.exp(np.linalg.slogdet(cov_unscaled)[1])
      scales = 10 ** np.arange(-50, 10, dtype=np.float32)
      min_accurate_scale = 10.0**-7 if mode == 'fast' else 10.0**-30
      for scale in scales:
        cov = scale * cov_unscaled
        cov_isotropic = fn(cov)

        # Check that cov_isotropic is finite.
        self.assertTrue(jnp.all(jnp.isfinite(cov_isotropic)))

        # Check that cov_isotropic is diagonal.
        expand_diag = jnp.vectorize(jnp.diag, signature='(d)->(d,d)')
        extract_diag = jnp.vectorize(jnp.diag, signature='(d,d)->(d)')
        np.testing.assert_array_equal(
            expand_diag(extract_diag(cov_isotropic)), cov_isotropic
        )

        if scale >= min_accurate_scale:
          # Test that cov_isotropic's determinant matches cov's.
          det = np.exp(np.linalg.slogdet(cov_isotropic)[1])
          det_true = det_unscaled * scale**d
          np.testing.assert_allclose(det, det_true, rtol=1e-5)
        else:
          # if the scale is too small to get right, just check that cov is
          # close to 0.
          self.assertLessEqual(jnp.max(jnp.abs(cov_isotropic)), 1e-5)

  @chex.all_variants()
  @parameterized.parameters(('fast',), ('accurate',))
  def test_isotropize_gradient_is_finite(self, mode):
    batch_size = 20
    fn = lambda z: jnp.sum(functools.partial(coord.isotropize, mode=mode)(z))
    grad_fn = self.variant(jax.vmap(jax.grad(fn)))
    # Construct some random Gaussians with dimensionalities in [1, 5].
    for d in range(1, 5):
      cov_unscaled = sample_covariance(random.PRNGKey(0), batch_size, d)
      if mode == 'fast':
        scales = 10 ** np.arange(-6, 10, dtype=np.float32)
      else:
        scales = 10 ** np.arange(-30, 10, dtype=np.float32)
      for scale in scales:
        cov = scale * cov_unscaled
        grad = grad_fn(cov)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))

  @parameterized.named_parameters(
      ('reciprocal', jnp.reciprocal),
      ('log', jnp.log),
      ('sqrt', jnp.sqrt),
      ('noop', None),
  )
  def test_construct_ray_warps_extents(self, fn):
    n = 100
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t_near = jnp.exp(jax.random.normal(key, [n]))
    key, rng = random.split(rng)
    t_far = t_near + jnp.exp(jax.random.normal(key, [n]))

    t_to_s, s_to_t = coord.construct_ray_warps(fn, t_near, t_far)

    np.testing.assert_allclose(
        t_to_s(t_near), jnp.zeros_like(t_near), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        t_to_s(t_far), jnp.ones_like(t_far), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        s_to_t(jnp.zeros_like(t_near)), t_near, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        s_to_t(jnp.ones_like(t_near)), t_far, atol=1e-5, rtol=1e-5
    )

  def test_construct_ray_warps_special_reciprocal(self):
    """Test fn=1/x against its closed form."""
    n = 100
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t_near = jnp.exp(jax.random.normal(key, [n]))
    key, rng = random.split(rng)
    t_far = t_near + jnp.exp(jax.random.normal(key, [n]))

    key, rng = random.split(rng)
    u = jax.random.uniform(key, [n])
    t = t_near * (1 - u) + t_far * u
    key, rng = random.split(rng)
    s = jax.random.uniform(key, [n])

    t_to_s, s_to_t = coord.construct_ray_warps(jnp.reciprocal, t_near, t_far)

    # Special cases for fn=reciprocal.
    s_to_t_ref = lambda s: 1 / (s / t_far + (1 - s) / t_near)
    t_to_s_ref = lambda t: (t_far * (t - t_near)) / (t * (t_far - t_near))

    np.testing.assert_allclose(t_to_s(t), t_to_s_ref(t), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(s_to_t(s), s_to_t_ref(s), atol=1e-5, rtol=1e-5)

  def test_expected_sin(self):
    normal_samples = random.normal(random.PRNGKey(0), (10000,))
    for mu, var in [(0, 1), (1, 3), (-2, 0.2), (10, 10)]:
      sin_mu = coord.expected_sin(mu, var)
      x = jnp.sin(jnp.sqrt(var) * normal_samples + mu)
      np.testing.assert_allclose(sin_mu, jnp.mean(x), atol=1e-2)

  @chex.all_variants()
  def test_integrated_pos_enc_when_degrees_are_large(self):
    min_deg = -100
    max_deg = 100
    mean = jnp.array([0.0])
    var = jnp.array([1.0])
    fn = functools.partial(
        coord.integrated_pos_enc, min_deg=min_deg, max_deg=max_deg
    )
    z = self.variant(fn)(mean, var)
    self.assertTrue(jnp.all(jnp.isfinite(z)))
    z0, z1 = tuple(z.reshape((-1, len(z) // 2)))
    np.testing.assert_array_equal(z0, 0)
    np.testing.assert_array_equal(z1[:80], 1.0)
    np.testing.assert_array_equal(z1[-80:], 0.0)

  @chex.all_variants()
  @parameterized.parameters((1,), (2,), (3,))
  def test_integrated_pos_enc_against_brute_force(self, num_dims):
    min_deg = -1
    max_deg = 4
    num_samples = 100000
    rng = random.PRNGKey(0)
    for _ in range(5):
      # Generate a coordinate's mean and covariance matrix.
      key, rng = random.split(rng)
      mean = random.normal(key, (num_dims,))
      key, rng = random.split(rng)
      half_cov = jax.random.normal(key, [num_dims] * 2)
      cov = half_cov @ half_cov.T
      var = jnp.diag(cov)

      # Generate an IPE.
      fn = functools.partial(
          coord.integrated_pos_enc, min_deg=min_deg, max_deg=max_deg
      )
      enc = self.variant(fn)(mean, var)

      # Draw samples, encode them, and take their mean.
      key, rng = random.split(rng)
      samples = random.multivariate_normal(key, mean, cov, [num_samples])
      enc_samples = coord.pos_enc(
          samples, min_deg, max_deg, append_identity=False
      )
      enc_gt = jnp.mean(enc_samples, 0)

      np.testing.assert_allclose(enc, enc_gt, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  absltest.main()
