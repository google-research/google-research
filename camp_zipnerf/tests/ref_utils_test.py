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

"""Tests for ref_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import ref_utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy.special


def generate_dir_enc_fn_scipy(deg_view):
  """Return spherical harmonics using scipy.special.sph_harm."""
  ml_array = ref_utils.get_ml_array(deg_view)

  def dir_enc_fn(theta, phi):
    de = [scipy.special.sph_harm(m, l, phi, theta) for m, l in ml_array.T]
    de = np.stack(de, axis=-1)
    # Split into real and imaginary parts.
    return np.concatenate([np.real(de), np.imag(de)], axis=-1)

  return dir_enc_fn


def old_l2_normalize(x, eps=jnp.finfo(jnp.float32).eps):
  """The L2 normalization used in the ref-nerf paper."""
  return x / jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=-1, keepdims=True), eps))


class RefUtilsTest(parameterized.TestCase):

  @chex.all_variants()
  @parameterized.parameters(list(10.0 ** np.arange(-40, 10)))
  def test_l2_normalize_gradients_are_finite(self, scale):
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    x = scale * (2 * random.uniform(key, shape=(10000, 3)) - 1)
    grad = self.variant(
        jax.vmap(jax.grad(lambda x: jnp.sum(ref_utils.l2_normalize(x))))
    )(x)
    np.testing.assert_equal(np.all(np.isfinite(grad)), True)

  @chex.all_variants()
  @parameterized.parameters(list(10.0 ** np.arange(-16, 10)))
  def test_l2_normalize_is_accurate(self, scale):
    # Construct the cartesian product of [-1, 0, 1]^3.
    xyz = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3), axis=-1).reshape(
        [-1, 3]
    )
    normals_true = np.nan_to_num(
        xyz / np.sqrt(np.sum(xyz**2, axis=-1, keepdims=True))
    )
    normals = self.variant(ref_utils.l2_normalize)(scale * xyz)
    np.testing.assert_allclose(normals, normals_true, rtol=1e-6)

  @chex.all_variants()
  @parameterized.parameters(list(10.0 ** np.arange(-16, 10)))
  def test_l2_normalize_gradient_is_accurate(self, scale):
    # Construct the cartesian product of [-1, 0, 1]^3.
    xyz = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3), axis=-1).reshape(
        [-1, 3]
    )

    # Ask Jax for the gradient of the x-normal. Note that we set grad_eps = 0.
    fn = jax.vmap(
        jax.grad(lambda z: ref_utils.l2_normalize(z, grad_eps=0)[Ellipsis, 0])
    )
    grad = self.variant(fn)(scale * xyz)[:, 0]

    # An analytical solution to the x-normal gradient.
    denom = np.sqrt(np.sum(xyz**2, axis=-1))
    grad_true = (xyz[:, 1] ** 2 + xyz[:, 2] ** 2) / (scale * denom**3)
    grad_true = np.nan_to_num(grad_true)

    np.testing.assert_allclose(grad, grad_true, atol=1e-6 / scale, rtol=1e-6)

  @chex.all_variants()
  @parameterized.parameters(list(10.0 ** np.arange(-16, 10)))
  def test_l2_normalize_gradient_matches_old_implementation(self, scale):
    # Construct the cartesian product of [-1, 0, 1]^3.
    xyz = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3), axis=-1).reshape(
        [-1, 3]
    )
    # Remove the all-zero entry.
    xyz = xyz[~np.all(xyz == 0, axis=-1), :]

    # Ask Jax for the gradient of the x-normal.
    fn = jax.vmap(jax.grad(lambda z: ref_utils.l2_normalize(z)[Ellipsis, 0]))
    grad = self.variant(fn)(scale * xyz)[:, 0]

    old_fn = jax.vmap(jax.grad(lambda z: old_l2_normalize(z)[Ellipsis, 0]))
    old_grad = old_fn(scale * xyz)[:, 0]

    np.testing.assert_allclose(grad, old_grad, atol=1e-4, rtol=1e-4)

  @parameterized.parameters((1e10,), (1e5,), (1,), (1e-5,), (1e-10,), (0.0))
  def test_orientation_loss_gradients_are_finite(self, scale):
    rng = random.PRNGKey(0)
    n, d = 10000, 3

    key, rng = random.split(rng)
    x_grad = scale * random.normal(key, shape=(n, d))

    key, rng = random.split(rng)
    v = ref_utils.l2_normalize(random.normal(key, shape=(n, d)))

    key, rng = random.split(rng)
    w = random.uniform(key, shape=(n,))

    def fn(x_grad):
      return ref_utils.orientation_loss(w, ref_utils.l2_normalize(x_grad), v)

    grad = jax.grad(fn)(x_grad)
    np.testing.assert_equal(np.all(np.isfinite(grad)), True)

  def test_reflection(self):
    """Make sure reflected vectors have the same angle from normals as input."""
    rng = random.PRNGKey(0)
    for shape in [(45, 3), (4, 7, 3)]:
      key, rng = random.split(rng)
      normals = random.normal(key, shape)
      key, rng = random.split(rng)
      directions = random.normal(key, shape)

      # Normalize normal vectors.
      normals = normals / (
          jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
      )

      reflected_directions = ref_utils.reflect(directions, normals)

      cos_angle_original = jnp.sum(directions * normals, axis=-1)
      cos_angle_reflected = jnp.sum(reflected_directions * normals, axis=-1)

      np.testing.assert_allclose(
          cos_angle_original, cos_angle_reflected, atol=1e-5, rtol=1e-5
      )

  def test_spherical_harmonics(self):
    """Make sure the fast spherical harmonics are accurate."""
    shape = (12, 11, 13)

    # Generate random points on sphere.
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    theta = random.uniform(key1, shape, minval=0.0, maxval=jnp.pi)
    phi = random.uniform(key2, shape, minval=0.0, maxval=2.0 * jnp.pi)

    # Convert to Cartesian coordinates.
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    xyz = jnp.stack([x, y, z], axis=-1)

    deg_view = 5
    de = ref_utils.generate_dir_enc_fn(deg_view)(xyz)
    de_scipy = generate_dir_enc_fn_scipy(deg_view)(theta, phi)

    np.testing.assert_allclose(
        de, de_scipy, atol=0.02, rtol=1e6
    )  # Only use atol.
    self.assertFalse(jnp.any(jnp.isnan(de)))


if __name__ == '__main__':
  absltest.main()
