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

"""Unit tests for math."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import math
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

# Some of the tests here will fail on GPU, because we're comparing against
# Numpy which runs on a CPU.
jax.config.update('jax_platform_name', 'cpu')


def safe_trig_harness(fn, max_exp):
  x = 10 ** np.linspace(-30, max_exp, 10000)
  x = np.concatenate([-x[::-1], np.array([0]), x])
  y_true = getattr(np, fn)(x)
  y = getattr(math, 'safe_' + fn)(x)
  return y_true, y


def positive_floats(n, include_zero=False, include_inf=False):
  lo = np.finfo(np.float32).tiny
  hi = np.finfo(np.float32).max
  log_x = np.linspace(np.log(lo), np.log(hi), n - include_inf - include_zero)
  x = np.float32(np.exp(log_x))
  x[0] = lo
  x[-1] = hi
  if include_zero:
    x = np.concatenate([np.array([np.float32(0)]), x])
  if include_inf:
    x = np.concatenate([x, np.array([np.float32(np.inf)])])
  return x


def all_floats(n, include_inf=False):
  x = positive_floats(n // 2, include_zero=False, include_inf=include_inf)
  x = np.concatenate([-x[::-1], np.array([0], dtype=np.float32), x])
  return x


class MathTest(parameterized.TestCase):

  @parameterized.parameters((1,), (1e-1,), (1e-2,), (1e-3))
  def test_laplace_cdf_gradient(self, beta):
    x = jnp.linspace(-1, 1, 10001)
    grad_laplace = jax.vmap(jax.grad(lambda z: math.laplace_cdf(z, beta)))(x)
    laplace_pdf = -(1 / beta) * (1 / beta) * 0.5 * jnp.exp(-jnp.abs(x) / beta)
    np.testing.assert_allclose(grad_laplace, laplace_pdf, atol=1e-5, rtol=1e-5)

  def test_plus_eps(self):
    x = all_floats(10001)[:-1]
    xp = math.plus_eps(x)
    np.testing.assert_array_less(x, xp)
    np.testing.assert_equal(bool(jnp.all(xp != x)), True)
    np.testing.assert_equal(bool(jnp.all(((xp - x) * 0.49 + x) == x)), True)

  def test_minus_eps(self):
    x = all_floats(10001)[1:]
    xn = math.minus_eps(x)
    np.testing.assert_array_less(xn, x)
    np.testing.assert_equal(bool(jnp.all(xn != x)), True)
    np.testing.assert_equal(bool(jnp.all(((xn - x) * 0.49 + x) == x)), True)

  def test_plus_eps_grad(self):
    x = all_floats(10001)[:-1]
    np.testing.assert_array_equal(jax.vmap(jax.grad(math.plus_eps))(x), 1.0)

  def test_minus_eps_grad(self):
    x = all_floats(10001)[1:]
    np.testing.assert_array_equal(jax.vmap(jax.grad(math.minus_eps))(x), 1.0)

  def test_clip_finite_nograd(self):
    x = all_floats(10001, include_inf=True)
    xc = math.clip_finite_nograd(x)
    np.testing.assert_array_equal(xc[1:-1], x[1:-1])
    np.testing.assert_array_equal(xc[-1], jnp.finfo(jnp.float32).max)
    np.testing.assert_array_equal(xc[0], jnp.finfo(jnp.float32).min)
    mult = 10.0
    g = jax.vmap(jax.grad(lambda z: math.clip_finite_nograd(mult * z)))(x)
    np.testing.assert_array_equal(g, mult)

  def test_clip_pos(self):
    x = all_floats(10001, include_inf=True)
    xp = math.clip_pos(x)
    np.testing.assert_array_less(0, xp)
    pos = x > 0
    non_pos = x <= 0
    np.testing.assert_array_equal(xp[non_pos], jnp.finfo(jnp.float32).tiny)
    np.testing.assert_array_equal(xp[pos], x[pos])
    mult = 10.0
    g = jax.vmap(jax.grad(lambda z: math.clip_pos(mult * z)))(x)
    np.testing.assert_array_equal(g[pos], mult)
    np.testing.assert_array_equal(g[non_pos], 0.0)

  def test_remove_zero(self):
    x = all_floats(10001, include_inf=True)
    xr = math.remove_zero(x)
    non_zero = x != 0
    np.testing.assert_array_equal(x[non_zero], xr[non_zero])
    is_zero = x == 0
    np.testing.assert_array_equal(xr[is_zero], np.finfo(np.float32).tiny)
    mult = 10.0
    g = jax.vmap(jax.grad(lambda z: math.remove_zero(mult * z)))(x)
    np.testing.assert_array_equal(g == mult, x != 0)

  def test_sin(self):
    """In [-1e10, 1e10] safe_sin and safe_cos are accurate."""
    for fn in ['sin', 'cos']:
      y_true, y = safe_trig_harness(fn, 10)
      self.assertLess(jnp.max(jnp.abs(y - y_true)), 1e-4)
      self.assertFalse(jnp.any(jnp.isnan(y)))

  def test_safe_exp_correct(self):
    """math.safe_exp() should match np.exp() for not-huge values."""
    x = all_floats(10001, include_inf=False)
    yg_true = jnp.exp(x)  # exp() is its own derivative.
    y = math.safe_exp(x)
    g = jax.vmap(jax.grad(math.safe_exp))(x)
    valid = np.isfinite(yg_true)
    np.testing.assert_array_equal(y[valid], yg_true[valid])
    np.testing.assert_allclose(g[valid], yg_true[valid], rtol=1e-5)

  def test_safe_exp_finite(self):
    """math.safe_exp() behaves reasonably for all inputs."""
    _, y, g = self._test_finite(math.safe_exp)
    # The derivative of exp() should be exp().
    np.testing.assert_allclose(y, g)
    # safe_exp()'s output and gradient should be monotonic.
    self.assertTrue(jnp.all(y[1:] >= y[:-1]))
    self.assertTrue(jnp.all(g[1:] >= g[:-1]))

  def test_safe_log_correct(self):
    """math.safe_log() should match np.log() for reasonable values."""
    x_log_true = np.linspace(-80, 80, 10001, dtype=np.float64)
    x = np.exp(x_log_true)
    x_log = math.safe_log(x)
    grad_x_log = jax.vmap(jax.grad(math.safe_log))(x)
    grad_x_log_true = jax.vmap(jax.grad(jnp.log))(x)
    np.testing.assert_allclose(x_log, x_log_true, rtol=1e-5)
    np.testing.assert_allclose(grad_x_log, grad_x_log_true)

  def test_safe_log_finite(self):
    """math.safe_exp() behaves reasonably for all inputs."""
    x, y, g = self._test_finite(math.safe_log)

    # log and its gradient on non-negative values should be monotonic.
    pos_idx = jnp.where(x > 0)[0][0]
    self.assertTrue(jnp.all(y[pos_idx + 1 :] >= y[pos_idx:-1]))
    self.assertTrue(jnp.all(g[pos_idx + 1 :] <= g[pos_idx:-1]))

  def test_safe_sqrt_correct(self):
    """math.safe_sqrt() should match np.sqrt() for reasonable values."""
    x = np.exp(np.linspace(-80, 80, 10001, dtype=np.float64))
    x_sqrt = math.safe_sqrt(x)
    x_sqrt_true = np.sqrt(x)
    grad_x_sqrt = jax.vmap(jax.grad(math.safe_sqrt))(x)
    grad_x_sqrt_true = jax.vmap(jax.grad(jnp.sqrt))(x)
    np.testing.assert_array_equal(math.safe_sqrt(0.0), 0.0)
    np.testing.assert_allclose(x_sqrt, x_sqrt_true, rtol=1e-5)
    np.testing.assert_allclose(grad_x_sqrt, grad_x_sqrt_true)

  def test_safe_log1p_correct(self):
    x = all_floats(10001, include_inf=False)
    x = x[x > -1]
    y, g = jax.vmap(jax.value_and_grad(math.safe_log1p))(x)
    y_true, g_true = jax.vmap(jax.value_and_grad(jnp.log1p))(x)
    np.testing.assert_array_equal(y, y_true)
    np.testing.assert_array_equal(g, g_true)

  def test_safe_log1p_out_of_bounds(self):
    x = all_floats(10001, include_inf=True)
    x = jnp.concatenate([x[x < -1], jnp.array([-1], dtype=np.float32)])
    y, g = jax.vmap(jax.value_and_grad(math.safe_log1p))(x)
    y_true, g_true = jax.vmap(jax.value_and_grad(jnp.log1p))(
        math.plus_eps(jnp.array([-1]))
    )
    np.testing.assert_array_equal(y, jnp.broadcast_to(y_true, y.shape))
    np.testing.assert_array_equal(g, jnp.broadcast_to(g_true, g.shape))

    y, g = jax.vmap(jax.value_and_grad(math.safe_log1p))(jnp.array([jnp.inf]))
    y_true, g_true = jax.vmap(jax.value_and_grad(jnp.log1p))(
        jnp.array([jnp.finfo(jnp.float32).max])
    )
    np.testing.assert_array_equal(y, jnp.broadcast_to(y_true, y.shape))
    np.testing.assert_array_equal(g, jnp.broadcast_to(g_true, g.shape))

  def test_safe_log1p_output_and_gradient_are_finite(self):
    x = all_floats(10001, include_inf=True)
    x = jnp.sort(jnp.concatenate([x, jnp.array([-1], dtype=np.float32)]))
    y, g = jax.vmap(jax.value_and_grad(math.safe_log1p))(x)
    self.assertTrue(jnp.all(jnp.isfinite(y)))
    self.assertTrue(jnp.all(jnp.isfinite(g)))
    self.assertTrue(jnp.all(y[1:] >= y[:-1]))
    self.assertTrue(jnp.all(g[1:] <= g[:-1]))
    self.assertTrue(jnp.all(g >= 0))

  def test_safe_arctan2(self):
    # Probe a wide range of floats, making sure to probe around -2, -1, 1, 2.
    xx = all_floats(101, include_inf=False)
    xx = jnp.unique(jnp.concatenate([xx - 2, xx - 1, xx, xx + 1, xx + 2]))
    x1 = np.tile(xx[:, None], [1, len(xx)]).flatten()
    x2 = np.tile(xx[None, :], [len(xx), 1]).flatten()

    # Construct some scaled versions of the arctan2 functions so that we probe
    # gradient correctness when the grad coming in is something other than 1.
    scale = 7.0
    safe_fn = lambda x1, x2: scale * math.safe_arctan2(x1, x2)
    true_fn = lambda x1, x2: scale * jnp.arctan2(x1, x2)

    # Get the value and gradient for safe_arctan2 and jnp.arctan2.
    y, (g1, g2) = jax.vmap(jax.value_and_grad(safe_fn, argnums=[0, 1]))(x1, x2)
    y_true, (g1_true, g2_true) = jax.vmap(
        jax.value_and_grad(true_fn, argnums=[0, 1])
    )(x1, x2)

    # The value and gradient of safe_arctan2 should always be finite.
    self.assertTrue(jnp.all(jnp.isfinite(y)))
    self.assertTrue(jnp.all(jnp.isfinite(g1)))
    self.assertTrue(jnp.all(jnp.isfinite(g2)))

    # The values should match exactly.
    np.testing.assert_array_equal(y, y_true)

    # The gradient should be close when the true gradient is finite.
    v = np.isfinite(g1_true)
    np.testing.assert_allclose(g1[v], g1_true[v], rtol=1e-3, atol=1e-30)
    v = np.isfinite(g2_true)
    np.testing.assert_allclose(g2[v], g2_true[v], rtol=1e-3, atol=1e-30)

    # Make sure the signs of the gradients match, even if the true gradient
    # is finite. This is a little awkward to write as you need to cover the
    # case where gradients are so close to zero that np.sign() may return 0 or
    # +/-1, so we just check that there aren't any cases where the signs are
    # different by more than 1.
    self.assertLessEqual(np.max(np.abs(np.sign(x2) - np.sign(g1))), 1)
    self.assertLessEqual(np.max(np.abs(np.sign(-x1) - np.sign(g2))), 1)

  def _test_finite(self, fn, num_samples=10000):
    x = np.float64(all_floats(num_samples))
    y = fn(x)
    g = jax.vmap(jax.grad(fn))(x)
    # `y` and `g` should both always be finite.
    self.assertTrue(jnp.all(jnp.isfinite(y)))
    self.assertTrue(jnp.all(jnp.isfinite(g)))
    return x, y, g

  def test_safe_sin_finite(self):
    """math.safe_sin() behaves reasonably for all inputs."""
    self._test_finite(math.safe_sin)

  def test_safe_cos_finite(self):
    """math.safe_cos() behaves reasonably for all inputs."""
    self._test_finite(math.safe_cos)

  @chex.all_variants()
  def test_safe_div(self):
    """math.safe_div() behaves reasonably for all inputs."""
    x = all_floats(50)
    n = np.tile(x[None, :], (len(x), 1))
    d = np.tile(x[:, None], (1, len(x)))
    r = np.float32(n) / np.float32(d)
    r_safe = np.array(self.variant(math.safe_div)(n, d))
    np.testing.assert_array_equal(r_safe[d == 0], 0)
    self.assertTrue(jnp.all(jnp.isfinite(r_safe)))
    mask = (
        (np.abs(d) >= np.finfo(np.float32).tiny)
        & (np.abs(n) >= np.finfo(np.float32).tiny)
        & np.isfinite(r)
    )
    np.testing.assert_allclose(r[mask], r_safe[mask], atol=1e-6, rtol=1e-6)

    scale = 7.0
    fn = self.variant(lambda n, d: scale * math.safe_div(n, d))
    dn, dd = jax.vmap(jax.vmap(jax.grad(fn, argnums=[0, 1])))(n, d)
    self.assertTrue(jnp.all(jnp.isfinite(dn)))
    self.assertTrue(jnp.all(jnp.isfinite(dd)))

    dn_true = scale / d
    valid = np.isfinite(dn_true)
    np.testing.assert_allclose(dn_true[valid], dn[valid], atol=1e-30)

    dd_true = -scale * r / d
    valid = np.isfinite(dd_true)
    np.testing.assert_allclose(dd_true[valid], dd[valid], atol=1e-30)

  def test_safe_sqrt_finite(self):
    """math.safe_sqrt() behaves reasonably for all inputs."""
    x, y, g = self._test_finite(math.safe_sqrt)

    # sqrt and its gradient on non-negative values should be monotonic.
    nonneg_idx = jnp.where(x >= 0)[0][0]
    self.assertTrue(jnp.all(y[nonneg_idx + 1 :] >= y[nonneg_idx:-1]))
    self.assertTrue(jnp.all(g[nonneg_idx + 1 :] <= g[nonneg_idx:-1]))

  @chex.all_variants()
  def test_safe_arccos_output_and_gradient_are_correct_and_finite(self):
    x = np.float32(np.arange(-2000, 2001) / 1000)
    valid = (-1 < x) & (x < 1)
    y, g = self.variant(jax.vmap(jax.value_and_grad(math.safe_arccos)))(x)
    y_ref, g_ref = jax.vmap(jax.value_and_grad(jnp.arccos))(x)

    # The values should match if x is valid.
    np.testing.assert_array_equal(y[valid], y_ref[valid])
    np.testing.assert_array_equal(y[x >= 1], 0.0)
    np.testing.assert_array_equal(y[x <= -1], np.pi)

    # The gradient should be finite and be zero outside of the valid range.
    np.testing.assert_array_equal(g[~valid], 0.0)
    np.testing.assert_array_equal(g[valid], g_ref[valid])
    self.assertTrue(jnp.all(jnp.isfinite(g)))

  @chex.all_variants()
  @parameterized.parameters(
      [-np.inf, -4, -1, -0.25, 0, 0.25, 0.5, 0.75, 1, 4, np.inf]
  )
  def test_power_ladder_round_trip(self, p):
    """Check that inv_power_ladder(power_ladder(x, p)) == x."""
    x = jnp.linspace(-10, 10, 1001)
    y = self.variant(math.power_ladder)(x, p)
    x_recon = self.variant(math.inv_power_ladder)(y, p)
    np.testing.assert_allclose(x, x_recon, atol=1e-4, rtol=1e-4)

  @parameterized.parameters([-np.inf, -1, 0, 1, np.inf])
  def test_power_ladder_special_cases(self, p):
    """Test power_ladder() against implementations of various special cases."""
    x = jnp.linspace(-10, 10, 1001)

    if p == 1:
      np.testing.assert_array_equal(math.power_ladder(x, 1), x)
    elif p == 0:
      np.testing.assert_allclose(
          math.power_ladder(x, 0),
          np.sign(x) * np.log(np.abs(x) + 1),
          atol=1e-6,
          rtol=1e-6,
      )
    elif p == -1:
      np.testing.assert_allclose(
          math.power_ladder(x, -1),
          (2 * x) / (2 + np.abs(x)),
          atol=1e-6,
          rtol=1e-6,
      )
    elif p == -np.inf:
      np.testing.assert_allclose(
          math.power_ladder(x, -np.inf),
          np.sign(x) * (1 - np.exp(-np.abs(x))),
          atol=1e-6,
          rtol=1e-6,
      )
    elif p == np.inf:
      np.testing.assert_allclose(
          math.power_ladder(x, np.inf),
          np.sign(x) * (jnp.exp(jnp.abs(x)) - 1),
          atol=1e-6,
          rtol=1e-6,
      )
    else:
      assert False

  def test_power_ladder_misc_properties(self):
    x = jnp.linspace(-10, 10, 1001)
    # Fun fact:
    np.testing.assert_array_equal(
        math.power_ladder(x, jnp.inf), math.inv_power_ladder(x, 0)
    )
    np.testing.assert_array_equal(
        math.power_ladder(x, 0), math.inv_power_ladder(x, jnp.inf)
    )

  @chex.all_variants()
  @parameterized.parameters(
      [-np.inf, -4, -1, -0.25, 0, 0.25, 0.5, 0.75, 1, 2, 4, np.inf]
  )
  def test_power_ladder_gradients_are_finite_and_correct(self, p):
    fn = self.variant(math.power_ladder)

    # The first derivative at 0 is 1.
    derivative1 = jax.grad(lambda x: fn(x, p))(0.0)
    np.testing.assert_allclose(derivative1, 1.0)

    # The second derivative at 0 is sign(p - 1).
    derivative2 = jax.grad(jax.grad(lambda x: math.power_ladder(x, p)))(0.0)
    np.testing.assert_allclose(derivative2, np.sign(p - 1))

    # Pick a maximum value where x^p is finite.
    if p == jnp.inf:
      max_val = 1e7
    else:
      max_val = (
          jnp.log(jnp.finfo(jnp.float32).max ** (1 / jnp.maximum(1, p))) - 1
      )
    x = np.exp(np.linspace(-90, jnp.log(max_val), 50, dtype=np.float64))
    x = jnp.concatenate([-x[::-1], jnp.array([0]), x])
    x = jnp.float32(x)

    # Gradient is finite and monotonic wrt x.
    d_x = jax.vmap(jax.grad(lambda x: math.power_ladder(x, p)))(x)
    self.assertTrue(jnp.all(jnp.isfinite(d_x)))
    self.assertTrue(jnp.all(d_x >= 0))

    if p not in [-np.inf, 0, 1, np.inf]:
      # Gradient is finite and monotonic wrt p.
      p_mat = jnp.full_like(x, float(p))
      d_p = jax.vmap(jax.grad(math.power_ladder, argnums=1))(x, p_mat)
      self.assertTrue(jnp.all(jnp.isfinite(d_p)))
      np.testing.assert_array_less(-1e-6, d_p[x > 0])
      np.testing.assert_array_less(d_p[x < 0], 1e-6)

  @chex.all_variants()
  @parameterized.parameters(
      [-np.inf, -4, -1, -0.25, 0, 0.25, 0.5, 0.75, 1, 4, np.inf]
  )
  def test_inv_power_ladder_gradients_are_finite_and_correct(self, p):
    fn = self.variant(math.inv_power_ladder)

    # The first derivative at 0 is 1.
    derivative1 = jax.grad(lambda x: fn(x, p))(0.0)
    np.testing.assert_allclose(derivative1, 1.0)

    # The second derivative at 0 is sign(p - 1).
    derivative2 = jax.grad(jax.grad(lambda x: fn(x, p)))(0.0)
    np.testing.assert_allclose(
        derivative2, np.sign(1 - p), atol=1e-6, rtol=1e-6
    )

    max_val = jnp.minimum(1e6, math.plus_eps(math.power_ladder_max_output(p)))
    x = np.exp(np.linspace(-90, jnp.log(max_val), 50, dtype=np.float64))
    x = jnp.concatenate([-x[::-1], jnp.array([0]), x])
    x = jnp.float32(x)

    # Gradient is finite and monotonic wrt x.
    val, d_x = jax.vmap(jax.value_and_grad(lambda x: fn(x, p)))(x)
    self.assertTrue(jnp.all(jnp.isfinite(val)))
    self.assertTrue(jnp.all(jnp.isfinite(d_x)))
    self.assertTrue(jnp.all(d_x >= 0))

    if p not in [-np.inf, 0, 1, np.inf]:
      # Gradient is finite and monotonic wrt p.
      p_mat = jnp.full_like(x, float(p))
      d_p = jax.vmap(jax.grad(fn, argnums=1))(x, p_mat)
      self.assertTrue(jnp.all(jnp.isfinite(d_p)))
      np.testing.assert_array_less(-1e-6, d_p[x < 0])
      np.testing.assert_array_less(d_p[x > 0], 1e-6)

  @chex.all_variants()
  def test_power_ladder_is_odd(self):
    n = 100
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = 10 * random.normal(key1, [n])
    p = random.normal(key2, [n])
    fn = self.variant(math.power_ladder)
    np.testing.assert_allclose(fn(x, p), -fn(-x, p))

  @chex.all_variants()
  @parameterized.parameters([-np.inf, -10, -4, -1, -0.25])
  def test_power_ladder_max_output(self, p):
    """Test that power_ladder_max_output(p) == power_ladder(np.inf, p)."""
    np.testing.assert_allclose(
        self.variant(math.power_ladder_max_output)(p),
        math.power_ladder(np.inf, p),
    )

  @chex.all_variants()
  def test_power_ladder_and_inverse_pre_post_multipliers_are_correct(self):
    n = 1000
    x, p, premult, postmult = tuple(random.normal(random.PRNGKey(0), [4, n]))
    kwargs = {'p': p, 'premult': premult, 'postmult': postmult}
    fn_fwd = self.variant(functools.partial(math.power_ladder, **kwargs))
    fn_inv = self.variant(functools.partial(math.inv_power_ladder, **kwargs))
    np.testing.assert_allclose(x, fn_inv(fn_fwd(x)), atol=0.005)

  def test_learning_rate_decay(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      lr_init = jnp.exp(random.normal(key) - 3)
      key, rng = random.split(rng)
      lr_final = lr_init * jnp.exp(random.normal(key) - 5)
      key, rng = random.split(rng)
      max_steps = int(jnp.ceil(100 + 100 * jnp.exp(random.normal(key))))

      lr_fn = functools.partial(
          math.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps,
      )

      # Test that the rate at the beginning is the initial rate.
      np.testing.assert_allclose(lr_fn(0), lr_init, atol=1e-5, rtol=1e-5)

      # Test that the rate at the end is the final rate.
      np.testing.assert_allclose(
          lr_fn(max_steps), lr_final, atol=1e-5, rtol=1e-5
      )

      # Test that the rate at the middle is the geometric mean of the two rates.
      np.testing.assert_allclose(
          lr_fn(max_steps / 2),
          jnp.sqrt(lr_init * lr_final),
          atol=1e-5,
          rtol=1e-5,
      )

      # Test that the rate past the end is the final rate
      np.testing.assert_allclose(
          lr_fn(max_steps + 100), lr_final, atol=1e-5, rtol=1e-5
      )

  def test_approx_erf(self):
    x = all_floats(100000)
    y = math.approx_erf(x)
    y_ref = jax.lax.erf(x)
    np.testing.assert_allclose(y, y_ref, atol=0.007)


if __name__ == '__main__':
  absltest.main()
