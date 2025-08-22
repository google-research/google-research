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

"""Tests for probability_flow."""
import unittest

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
from score_sde import sde_lib
from score_sde import utils

from score_prior import probability_flow

# Parameters for shape of dummy data:
_IMAGE_SIZE = 8
_N_CHANNELS = 1
_IMAGE_SHAPE = (_IMAGE_SIZE, _IMAGE_SIZE, _N_CHANNELS)
_IMAGE_DIM = _IMAGE_SIZE * _IMAGE_SIZE * _N_CHANNELS

# Tolerance for checking all-closeness of arrays:
_RTOL = 1e-6
# Tolerance for checking all-closeness of ODE results:
_ODE_RTOL = 0.001
_ODE_ATOL = 0.0001


class DummySDE(sde_lib.SDE):
  """A dummy SDE with a simple drift and diffusion.

  Drift: f(x, t) = x
  Diffusion: g(t) = sqrt(2 * x)
  """

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    return x, jnp.sqrt(2 * t)

  def marginal_prob(self, x, t):
    pass

  def prior_sampling(self, rng, shape):
    pass

  def prior_logp(self, z):
    pass


class ProbabilityFlowTest(unittest.TestCase):
  """Tests for probability_flow."""

  def setUp(self):
    super().setUp()
    self.sde = DummySDE(N=1000)
    self.score_fn = lambda x, t_batch: utils.batch_mul(t_batch / 5, x**2)
    self.prob_flow = probability_flow.ProbabilityFlow(
        self.sde,
        self.score_fn,
        solver=diffrax.Heun(),
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        n_trace_estimates=5)

  def test_drift(self):
    """Test drift of the PF-ODE.

    For the dummy SDE and score function, the actual drift is given by:
    pf_drift(x, t) = f(x, t) - 0.5 * g(t)**2 * score(x, t)
                   = x - 0.5 * (2 * t) * (t * x**2 / 5)
                   = x - t * t * x**2 / 5
                   = x - t**2 * x**2 / 5
    """
    rng = jax.random.PRNGKey(0)
    # We'll use a mini-batch size of 2.
    x = jax.random.normal(rng, (2, *_IMAGE_SHAPE))
    t_batch = jnp.ones(2) * 0.8

    drift = self.prob_flow.drift_fn(x, t_batch)

    # Compute reference probability flow ODE drift.
    reference_drift = x - utils.batch_mul(t_batch**2, x**2) / 5

    np.testing.assert_allclose(drift, reference_drift, rtol=_RTOL)

  def test_divergence(self):
    """Test divergence of the drift of the PF-ODE.

    For the dummy SDE and score function, the actual divergence is given by:
    pf_div(x, t) = sum(ones_like(x) - (2 / 5) * t**2 * x)
    """
    # Use two test images that have different structures.
    rng = jax.random.PRNGKey(0)
    x1 = jax.random.normal(rng, _IMAGE_SHAPE)
    x2 = 0.5 * jnp.arange(_IMAGE_DIM).reshape(_IMAGE_SHAPE)
    x = jnp.concatenate((x1[None, Ellipsis], x2[None, Ellipsis]))

    # Diffusion time:
    t = 0.8
    t_batch = jnp.ones(2) * t

    # Compute divergence with trace estimation.
    rng, step_rng = jax.random.split(rng)
    div = self.prob_flow.div_fn(step_rng, x, t_batch)

    # Compute reference divergence.
    ref_div = jnp.sum(
        jnp.ones((2, *_IMAGE_SHAPE)) - (2 / 5) * t**2 * x, axis=(1, 2, 3))

    np.testing.assert_allclose(div, ref_div, rtol=_RTOL)

  def test_ode(self):
    """Test the ODE wth x only."""
    # We'll use a mini-batch size of 1.
    rng = jax.random.PRNGKey(1)
    x_init = jax.random.normal(rng, (1, *_IMAGE_SHAPE))

    def ode_func(t, x):
      # We'll use the true drift.
      dx = x - t**2 * x**2 / 5
      return dx

    x_final_ref = integrate.solve_ivp(
        ode_func, (0., 1.), x_init.reshape(-1)).y[:, -1]
    x_final_ref = x_final_ref.reshape(1, *_IMAGE_SHAPE)

    x_final, _ = self.prob_flow.ode(x_init, t0=0., t1=1., dt0=0.01)
    np.testing.assert_allclose(
        x_final, x_final_ref, rtol=_ODE_RTOL, atol=_ODE_ATOL)

  def test_ode_with_logp(self):
    """Test the ODE with x and negative change in logp(x)."""
    # We'll use a mini-batch size of 1.
    rng = jax.random.PRNGKey(0)
    x_init = jax.random.normal(rng, (1, *_IMAGE_SHAPE))

    def ode_func(t, state):
      x = state[:-1]

      # We'll use the true drift and divergence.
      dx = x - t**2 * x**2 / 5
      dy = jnp.sum(jnp.ones(x.shape) - (2 / 5) * t**2 * x)

      return np.concatenate((dx, [dy]))

    final = integrate.solve_ivp(
        ode_func, (1., 0.), np.concatenate((x_init.reshape(-1), [0.]))).y[:, -1]
    x_final_ref = final[:-1].reshape(1, *_IMAGE_SHAPE)
    logp_final_ref = final[-1]

    rng = jax.random.PRNGKey(0)
    x_final, logp_final, _ = self.prob_flow.ode_with_logp(
        rng, x_init, t0=1., t1=0., dt0=-0.01)
    np.testing.assert_allclose(
        x_final, x_final_ref, rtol=_ODE_RTOL, atol=_ODE_ATOL)
    np.testing.assert_allclose(
        logp_final, logp_final_ref, rtol=_ODE_RTOL, atol=_ODE_ATOL)


if __name__ == '__main__':
  unittest.main()
