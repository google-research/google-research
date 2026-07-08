# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Test differentiable assimilation: 1D-Var finds cloud ice water content."""

# pylint: disable=invalid-name,g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_rtm import AtmosphereState
from jax_rtm import GeometryState
from jax_rtm import GOES_ABI_CONFIG
from jax_rtm import simulate_pixel
from jax_rtm import SurfaceState
import numpy as np

# Enable float64 for JAX (critical for radiative transfer precision)
jax.config.update("jax_enable_x64", True)


def get_us_standard_atmosphere(n_layers=50):
  """Generates a 50-layer US Standard Atmosphere profile."""
  pressures = jnp.logspace(jnp.log10(0.1), jnp.log10(1000.0), n_layers)

  # Scale height for pressure-to-height mapping
  H = 7.4  # km
  z = -H * jnp.log(pressures / 1013.25)

  # Piece-wise linear temperature profile based on height z (in km)
  def get_t(h):
    return jnp.where(
        h < 11.0,
        288.15 - 6.5 * h,
        jnp.where(
            h < 20.0,
            216.65,
            jnp.where(
                h < 32.0,
                216.65 + 1.0 * (h - 20.0),
                jnp.where(
                    h < 47.0,
                    228.65 + 2.8 * (h - 32.0),
                    jnp.where(
                        h < 51.0,
                        270.65,
                        jnp.where(
                            h < 71.0,
                            270.65 - 2.8 * (h - 51.0),
                            214.65 - 2.0 * (h - 71.0),
                        ),
                    ),
                ),
            ),
        ),
    )

  temperatures = jax.vmap(get_t)(z)

  # Rayleigh optical depth distributed by pressure
  tau_total = 0.01
  dp = jnp.diff(jnp.concatenate([jnp.array([0.0]), pressures]))
  tau_rayleigh = tau_total * dp / 1000.0

  return pressures, temperatures, tau_rayleigh


class AssimilationTest(absltest.TestCase):

  def test_1d_var_ciwc_recovery(self):
    """Test 1D-Var retrieval of contrail ice water content (ciwc)."""
    # Use 20 layers to get a realistic vertical level around 230 hPa
    # (flight altitude)
    n_layers = 20
    pressures, temperatures, tau_rayleigh = get_us_standard_atmosphere(n_layers)

    # Base atmosphere state (clear sky)
    p_prof = pressures * 100.0  # Convert hPa to Pa
    T_prof = temperatures
    # Simple humidity profile matching layers
    q_prof = jnp.logspace(jnp.log10(1e-3), jnp.log10(1e-6), n_layers)
    clwc = jnp.zeros(n_layers)
    cswc = jnp.zeros(n_layers)
    crwc = jnp.zeros(n_layers)
    tau_ray_prof = tau_rayleigh
    omega_ray_prof = jnp.ones(n_layers)

    # Surface state (Land, no snow)
    is_land = jnp.array(True)
    sd = jnp.array(0.0)
    skt = jnp.array(288.15)
    lat = jnp.array(30.0)
    u10 = jnp.array(0.0)
    v10 = jnp.array(0.0)

    # Geometry
    mu_0 = jnp.array(0.5)
    mu_view = jnp.array(0.8)

    # True state: inject some ice water content at layer 16 (approx 234 hPa)
    true_ciwc_val = 1e-4
    active_layer = 16

    @jax.jit
    def run_simulation(log_ciwc_val):
      ciwc_val = jnp.exp(log_ciwc_val)
      ciwc_active = jnp.zeros(n_layers).at[active_layer].set(ciwc_val)

      atmosphere = AtmosphereState(
          p_prof=p_prof,
          T_prof=T_prof,
          q_prof=q_prof,
          clwc=clwc,
          ciwc_nat=ciwc_active,
          cswc=cswc,
          crwc=crwc,
          tau_ray_prof=tau_ray_prof,
          omega_ray_prof=omega_ray_prof,
      )
      surface = SurfaceState(
          is_land=is_land,
          sd=sd,
          skt=skt,
          lat=lat,
          u10=u10,
          v10=v10,
      )
      geometry = GeometryState(
          mu_0=mu_0,
          mu_view=mu_view,
      )

      bt_84, bt_103, bt_123, _ = simulate_pixel(
          atmosphere,
          surface,
          geometry,
          n_streams=4,
          num_doubling_steps=10,
          satellite_config=GOES_ABI_CONFIG,
      )
      return jnp.stack([bt_84, bt_103, bt_123])

    # Generate "true" observations (noise-free for perfect recovery)
    true_log_ciwc = jnp.log(true_ciwc_val)
    obs_bts = run_simulation(true_log_ciwc)

    @jax.jit
    def cost_function(log_ciwc_guess):
      sim_bts = run_simulation(log_ciwc_guess)
      return jnp.mean((sim_bts - obs_bts) ** 2)

    # Print for debugging to verify physical signal
    clear_bts = run_simulation(jnp.log(1e-15))
    guess_bts = run_simulation(jnp.log(1e-5))
    print(f"DEBUG: Clear sky BTs : {clear_bts}")
    print(f"DEBUG: Guess sky BTs : {guess_bts}")
    print(f"DEBUG: Observed BTs  : {obs_bts}")
    print(f"DEBUG: Diff Obs-Clear: {obs_bts - clear_bts} (Should be > 0.1 K)")
    print(f"DEBUG: Diff Guess-Clr: {guess_bts - clear_bts}")

    # Initial guess: thin cloud (10 ug/kg)
    x0 = jnp.log(1e-5)

    # 1. Verify gradients are non-zero and substantial
    grad_fn = jax.grad(cost_function)
    gradient = grad_fn(x0)
    print(f"Gradient at x0: {gradient}")
    self.assertGreater(jnp.abs(gradient), 1e-3)

    # 2. Verify a gradient step reduces the cost (use a conservative step to
    # avoid explosion)
    step = 0.001
    x_new = x0 - step * gradient
    cost_old = cost_function(x0)
    cost_new = cost_function(x_new)
    print(
        f"Cost old: {cost_old:.2e}, Cost new: {cost_new:.2e} (with step {step})"
    )
    self.assertLess(cost_new, cost_old)

    # 3. Run adaptive gradient descent with update clipping.
    # We clip the update step in log-space to prevent jumping over the basin
    # into the flat saturation plateau.
    x = x0
    lr = 0.003  # Start bold
    c_old = cost_old
    # Maximum change in log-space (limits change in ciwc to factor of ~4.5 per
    # step)
    max_step_size = 1.5
    print(
        "Starting adaptive optimization with clipping | Initial Cost:"
        f" {c_old:.2e}"
    )

    for i in range(40):
      g = grad_fn(x)

      # Propose step with update clipping
      update = lr * g
      clipped_update = np.clip(update, -max_step_size, max_step_size)
      x_next = x - clipped_update
      c_next = cost_function(x_next)

      if c_next < c_old:
        # Success: accept step and slightly increase learning rate
        x = x_next
        c_old = c_next
        print(
            f"Step {i+1:02d} | ciwc: {jnp.exp(x):.2e} | Cost: {c_old:.2e} |"
            f" Grad: {g:.2f} | LR: {lr:.5f} | Update: {clipped_update:.3f}"
            " (Accepted)"
        )
        lr = lr * 1.05
      else:
        # Failure (overshot despite clipping): reject step, shrink learning
        # rate, and retry
        lr = lr * 0.5
        print(
            f"Step {i+1:02d} | Rejected (Cost {c_next:.2e} > {c_old:.2e}) |"
            f" Shrinking LR to {lr:.5f}"
        )

    final_ciwc = jnp.exp(x)
    print(f"True ciwc: {true_ciwc_val:.2e}, Recovered: {final_ciwc:.2e}")

    # Assert that we recovered the true value within 5% tolerance
    np.testing.assert_allclose(final_ciwc, true_ciwc_val, rtol=0.05)
    self.assertLess(cost_function(x), 1e-4)


if __name__ == "__main__":
  absltest.main()
