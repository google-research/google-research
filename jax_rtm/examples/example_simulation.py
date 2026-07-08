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

"""Example script running a JAX-RTM simulation using the modular library."""

# pylint: disable=invalid-name,g-import-not-at-top

import json
import os
import sys

# Add parent directory to path to allow importing jax_rtm locally
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from absl import app
from absl import flags
import jax.numpy as jnp
import jax_rtm
import matplotlib.pyplot as plt
import numpy as np

_RESOLUTION = flags.DEFINE_integer(
    "resolution", 85, "Grid resolution: 85 (85x85) or 339 (339x339)"
)
flags.register_validator(
    "resolution",
    lambda val: val in (85, 339),
    message="Resolution must be either 85 or 339.",
)


def main(argv):
  del argv  # Unused
  # Paths relative to this script
  script_dir = os.path.dirname(os.path.abspath(__file__))
  package_dir = os.path.abspath(os.path.join(script_dir, ".."))

  params_path = os.path.join(package_dir, "data", "params_992.json")

  if _RESOLUTION.value == 339:
    weather_file = "weather_339x339.npz"
    output_name = "simulated_ash_rgb_339x339.png"
  elif _RESOLUTION.value == 85:
    weather_file = "weather_85x85.npz"
    output_name = "simulated_ash_rgb_85x85.png"
  else:
    # Unreachable due to validator, but good as fallback
    raise ValueError(f"Resolution {_RESOLUTION.value} is not implemented.")

  # Check multiple locations to support both internal tests and public
  # execution
  internal_weather_path = os.path.join(
      package_dir, "google", "test_data", weather_file
  )
  public_weather_path = os.path.join(package_dir, "data", weather_file)

  if os.path.exists(internal_weather_path):
    weather_path = internal_weather_path
  else:
    weather_path = public_weather_path
  output_png = os.path.join(script_dir, output_name)

  print(f"Loading calibrated parameters from {params_path}...")
  with open(params_path, "r") as f:
    params = json.load(f)

  print(f"Loading sample weather data from {weather_path}...")
  weather = np.load(weather_path)
  T_prof = weather["T"]
  q_prof = weather["q"]
  ciwc = weather["ciwc"]
  clwc = weather["clwc"]
  crwc = weather["crwc"]
  cswc = weather["cswc"]
  sd = np.nan_to_num(weather["sd"], nan=0.0)
  skt = weather["skt"]
  p_prof = weather["p"] * 100.0  # hPa to Pa
  lsm = weather["lsm"]

  height, width, n_levels = T_prof.shape
  n_pixels = height * width
  print(
      f"Grid shape: {height}x{width} ({n_pixels} total pixels), {n_levels}"
      " vertical levels."
  )

  # Construct coordinate grids
  lats = np.linspace(0, 60, height)[:, None]
  lat_grid = np.broadcast_to(lats, (height, width)).flatten()
  mu_view = np.linspace(0.5, 1.0, width)[None, :]
  mu_view_grid = np.broadcast_to(mu_view, (height, width)).flatten()
  is_land_grid = (lsm > 0.5).flatten()

  # Flatten weather inputs for batching
  T_flat = T_prof.reshape(n_pixels, n_levels)
  q_flat = q_prof.reshape(n_pixels, n_levels)
  ciwc_flat = ciwc.reshape(n_pixels, n_levels)
  clwc_flat = clwc.reshape(n_pixels, n_levels)
  crwc_flat = crwc.reshape(n_pixels, n_levels)
  cswc_flat = cswc.reshape(n_pixels, n_levels)
  p_flat = p_prof.reshape(n_pixels, n_levels)
  sd_flat = sd.flatten()
  skt_flat = skt.flatten()

  # Zero arrays for unused physical inputs
  tau_ray_flat = np.zeros_like(T_flat)
  omega_ray_flat = np.zeros_like(T_flat)
  # Note: Setting wind components to zero still runs the Cox-Munk rough ocean
  # integration in microphysics.py, falling back to a minimum background
  # slope variance of 0.003 (calm sea roughness floor). In production,
  # these should be populated with real wind fields.
  u10_flat = np.zeros_like(skt_flat)
  v10_flat = np.zeros_like(skt_flat)

  # Package flat arrays into CRTM-style structured states
  atmosphere = jax_rtm.AtmosphereState(
      p_prof=jnp.array(p_flat),
      T_prof=jnp.array(T_flat),
      q_prof=jnp.array(q_flat),
      clwc=jnp.array(clwc_flat),
      ciwc_nat=jnp.array(ciwc_flat),
      cswc=jnp.array(cswc_flat),
      crwc=jnp.array(crwc_flat),
      tau_ray_prof=jnp.array(tau_ray_flat),
      omega_ray_prof=jnp.array(omega_ray_flat),
  )
  surface = jax_rtm.SurfaceState(
      is_land=jnp.array(is_land_grid),
      sd=jnp.array(sd_flat),
      skt=jnp.array(skt_flat),
      lat=jnp.array(lat_grid),
      u10=jnp.array(u10_flat),
      v10=jnp.array(v10_flat),
  )
  geometry = jax_rtm.GeometryState(
      mu_0=jnp.ones_like(skt_flat) * 0.5,  # mu_0 is 0.5 for all pixels
      mu_view=jnp.array(mu_view_grid),
  )

  print("Initializing JAX batch simulator...")
  # get_batch_simulator returns a JIT-compiled function
  batch_sim = jax_rtm.get_batch_simulator(
      params, n_streams=16, num_doubling_steps=14, mu_0=0.5
  )

  print("Executing simulation (first call will trigger JIT compilation)...")
  t0 = os.times().elapsed

  # Run the simulation on the whole grid in one batch using the structured
  # states
  b_84, b_103, b_123, _ = batch_sim(atmosphere, surface, geometry)

  # Force execution to measure time
  b_84.block_until_ready()
  elapsed = os.times().elapsed - t0
  print(f"Simulation completed in {elapsed:.2f}s.")

  # Reshape back to 2D
  sim_84 = np.array(b_84).reshape(height, width)
  sim_103 = np.array(b_103).reshape(height, width)
  sim_123 = np.array(b_123).reshape(height, width)

  print("Composing Ash RGB image...")
  rgb = jax_rtm.ash_rgb_compositor(sim_84, sim_103, sim_123)
  rgb_np = np.array(jnp.nan_to_num(rgb, nan=0.0))

  print(f"Saving composed image to {output_png}...")
  try:
    plt.imsave(output_png, rgb_np, format="png")
    print("Success!")
  except PermissionError:
    fallback_png = os.path.join("/tmp", output_name)
    print(
        f"Permission denied writing to {output_png} (likely in a read-only"
        " sandbox)."
    )
    print(f"Falling back to saving to: {fallback_png}...")
    plt.imsave(fallback_png, rgb_np, format="png")
    print("Success (saved to fallback path)!")


if __name__ == "__main__":
  app.run(main)
