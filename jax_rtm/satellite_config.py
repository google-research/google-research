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

# pylint: skip-file
import json
import os
import flax
import jax.numpy as jnp
import numpy as np

_package_dir = os.path.dirname(os.path.realpath(__file__))
_params_path = os.path.join(_package_dir, "data/params_992.json")
if os.path.exists(_params_path):
  with open(_params_path, "r") as f:
    _loaded_params = json.load(f)
else:
  _loaded_params = {}


@flax.struct.dataclass
class AtmosphereState:
  p_prof: jnp.ndarray
  T_prof: jnp.ndarray
  q_prof: jnp.ndarray
  clwc: jnp.ndarray
  ciwc_nat: jnp.ndarray
  cswc: jnp.ndarray
  crwc: jnp.ndarray
  tau_ray_prof: jnp.ndarray
  omega_ray_prof: jnp.ndarray


@flax.struct.dataclass
class SurfaceState:
  is_land: jnp.ndarray
  sd: jnp.ndarray
  skt: jnp.ndarray
  lat: jnp.ndarray
  u10: jnp.ndarray
  v10: jnp.ndarray


@flax.struct.dataclass
class GeometryState:
  mu_0: jnp.ndarray
  mu_view: jnp.ndarray


@flax.struct.dataclass
class SatelliteConfig:
  """Configuration container for satellite-specific spectral properties.

  Registered as a JAX PyTree via flax.struct.dataclass to allow native tracing,
  differentiation, and parameter overrides using .replace().
  """

  wavelengths: np.ndarray
  em_snow: np.ndarray
  em_land: np.ndarray
  em_ocean: np.ndarray
  wv_linear: np.ndarray
  wv_quadratic: np.ndarray
  tau_co2: float
  tau_o3: float
  ocean_em_floor: float
  ice_sizing_scheme: str
  ice_spectral_shift: np.ndarray
  use_linear_stream_interp: bool
  grid_spec: dict
  wv_t_dep: float
  t_land_bias: float
  t_water_bias: float


# Preconfigured Satellite Spectral Configurations using static NumPy arrays
# to prevent global JAX initialization flag errors at module load time.
GOES_ABI_CONFIG = SatelliteConfig(
    wavelengths=np.array([8.4, 10.3, 12.3]),
    em_snow=np.array(_loaded_params.get("em_snow", [0.986, 0.990, 0.990])),
    em_land=np.array(_loaded_params.get("em_land", [0.962, 0.980, 0.976])),
    em_ocean=np.array(_loaded_params.get("em_ocean", [0.982, 0.990, 0.984])),
    wv_linear=np.array(
        _loaded_params.get("wv_linear", [0.0165, 0.0110, 0.0230])
    ),
    wv_quadratic=np.array(
        _loaded_params.get("wv_quadratic", [0.000190, 0.000140, 0.000540])
    ),
    tau_co2=_loaded_params.get("tau_co2", 0.050),
    tau_o3=_loaded_params.get("tau_o3", 0.0090),
    ocean_em_floor=_loaded_params.get("ocean_em_floor", 0.772),
    ice_sizing_scheme="wyser",
    ice_spectral_shift=np.array(
        _loaded_params.get("ice_spectral_shift", [0.885, 1.115, 1.140])
    ),
    use_linear_stream_interp=True,
    grid_spec={"ds_ratio_base": 1.0, "crop_size": 339},
    wv_t_dep=_loaded_params.get("wv_t_dep", 0.060),
    t_land_bias=_loaded_params.get("t_land_bias", -0.64),
    t_water_bias=_loaded_params.get("t_water_bias", -0.34),
)


SATELLITE_CONFIGS = {
    "goes": GOES_ABI_CONFIG,
}
