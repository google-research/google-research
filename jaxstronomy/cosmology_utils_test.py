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

"""Tests for cosmology_utils.py.

Expected values are drawn from colossus:
https://bdiemer.bitbucket.io/colossus/cosmology_cosmology.html.
"""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from jaxstronomy import cosmology_utils

# Using immutabledict is good practice, although we will have to cast to dict
# everywhere because jax does not accept immutabledict.
COSMOLOGY_PARAMS_INIT = immutabledict({
    'omega_m_zero': 0.3089,
    'omega_b_zero': 0.0486,
    'omega_de_zero': 0.6910088292453472,
    'omega_rad_zero': 9.117075466e-5,
    'temp_cmb_zero': 2.7255,
    'hubble_constant': 67.74,
    'n_s': 0.9667,
    'sigma_eight': 0.8159,
})
COSMOLOGY_PARAMS_LENSTRONOMY = immutabledict({
    'omega_m_zero': 0.30966,
    'omega_b_zero': 0.0486,
    'omega_de_zero': 0.688846306,
    'omega_rad_zero': 0.0014937,
    'temp_cmb_zero': 2.7255,
    'hubble_constant': 67.66,
    'n_s': 0.9667,
    'sigma_eight': 0.8159,
})


def _prepare_cosmology_params(
    cosmology_params_init,
    z_lookup_max,
    dz,
    r_min=1e-4,
    r_max=1e3,
    n_r_bins=2,
):
  # Only generate a lookup table for values we need.
  # When 0,0 is specified for the two z values, need to select a small non-zero
  # values to generate a non-empty table.
  z_lookup_max = max(z_lookup_max, 1e-7)
  dz = max(dz, 1e-7)
  return cosmology_utils.add_lookup_tables_to_cosmology_params(
      dict(cosmology_params_init), z_lookup_max, dz, r_min, r_max, n_r_bins)


def _prepare_x_y():
  rng = jax.random.PRNGKey(3)
  rng_x, rng_y = jax.random.split(rng)
  x = jax.random.normal(rng_x, shape=(3,)) * 1e3
  y = jax.random.normal(rng_y, shape=(3,)) * 1e3
  return x, y


def _prepare_alpha_x_alpha_y():
  rng = jax.random.PRNGKey(2)
  rng_x, rng_y = jax.random.split(rng)
  alpha_x = jax.random.normal(rng_x, shape=(3,)) * 2
  alpha_y = jax.random.normal(rng_y, shape=(3,)) * 2
  return alpha_x, alpha_y


def _prepare__sigma_k_integrand_expected(radius):
  if radius == 1e-2:
    return jnp.array([
        3.127437136035467e-06, 1.820751207237299e-05, 5.2541055609899326e-05,
        0.00010306278738724287, 0.00010775093864571042
    ])
  elif radius == 1e-1:
    return jnp.array([
        3.127323720477859e-06, 1.8185657328033008e-05, 5.04866815266023e-05,
        2.3545351279405076e-05, 1.5387621097147294e-08
    ])
  elif radius == 1e0:
    return jnp.array([
        3.116000043964669e-06, 1.611047820558932e-05, 1.534716000730611e-10,
        1.2524276267847e-09, 1.6400363103856087e-12
    ])
  else:
    raise ValueError(f'Unsupported radius = {radius}')


def _prepare_reduced_to_physical_expected(z_lens, z_source):
  if z_lens == 0.1 and z_source == 0.5:
    return jnp.array([-0.6492834, 4.845596, -3.8608627])
  elif z_lens == 0.5 and z_source == 1.0:
    return jnp.array([-1.1832292, 8.830428, -7.035888])
  elif z_lens == 0.5 and z_source == 2.0:
    return jnp.array([-0.7973697, 5.9507623, -4.7414346])
  else:
    raise ValueError(f'Unsupported z_lens and z_source = ({z_lens},{z_source})')


def _preapre_comoving_to_angle_expected(z_lens):
  if z_lens == 0.1:
    return jnp.array([[-3.1144106, -1.2556583, 0.8742881],
                      [1.524639, -4.0299582, 3.8287995]])
  elif z_lens == 0.5:
    return jnp.array([[-0.6921371, -0.2790537, 0.19429912],
                      [0.33883113, -0.8956057, 0.85090077]])
  elif z_lens == 1.0:
    return jnp.array([[-0.39674094, -0.15995677, 0.11137448],
                      [0.19422188, -0.51337147, 0.48774606]])


def _prepare_expected_comoving_distance(z_min, z_max):
  if z_min == 0.0 and z_max == 0.0:
    expected = 0.0
  elif z_min == 0.0 and z_max == 1.283:
    expected = 2.740512775683e3
  elif z_min == 0.0 and z_max == 20.0:
    expected = 7.432211484621e3
  elif z_min == 0.5 and z_max == 1.283:
    expected = 1.4225931620767858e3
  else:
    raise ValueError(f'Unsupported z_min and z_max = ({z_min},{z_max})')

  # Expected is in units of /h
  return expected / (COSMOLOGY_PARAMS_INIT['hubble_constant'] / 100.)


def _prepare_expected_angular_diameter_distance(z):
  return _prepare_expected_comoving_distance(0.0, z) / (1 + z)


def _prepare_expected_kpc_per_arcsecond(z):
  if z == 0.0:
    return 0.0
  elif z == 1.283:
    return 8.591241981530676
  elif z == 20.0:
    return 2.5329606272180145
  else:
    raise ValueError(f'Unsupported z = {z}')


class CosmologyUtilsTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of various cosmology functions."""

  @chex.all_variants
  def test__e_z(self):
    z_values = jnp.array([0.0, 1.283, 20.0])
    expected = jnp.array([1.0, 2.090250748388e+00, 5.365766383626e+01])

    e_z = self.variant(cosmology_utils._e_z)

    np.testing.assert_allclose(
        e_z(dict(COSMOLOGY_PARAMS_INIT), z_values), expected)

  @chex.all_variants
  def test__e_z_rad_to_dark(self):
    z_values = jnp.array([0.0, 1.283, 20.0])
    expected = jnp.array([1.0, 2.0896800302592498, 53.49218634529721])

    e_z_rad_to_dark = self.variant(cosmology_utils._e_z_rad_to_dark)

    np.testing.assert_allclose(
        e_z_rad_to_dark(dict(COSMOLOGY_PARAMS_INIT), z_values), expected)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_min_{z_min}_z_max_{z_max}', z_min, z_max)
      for z_min, z_max in zip([0.0, 0.0, 0.0, 0.5], [0.0, 1.283, 20.0, 1.283])
  ])
  def test__comoving_distance_numerical(self, z_min, z_max):
    expected = _prepare_expected_comoving_distance(z_min, z_max)
    comoving_distance = self.variant(
        cosmology_utils._comoving_distance_numerical)

    np.testing.assert_allclose(
        comoving_distance(dict(COSMOLOGY_PARAMS_INIT), z_min, z_max),
        expected,
        rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([(f'_r_{r}', r) for r in [1e-2, 1e-1, 1e0]])
  def test__sigma_k_integrand(self, r):
    log_k = jnp.linspace(-2, 5, 5)
    expected = _prepare__sigma_k_integrand_expected(r)

    sigma_k_integrand = self.variant(cosmology_utils._sigma_k_integrand)

    np.testing.assert_allclose(
        sigma_k_integrand(dict(COSMOLOGY_PARAMS_INIT), log_k, r),
        expected,
        rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_r_{r}', r, expected)
      for r, expected in zip([1e-2, 1e-1, 1e0], [0.004892, 0.002956, 0.001396])
  ])
  def test__sigma_numerical(self, r, expected):
    sigma_numerical = self.variant(cosmology_utils._sigma_numerical)

    np.testing.assert_allclose(
        sigma_numerical(dict(COSMOLOGY_PARAMS_INIT), r), expected, rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_se_{se}', se, expected) for se, expected in zip(
          [0.82, 0.88], [8.60876269756018, 9.238672163235316])
  ])
  def test__sigma_norm(self, se, expected):
    # The comparison is to the normalzied sigma_numerical calculation.
    radius = 1e-2
    cosmology_params = dict(COSMOLOGY_PARAMS_INIT)
    cosmology_params['sigma_eight'] = se
    sigma_norm = self.variant(cosmology_utils._sigma_norm)
    sigma_numerical = self.variant(cosmology_utils._sigma_numerical)

    np.testing.assert_allclose((sigma_norm(cosmology_params) *
                                sigma_numerical(cosmology_params, radius)),
                               expected,
                               rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_{z}', z, expected) for z, expected in zip(
          [0.0, 0.1, 0.2, 0.3], [0.7843, 0.7442, 0.7060, 0.6698])
  ])
  def test___growth_factor_exact_unormalized(self, z, expected):
    growth_factor = self.variant(
        cosmology_utils._growth_factor_exact_unormalized)

    self.assertAlmostEqual(
        growth_factor(dict(COSMOLOGY_PARAMS_INIT), z), expected, places=3)

  @parameterized.named_parameters([
      (f'_z_l_max_{z_l_max}_dz_{dz}_r_max_{r_max}', z_l_max, dz, r_max)
      for z_l_max, dz, r_max in zip([1.01, 2.02], [0.01, 0.02], [1e-2, 1e1])
  ])
  def test_add_lookup_tables_to_cosmology_params(self, z_l_max, dz, r_max):
    cosmology_params = (
        cosmology_utils.add_lookup_tables_to_cosmology_params(
            dict(COSMOLOGY_PARAMS_INIT), z_l_max, dz, 1e-3, r_max, 2))

    # For equivalent precision must also be jitted.
    comoving_distance_numerical = jax.jit(
        cosmology_utils._comoving_distance_numerical)
    sigma_numerical = jax.jit(cosmology_utils._sigma_numerical)
    sigma_norm = jax.jit(cosmology_utils._sigma_norm)
    growth_factor = jax.jit(cosmology_utils._growth_factor_exact_unormalized)

    # Tests for comoving distance lookup table.

    # Diagonal should always be zero
    np.testing.assert_allclose(
        jnp.diag(cosmology_params['comoving_lookup_table']),
        jnp.zeros(len(cosmology_params['comoving_lookup_table'])),
        rtol=0,
        atol=1e-7)

    # Test a few values. Low precision from float 32.
    i = 0
    j = 30
    self.assertAlmostEqual(
        cosmology_params['comoving_lookup_table'][i, j],
        comoving_distance_numerical(
            dict(COSMOLOGY_PARAMS_INIT), dz * i, dz * j),
        places=3)
    i = 12
    j = 37
    self.assertAlmostEqual(
        cosmology_params['comoving_lookup_table'][i, j],
        comoving_distance_numerical(cosmology_params, dz * i, dz * j),
        places=3)

    # Check z_min > z_max
    i = 42
    j = 15
    self.assertAlmostEqual(
        cosmology_params['comoving_lookup_table'][i, j],
        comoving_distance_numerical(cosmology_params, dz * i, dz * j),
        places=3)

    # Test sigma lookup table.
    self.assertAlmostEqual(
        cosmology_params['sigma_lookup_table'][1],
        sigma_numerical(cosmology_params, r_max) * sigma_norm(cosmology_params),
        places=3)

    # Test growth factor.
    i = 12
    growth_norm = 1.0 / growth_factor(cosmology_params, 0.0)
    self.assertAlmostEqual(
        cosmology_params['growth_lookup_table'][i],
        growth_factor(cosmology_params, dz * i) * growth_norm,
        places=3)

    i = 42
    self.assertAlmostEqual(
        cosmology_params['growth_lookup_table'][i],
        growth_factor(cosmology_params, dz * i) * growth_norm,
        places=3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_min_{z_min}_z_max_{z_max}', z_min, z_max)
      for z_min, z_max in zip([0.1110, 0.2102], [0.2001, 0.3004])
  ])
  def test_comoving_distance(self, z_min, z_max):
    # Test that inerpolation returns approximately the correct result even when
    # the sampling doesn't fall perfectly on the grid.
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, 0.31,
                                                 0.01)

    comoving_distance = self.variant(cosmology_utils.comoving_distance)
    comoving_distance_numerical = jax.jit(
        cosmology_utils._comoving_distance_numerical)

    np.testing.assert_allclose(
        comoving_distance(cosmology_params, z_min, z_max),
        comoving_distance_numerical(cosmology_params, z_min, z_max),
        rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.0, 1.283, 20.0]])
  def test_angular_diameter_distance(self, z):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z, z)
    expected = _prepare_expected_angular_diameter_distance(z)
    angular_diameter_distance = self.variant(
        cosmology_utils.angular_diameter_distance)

    np.testing.assert_allclose(
        angular_diameter_distance(cosmology_params, z), expected, rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_min_{z_min}_z_max_{z_max}', z_min, z_max)
      for z_min, z_max in zip([0.0, 0.0, 0.0], [0.0, 1.283, 20.0])
  ])
  def test_angular_diameter_distance_between(self, z_min, z_max):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z_max,
                                                 z_max)
    expected = _prepare_expected_comoving_distance(z_min, z_max) / (1 + z_max)
    angular_diameter_distance_between = self.variant(
        cosmology_utils.angular_diameter_distance_between)

    np.testing.assert_allclose(
        angular_diameter_distance_between(cosmology_params, z_min, z_max),
        expected,
        rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.0, 1.283, 20.0]])
  def test_kpc_per_arcsecond(self, z):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z, z)
    expected = _prepare_expected_kpc_per_arcsecond(z)
    kpc_per_arcsecond = self.variant(cosmology_utils.kpc_per_arcsecond)

    np.testing.assert_allclose(
        kpc_per_arcsecond(cosmology_params, z), expected, rtol=1e-4)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_lens_{z_lens}', z_lens) for z_lens in [0.1, 0.5, 1.0]
  ])
  def test_comoving_to_angle(self, z_lens):
    x, y = _prepare_x_y()
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_LENSTRONOMY,
                                                 z_lens, z_lens)
    expected = _preapre_comoving_to_angle_expected(z_lens)

    comoving_to_angle = self.variant(cosmology_utils.comoving_to_angle)

    np.testing.assert_allclose(
        jnp.array(comoving_to_angle(x, y, cosmology_params, z_lens)),
        expected,
        rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_l_{z_lens}_z_s_{z_source}', z_lens, z_source)
      for z_lens, z_source in zip([0.1, 0.5, 0.5], [0.5, 1.0, 2.0])
  ])
  def test_reduced_to_physical(self, z_lens, z_source):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_LENSTRONOMY,
                                                 z_source, z_lens)
    reduced, _ = _prepare_alpha_x_alpha_y()
    expected = _prepare_reduced_to_physical_expected(z_lens, z_source)

    reduced_to_physical = self.variant(cosmology_utils.reduced_to_physical)

    # Division and multiplication by two quantities with errors in 1e-6, so need
    # to increase rtol for this test significantly.
    np.testing.assert_allclose(
        reduced_to_physical(reduced, cosmology_params, z_lens, z_source),
        expected,
        rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omega_m_{omega_m_zero}_z_{z}', omega_m_zero, z, expected)
      for omega_m_zero, z, expected in zip([0.2, 0.3, 0.4], [0.2, 0.4, 1.2],
                                           [95.917, 228.468, 1182.084])
  ])
  def test_rho_matter(self, omega_m_zero, z, expected):
    rho_matter = self.variant(cosmology_utils.rho_matter)

    self.assertAlmostEqual(
        rho_matter({'omega_m_zero': omega_m_zero}, z) / 1e9, expected, places=3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_z_{z}', z, expected)
      for z, expected in zip([0.2, 0.3, 0.4], [1.6788, 1.6800, 1.6809])
  ])
  def test_collapse_overdensity(self, z, expected):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z, z)
    collapse_overdensity = self.variant(cosmology_utils.collapse_overdensity)

    self.assertAlmostEqual(
        collapse_overdensity(cosmology_params, z), expected, places=3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_m_log10_{m_log10}', m_log10, expected)
      for m_log10, expected in zip([7, 8, 9], [0.0303, 0.06530, 0.14069])
  ])
  def test_lagrangian_radius(self, m_log10, expected):
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, 0.1,
                                                 0.1)
    lagrangian_radius = self.variant(cosmology_utils.lagrangian_radius)

    self.assertAlmostEqual(
        lagrangian_radius(cosmology_params, 10**m_log10), expected, places=3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_r_{r}_z_{z}', r, z, expected) for r, z, expected in zip(
          [1e-2, 1e-1, 1e0], [0.1, 0.2, 0.3],
          [8.128724384821487, 4.660020562550168, 2.08730668354650])
  ])
  def test_sigma_tophat(self, r, z, expected):
    # Minimal lookup tables to reproduce desired results.
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z, z,
                                                 1e-3, r, 2)

    sigma_tophat = self.variant(cosmology_utils.sigma_tophat)

    # Test that inerpolation behaves as expected.
    self.assertAlmostEqual(
        sigma_tophat(cosmology_params, r, z), expected, places=2)
    self.assertAlmostEqual(
        sigma_tophat(cosmology_params, r + 1e-5, z), expected, places=2)
    self.assertAlmostEqual(
        sigma_tophat(cosmology_params, r - 1e-5, z), expected, places=2)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_r_{r}_z_{z}', r, z, expected) for r, z, expected in zip(
          [1e-2, 1e-1, 1e0], [0.1, 0.2, 0.3],
          [-0.1840887668734233, -0.26100267465205956, -0.4077157354139687])
  ])
  def test_derivative_log_sigma_log_r(self, r, z, expected):
    # Minimal lookup tables to reproduce desired results.
    dr = 1e-4
    cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z, z,
                                                 r - 2 * dr, r + 2 * dr, 5)

    derivative_log_sigma_log_r = self.variant(
        cosmology_utils.derivative_log_sigma_log_r)

    self.assertAlmostEqual(
        derivative_log_sigma_log_r(cosmology_params, r, z), expected, places=2)


if __name__ == '__main__':
  absltest.main()
