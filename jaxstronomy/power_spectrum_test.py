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
import jax.numpy as jnp
import numpy as np
from jaxstronomy import power_spectrum


def _prepare__calculate_z_k_eq_expected(omega_m_h_squared, temp_cmb_scaled):
  if omega_m_h_squared == 0.2 and temp_cmb_scaled == 1.0:
    return (5000.0, 0.01492)
  elif omega_m_h_squared == 0.3 and temp_cmb_scaled == 0.9:
    return (11431.184270690443, 0.02762962962962963)
  elif omega_m_h_squared == 0.4 and temp_cmb_scaled == 1.1:
    return (6830.134553650705, 0.024661157024793386)
  else:
    raise ValueError('Unsupported omega_m_h_squared and temp_cmb_scaled ' +
                     f'= ({omega_m_h_squared},{temp_cmb_scaled})')


def _prepare__transfer_hypergeometric_terms_expected(omega_m_h_squared,
                                                     omega_m_zero,
                                                     omega_cdm_zero,
                                                     omega_ratio):
  if (omega_m_h_squared == 0.3 and omega_m_zero == 1.5 and
      omega_cdm_zero == 1.0 and omega_ratio == 0.3):
    return (0.5325242225193926, 1.4697116651237383)
  elif (omega_m_h_squared == 0.4 and omega_m_zero == 2.0 and
        omega_cdm_zero == 1.5 and omega_ratio == 0.6):
    return (0.22393956565237721, 1.3160829068010944)
  else:
    raise ValueError('Unsupported omega_m_h_squared, omega_m_zero, ' +
                     'omega_cdm_zero, and omega_ratio ' +
                     f'= ({omega_m_h_squared},{omega_m_zero}' +
                     f'= {omega_cdm_zero},{omega_ratio})')


def _prepare__transfer_cdm_expected(omega_m_h_squared, temp_cmb_scaled, alpha_c,
                                    beta_c):
  if (omega_m_h_squared == 0.3 and temp_cmb_scaled == 1.0 and alpha_c == 0.5 and
      beta_c == 1.3):
    return jnp.array([
        1.00000000e+00, 9.99999999e-01, 9.69724390e-01, 4.95379216e-05,
        4.30261459e-12
    ])
  elif (omega_m_h_squared == 0.4 and temp_cmb_scaled == 0.9 and
        alpha_c == 0.3 and beta_c == 1.4):
    return jnp.array([
        1.00000000e+00, 9.99999999e-01, 9.86159341e-01, 7.37936223e-05,
        6.77628676e-12
    ])
  else:
    raise ValueError('Unsupported omega_m_h_squared, temp_cmb_scaled, ' +
                     'alpha_c and beta_c ' +
                     f'= ({omega_m_h_squared},{temp_cmb_scaled}' +
                     f'= {alpha_c},{beta_c})')


def _prepare__transfer_baryon_expected(omega_m_h_squared, omega_b_h_squared,
                                       temp_cmb_scaled):
  if (omega_m_h_squared == 0.3 and omega_b_h_squared == 0.1 and
      temp_cmb_scaled == 1.0):
    return jnp.array([
        1.00000000e+00, 1.00000000e+00, 9.99999524e-01, 9.67969220e-01,
        -4.66377586e-06
    ])
  elif (omega_m_h_squared == 0.4 and omega_b_h_squared == 0.2 and
        temp_cmb_scaled == 0.9):
    return jnp.array([
        1.00000000e+00, 1.00000000e+00, 9.99999819e-01, 9.85741428e-01,
        -6.66151765e-05
    ])
  else:
    raise ValueError('Unsupported omega_m_h_squared, omega_b_h_squared, ' +
                     'temp_cmb_scaled ' +
                     f'= ({omega_m_h_squared},{omega_b_h_squared}' +
                     f'= {temp_cmb_scaled})')


def _prepare_transfer_function_expected(omega_m_zero, omega_b_zero,
                                        temp_cmb_zero, hubble_constant):
  if (omega_m_zero == 0.3 and omega_b_zero == 0.1 and temp_cmb_zero == 2.7 and
      hubble_constant == 70.0):
    return jnp.array([
        6.86652838e-01, 1.37657267e-01, 1.08448279e-02, 6.32503650e-04,
        2.97632697e-05
    ])
  elif (omega_m_zero == 0.4 and omega_b_zero == 0.2 and temp_cmb_zero == 2.2 and
        hubble_constant == 68.0):
    return jnp.array([
        8.24170157e-01, 1.61795654e-01, 1.73111531e-02, 1.03541668e-03,
        5.03621680e-05
    ])
  else:
    raise ValueError('Unsupported omega_m_zero, omega_b_zero, ' +
                     'temp_cmb_zero, and hubble_constant ' +
                     f'= ({omega_m_zero},{omega_b_zero}' +
                     f'= {temp_cmb_zero},{hubble_constant})')


def _prepare_matter_power_spectrum_expected(omega_m_zero, omega_b_zero,
                                            temp_cmb_zero, hubble_constant,
                                            n_s):
  if (omega_m_zero == 0.3 and omega_b_zero == 0.1 and temp_cmb_zero == 2.7 and
      hubble_constant == 70.0 and n_s == 0.96):
    return jnp.array([
        7.33307553e-03, 2.08407101e-03, 1.12568228e-04, 2.35482862e-06,
        2.92068702e-08
    ])
  elif (omega_m_zero == 0.4 and omega_b_zero == 0.2 and temp_cmb_zero == 2.2 and
        hubble_constant == 68.0 and n_s == 0.98):
    return jnp.array([
        8.75279731e-03, 3.64332269e-03, 2.73934825e-04, 6.90003260e-06,
        9.60384043e-08
    ])
  else:
    raise ValueError('Unsupported omega_m_zero, omega_b_zero, ' +
                     'temp_cmb_zero, and hubble_constant ' +
                     f'= ({omega_m_zero},{omega_b_zero}' +
                     f'= {temp_cmb_zero},{hubble_constant})')


class PowerSpectrumTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of various cosmology functions."""

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_tc_scaled_{tc_scaled}', omhs, tc_scaled)
      for omhs, tc_scaled in zip([0.2, 0.3, 0.4], [1.0, 0.9, 1.1])
  ])
  def test__calculate_z_k_eq(self, omhs, tc_scaled):
    calculate_z_k_eq = self.variant(power_spectrum._calculate_z_k_eq)
    z_eq, k_eq = calculate_z_k_eq(omhs, tc_scaled)
    z_expected, k_expected = _prepare__calculate_z_k_eq_expected(
        omhs, tc_scaled)

    self.assertAlmostEqual(z_eq, z_expected, places=2)
    self.assertAlmostEqual(k_eq, k_expected, places=4)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_obhs_{obhs}_tc_scaled_{tc_scaled}', obhs, tc_scaled, z, expected)
      for obhs, tc_scaled, z, expected in zip([0.2, 0.3], [1.0, 0.9],
                                              [0.1, 0.2], [63000.0, 72016.47])
  ])
  def test__baryon_to_photon_ratio(self, obhs, tc_scaled, z, expected):
    baryon_to_photon_ratio = self.variant(
        power_spectrum._baryon_to_photon_ratio)

    self.assertAlmostEqual(
        baryon_to_photon_ratio(obhs, tc_scaled, z), expected, places=1)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_obhs_{obhs}', omhs, obhs, expected)
      for omhs, obhs, expected in zip([0.3, 0.4], [0.1, 0.2],
                                      [1099.89, 1133.71])
  ])
  def test__calculate_z_drag(self, omhs, obhs, expected):
    calculate_z_drag = self.variant(power_spectrum._calculate_z_drag)

    self.assertAlmostEqual(calculate_z_drag(omhs, obhs), expected, places=2)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_obhs_{obhs}_tc_scaled_{tc_scaled}', omhs, obhs, tc_scaled,
       expected) for omhs, obhs, tc_scaled, expected in zip(
           [0.3, 0.4], [0.1, 0.2], [1.0, 0.9], [93.398, 68.300])
  ])
  def test__calculate_sound_horizon(self, omhs, obhs, tc_scaled, expected):
    calculate_sound_horizon = self.variant(
        power_spectrum._calculate_sound_horizon)

    self.assertAlmostEqual(
        calculate_sound_horizon(omhs, obhs, tc_scaled), expected, places=2)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_omz_{omz}_ocz_{ocz}_omr_{omr}', omhs, omz, ocz, omr)
      for omhs, omz, ocz, omr in zip([0.3, 0.4], [1.5, 2.0], [1.0, 1.5],
                                     [0.3, 0.6])
  ])
  def test__transfer_hypergeometric_terms(self, omhs, omz, ocz, omr):
    transfer_hypergeometric_terms = self.variant(
        power_spectrum._transfer_hypergeometric_terms)
    alpha_c_expected, beta_c_expected = (
        _prepare__transfer_hypergeometric_terms_expected(omhs, omz, ocz, omr))

    alpha_c, beta_c = transfer_hypergeometric_terms(omhs, omz, ocz, omr)
    self.assertAlmostEqual(alpha_c, alpha_c_expected, places=3)
    self.assertAlmostEqual(beta_c, beta_c_expected, places=3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_tc_scaled_{tc_scaled}_ac_{ac}_bc_{bc}', omhs, tc_scaled,
       ac, bc) for omhs, tc_scaled, ac, bc in zip([0.3, 0.4], [1.0, 0.9],
                                                  [0.5, 0.3], [1.3, 1.4])
  ])
  def test__transfer_cdm(self, omhs, tc_scaled, ac, bc):
    transfer_cdm = self.variant(power_spectrum._transfer_cdm)
    expected = _prepare__transfer_cdm_expected(omhs, tc_scaled, ac, bc)
    k = jnp.logspace(-10, 5, 5)

    np.testing.assert_allclose(
        transfer_cdm(k, omhs, tc_scaled, ac, bc), expected, rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omhs_{omhs}_obhs_{obhs}_tc_scaled_{tc_scaled}', omhs, obhs, tc_scaled)
      for omhs, obhs, tc_scaled in zip([0.3, 0.4], [0.1, 0.2], [1.0, 0.9])
  ])
  def test__transfer_baryon(self, omhs, obhs, tc_scaled):
    transfer_baryon = self.variant(power_spectrum._transfer_baryon)
    k = jnp.logspace(-10, 0, 5)
    expected = _prepare__transfer_baryon_expected(omhs, obhs, tc_scaled)

    np.testing.assert_allclose(
        transfer_baryon(k, omhs, obhs, tc_scaled), expected, rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omz_{omz}_obz_{obz}_tcz_{tcz}_hc_{hc}', omz, obz, tcz, hc)
      for omz, obz, tcz, hc in zip([0.3, 0.4], [0.1, 0.2], [2.7, 2.2],
                                   [70.0, 68.0])
  ])
  def test_transfer_function(self, omz, obz, tcz, hc):
    transfer_function = self.variant(power_spectrum.transfer_function)
    k = jnp.logspace(-2, 1, 5)
    expected = _prepare_transfer_function_expected(omz, obz, tcz, hc)

    np.testing.assert_allclose(
        transfer_function(k, omz, obz, tcz, hc), expected, rtol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_omz_{omz}_obz_{obz}_tcz_{tcz}_hc_{hc}_ns_{ns}', omz, obz, tcz, hc, ns)
      for omz, obz, tcz, hc, ns in zip([0.3, 0.4], [0.1, 0.2], [2.7, 2.2],
                                       [70.0, 68.0], [0.96, 0.98])
  ])
  def test_matter_power_spectrum(self, omz, obz, tcz, hc, ns):
    matter_power_spectrum = self.variant(power_spectrum.matter_power_spectrum)
    k = jnp.logspace(-2, 1, 5)
    expected = _prepare_matter_power_spectrum_expected(omz, obz, tcz, hc, ns)

    np.testing.assert_allclose(
        matter_power_spectrum(k, omz, obz, tcz, hc, ns), expected, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
