# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for symbolic.xc_functionals."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import evaluators
from symbolic_functionals.syfes.symbolic import xc_functionals
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import mgga
from symbolic_functionals.syfes.xc import xc

jax.config.update('jax_enable_x64', True)


def _get_expected_e_xc_unpolarized(rho_and_derivs, xc_name):
  """Gets expected XC energy density from unpolarized rho and derivatives."""
  rho, grad_x, grad_y, grad_z, _, tau = rho_and_derivs
  sigma = grad_x ** 2 + grad_y ** 2 + grad_z ** 2
  if xc_name == 'B97':
    return (gga.e_x_b97_unpolarized(rho, sigma)
            + gga.e_c_b97_unpolarized(rho, sigma))
  elif xc_name == 'wB97M-V':
    return mgga.e_xc_wb97mv_unpolarized(rho, sigma, tau)
  else:
    raise ValueError


def _get_expected_e_xc_polarized(rho_and_derivs, xc_name):
  """Gets expected XC energy density from spin polarized rho and derivatives."""
  rho_a, grad_x_a, grad_y_a, grad_z_a, _, tau_a = rho_and_derivs[0]
  rho_b, grad_x_b, grad_y_b, grad_z_b, _, tau_b = rho_and_derivs[1]
  sigma_aa = grad_x_a ** 2 + grad_y_a ** 2 + grad_z_a ** 2
  sigma_ab = grad_x_a * grad_x_b + grad_y_a * grad_y_b + grad_z_a * grad_z_b
  sigma_bb = grad_x_b ** 2 + grad_y_b ** 2 + grad_z_b ** 2
  if xc_name == 'B97':
    return (gga.e_x_b97_polarized(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)
            + gga.e_c_b97_polarized(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb))
  elif xc_name == 'wB97M-V':
    return mgga.e_xc_wb97mv_polarized(
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b)
  else:
    raise ValueError


class XCFunctionalTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_eval_exc_with_b97_u(self, use_jax):
    rho_and_derivs = np.random.rand(6, 5)

    quantities, lda_energies = evaluators.parse_rho_and_derivs(
        0.5 * rho_and_derivs, omega=0., polarized=False)  # 0.5: spin factor
    parameters = copy.deepcopy(xc_functionals.B97_PARAMETERS)
    u_x = gga.u_b97(
        np.sqrt(quantities['x2']),
        gamma=parameters['parameters_x'].pop('gamma'),
        polarized=True)
    u_css = gga.u_b97(
        np.sqrt(quantities['x2']),
        gamma=parameters['parameters_css'].pop('gamma'),
        polarized=True)
    u_cos = gga.u_b97(
        np.sqrt(quantities['x2']),
        gamma=parameters['parameters_cos'].pop('gamma'),
        polarized=True)

    e_xc = xc_functionals.b97_u.eval_exc(
        **parameters,
        e_lda_x=lda_energies['x'],
        e_lda_css=lda_energies['css'],
        e_lda_cos=lda_energies['cos'],
        features_x={'u': u_x},
        features_css={'u': u_css},
        features_cos={'u': u_cos},
        use_jax=use_jax
    )

    np.testing.assert_allclose(
        e_xc, _get_expected_e_xc_unpolarized(rho_and_derivs, xc_name='B97'))

  @parameterized.parameters(
      (xc_functionals.b97_x2, xc_functionals.B97_PARAMETERS, False),
      (xc_functionals.b97_x2, xc_functionals.B97_PARAMETERS, True),
      (xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM,
       False),
      (xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM,
       True),
      )
  def test_eval_exc_with_b97_x2_unpolarized(self, b97_x2, parameters, use_jax):
    rho_and_derivs = np.random.rand(6, 5)
    quantities, lda_energies = evaluators.parse_rho_and_derivs(
        0.5 * rho_and_derivs, omega=0., polarized=False)  # 0.5: spin factor

    e_xc = b97_x2.eval_exc(
        **parameters,
        e_lda_x=lda_energies['x'],
        e_lda_css=lda_energies['css'],
        e_lda_cos=lda_energies['cos'],
        features={'x2': quantities['x2']},
        use_jax=use_jax
    )

    np.testing.assert_allclose(
        e_xc, _get_expected_e_xc_unpolarized(rho_and_derivs, xc_name='B97'))

  @parameterized.parameters(
      (xc_functionals.b97_x2, xc_functionals.B97_PARAMETERS, False),
      (xc_functionals.b97_x2, xc_functionals.B97_PARAMETERS, True),
      (xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM,
       False),
      (xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM,
       True),
      )
  def test_eval_exc_with_b97_x2_polarized(self, b97_x2, parameters, use_jax):
    rho_and_derivs = np.random.rand(2, 6, 5)
    quantities, lda_energies = evaluators.parse_rho_and_derivs(
        rho_and_derivs, omega=0., polarized=True)

    e_xc = np.sum(b97_x2.eval_exc(
        **parameters,
        e_lda_x=lda_energies['x'],
        e_lda_css=lda_energies['css'],
        e_lda_cos=lda_energies['cos'],
        features={'x2': quantities['x2']},
        use_jax=use_jax
    ).reshape(-1, 3), axis=1)

    np.testing.assert_allclose(
        e_xc, _get_expected_e_xc_polarized(rho_and_derivs, xc_name='B97'))

  @parameterized.parameters(
      (xc_functionals.wb97mv, xc_functionals.WB97MV_PARAMETERS),
      (xc_functionals.wb97mv_short, xc_functionals.WB97MV_PARAMETERS_UTRANSFORM)
      )
  def test_eval_exc_with_wb97mv_unpolarized(self, wb97mv_form, parameters):
    rho_and_derivs = np.random.rand(6, 5)
    quantities, lda_energies = evaluators.parse_rho_and_derivs(
        0.5 * rho_and_derivs, omega=0.3, polarized=False)  # 0.5: spin factor

    e_xc = wb97mv_form.eval_exc(
        **parameters,
        e_lda_x=lda_energies['x'],
        e_lda_css=lda_energies['css'],
        e_lda_cos=lda_energies['cos'],
        features={'w': quantities['w'], 'x2': quantities['x2']},
    )

    np.testing.assert_allclose(
        e_xc, _get_expected_e_xc_unpolarized(rho_and_derivs, xc_name='wB97M-V'))

  @parameterized.parameters(
      (xc_functionals.wb97mv, xc_functionals.WB97MV_PARAMETERS),
      (xc_functionals.wb97mv_short, xc_functionals.WB97MV_PARAMETERS_UTRANSFORM)
      )
  def test_eval_exc_with_wb97mv_polarized(self, wb97mv_form, parameters):
    rho_and_derivs = np.random.rand(2, 6, 5)
    quantities, lda_energies = evaluators.parse_rho_and_derivs(
        rho_and_derivs, omega=0.3, polarized=True)

    e_xc = np.sum(wb97mv_form.eval_exc(
        **parameters,
        e_lda_x=lda_energies['x'],
        e_lda_css=lda_energies['css'],
        e_lda_cos=lda_energies['cos'],
        features={'w': quantities['w'], 'x2': quantities['x2']},
    ).reshape(-1, 3), axis=1)

    np.testing.assert_allclose(
        e_xc, _get_expected_e_xc_polarized(rho_and_derivs, xc_name='wB97M-V'))

  def test_convert_xc_functional_to_and_from_dict(self):
    self.assertEqual(
        xc_functionals.b97_u,
        xc_functionals.XCFunctional.from_dict(
            xc_functionals.b97_u.to_dict()))

  def test_make_isomorphic_copy_of_wb97mv_short(self):
    feature_names = ['rho', 'x2', 'w']
    num_shared_parameters = 10
    num_variables = 10

    wb97mv_copy = xc_functionals.wb97mv_short.make_isomorphic_copy(
        feature_names_x=feature_names,
        feature_names_css=feature_names,
        feature_names_cos=feature_names,
        num_shared_parameters=num_shared_parameters,
        num_variables=num_variables)

    self.assertEqual(
        wb97mv_copy.f_x,
        enhancement_factors.f_x_wb97mv_short.make_isomorphic_copy(
            feature_names=feature_names,
            num_shared_parameters=num_shared_parameters,
            num_variables=num_variables))
    self.assertEqual(
        wb97mv_copy.f_css,
        enhancement_factors.f_css_wb97mv_short.make_isomorphic_copy(
            feature_names=feature_names,
            num_shared_parameters=num_shared_parameters,
            num_variables=num_variables))
    self.assertEqual(
        wb97mv_copy.f_cos,
        enhancement_factors.f_cos_wb97mv_short.make_isomorphic_copy(
            feature_names=feature_names,
            num_shared_parameters=num_shared_parameters,
            num_variables=num_variables))

  @parameterized.parameters(
      list(range(len(xc_functionals.XCFunctional._signature_features['rho']))))
  def test_get_signature_of_lda(self, rho_index):
    rho = xc_functionals.XCFunctional._signature_features['rho'][rho_index]
    expected_e_xc = (lda.e_x_lda_unpolarized(2 * rho)  # 2: spin factor
                     + lda.e_c_lda_unpolarized(2 * rho))

    mock_random_state = mock.Mock()
    mock_random_state.choice = mock.Mock(side_effect=[[rho_index]])
    e_xc = xc_functionals.lda_functional.get_signature(
        {}, {}, {},
        num_feature_samples=1,
        signature='e_xc',
        random_state=mock_random_state)

    np.testing.assert_allclose(e_xc, expected_e_xc)

  @parameterized.parameters((0, 1.), (1, 10.))
  def test_get_signature_of_b97(self, rho_index, sigma):
    rho = xc_functionals.XCFunctional._signature_features['rho'][rho_index]
    x2 = (.5 * np.sqrt(sigma) / rho ** (4 / 3)) ** 2  # .5: spin factor
    expected_e_xc = (gga.e_x_b97_unpolarized(2 * rho, sigma)
                     + gga.e_c_b97_unpolarized(2 * rho, sigma))

    mock_random_state = mock.Mock()
    mock_random_state.choice = mock.Mock(
        side_effect=[[rho_index], np.array([x2])])
    e_xc = xc_functionals.b97_x2.get_signature(
        **xc_functionals.B97_PARAMETERS,
        num_feature_samples=1,
        signature='e_xc',
        random_state=mock_random_state)

    np.testing.assert_allclose(e_xc, expected_e_xc)

  def test_equivalent_functional_forms_have_identical_fingerprint(self):
    self.assertEqual(
        xc_functionals.wb97mv.get_fingerprint(),
        xc_functionals.wb97mv_short.get_fingerprint())

  @parameterized.parameters(
      (xc_functionals.empty_functional, xc_functionals.b97x_u_short),
      (xc_functionals.b97x_u_short, xc_functionals.b97_u),
      (xc_functionals.b97_u, xc_functionals.b97_x2),
      (xc_functionals.b97_x2, xc_functionals.wb97mv),)
  def test_different_functional_forms_have_different_fingerprint(
      self, functional1, functional2):
    self.assertNotEqual(
        functional1.get_fingerprint(), functional2.get_fingerprint())

  def test_get_fingerprint_with_unsupported_feature(self):
    with self.assertRaisesRegex(
        ValueError, 'Evaluating signature is not supported for feature t'):
      xc_functionals.empty_functional.make_isomorphic_copy(
          feature_names_x=['t']).get_signature({}, {}, {})

  @parameterized.parameters(
      ('b97', 0.0,
       xc_functionals.b97_x2,
       xc_functionals.B97_PARAMETERS),
      ('b97', 0.0,
       xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM),
      ('wb97m_v', 0.3,
       xc_functionals.wb97mv,
       xc_functionals.WB97MV_PARAMETERS),
      ('wb97m_v', 0.3,
       xc_functionals.wb97mv_short,
       xc_functionals.WB97MV_PARAMETERS_UTRANSFORM),
      )
  def test_make_eval_exc_unpolarized(self, xc_name, omega, xc_fun, parameters):
    rho_and_derivs = np.random.rand(6, 5)
    eval_xc_ref = xc.make_eval_xc(xc_name)
    eval_xc = xc_fun.make_eval_xc(omega=omega, **parameters)

    eps_xc, (vrho, vsigma, _, vtau), _, _ = eval_xc(
        '', rho_and_derivs, spin=0, relativity=0, deriv=1, verbose=None)
    eps_xc_ref, (vrho_ref, vsigma_ref, _, vtau_ref), _, _ = eval_xc_ref(
        '', rho_and_derivs, spin=0, relativity=0, deriv=1, verbose=None)

    np.testing.assert_allclose(eps_xc, eps_xc_ref)
    np.testing.assert_allclose(vrho, vrho_ref)
    np.testing.assert_allclose(vsigma, vsigma_ref)
    np.testing.assert_allclose(vtau, vtau_ref)

  @parameterized.parameters(
      ('b97', 0.0,
       xc_functionals.b97_x2,
       xc_functionals.B97_PARAMETERS),
      ('b97', 0.0,
       xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM),
      ('wb97m_v', 0.3,
       xc_functionals.wb97mv,
       xc_functionals.WB97MV_PARAMETERS),
      ('wb97m_v', 0.3,
       xc_functionals.wb97mv_short,
       xc_functionals.WB97MV_PARAMETERS_UTRANSFORM),
      )
  def test_make_eval_exc_polarized(self, xc_name, omega, xc_fun, parameters):
    rho_and_derivs = np.random.rand(2, 6, 5)
    eval_xc_ref = xc.make_eval_xc(xc_name)
    eval_xc = xc_fun.make_eval_xc(omega=omega, **parameters)

    eps_xc, (vrho, vsigma, _, vtau), _, _ = eval_xc(
        '', rho_and_derivs, spin=1, relativity=0, deriv=1, verbose=None)
    eps_xc_ref, (vrho_ref, vsigma_ref, _, vtau_ref), _, _ = eval_xc_ref(
        '', rho_and_derivs, spin=1, relativity=0, deriv=1, verbose=None)

    np.testing.assert_allclose(eps_xc, eps_xc_ref)
    np.testing.assert_allclose(vrho, vrho_ref)
    np.testing.assert_allclose(vsigma, vsigma_ref)
    np.testing.assert_allclose(vtau, vtau_ref)

if __name__ == '__main__':
  absltest.main()
