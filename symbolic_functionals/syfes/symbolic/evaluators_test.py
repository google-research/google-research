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

"""Tests for symbolic.evaluators."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from symbolic_functionals.syfes import loss
from symbolic_functionals.syfes.symbolic import evaluators
from symbolic_functionals.syfes.symbolic import xc_functionals
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda

jax.config.update('jax_enable_x64', True)


class ParserTest(parameterized.TestCase):

  def test_combine_arrays(self):
    np.testing.assert_allclose(
        evaluators.combine_arrays(
            np.arange(1, 4), np.arange(4, 7), np.arange(7, 10)),
        np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]))

  def test_parse_rho_and_derivs_unpolarized(self):
    num_grids = 10
    rho_and_derivs = np.zeros([6, num_grids])

    features, lda_energies = evaluators.parse_rho_and_derivs(
        rho_and_derivs, omega=0., polarized=False)

    for feature_name in ['rho', 'x2', 'u']:
      np.testing.assert_allclose(
          features[feature_name], np.zeros(num_grids), atol=1e-7)
    for feature_name in ['w']:
      np.testing.assert_allclose(features[feature_name], -np.ones(num_grids))
    for lda_energy in lda_energies.values():
      np.testing.assert_allclose(lda_energy, np.zeros(num_grids))

  def test_parse_rho_and_derivs_polarized(self):
    num_grids = 10
    rho_and_derivs = np.zeros([2, 6, num_grids])

    features, lda_energies = evaluators.parse_rho_and_derivs(
        rho_and_derivs, omega=0., polarized=True)

    for feature_name in ['rho', 'x2', 'u']:
      np.testing.assert_allclose(
          features[feature_name], np.zeros(3 * num_grids), atol=1e-7)
    for feature_name in ['w']:
      np.testing.assert_allclose(
          features[feature_name], -np.ones(3 * num_grids))
    for lda_energy in lda_energies.values():
      np.testing.assert_allclose(lda_energy, np.zeros(3 * num_grids))

  @parameterized.parameters(((3, 4),), ((2, 6, 6),), ((10,),))
  def test_parse_rho_and_derivs_unpolarized_with_wrong_shape(self, shape):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for rho_and_derivs. '
                    rf'Expected \(6, \*\), got \({shape}\)'):
      evaluators.parse_rho_and_derivs(
          np.zeros(shape), omega=0., polarized=False)

  @parameterized.parameters(((1, 6, 10),), ((2, 7, 10),), ((6, 10),), ((10,),))
  def test_parse_rho_and_derivs_polarized_with_wrong_shape(self, shape):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for rho_and_derivs. '
                    rf'Expected \(2, 6, \*\), got \({shape}\)'):
      evaluators.parse_rho_and_derivs(
          np.zeros(shape), omega=0., polarized=True)

  def test_parse_rho_and_derivs_spin_consistency(self):
    num_grids = 10
    rho_and_derivs_unpolarized = np.random.rand(6, num_grids)
    rho_and_derivs_polarized = np.repeat(
        rho_and_derivs_unpolarized[np.newaxis, :, :], repeats=2, axis=0)

    features_unpolarized, lda_energies_unpolarized = (
        evaluators.parse_rho_and_derivs(
            rho_and_derivs_unpolarized,
            omega=0.3,
            polarized=False))
    features_polarized, lda_energies_polarized = (
        evaluators.parse_rho_and_derivs(
            rho_and_derivs_polarized,
            omega=0.3,
            polarized=True))

    for feature_name in features_unpolarized:
      np.testing.assert_allclose(
          np.stack([features_unpolarized[feature_name]] * 3, axis=1),
          features_polarized[feature_name].reshape([-1, 3]))

    for component in ['x', 'css']:
      np.testing.assert_allclose(
          np.stack([
              lda_energies_unpolarized[component],
              lda_energies_unpolarized[component],
              np.zeros(num_grids)], axis=1),
          2 * lda_energies_polarized[component].reshape([-1, 3]))
    np.testing.assert_allclose(
        np.stack([
            np.zeros(num_grids),
            np.zeros(num_grids),
            lda_energies_unpolarized['cos'],
            ], axis=1),
        lda_energies_polarized['cos'].reshape([-1, 3]))

  @parameterized.parameters((10, 5), (10, 0), (0, 5))
  def test_parse_ks_info(self, num_mols_unpolarized, num_mols_polarized):
    num_grids_unpolarized = np.random.randint(5, size=num_mols_unpolarized)
    num_grids_polarized = np.random.randint(5, size=num_mols_polarized)
    ks_info_unpolarized = [
        {'weights': np.random.rand(num_grids),
         'rho': np.random.rand(6, num_grids)}
        for num_grids in num_grids_unpolarized]
    ks_info_polarized = [
        {'weights': np.random.rand(num_grids),
         'rho': np.random.rand(2, 6, num_grids)}
        for num_grids in num_grids_polarized]
    # test the XC energy of a functional with enhancement factors
    # F_x = F_css = F_cos = Identity(rho_sigma)
    # XC energy = sum weights * e_lda * rho_sigma
    expected_xc_energy = 0.
    for ks_info in ks_info_unpolarized:
      rho = ks_info['rho'][0, :]
      expected_xc_energy += np.sum(
          ks_info['weights'] * (0.5 * rho) * (
              lda.e_x_lda_unpolarized(rho) + lda.e_c_lda_unpolarized(rho)))
    for ks_info in ks_info_polarized:
      rho_a = ks_info['rho'][0, 0, :]
      rho_b = ks_info['rho'][1, 0, :]
      e_lda_x_a = 0.5 * lda.e_x_lda_unpolarized(2 * rho_a)
      e_lda_x_b = 0.5 * lda.e_x_lda_unpolarized(2 * rho_b)
      e_lda_css_a, e_lda_css_b, e_lda_cos = lda.decomposed_e_c_lda_polarized(
          rho_a, rho_b)
      expected_xc_energy += np.sum(
          ks_info['weights'] * (
              rho_a * (e_lda_x_a + e_lda_css_a)
              + rho_b * (e_lda_x_b + e_lda_css_b)
              + 0.5 * (rho_a + rho_b) * e_lda_cos))

    results = evaluators.parse_ks_info(
        ks_info_unpolarized=ks_info_unpolarized,
        ks_info_polarized=ks_info_polarized,
        feature_names_x=['rho'],
        feature_names_css=[],
        feature_names_cos=[],
        omega=0.)

    xc_energy = np.sum(results['rho_weights'] * results['features_x']['rho'] * (
        results['e_lda_x'] + results['e_lda_css'] + results['e_lda_cos']
        ))
    self.assertAlmostEqual(xc_energy, expected_xc_energy)


def _make_evaluator(num_mols, num_targets, eval_mode):
  """Makes an evaluator instance for testing."""
  # B97 nonlinear parameters
  gamma_x, gamma_css, gamma_cos = 0.004, 0.2, 0.006

  num_grids_for_mols = [np.random.randint(5, 10) for _ in range(num_mols)]
  grids_indices = [0] + list(np.cumsum(np.array(num_grids_for_mols)))
  num_grids_all = np.sum(num_grids_for_mols)

  rho = np.random.rand(num_grids_all)
  sigma = np.random.rand(num_grids_all)
  x = gga.get_reduced_density_gradient(rho, sigma)
  e_lda_x = lda.e_x_lda_unpolarized(rho)
  e_lda_css, e_lda_cos = lda.decomposed_e_c_lda_unpolarized(rho)
  u_x = gga.u_b97(x, gamma=gamma_x)
  u_css = gga.u_b97(x, gamma=gamma_css)
  u_cos = gga.u_b97(x, gamma=gamma_cos)

  rho_weights = np.random.rand(num_grids_all)
  formula_matrix = np.random.rand(num_targets, num_mols)
  sample_weights = np.random.rand(num_targets)
  expected_exc_weighted = rho_weights * (
      gga.e_x_b97_unpolarized(rho, sigma) +
      gga.e_c_b97_unpolarized(rho, sigma))
  targets = formula_matrix @ jnp.array([
      jnp.sum(expected_exc_weighted[grids_indices[i]:grids_indices[i+1]])
      for i in range(num_mols)
  ])

  return evaluators.Evaluator(
      num_grids_for_mols=num_grids_for_mols,
      rho_weights=rho_weights,
      formula_matrix=formula_matrix,
      targets=targets,
      sample_weights=sample_weights,
      e_lda_x=e_lda_x,
      e_lda_css=e_lda_css,
      e_lda_cos=e_lda_cos,
      features_x={'u': u_x},
      features_css={'u': u_css},
      features_cos={'u': u_cos},
      eval_mode=eval_mode)


class EvaluatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_mols = 10
    self.num_targets = 5

  def test_initialize_evaluator_with_wrong_feature_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for features. Expected \(20,\), got \(30,\)'):
      evaluators.Evaluator(
          num_grids_for_mols=[5, 5, 5, 5],
          rho_weights=np.zeros(20),
          formula_matrix=np.zeros([2, 4]),
          targets=np.zeros(2),
          sample_weights=np.zeros(2),
          e_lda_x=np.zeros(20),
          e_lda_css=np.zeros(20),
          e_lda_cos=np.zeros(20),
          features={'u': np.zeros(30)})

  def test_initialize_evaluator_with_wrong_rho_weights_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for rho_weights. '
                    r'Expected \(20,\), got \(30,\)'):
      evaluators.Evaluator(
          num_grids_for_mols=[5, 5, 5, 5],
          rho_weights=np.zeros(30),
          formula_matrix=np.zeros([2, 4]),
          targets=np.zeros(2),
          sample_weights=np.zeros(2),
          e_lda_x=np.zeros(20),
          e_lda_css=np.zeros(20),
          e_lda_cos=np.zeros(20),
          features={'u': np.zeros(20)})

  def test_initialize_evaluator_with_wrong_e_lda_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for e_lda. '
                    r'Expected \(20,\), got \(30,\)'):
      evaluators.Evaluator(
          num_grids_for_mols=[5, 5, 5, 5],
          rho_weights=np.zeros(20),
          formula_matrix=np.zeros([2, 4]),
          targets=np.zeros(2),
          sample_weights=np.zeros(2),
          e_lda_x=np.zeros(30),
          e_lda_css=np.zeros(30),
          e_lda_cos=np.zeros(30),
          features={'u': np.zeros(20)})

  def test_initialize_evaluator_with_wrong_formula_matrix_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for formula_matrix. '
                    r'Expected \(2, 4\), got \(3, 4\)'):
      evaluators.Evaluator(
          num_grids_for_mols=[5, 5, 5, 5],
          rho_weights=np.zeros(20),
          formula_matrix=np.zeros([3, 4]),
          targets=np.zeros(2),
          sample_weights=np.zeros(2),
          e_lda_x=np.zeros(20),
          e_lda_css=np.zeros(20),
          e_lda_cos=np.zeros(20),
          features={'u': np.zeros(20)})

  def test_initialize_evaluator_with_wrong_sample_weights_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for sample_weights. '
                    r'Expected \(2,\), got \(3,\)'):
      evaluators.Evaluator(
          num_grids_for_mols=[5, 5, 5, 5],
          rho_weights=np.zeros(20),
          formula_matrix=np.zeros([2, 4]),
          targets=np.zeros(2),
          sample_weights=np.zeros(3),
          e_lda_x=np.zeros(20),
          e_lda_css=np.zeros(20),
          e_lda_cos=np.zeros(20),
          features={'u': np.zeros(20)})

  @parameterized.parameters('onp', 'jnp', 'jit')
  def test_eval_xc_energies_for_mols(self, eval_mode):
    evaluator = _make_evaluator(
        num_mols=self.num_mols,
        num_targets=self.num_targets,
        eval_mode=eval_mode)

    xc_energies_for_mols = evaluator.eval_xc_energies_for_mols(
        xc_functionals.b97_u,
        parameters_x={'c0': 0., 'c1': 0., 'c2': 0.},
        parameters_css={'c0': 0., 'c1': 0., 'c2': 0.},
        parameters_cos={'c0': 0., 'c1': 0., 'c2': 0.},)

    self.assertEqual(xc_energies_for_mols.shape, (self.num_mols,))

  @parameterized.parameters('onp', 'jnp', 'jit')
  def test_evaluate_b97_functional(self, eval_mode):
    # B97 linear coefficients
    coeffs_x, coeffs_css, coeffs_cos = (
        np.array([0.8094, 0.5073, 0.7481]),
        np.array([0.1737, 2.3487, -2.4868]),
        np.array([0.9454, 0.7471, -4.5961]))
    evaluator = _make_evaluator(
        num_mols=self.num_mols,
        num_targets=self.num_targets,
        eval_mode=eval_mode)

    wrmsd = evaluator.get_eval_wrmsd(xc_functionals.b97_u)(
        parameters_x={
            'c0': coeffs_x[0],
            'c1': coeffs_x[1],
            'c2': coeffs_x[2]
        },
        parameters_css={
            'c0': coeffs_css[0],
            'c1': coeffs_css[1],
            'c2': coeffs_css[2]
        },
        parameters_cos={
            'c0': coeffs_cos[0],
            'c1': coeffs_cos[1],
            'c2': coeffs_cos[2]
        })

    self.assertAlmostEqual(wrmsd, 0.)

  def test_eval_baseline(self):
    evaluator = _make_evaluator(
        num_mols=self.num_mols, num_targets=self.num_targets, eval_mode='onp')

    evaluator.eval_baselines(['empty'])

    self.assertAlmostEqual(
        evaluator.baselines['empty'],
        (evaluators.HARTREE_TO_KCALPERMOLE
         * loss.weighted_root_mean_square_deviation(
             evaluator.targets, 0., evaluator.sample_weights)))

  def test_eval_baseline_wrong_argument(self):
    with self.assertRaisesRegex(
        ValueError, 'Unknown baseline: unknown_baseline'):
      _make_evaluator(
          num_mols=self.num_mols,
          num_targets=self.num_targets,
          eval_mode='onp').eval_baselines(['unknown_baseline'])


class GridEvaluatorTest(parameterized.TestCase):

  def test_initialize_grid_evaluator_with_wrong_feature_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for features. Expected \(10,\), got \(20,\)'):
      evaluators.GridEvaluator(
          features={'u': np.zeros(20)},
          weights=np.zeros(10),
          targets=np.zeros(10),
          e_lda_x=np.zeros(10),
          e_lda_css=np.zeros(10),
          e_lda_cos=np.zeros(10),
          signature='f_xc')

  def test_initialize_grid_evaluator_with_wrong_e_lda_shape(self):
    with self.assertRaisesRegex(
        ValueError, r'Wrong shape for e_lda. Expected \(10,\), got \(20,\)'):
      evaluators.GridEvaluator(
          features={'u': np.zeros(10)},
          weights=np.zeros(10),
          targets=np.zeros(10),
          e_lda_x=np.zeros(10),
          e_lda_css=np.zeros(20),
          e_lda_cos=np.zeros(10),
          signature='f_xc')

  def test_initialize_grid_evaluator_with_wrong_eval_mode(self):
    with self.assertRaisesRegex(
        ValueError,
        'Unknown eval_mode np, supported values are onp, jnp and jit'):
      evaluators.GridEvaluator(
          weights=np.zeros(10),
          targets=np.zeros(10),
          e_lda_x=np.zeros(10),
          e_lda_css=np.zeros(10),
          e_lda_cos=np.zeros(10),
          signature='f_xc',
          features={},
          eval_mode='np')

  def test_initialize_grid_evaluator_with_wrong_signature(self):
    with self.assertRaisesRegex(
        ValueError,
        'Unknown signature xc, supported values are e_xc and f_xc'):
      evaluators.GridEvaluator(
          weights=np.zeros(10),
          targets=np.zeros(10),
          e_lda_x=np.zeros(10),
          e_lda_css=np.zeros(10),
          e_lda_cos=np.zeros(10),
          signature='xc',
          features={},
          eval_mode='onp')

  @parameterized.parameters('onp', 'jnp', 'jit')
  def test_eval_wrmsd_e_xc(self, eval_mode):
    num_grids = 10
    raw_weights = np.random.rand(num_grids)

    evaluator = evaluators.GridEvaluator(
        weights=raw_weights * np.sqrt(num_grids / np.sum(raw_weights ** 2)),
        targets=np.ones(num_grids),  # targets are LDA enhancement factors 1.
        e_lda_x=np.random.rand(num_grids),
        e_lda_css=np.random.rand(num_grids),
        e_lda_cos=np.random.rand(num_grids),
        features={},
        signature='f_xc',
        eval_mode=eval_mode)

    self.assertAlmostEqual(
        evaluator.get_eval_wrmsd(xc_functionals.empty_functional)({}, {}, {}),
        1.)
    self.assertAlmostEqual(
        evaluator.get_eval_wrmsd(xc_functionals.lda_functional)({}, {}, {}),
        0.)


if __name__ == '__main__':
  absltest.main()
