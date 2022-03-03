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

"""Tests for symbolic.optimizer."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import evaluators
from symbolic_functionals.syfes.symbolic import instructions
from symbolic_functionals.syfes.symbolic import optimizers
from symbolic_functionals.syfes.symbolic import xc_functionals

jax.config.update('jax_enable_x64', True)


class CMAESOptimizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.functional = xc_functionals.XCFunctional(
        # objective = (x + 1) ** 2 + (y + 1) ** 2
        f_x=enhancement_factors.EnhancementFactor(
            shared_parameter_names=['x'],
            variable_names=['var1', 'var2', 'enhancement_factor'],
            instruction_list=[
                instructions.AdditionBy1Instruction('var1', 'x'),
                instructions.Power2Instruction('var2', 'var1'),
                instructions.AdditionInstruction(
                    'enhancement_factor', 'enhancement_factor', 'var2'),
            ]),
        f_css=enhancement_factors.EnhancementFactor(
            shared_parameter_names=['y'],
            variable_names=['var1', 'var2', 'enhancement_factor'],
            instruction_list=[
                instructions.AdditionBy1Instruction('var1', 'y'),
                instructions.Power2Instruction('var2', 'var1'),
                instructions.AdditionInstruction(
                    'enhancement_factor', 'enhancement_factor', 'var2'),
            ]),
        f_cos=enhancement_factors.EnhancementFactor(
            variable_names=['enhancement_factor']),
        )
    self.evaluator = evaluators.Evaluator(
        num_grids_for_mols=[5, 5],
        rho_weights=np.ones(10) * 0.2,
        formula_matrix=np.eye(2),
        targets=np.zeros(2),
        sample_weights=np.ones(2),
        e_lda_x=np.ones(10),
        e_lda_css=np.ones(10),
        e_lda_cos=np.zeros(10),
        features={})
    self.optimizer = optimizers.CMAESOptimizer(
        evaluator=self.evaluator,
        initial_parameters_mean=0.,
        initial_parameters_std=0.,
        sigma0=1.)

  def test_constructor_with_unknown_kwarg(self):
    with self.assertRaisesRegex(
        ValueError, 'Unknown hyperparameter: unknown_flag'):
      optimizers.CMAESOptimizer(evaluator=self.evaluator, unknown_flag=None)

  @parameterized.parameters(
      (50., None, None, True),  # terminate due to abnormal wrmsd value
      (1., 101, 2., True),  # terminate due to large wrmsd after certain steps
      (1., 101, 0.5, False),  # proceed without termiantion
      )
  def test_termination_callback(
      self, fit, countevals, histbest, expected_terminate):
    mock_cma_es = mock.Mock()  # mock CMA-ES results
    mock_cma_es.fit.fit = fit
    mock_cma_es.countevals = countevals
    mock_cma_es.fit.histbest = [histbest]

    termination_callback = self.optimizer.make_termination_callback(
        early_termination_abnormal_wrmsd=10.,
        early_termination_num_fevals=100,
        early_termination_wrmsd=1.)

    self.assertEqual(termination_callback(mock_cma_es), expected_terminate)

  def test_get_objective(self):
    objective = self.optimizer.get_objective(self.functional)

    self.assertAlmostEqual(
        # (1 + 1) ** 2 + (1 + 1) ** 2 = 8
        objective([1., 1.]), 8. * evaluators.HARTREE_TO_KCALPERMOLE)

  def test_optimize(self):
    results = self.optimizer.optimize(
        self.optimizer.get_objective(self.functional),
        initial_guess=[0., 0.])

    np.testing.assert_allclose(results['xbest'], [-1., -1.])

  def test_optimize_with_bounds(self):
    self.optimizer.hyperparameters['bounds'] = [-0.5, 0.5]
    self.optimizer.sigma0 = 0.1

    results = self.optimizer.optimize(
        self.optimizer.get_objective(self.functional),
        initial_guess=[0., 0.])

    np.testing.assert_allclose(results['xbest'], [-0.5, -0.5])

  def test_optimize_with_l1_penalty(self):
    # (x + 1) ** 2 + (y + 1) ** 2 + 0.5 |x| + 0.5 |y|
    self.optimizer.l1_penalty = 0.5 * evaluators.HARTREE_TO_KCALPERMOLE

    results = self.optimizer.optimize(
        self.optimizer.get_objective(self.functional),
        initial_guess=[0., 0.])

    np.testing.assert_allclose(results['xbest'], [-0.75, -0.75])

  def test_optimize_with_l2_penalty(self):
    # (x + 1) ** 2 + (y + 1) ** 2 + 0.5 * x ** 2 + 0.5 * y ** 2
    self.optimizer.l2_penalty = 0.5 * evaluators.HARTREE_TO_KCALPERMOLE

    results = self.optimizer.optimize(
        self.optimizer.get_objective(self.functional),
        initial_guess=[0., 0.])

    np.testing.assert_allclose(results['xbest'], [-2 / 3, -2 / 3])

  def test_run_optimization(self):
    with mock.patch.object(
        optimizers.CMAESOptimizer, 'optimize',
        side_effect=[
            {'xbest': [0., 1.], 'fbest': 3.},
            {'xbest': [0., 2.], 'fbest': 1.},
            {'xbest': [0., 3.], 'fbest': 2.},
            {'xbest': [0., 4.], 'fbest': np.nan}]):
      optimizer = optimizers.CMAESOptimizer(evaluator=self.evaluator)

      results = optimizer.run_optimization(
          functional=self.functional, num_trials=4)

      self.assertAlmostEqual(results['fbest'], 1.)
      np.testing.assert_allclose(results['xbest'], [0., 2.])
      np.testing.assert_allclose(results['wrmsd_trials'], [3., 1., 2., np.nan])


if __name__ == '__main__':
  absltest.main()
