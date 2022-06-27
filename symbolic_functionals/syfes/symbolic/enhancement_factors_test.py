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

"""Tests for symbolic.enhancement_factors."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import sympy

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import instructions
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import mgga

jax.config.update('jax_enable_x64', True)


class EnhancementFactorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_features = 2
    self.num_shared_parameters = 2
    self.num_variables = 3

    self.features = {
        f'feature_{i}': np.random.rand(5) for i in range(self.num_features)
    }
    self.shared_parameters = {
        f'shared_parameter_{i}': np.random.rand()
        for i in range(self.num_shared_parameters)
    }
    self.bound_parameters = {'gamma_utransform': np.random.rand()}
    self.parameters = {**self.shared_parameters, **self.bound_parameters}
    self.variables = {
        f'variable_{i}': np.zeros(5) for i in range(self.num_variables - 1)
    }
    self.variables.update({'enhancement_factor': np.zeros(5)})

    self.enhancement_factor = enhancement_factors.EnhancementFactor(
        feature_names=list(self.features.keys()),
        shared_parameter_names=list(self.shared_parameters.keys()),
        variable_names=list(self.variables.keys()),
        instruction_list=[
            instructions.MultiplicationInstruction(
                'variable_0', 'feature_0', 'shared_parameter_0'),
            instructions.AdditionInstruction(
                'variable_1', 'feature_1', 'shared_parameter_1'),
            instructions.AdditionInstruction(
                'variable_1', 'variable_1', 'variable_0'),
            instructions.Power2Instruction('enhancement_factor', 'variable_1'),
            instructions.UTransformInstruction(
                'enhancement_factor', 'enhancement_factor')
        ])

  def test_constructor(self):
    self.assertEqual(self.enhancement_factor.num_features, self.num_features)
    self.assertEqual(self.enhancement_factor.num_parameters,
                     self.num_shared_parameters + 1)  # 1 from UTransform
    self.assertEqual(self.enhancement_factor.num_variables, self.num_variables)

  def test_constructor_without_enhancement_factor_in_variable_names(self):
    with self.assertRaisesRegex(
        ValueError, '"enhancement_factor" not found in variable_names.'):
      enhancement_factors.EnhancementFactor(
          feature_names=[],
          shared_parameter_names=[],
          variable_names=[],
          instruction_list=[])

  def test_constructor_with_repeated_name(self):
    with self.assertRaisesRegex(ValueError, 'Repeated names found in input.'):
      enhancement_factors.EnhancementFactor(
          feature_names=['var'],
          shared_parameter_names=['var'],
          variable_names=['enhancement_factor'],
          instruction_list=[])

  def test_constructor_with_wrong_instruction_type(self):
    with self.assertRaisesRegex(
        TypeError, r"1 is of type <class 'int'>, not an "
                   'instance of instructions.Instruction'):
      enhancement_factors.EnhancementFactor(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[1])

  @parameterized.parameters(
      (instructions.Power2Instruction('variable_0', 'var'),
       (r'Instruction variable_0 = var \*\* 2 contains invalid input argument '
        'var')),
      (instructions.AdditionInstruction('variable_0', 'shared_parameter_1',
                                        'gamma_utransform'),
       (r'Instruction variable_0 = shared_parameter_1 \+ gamma_utransform '
        'contains invalid input argument gamma_utransform')),
  )
  def test_constructor_with_invalid_input(self, instruction, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      enhancement_factors.EnhancementFactor(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[instruction])

  @parameterized.parameters(
      (instructions.Power2Instruction('feature_0', 'shared_parameter_0'),
       (r'Instruction feature_0 = shared_parameter_0 \*\* 2 contains '
        'invalid output argument feature_0')),
      (instructions.AdditionInstruction(
          'feature_1', 'shared_parameter_1', 'variable_1'),
       (r'Instruction feature_1 = shared_parameter_1 \+ variable_1 contains '
        'invalid output argument feature_1')
       ),
      (instructions.Power4Instruction(
          'bound_parameter_1', 'shared_parameter_1'),
       (r'Instruction bound_parameter_1 = shared_parameter_1 \*\* 4 contains '
        'invalid output argument bound_parameter_1')
       ),
  )
  def test_constructor_with_invalid_output(self, instruction, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      enhancement_factors.EnhancementFactor(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[instruction])

  @parameterized.parameters(False, True)
  def test_eval(self, use_jax):
    tmp = (
        (self.features['feature_0'] * self.parameters['shared_parameter_0']) +
        (self.features['feature_1'] + self.parameters['shared_parameter_1']))
    tmp = self.parameters['gamma_utransform'] * tmp ** 2
    expected_f = tmp / (1. + tmp)

    f = self.enhancement_factor.eval(
        self.features, self.parameters, use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)

  @parameterized.parameters(False, True)
  def test_b97_u_enhancement_factor(self, use_jax):
    gamma_x = 0.004
    coeffs_x = 0.8094, 0.5073, 0.7481
    x = np.random.rand(5)
    u = gga.u_b97(x, gamma=gamma_x)
    expected_f = gga.f_b97(x)

    f = enhancement_factors.f_b97_u.eval(
        features={'u': u},
        parameters={
            'c0': coeffs_x[0],
            'c1': coeffs_x[1],
            'c2': coeffs_x[2],
        },
        use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)

  @parameterized.parameters(False, True)
  def test_b97_u_short_enhancement_factor(self, use_jax):
    gamma_x = 0.004
    coeffs_x = 0.8094, 0.5073, 0.7481
    x = np.random.rand(5)
    u = gga.u_b97(x, gamma=gamma_x)
    expected_f = gga.f_b97(x)

    f = enhancement_factors.f_b97_u_short.eval(
        features={'u': u},
        parameters={
            'c0': coeffs_x[0],
            'c1': coeffs_x[1],
            'c2': coeffs_x[2],
        },
        use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)

  @parameterized.parameters(False, True)
  def test_b97_x2_enhancement_factor(self, use_jax):
    gamma_x = 0.004
    coeffs_x = 0.8094, 0.5073, 0.7481
    x = np.random.rand(5)
    x2 = (1 / 2)**(-2 / 3) * x**2
    expected_f = gga.f_b97(x)

    f = enhancement_factors.f_b97_x2.eval(
        features={'x2': x2},
        parameters={
            'c0': coeffs_x[0],
            'c1': coeffs_x[1],
            'c2': coeffs_x[2],
            'gamma': gamma_x
        },
        use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)

  @parameterized.parameters(False, True)
  def test_b97_x2_short_enhancement_factor(self, use_jax):
    gamma_x = 0.004
    coeffs_x = 0.8094, 0.5073, 0.7481
    x = np.random.rand(5)
    x2 = (1 / 2)**(-2 / 3) * x**2
    expected_f = gga.f_b97(x)

    f = enhancement_factors.f_b97_x2_short.eval(
        features={'x2': x2},
        parameters={
            'c0': coeffs_x[0],
            'c1': coeffs_x[1],
            'c2': coeffs_x[2],
            'gamma_utransform': gamma_x
        },
        use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)

  @parameterized.parameters(
      (enhancement_factors.f_x_wb97mv,
       enhancement_factors.f_css_wb97mv,
       enhancement_factors.f_cos_wb97mv,
       'gamma'),
      (enhancement_factors.f_x_wb97mv_short,
       enhancement_factors.f_css_wb97mv_short,
       enhancement_factors.f_cos_wb97mv_short,
       'gamma_utransform'),
      )
  def test_wb97mv_enhancement_factors(self,
                                      f_x_wb97mv,
                                      f_css_wb97mv,
                                      f_cos_wb97mv,
                                      gamma_key):
    rho = np.random.rand(5)
    x = np.random.rand(5)
    tau = np.random.rand(5)
    x2 = (1 / 2)**(-2 / 3) * x**2
    t = mgga.get_mgga_t(rho, tau, polarized=False)
    w = (t - 1) / (t + 1)
    expected_f_x = mgga.f_b97m(
        x, t, gamma=mgga.WB97MV_PARAMS['gamma_x'],
        power_series=mgga.WB97MV_PARAMS['power_series_x'], polarized=False)
    expected_f_css = mgga.f_b97m(
        x, t, gamma=mgga.WB97MV_PARAMS['gamma_ss'],
        power_series=mgga.WB97MV_PARAMS['power_series_ss'], polarized=False)
    expected_f_cos = mgga.f_b97m(
        x, t, gamma=mgga.WB97MV_PARAMS['gamma_os'],
        power_series=mgga.WB97MV_PARAMS['power_series_os'], polarized=False)

    f_x = f_x_wb97mv.eval(
        features={'x2': x2, 'w': w},
        parameters={
            'c00': mgga.WB97MV_PARAMS['power_series_x'][0][2],
            'c10': mgga.WB97MV_PARAMS['power_series_x'][1][2],
            'c01': mgga.WB97MV_PARAMS['power_series_x'][2][2],
            gamma_key: mgga.WB97MV_PARAMS['gamma_x']})
    f_css = f_css_wb97mv.eval(
        features={'x2': x2, 'w': w},
        parameters={
            'c00': mgga.WB97MV_PARAMS['power_series_ss'][0][2],
            'c10': mgga.WB97MV_PARAMS['power_series_ss'][1][2],
            'c20': mgga.WB97MV_PARAMS['power_series_ss'][2][2],
            'c43': mgga.WB97MV_PARAMS['power_series_ss'][3][2],
            'c04': mgga.WB97MV_PARAMS['power_series_ss'][4][2],
            gamma_key: mgga.WB97MV_PARAMS['gamma_ss']})
    f_cos = f_cos_wb97mv.eval(
        features={'x2': x2, 'w': w},
        parameters={
            'c00': mgga.WB97MV_PARAMS['power_series_os'][0][2],
            'c10': mgga.WB97MV_PARAMS['power_series_os'][1][2],
            'c20': mgga.WB97MV_PARAMS['power_series_os'][2][2],
            'c60': mgga.WB97MV_PARAMS['power_series_os'][3][2],
            'c21': mgga.WB97MV_PARAMS['power_series_os'][4][2],
            'c61': mgga.WB97MV_PARAMS['power_series_os'][5][2],
            gamma_key: mgga.WB97MV_PARAMS['gamma_os']})

    np.testing.assert_allclose(f_x, expected_f_x)
    np.testing.assert_allclose(f_css, expected_f_css)
    np.testing.assert_allclose(f_cos, expected_f_cos)

  def test_convert_enhancement_factor_to_and_from_dict(self):
    self.assertEqual(
        self.enhancement_factor,
        enhancement_factors.EnhancementFactor.from_dict(
            self.enhancement_factor.to_dict()))

  @parameterized.parameters(
      enhancement_factors.f_empty,
      enhancement_factors.f_lda,
      enhancement_factors.f_b97_u,
      enhancement_factors.f_b97_u_short,
      enhancement_factors.f_b97_x2,
      enhancement_factors.f_b97_x2_short,
      enhancement_factors.f_x_wb97mv,
      enhancement_factors.f_css_wb97mv,
      enhancement_factors.f_cos_wb97mv,
      enhancement_factors.f_x_wb97mv_short,
      enhancement_factors.f_css_wb97mv_short,
      enhancement_factors.f_cos_wb97mv_short,
  )
  def test_make_isomorphic_copy(self, enhancement_factor):
    features = {
        feature_name: np.random.rand(5)
        for feature_name in enhancement_factor.feature_names
    }
    shared_parameters = {
        parameter_name: np.random.rand()
        for parameter_name in enhancement_factor.shared_parameter_names
    }
    renamed_shared_parameters = {
        (enhancement_factor._isomorphic_copy_shared_parameter_prefix
         + str(index)): value
        for index, value in enumerate(shared_parameters.values())
    }
    bound_parameters = {
        parameter_name: np.random.rand()
        for parameter_name in enhancement_factor.bound_parameter_names
    }

    enhancement_factor_copy = enhancement_factor.make_isomorphic_copy()

    np.testing.assert_allclose(
        enhancement_factor.eval(
            features=features, parameters={
                **shared_parameters, **bound_parameters}),
        enhancement_factor_copy.eval(
            features=features, parameters={
                **renamed_shared_parameters, **bound_parameters})
        )

  def test_make_isomorphic_copy_of_f_x_wb97mv_short(self):
    f_x_wb97mv_copy = enhancement_factors.f_x_wb97mv_short.make_isomorphic_copy(
        feature_names=['rho', 'x2', 'w'],
        num_shared_parameters=10,
        num_variables=10)

    self.assertEqual(f_x_wb97mv_copy.feature_names, ['rho', 'x2', 'w'])
    self.assertEqual(f_x_wb97mv_copy.num_shared_parameters, 10)
    self.assertEqual(
        f_x_wb97mv_copy.shared_parameter_names,
        [f_x_wb97mv_copy._isomorphic_copy_shared_parameter_prefix + str(index)
         for index in range(10)])
    self.assertEqual(
        f_x_wb97mv_copy.variable_names,
        [f_x_wb97mv_copy._isomorphic_copy_variable_prefix + str(index)
         for index in range(9)] + ['enhancement_factor'])

  def test_make_isomorphic_copy_enhancement_factor_variable_location(self):
    f_x_wb97mv_shuffled = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)
    f_x_wb97mv_shuffled.variable_names.remove('enhancement_factor')
    f_x_wb97mv_shuffled.variable_names.insert(
        np.random.randint(len(f_x_wb97mv_shuffled.variable_names)),
        'enhancement_factor')
    self.assertEqual(
        enhancement_factors.f_x_wb97mv_short.make_isomorphic_copy(),
        f_x_wb97mv_shuffled.make_isomorphic_copy())

  def test_make_isomorphic_copy_repeated_feature_names(self):
    with self.assertRaisesRegex(
        ValueError, 'Repeated feature names'):
      enhancement_factors.f_b97_u.make_isomorphic_copy(
          feature_names=['u', 'u'])

  def test_make_isomorphic_copy_wrong_feature_names(self):
    with self.assertRaisesRegex(
        ValueError,
        r"feature_names \['rho', 'x2'\] is not a superset of feature_names of "
        r"current instance \['w', 'x2'\]"):
      enhancement_factors.f_x_wb97mv.make_isomorphic_copy(
          feature_names=['rho', 'x2'])

  def test_make_isomorphic_copy_wrong_num_shared_parameters(self):
    with self.assertRaisesRegex(
        ValueError, 'num_shared_parameters 5 is smaller than '
        'that of current instance 6'):
      enhancement_factors.f_cos_wb97mv_short.make_isomorphic_copy(
          num_shared_parameters=5)

  def test_make_isomorphic_copy_wrong_num_variables(self):
    with self.assertRaisesRegex(
        ValueError, 'num_variables 3 is smaller than '
        'that of current instance 5'):
      enhancement_factors.f_cos_wb97mv_short.make_isomorphic_copy(
          num_variables=3)

  @parameterized.parameters(
      (enhancement_factors.f_b97_u, 3),
      (enhancement_factors.f_b97_u_short, 3),
      (enhancement_factors.f_b97_x2, 4),
      (enhancement_factors.f_b97_x2_short, 4),
      (enhancement_factors.f_x_wb97mv_short, 4),)
  def test_num_used_parameters(
      self, enhancement_factor, expected_num_used_parameters):
    self.assertEqual(enhancement_factor.num_used_parameters,
                     expected_num_used_parameters)
    self.assertEqual(
        enhancement_factor.make_isomorphic_copy(
            num_shared_parameters=20).num_used_parameters,
        expected_num_used_parameters)

  def test_get_symbolic_expression(self):
    c0, c1, c2, gamma, x = sympy.symbols(
        'c0 c1 c2 gamma_utransform x')
    self.assertEqual(
        enhancement_factors.f_b97_x2_short.get_symbolic_expression(
            latex=False, simplify=False),
        (c0 + c1 * gamma * x ** 2 / (gamma * x ** 2 + 1.)
         + c2 * gamma ** 2 * x ** 4 / (gamma * x ** 2 + 1.) ** 2))

  def test_get_symbolic_expression_latex(self):
    self.assertEqual(
        enhancement_factors.f_b97_x2_short.get_symbolic_expression(
            latex=True, simplify=False),
        r'c_{0} + \frac{c_{1} \gamma_{u} x^{2}}{\gamma_{u} x^{2} + 1.0} + '
        r'\frac{c_{2} \gamma_{u}^{2} x^{4}}{\left(\gamma_{u} x^{2} + '
        r'1.0\right)^{2}}')


if __name__ == '__main__':
  absltest.main()
