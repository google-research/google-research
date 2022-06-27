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

"""Tests for symbolic.instructions."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import sympy

from symbolic_functionals.syfes.symbolic import instructions
from symbolic_functionals.syfes.xc import gga

jax.config.update('jax_enable_x64', True)


class InstructionTest(parameterized.TestCase):

  # general tests
  def test_instructions_do_not_have_repeated_bound_parameters(self):
    bound_parameters = list(itertools.chain(
        *[instruction_class.get_bound_parameters()
          for instruction_class in instructions.INSTRUCTION_CLASSES.values()]))
    self.assertEqual(len(bound_parameters), len(set(bound_parameters)))

  @parameterized.parameters(
      *instructions.Instruction.__subclasses__())
  def test_class_attributes(self, instruction_class):
    self.assertIn(instruction_class.get_num_inputs(), [1, 2])
    self.assertGreaterEqual(instruction_class.get_num_bound_parameters(), 0)

  def test_constructor_with_wrong_number_of_arguments(self):
    with self.assertRaisesRegex(
        ValueError,
        'Power4Instruction: wrong number of arguments. Expected 2, got 3'):
      instructions.Power4Instruction('a', 'b', 'c')

  @parameterized.parameters(
      (instructions.Power2Instruction('a', 'b'),
       instructions.Power2Instruction('a', 'b')),
      (instructions.AdditionInstruction('a', 'b', 'c'),
       instructions.AdditionInstruction('a', 'b', 'c')),
      (instructions.UTransformInstruction('a', 'b'),
       instructions.UTransformInstruction('a', 'b')))
  def test_eq(self, instruction1, instruction2):
    self.assertEqual(instruction1, instruction2)

  @parameterized.parameters(
      (instructions.AdditionInstruction('a', 'b', 'c'),
       instructions.MultiplicationInstruction('a', 'b', 'c')),
      (instructions.AdditionInstruction('a', 'b', 'c'),
       instructions.AdditionInstruction('a', 'b', 'd')),
      (instructions.AdditionInstruction('a', 'b', 'c'),
       instructions.Power2Instruction('a', 'b'))
      )
  def test_eq_false(self, instruction1, instruction2):
    self.assertNotEqual(instruction1, instruction2)

  # test instructions without bound parameters
  @parameterized.parameters(
      (instructions.AdditionBy1Instruction, lambda a: a + 1.),
      (instructions.Power2Instruction, lambda a: a ** 2),
      (instructions.Power3Instruction, lambda a: a ** 3),
      (instructions.Power4Instruction, lambda a: a ** 4),
      (instructions.Power6Instruction, lambda a: a ** 6),
      (instructions.SquareRootInstruction, np.sqrt),
      (instructions.CubeRootInstruction, np.cbrt),
      (instructions.Log1PInstruction, np.log1p),
      (instructions.ExpInstruction, np.exp),
  )
  def test_unary_instruction(self, instruction_class, function):
    workspace = {'a': np.random.rand(),}
    instruction_class('b', 'a').apply(workspace, use_jax=False)
    self.assertAlmostEqual(workspace['b'], function(workspace['a']))
    instruction_class('b', 'a').apply(workspace, use_jax=True)
    self.assertAlmostEqual(workspace['b'], function(workspace['a']))

  @parameterized.parameters(
      (instructions.AdditionInstruction, lambda a, b: a + b),
      (instructions.SubtractionInstruction, lambda a, b: a - b),
      (instructions.MultiplicationInstruction, lambda a, b: a * b),
      (instructions.DivisionInstruction, lambda a, b: a / b),
  )
  def test_binary_instruction(self, instruction_class, function):
    workspace = {'a': np.random.rand(), 'b': np.random.rand()}
    instruction_class('c', 'a', 'b').apply(workspace, use_jax=False)
    self.assertAlmostEqual(
        workspace['c'], function(workspace['a'], workspace['b']))
    instruction_class('c', 'a', 'b').apply(workspace, use_jax=False)
    self.assertAlmostEqual(
        workspace['c'], function(workspace['a'], workspace['b']))

  @parameterized.parameters(
      instructions.AdditionBy1Instruction,
      instructions.Power2Instruction,
      instructions.Power3Instruction,
      instructions.Power4Instruction,
      instructions.Power6Instruction,
      instructions.SquareRootInstruction,
      instructions.CubeRootInstruction,
      instructions.Log1PInstruction,
      instructions.ExpInstruction)
  def test_unary_instruction_apply(self, instruction_class):
    input1 = np.random.rand()
    instruction = instruction_class('output', 'input1')
    workspace = {'input1': input1}
    symbolic_workspace = {'input1': sympy.Symbol('input1')}

    instruction.apply(workspace)
    instruction.sympy_apply(symbolic_workspace)

    self.assertAlmostEqual(
        workspace['output'],
        symbolic_workspace['output'].subs({'input1': input1}))

  @parameterized.parameters(
      instructions.AdditionInstruction,
      instructions.SubtractionInstruction,
      instructions.MultiplicationInstruction,
      instructions.DivisionInstruction)
  def test_binary_instruction_apply(self, instruction_class):
    input1 = np.random.rand()
    input2 = np.random.rand()
    instruction = instruction_class('output', 'input1', 'input2')
    workspace = {'input1': input1, 'input2': input2}
    symbolic_workspace = {
        'input1': sympy.Symbol('input1'), 'input2': sympy.Symbol('input2')}

    instruction.apply(workspace)
    instruction.sympy_apply(symbolic_workspace)

    self.assertAlmostEqual(
        workspace['output'],
        symbolic_workspace['output'].subs(
            {'input1': input1, 'input2': input2}))

  # test instructions with bound parameters
  @parameterized.parameters(
      [0., 0., 0.],
      [0., 1., 0.],
      [1., 0., 0.],
      [1., 1., .5],
      )
  def test_utransform_instruction(self, x, gamma_utransform, expected_y):
    workspace = {'x': x, 'gamma_utransform': gamma_utransform}
    instructions.UTransformInstruction('output', 'x').apply(workspace)
    self.assertAlmostEqual(workspace['output'], expected_y)

  @parameterized.parameters(False, True)
  def test_pbex_instruction(self, use_jax):
    workspace = {
        'x': np.random.rand(5),
        'kappa_pbex': np.random.rand(),
        'mu_pbex': np.random.rand()}

    instructions.PBEXInstruction('output', 'x').apply(
        workspace, use_jax=use_jax)

    np.testing.assert_allclose(
        workspace.pop('output'), gga.f_x_pbe(
            **{key.partition('_')[0]: value for key, value in workspace.items()
               }))

  @parameterized.parameters(False, True)
  def test_rpbex_instruction(self, use_jax):
    workspace = {
        'x': np.random.rand(5),
        'kappa_rpbex': np.random.rand(),
        'mu_rpbex': np.random.rand()}

    instructions.RPBEXInstruction('output', 'x').apply(
        workspace, use_jax=use_jax)

    np.testing.assert_allclose(
        workspace.pop('output'), gga.f_x_rpbe(
            **{key.partition('_')[0]: value for key, value in workspace.items()
               }))

  @parameterized.parameters(False, True)
  def test_b88_instruction(self, use_jax):
    workspace = {
        'x': np.random.rand(5),
        'beta_b88x': np.random.rand()}

    instructions.B88XInstruction('output', 'x').apply(
        workspace, use_jax=use_jax)

    np.testing.assert_allclose(
        workspace.pop('output'), gga.f_x_b88(
            **{key.partition('_')[0]: value for key, value in workspace.items()
               }))

  @parameterized.parameters(False, True)
  def test_pbec_instruction(self, use_jax):
    workspace = {
        'rho': np.random.rand(5),
        'sigma': np.random.rand(5),
        'beta_pbec': np.random.rand(),
        'gamma_pbec': np.random.rand()}

    instructions.PBECInstruction('output', 'rho', 'sigma').apply(
        workspace, use_jax=use_jax)

    np.testing.assert_allclose(
        workspace.pop('output'), gga.e_c_pbe_unpolarized(
            **{key.partition('_')[0]: value for key, value in workspace.items()
               }))

  # test conversion
  @parameterized.parameters(
      *instructions.Instruction.__subclasses__())
  def test_convert_instruction_to_and_from_list(self, instruction_class):
    instruction = instruction_class(
        *[f'arg{i}' for i in range(instruction_class.get_num_args())])

    instruction_from_list = instructions.Instruction.from_list(
        instruction.to_list())

    self.assertEqual(instruction, instruction_from_list)

  def test_from_list_with_wrong_instruction_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Invalid instruction class name: UnknownInstruction'):
      instructions.Instruction.from_list(
          ['UnknownInstruction', 'a', 'b', 'c'])

  # test helper functions
  def test_is_unary_instruction_name(self):
    self.assertTrue(
        instructions.is_unary_instruction_name('Power2Instruction'))
    self.assertTrue(
        instructions.is_unary_instruction_name('UTransformInstruction'))
    self.assertFalse(
        instructions.is_unary_instruction_name('AdditionInstruction'))

  def test_is_binary_instruction_name(self):
    self.assertTrue(
        instructions.is_binary_instruction_name('MultiplicationInstruction'))
    self.assertTrue(
        instructions.is_binary_instruction_name('PBECInstruction'))
    self.assertFalse(
        instructions.is_binary_instruction_name('Additionby1Instruction'))

  def test_get_unary_instruction_names_from_list(self):
    self.assertEqual(
        instructions.get_unary_instruction_names_from_list(
            ['AdditionInstruction', 'Power2Instruction', 'UnknownInstruction']),
        ['Power2Instruction'])

  def test_get_binary_instruction_names_from_list(self):
    self.assertEqual(
        instructions.get_binary_instruction_names_from_list(
            ['AdditionInstruction', 'Power2Instruction', 'UnknownInstruction']),
        ['AdditionInstruction'])

  @parameterized.parameters(0, 1, 2, 3)
  def test_get_instruction_names_with_signature_num_inputs(self, num_inputs):
    instruction_names = instructions.get_instruction_names_with_signature(
        num_inputs=num_inputs)
    for instruction_name in instruction_names:
      self.assertEqual(
          instructions.INSTRUCTION_CLASSES[instruction_name].get_num_inputs(),
          num_inputs)

  @parameterized.parameters(0, 1, 2, 3)
  def test_get_instruction_names_with_num_bound_parameters(
      self, num_bound_parameters):
    instruction_names = instructions.get_instruction_names_with_signature(
        num_bound_parameters=num_bound_parameters)
    for instruction_name in instruction_names:
      self.assertEqual(
          instructions.INSTRUCTION_CLASSES[
              instruction_name].get_num_bound_parameters(),
          num_bound_parameters)

  @parameterized.parameters(0, 1, 2, 3)
  def test_get_instruction_names_with_max_num_bound_parameters(
      self, max_num_bound_parameters):
    instruction_names = instructions.get_instruction_names_with_signature(
        max_num_bound_parameters=max_num_bound_parameters)
    for instruction_name in instruction_names:
      self.assertLessEqual(
          instructions.INSTRUCTION_CLASSES[
              instruction_name].get_num_bound_parameters(),
          max_num_bound_parameters)

if __name__ == '__main__':
  absltest.main()
