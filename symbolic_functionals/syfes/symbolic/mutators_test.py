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

"""Tests for symbolic.mutators."""

import copy
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import instructions
from symbolic_functionals.syfes.symbolic import mutators
from symbolic_functionals.syfes.symbolic import xc_functionals

jax.config.update('jax_enable_x64', True)


class EnhancementFactorMutatorTest(parameterized.TestCase):

  # tests on initialization
  def test_initialization_with_unknown_instruction(self):
    with self.assertRaisesRegex(
        ValueError, 'Unknown instruction: UnknownInstruction'):
      mutators.EnhancementFactorMutator(
          instruction_pool={'UnknownInstruction': 1.})

  def test_initialization_with_unnormalized_probability_for_instructions(self):
    with self.assertRaisesRegex(
        ValueError, 'Instruction probabilities are not normalized to 1'):
      mutators.EnhancementFactorMutator(
          instruction_pool={'AdditionInstruction': 1., 'Power2Instruction': 1.})

  def test_initialization_with_negative_probability_for_instructions(self):
    with self.assertRaisesRegex(
        ValueError, 'Instruction pool contains negative probabilities'):
      mutators.EnhancementFactorMutator(
          instruction_pool={
              'AdditionInstruction': -0.2,
              'Power2Instruction': 0.6,
              'MultiplicationInstruction': 0.6})

  def test_initialization_with_unknown_mutation(self):
    with self.assertRaisesRegex(
        ValueError, 'Unknown mutation type: apply_unknown_mutation'):
      mutators.EnhancementFactorMutator(
          mutation_pool={'apply_unknown_mutation': 1.})

  def test_initialization_with_unnormalized_probability_for_mutations(self):
    with self.assertRaisesRegex(
        ValueError, 'Mutation probabilities are not normalized to 1'):
      mutators.EnhancementFactorMutator(
          mutation_pool={'insert_instruction': 1., 'remove_instruction': 1.})

  def test_initialization_with_negative_probability_for_mutations(self):
    with self.assertRaisesRegex(
        ValueError, 'Mutation pool contains negative probabilities'):
      mutators.EnhancementFactorMutator(
          mutation_pool={
              'insert_instruction': -0.2,
              'replace_instruction': 0.6,
              'change_argument': 0.6})

  # tests on helper functions
  def test_get_random_mutation_type(self):
    mutator = mutators.EnhancementFactorMutator(
        mutation_pool={
            'insert_instruction': 0.25,
            'remove_instruction': 0.25,
            'replace_instruction': 0.25,
            'change_argument': 0.25},
        seed=3)

    mutation_types = [mutator.get_random_mutation_type() for _ in range(10)]

    self.assertEqual(mutation_types, [
        'replace_instruction',
        'replace_instruction',
        'remove_instruction',
        'replace_instruction',
        'change_argument',
        'change_argument',
        'insert_instruction',
        'insert_instruction',
        'insert_instruction',
        'remove_instruction'])

  @parameterized.parameters(0, 1, 2)
  def test_get_random_instruction_name(self, max_num_bound_parameters):
    mutator = mutators.EnhancementFactorMutator(
        max_num_bound_parameters=max_num_bound_parameters)

    instruction_class = instructions.INSTRUCTION_CLASSES[
        mutator.get_random_instruction_name(existing_bound_parameters=[])]

    self.assertLessEqual(
        instruction_class.get_num_bound_parameters(), max_num_bound_parameters)

  @parameterized.parameters(
      (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
  )
  def test_get_random_instruction_name_num_inputs(self,
                                                  num_inputs,
                                                  max_num_bound_parameters):
    mutator = mutators.EnhancementFactorMutator(
        max_num_bound_parameters=max_num_bound_parameters)

    instruction_class = instructions.INSTRUCTION_CLASSES[
        mutator.get_random_instruction_name(
            existing_bound_parameters=[], num_inputs=num_inputs)]

    self.assertEqual(instruction_class.get_num_inputs(), num_inputs)
    self.assertLessEqual(
        instruction_class.get_num_bound_parameters(), max_num_bound_parameters)

  def test_get_random_instruction_name_existing_bound_parameters(self):
    mutator = mutators.EnhancementFactorMutator(
        instruction_pool={'UTransformInstruction': 0.5, 'B88XInstruction': 0.5},
        max_num_bound_parameters=1)

    instruction_name = mutator.get_random_instruction_name(
        existing_bound_parameters=['gamma_utransform'], num_inputs=1)

    self.assertEqual(instruction_name, 'UTransformInstruction')

  def test_get_random_instruction_name_wrong_instruction_pool(self):
    mutator = mutators.EnhancementFactorMutator(
        instruction_pool={'PBECInstruction': 1.0},
        max_num_bound_parameters=2)

    with self.assertRaisesRegex(
        ValueError, 'No instruction in instruction pool satisfies conditions: '
        r"num_inputs = 2, existing_bound_parameters = \['gamma_utransform'\]"):
      mutator.get_random_instruction_name(
          existing_bound_parameters=['gamma_utransform'], num_inputs=2)

    with self.assertRaisesRegex(
        ValueError, 'No instruction in instruction pool satisfies conditions: '
        r'num_inputs = 1, existing_bound_parameters = \[\]'):
      mutator.get_random_instruction_name(
          existing_bound_parameters=[], num_inputs=1)

  def test_get_random_instruction_name_wrong_bound_parameter(self):
    mutator = mutators.EnhancementFactorMutator(max_num_bound_parameters=0)

    with self.assertRaisesRegex(
        ValueError, 'No instruction in instruction pool satisfies conditions: '
        r"num_inputs = 2, existing_bound_parameters = \['gamma_utransform'\]"):
      mutator.get_random_instruction_name(
          existing_bound_parameters=['gamma_utransform'], num_inputs=2)

  @parameterized.parameters(
      *instructions.Instruction.__subclasses__()
  )
  def test_get_random_instruction(self, instruction_class):
    mutator = mutators.EnhancementFactorMutator(
        instruction_pool={instruction_class.__name__: 1.0})
    new_instruction = mutator.get_random_instruction(
        enhancement_factor=enhancement_factors.f_empty)
    self.assertIsInstance(new_instruction, instruction_class)

  # tests on mutations
  def test_mutate_yield_correct_length_of_instruction_list(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        side_effect=['insert_instruction', 'remove_instruction',
                     'replace_instruction', 'change_argument']):
      mutator = mutators.EnhancementFactorMutator()

      enhancement_factor, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factors.f_x_wb97mv_short,
          verbose=False)
      self.assertEqual(mutation_type, 'insert_instruction')
      self.assertEqual(enhancement_factor.num_instructions, 7)

      enhancement_factor, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factor, verbose=False)
      self.assertEqual(mutation_type, 'remove_instruction')
      self.assertEqual(enhancement_factor.num_instructions, 6)

      enhancement_factor, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factor, verbose=False)
      self.assertEqual(mutation_type, 'replace_instruction')
      self.assertEqual(enhancement_factor.num_instructions, 6)

      enhancement_factor, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factor, verbose=False)
      self.assertEqual(mutation_type, 'change_argument')
      self.assertEqual(enhancement_factor.num_instructions, 6)

  def test_mutate_with_empty_instruction_list(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        return_value='remove_instruction'):
      mutator = mutators.EnhancementFactorMutator()
      _, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factors.f_empty, verbose=False)
      self.assertEqual(mutation_type, 'insert_instruction')

  def test_mutate_with_empty_instruction_list_no_insertion(self):
    mutator = mutators.EnhancementFactorMutator(
        mutation_pool={'remove_instruction': 1.0})
    with self.assertRaisesRegex(
        ValueError,
        'Mutation cannot proceed on empty instruction list with '
        'zero insertion probability'):
      mutator.mutate(
          enhancement_factor=enhancement_factors.f_empty, verbose=False)

  def test_mutate_until_maximum_number_of_instructions(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        side_effect=['insert_instruction', 'replace_instruction']):
      mutator = mutators.EnhancementFactorMutator(max_num_instructions=6)
      new_enhancement_factor, mutation_type, _, _ = mutator.mutate(
          enhancement_factor=enhancement_factors.f_x_wb97mv_short,
          verbose=False)
      self.assertEqual(mutation_type, 'replace_instruction')
      self.assertEqual(new_enhancement_factor.num_instructions, 6)

  def test_mutate_until_maximum_number_of_instructions_only_insertion(self):
    mutator = mutators.EnhancementFactorMutator(
        mutation_pool={'insert_instruction': 1.0},
        max_num_instructions=6)
    with self.assertRaisesRegex(
        ValueError,
        'Mutation cannot proceed on max_num_instructions with '
        'only insertions allowed'):
      mutator.mutate(
          enhancement_factor=enhancement_factors.f_x_wb97mv_short,
          verbose=False)

  def test_mutate_beyond_maximum_number_of_instructions(self):
    mutator = mutators.EnhancementFactorMutator(max_num_instructions=5)
    with self.assertRaisesRegex(
        ValueError,
        'Mutation cannot proceed with instruction_list longer '
        'than max_num_instructions'):
      mutator.mutate(
          enhancement_factor=enhancement_factors.f_x_wb97mv_short,
          verbose=False)

  def test_mutate_beyond_maximum_number_of_bound_parameters(self):
    mutator = mutators.EnhancementFactorMutator(max_num_bound_parameters=0)
    with self.assertRaisesRegex(
        ValueError,
        'Mutation cannot proceed with number of bound parameters greater '
        'than max_num_bound_parameters'):
      mutator.mutate(
          enhancement_factor=enhancement_factors.f_x_wb97mv_short,
          verbose=False)

  def test_insert_instruction(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        return_value='insert_instruction'):
      mutator = mutators.EnhancementFactorMutator()
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_enhancement_factor, mutation_type, instruction_index, change = (
          mutator.mutate(enhancement_factor=enhancement_factor, verbose=False))

      self.assertEqual(enhancement_factor, enhancement_factors.f_x_wb97mv_short)
      self.assertEqual(mutation_type, 'insert_instruction')
      self.assertIsNone(change[0])
      self.assertEqual(
          change[1],
          new_enhancement_factor.instruction_list[instruction_index])
      new_enhancement_factor.instruction_list.pop(instruction_index)
      self.assertEqual(
          new_enhancement_factor, enhancement_factors.f_x_wb97mv_short)

  def test_remove_instruction(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        return_value='remove_instruction'):
      mutator = mutators.EnhancementFactorMutator()
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_enhancement_factor, mutation_type, instruction_index, change = (
          mutator.mutate(enhancement_factor=enhancement_factor, verbose=False))

      self.assertEqual(enhancement_factor, enhancement_factors.f_x_wb97mv_short)
      self.assertEqual(mutation_type, 'remove_instruction')
      self.assertEqual(
          change[0],
          enhancement_factors.f_x_wb97mv_short.instruction_list[
              instruction_index])
      self.assertIsNone(change[1])
      new_enhancement_factor.instruction_list.insert(
          instruction_index, change[0])
      self.assertEqual(
          new_enhancement_factor, enhancement_factors.f_x_wb97mv_short)

  def test_replace_instruction(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        return_value='replace_instruction'):
      mutator = mutators.EnhancementFactorMutator()
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_enhancement_factor, mutation_type, instruction_index, change = (
          mutator.mutate(enhancement_factor=enhancement_factor, verbose=False))

      self.assertEqual(enhancement_factor, enhancement_factors.f_x_wb97mv_short)
      self.assertEqual(mutation_type, 'replace_instruction')
      self.assertEqual(
          change[0],
          enhancement_factors.f_x_wb97mv_short.instruction_list[
              instruction_index])
      self.assertEqual(
          change[1],
          new_enhancement_factor.instruction_list[instruction_index])
      self.assertEqual(change[0].args, change[1].args)
      new_enhancement_factor.instruction_list[instruction_index] = change[0]
      self.assertEqual(
          new_enhancement_factor, enhancement_factors.f_x_wb97mv_short)

  def test_change_argument(self):
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_mutation_type',
        return_value='change_argument'):
      mutator = mutators.EnhancementFactorMutator()
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_enhancement_factor, mutation_type, instruction_index, change = (
          mutator.mutate(enhancement_factor=enhancement_factor, verbose=False))

      self.assertEqual(enhancement_factor, enhancement_factors.f_x_wb97mv_short)
      self.assertEqual(mutation_type, 'change_argument')
      self.assertEqual(
          change[0],
          enhancement_factors.f_x_wb97mv_short.instruction_list[
              instruction_index])
      self.assertEqual(
          change[1],
          new_enhancement_factor.instruction_list[instruction_index])
      self.assertEqual(type(change[0]), type(change[1]))
      new_enhancement_factor.instruction_list[instruction_index] = change[0]
      self.assertEqual(
          new_enhancement_factor, enhancement_factors.f_x_wb97mv_short)

  def test_randomize_instruction_list(self):
    expected_instruction_list = [
        instructions.UTransformInstruction('u', 'x2'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'c00', 'enhancement_factor'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c10', 'w'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c01', 'u'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
    ]
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_instruction',
        side_effect=expected_instruction_list):
      mutator = mutators.EnhancementFactorMutator(
          mutation_pool={'randomize_instruction_list': 1.})
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_enhancement_factor, mutation_type, _, _ = (
          mutator.mutate(enhancement_factor=enhancement_factor, verbose=False))

      self.assertEqual(enhancement_factor, enhancement_factors.f_x_wb97mv_short)
      self.assertEqual(mutation_type, 'randomize_instruction_list')
      self.assertEqual(
          new_enhancement_factor.instruction_list, expected_instruction_list)

  def test_randomize_instruction_list_fixed_num_instructions(self):
    expected_instruction_list = [
        instructions.AdditionInstruction('enhancement_factor', 'c10', 'u'),
        instructions.MultiplicationInstruction(
            'enhancement_factor', 'c10', 'u'),
        instructions.DivisionInstruction('enhancement_factor', 'c10', 'u')]
    with mock.patch.object(
        mutators.EnhancementFactorMutator, 'get_random_instruction',
        side_effect=expected_instruction_list):
      mutator = mutators.EnhancementFactorMutator(
          mutation_pool={'randomize_instruction_list': 1.})
      enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

      new_instruction_list, _, _, _ = mutator.randomize_instruction_list(
          enhancement_factor, num_instructions=2)

      self.assertEqual(new_instruction_list, expected_instruction_list[:2])

  @parameterized.parameters(0, 2, 4)
  def test_randomize_instruction_list_max_num_bound_parameters(
      self, max_num_bound_parameters):
    mutator = mutators.EnhancementFactorMutator(
        mutation_pool={'randomize_instruction_list': 1.},
        max_num_instructions=10,
        max_num_bound_parameters=max_num_bound_parameters)
    enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

    new_instruction_list, _, _, _ = mutator.randomize_instruction_list(
        enhancement_factor)

    self.assertLessEqual(
        len(set(itertools.chain(
            *[instruction.get_bound_parameters()
              for instruction in new_instruction_list]))),
        max_num_bound_parameters)

  def test_randomize_instruction_list_with_fixed_instructions(self):
    mutator = mutators.EnhancementFactorMutator(
        mutation_pool={'randomize_instruction_list': 1.},
        num_fixed_instructions=1)
    enhancement_factor = copy.deepcopy(enhancement_factors.f_x_wb97mv_short)

    with self.assertRaisesRegex(
        ValueError, 'randomize_instruction_list cannot be applied with '
        'fixed instructions'):
      mutator.randomize_instruction_list(enhancement_factor)


class XCFunctionalMutatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mutator_x = mutators.EnhancementFactorMutator()
    self.mutator_css = mutators.EnhancementFactorMutator()
    self.mutator_cos = mutators.EnhancementFactorMutator()

  def test_initialization_with_wrong_probability_length(self):
    with self.assertRaisesRegex(
        ValueError,
        'Wrong length for component_mutation_probabilities. Expected 3, got 2'):
      mutators.XCFunctionalMutator(
          self.mutator_x, self.mutator_css, self.mutator_cos,
          component_mutation_probabilities=[0.5, 0.5])

  def test_initialization_with_unnormalized_probability(self):
    with self.assertRaisesRegex(
        ValueError, 'component_mutation_probabilities not normalized to 1'):
      mutators.XCFunctionalMutator(
          self.mutator_x, self.mutator_css, self.mutator_cos,
          component_mutation_probabilities=[0.4, 0.4, 0.4])

  def test_initialization_with_negative_probability(self):
    with self.assertRaisesRegex(
        ValueError,
        'component_mutation_probabilities contains negative probabilities'):
      mutators.XCFunctionalMutator(
          self.mutator_x, self.mutator_css, self.mutator_cos,
          component_mutation_probabilities=[-0.2, 0.6, 0.6])

  def test_get_random_component(self):
    mutator = mutators.XCFunctionalMutator(
        self.mutator_x, self.mutator_css, self.mutator_cos, seed=1)
    self.assertEqual(
        [mutator.get_random_component() for _ in range(10)],
        ['f_css', 'f_cos', 'f_x', 'f_x', 'f_x',
         'f_x', 'f_x', 'f_css', 'f_css', 'f_css'])

  @parameterized.parameters('f_x', 'f_css', 'f_cos')
  def test_mutate(self, component):
    with mock.patch.object(
        mutators.XCFunctionalMutator,
        'get_random_component',
        return_value=component), mock.patch.object(
            mutators.EnhancementFactorMutator,
            'get_random_mutation_type',
            return_value='remove_instruction'):
      mutator = mutators.XCFunctionalMutator(
          mutator_x=mutators.EnhancementFactorMutator(),
          mutator_css=mutators.EnhancementFactorMutator(),
          mutator_cos=mutators.EnhancementFactorMutator())
      functional = copy.deepcopy(xc_functionals.b97_u)

      new_functional, mutated_component, _, instruction_index, change = (
          mutator.mutate(functional, verbose=False))

      self.assertEqual(functional, xc_functionals.b97_u)
      self.assertEqual(mutated_component, component)
      new_enhancement_factor = getattr(new_functional, component)
      self.assertEqual(new_enhancement_factor.num_instructions, 4)
      new_enhancement_factor.instruction_list.insert(
          instruction_index, change[0])
      self.assertEqual(new_functional, functional)


if __name__ == '__main__':
  absltest.main()
