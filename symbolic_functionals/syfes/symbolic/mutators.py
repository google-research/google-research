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

"""Mutators for enhancement factors and exchange-correlation functionals.

The EnhancementFactorMutator/XCFunctionalMutator class defined in this module
provides a mutate method, which takes an EnhancementFactor/XCFunctional
instance as input and returns a mutated EnhancementFactor/XCFunctional instance
as output. During the mutation process, EnhancementFactor/XCFunctional instance
are treated as immutable objects and no changes are made in-place.
"""

import copy

from absl import logging
import numpy as np

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import instructions
from symbolic_functionals.syfes.symbolic import xc_functionals


def _remove_small_probabilities(pool):
  """Remove small probabilities from instruction or mutation pool dictionary."""
  keys_to_remove = [key for key, value in pool.items() if value < 1e-8]
  for key in keys_to_remove:
    pool.pop(key)


class EnhancementFactorMutator:
  """Mutator for enhancement factor.

  Mutations on an EnhancementFactor instance are performed by inserting,
  removing, replacing and changing arguments of instructions in the
  instruction_list. Probabilities for instructions and mutation rules
  are specified by instruction_pool and mutation_pool, respectively.
  New instructions generated during the mutations obey the convention in
  the definition of enhancement factor that input arguments can be any names
  known to the enhancement factor, while output argument are limited to
  variable names.
  """

  _default_instruction_pool = {
      instruction_name: 1. / len(instructions.INSTRUCTION_CLASSES)
      for instruction_name in instructions.INSTRUCTION_CLASSES.keys()
  }

  _default_mutation_pool = {
      'insert_instruction': 0.25,
      'remove_instruction': 0.25,
      'replace_instruction': 0.25,
      'change_argument': 0.25,
  }

  def __init__(self,
               instruction_pool=None,
               mutation_pool=None,
               max_num_instructions=None,
               max_num_bound_parameters=None,
               num_fixed_instructions=0,
               seed=None):
    """Initializes mutator.

    Args:
      instruction_pool: Dict {instruction_name: instruction_probability},
        the pool of possible instructions for insertion and replacement.
      mutation_pool: Dict {mutation_name: mutation_probability}, the pool
        of possible mutation rules. Mutation rules are implemented as methods
        of this class.
      max_num_instructions: Integer, the maximum number of instructions.
        No more instructions will be inserted to the instruction_list beyond
        this number.
      max_num_bound_parameters: Integer, the maximum number of bound parameters.
      num_fixed_instructions: Integer, the number of fixed instructions.
        Mutator will not mutate the first num_fixed_instructions instructions
        of enhancement factors.
      seed: Integer, the random seed.
    """
    self.instruction_pool = instruction_pool or self._default_instruction_pool
    for instruction_name in self.instruction_pool:
      if instruction_name not in instructions.INSTRUCTION_CLASSES:
        raise ValueError(f'Unknown instruction: {instruction_name}')
    if abs(sum(self.instruction_pool.values()) - 1) > 1e-8:
      raise ValueError('Instruction probabilities are not normalized to 1')
    if any(probability < 0. for probability in self.instruction_pool.values()):
      raise ValueError('Instruction pool contains negative probabilities')
    _remove_small_probabilities(self.instruction_pool)

    self.mutation_pool = mutation_pool or self._default_mutation_pool
    for mutation_type in self.mutation_pool:
      if not hasattr(self, mutation_type):
        raise ValueError(f'Unknown mutation type: {mutation_type}')
    if abs(sum(self.mutation_pool.values()) - 1) > 1e-8:
      raise ValueError('Mutation probabilities are not normalized to 1')
    if any(probability < 0. for probability in self.mutation_pool.values()):
      raise ValueError('Mutation pool contains negative probabilities')
    _remove_small_probabilities(self.mutation_pool)

    if max_num_instructions is None:
      self.max_num_instructions = float('inf')
    else:
      self.max_num_instructions = max_num_instructions

    if max_num_bound_parameters is None:
      self.max_num_bound_parameters = float('inf')
    else:
      self.max_num_bound_parameters = max_num_bound_parameters

    self.num_fixed_instructions = num_fixed_instructions

    self.random_state = np.random.RandomState(seed=seed)

  @property
  def instruction_names(self):
    """List of instruction names."""
    return list(self.instruction_pool.keys())

  @property
  def instruction_probabilities(self):
    """List of probabilities for instructions."""
    return list(self.instruction_pool.values())

  @property
  def mutation_types(self):
    """List of mutation types."""
    return list(self.mutation_pool.keys())

  @property
  def mutation_probabilities(self):
    """List of probabilities for mutation types."""
    return list(self.mutation_pool.values())

  def mutate(self, enhancement_factor, verbose=True):
    """Mutates a single instruction in the enhancement factor.

    Once the instruction_list become empty, only insert_instruction will be
    performed; once the instruction_list hits self.max_num_instructions limit,
    only mutations other than insert_instruction will be performed.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the enhancement factor to be mutated. enhancement_factor will
        not be modified in-place.
      verbose: Boolean, if True, prints the log of mutation.

    Returns:
      new_enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the mutated enhancement factor.
      mutation_type: String, the type of mutation performed.
      instruction_index: Integer, the index of mutated instruction.
      change: Tuple of two instances of instructions.Instruction.
        * (None, new_instruction) for insert_instruction
        * (old_instruction, None) for remove_instruction
        * (old_instruction, new_instruction) for replace_instruction or
          change_argument

    Raises:
      ValueError, if instruction_list is empty and probability of
        insert_instruction is zero, or the length of instruction_list hits
        self.max_num_instructions limit and only insert_instruction mutation
        is allowed.
    """
    # determine mutation type
    if enhancement_factor.num_instructions == self.num_fixed_instructions:
      if self.mutation_pool.get('insert_instruction', 0.) < 1e-8:
        raise ValueError('Mutation cannot proceed on empty instruction list '
                         'with zero insertion probability')
      mutation_type = 'insert_instruction'

    elif enhancement_factor.num_instructions > self.max_num_instructions:
      raise ValueError('Mutation cannot proceed with instruction_list longer '
                       'than max_num_instructions.')

    elif (enhancement_factor.num_bound_parameters
          > self.max_num_bound_parameters):
      raise ValueError('Mutation cannot proceed with number of bound '
                       'parameters greater than max_num_bound_parameters.')

    elif enhancement_factor.num_instructions == self.max_num_instructions:
      if abs(self.mutation_pool.get('insert_instruction', 0.) - 1.) < 1e-8:
        raise ValueError('Mutation cannot proceed on max_num_instructions '
                         'with only insertions allowed')
      mutation_type = 'insert_instruction'
      while mutation_type == 'insert_instruction':
        mutation_type = self.get_random_mutation_type()

    else:
      mutation_type = self.get_random_mutation_type()

    # execute mutation
    new_instruction_list, instruction_index, change, message = getattr(
        self, mutation_type)(enhancement_factor)
    if verbose:
      logging.info(message)

    return (enhancement_factors.EnhancementFactor(
        feature_names=enhancement_factor.feature_names,
        shared_parameter_names=enhancement_factor.shared_parameter_names,
        variable_names=enhancement_factor.variable_names,
        instruction_list=new_instruction_list), mutation_type,
            instruction_index, change)

  def get_random_mutation_type(self):
    """Chooses a random mutation type based on mutation probabilities.

    Returns:
      String, the chosen mutation type.
    """
    return self.random_state.choice(
        self.mutation_types, p=self.mutation_probabilities)

  def get_random_instruction_index(self,
                                   enhancement_factor,
                                   allow_last_index=False):
    """Chooses a random instruction index for mutation.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.
      allow_last_index: Boolean, if True, allow index to be
        enhancement_factor.num_instructions, useful for inserting instructions.

    Returns:
      Integer, the chosen instruction index.
    """
    return self.random_state.randint(
        self.num_fixed_instructions,
        enhancement_factor.num_instructions + (1 if allow_last_index else 0))

  def get_random_instruction_name(self,
                                  existing_bound_parameters,
                                  num_inputs=None):
    """Gets a random instruction class name from instruction pool.

    The instruction will be chosen such that the total number of bound
    parameters will not exceed self.max_num_bound_parameters. In order to
    garantee this, the new instruction must satisfies at least one of the
    conditions:
      * It contains bound parameters in the existing_bound_parameters list, so
      it does not add new bound parameters.
      * The number of bound parameters of new instruction, when added to the
      number of existing bound parameters, is less than or equal to
      self.max_num_bound_parameters.

    Args:
      existing_bound_parameters: Sequence of strings, the names of existing
        bound parameters.
      num_inputs: Integer, if present, specifies the number of input arguments.
        Defaults to no constraint on number of input arguments.

    Returns:
      String, the class name of instruction.
    """
    candidates = instructions.get_instruction_names_with_signature(
        num_inputs=num_inputs,
        max_num_bound_parameters=(
            self.max_num_bound_parameters - len(existing_bound_parameters)),
        instructions=self.instruction_pool)

    # find instructions with bound parameters in the existing_bound_parameters
    # list, these instructions do not add new bound parameters and only need
    # to satisfy num_inputs constraint
    for bound_parameter in existing_bound_parameters:
      instruction_name = instructions.BOUND_PARAMETER_ASSOCIATION[
          bound_parameter]
      if instruction_name not in self.instruction_pool:
        continue
      if num_inputs is not None and (
          num_inputs != instructions.INSTRUCTION_CLASSES[
              instruction_name].get_num_inputs()):
        continue
      candidates.append(instruction_name)

    candidates = list(set(candidates))

    if not candidates:
      raise ValueError(
          'No instruction in instruction pool satisfies conditions: '
          f'num_inputs = {num_inputs}, '
          f'existing_bound_parameters = {existing_bound_parameters}')

    probabilities = [self.instruction_pool[instruction_name]
                     for instruction_name in candidates]
    assert probabilities and np.sum(probabilities) > 1e-8
    probabilities /= np.sum(probabilities)

    return self.random_state.choice(candidates, p=probabilities)

  def get_random_instruction(self,
                             enhancement_factor,
                             existing_bound_parameters=None):
    """Gets a random instruction with random arguments.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.
      existing_bound_parameters: Sequence of strings, the names of existing
        bound parameters. Defaults to those of enhancement_factor.

    Returns:
      Instance of instructions.Instruction, the resulting random instruction.
    """
    if existing_bound_parameters is None:
      existing_bound_parameters = enhancement_factor.bound_parameter_names
    instruction_class = instructions.INSTRUCTION_CLASSES[
        self.get_random_instruction_name(
            existing_bound_parameters=existing_bound_parameters)]
    output = self.random_state.choice(enhancement_factor.variable_names)
    inputs = [
        self.random_state.choice(enhancement_factor.allowed_input_names)
        for _ in range(instruction_class.get_num_inputs())]
    return instruction_class(output, *inputs)

  def insert_instruction(self, enhancement_factor):
    """One of mutation rules: inserts a random instruction.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.

    Returns:
      new_instruction_list: List of instructions.Instruction instances, the
        new instruction list after insertion.
      instruction_index: Integer, the index of inserted instruction.
      change: Tuple of (None, instructions.Instruction), change of instructions.
      message: String, the log of the mutation.
    """
    new_instruction_list = copy.deepcopy(enhancement_factor.instruction_list)

    instruction_index = self.get_random_instruction_index(
        enhancement_factor, allow_last_index=True)
    new_instruction = self.get_random_instruction(enhancement_factor)
    new_instruction_list.insert(instruction_index, new_instruction)

    return (new_instruction_list, instruction_index, (None, new_instruction),
            'EnhancementFactorMutator: inserted instruction at index '
            f'{instruction_index}: {new_instruction}')

  def remove_instruction(self, enhancement_factor):
    """One of mutation rules: removes a random instruction.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.

    Returns:
      new_instruction_list: List of instructions.Instruction instances, the
        new instruction list after removal.
      instruction_index: Integer, the index of removed instruction.
      change: Tuple of (instructions.Instruction, None), change of instructions.
      message: String, the log of the mutation.
    """
    new_instruction_list = copy.deepcopy(enhancement_factor.instruction_list)

    instruction_index = self.get_random_instruction_index(enhancement_factor)
    old_instruction = new_instruction_list.pop(instruction_index)

    return (new_instruction_list, instruction_index, (old_instruction, None),
            'EnhancementFactorMutator: removed instruction at index '
            f'{instruction_index}')

  def replace_instruction(self, enhancement_factor):
    """One of mutation rules: replace a random instruction.

    Input and output arguments will not be changed. The new instruction may
    carry different bound parameters from the old instruction.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.

    Returns:
      new_instruction_list: List of instructions.Instruction instances, the
        new instruction list after replacement.
      instruction_index: Integer, the index of replaced instruction.
      change: Tuple of 2 instructions.Instruction instances, old and new
        instructions.
      message: String, the log of the mutation.
    """
    new_instruction_list = copy.deepcopy(enhancement_factor.instruction_list)

    instruction_index = self.get_random_instruction_index(enhancement_factor)
    old_instruction = new_instruction_list[instruction_index]

    new_instruction_name = self.get_random_instruction_name(
        existing_bound_parameters=enhancement_factor.bound_parameter_names,
        num_inputs=old_instruction.get_num_inputs())
    new_instruction = instructions.INSTRUCTION_CLASSES[new_instruction_name](
        *old_instruction.args)

    new_instruction_list[instruction_index] = new_instruction

    return (new_instruction_list, instruction_index, (old_instruction,
                                                      new_instruction),
            f'EnhancementFactorMutator: replaced instruction at index '
            f'{instruction_index}. {old_instruction} -> {new_instruction}')

  def change_argument(self, enhancement_factor):
    """One of mutation rules: change an argument for a random instruction.

    The argument is randomly chosen and can be input or output argument.
    Bound parameters will not be changed.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.

    Returns:
      new_instruction_list: List of instructions.Instruction instances, the
        new instruction list after change of argument.
      instruction_index: Integer, the index of altered instruction.
      change: Tuple of 2 instructions.Instruction instances, old and new
        instructions.
      message: String, the log of the mutation.
    """
    new_instruction_list = copy.deepcopy(enhancement_factor.instruction_list)

    instruction_index = self.get_random_instruction_index(enhancement_factor)
    old_instruction = new_instruction_list[instruction_index]

    new_arguments = list(old_instruction.args)
    mutate_arg_index = self.random_state.randint(old_instruction.get_num_args())
    if mutate_arg_index == 0:
      # change output argument
      new_arguments[0] = self.random_state.choice(
          list(enhancement_factor.variable_names))
    else:
      # change input argument
      new_arguments[mutate_arg_index] = self.random_state.choice(
          list(enhancement_factor.allowed_input_names))

    new_instruction = old_instruction.__class__(*new_arguments)
    new_instruction_list[instruction_index] = new_instruction

    return (new_instruction_list, instruction_index, (old_instruction,
                                                      new_instruction),
            f'EnhancementFactorMutator: chaged argument {mutate_arg_index} of '
            f'instruction at index {instruction_index}. '
            f'{old_instruction} -> {new_instruction}')

  def randomize_instruction_list(self,
                                 enhancement_factor,
                                 num_instructions=None):
    """One of mutation rules: randomize the entire instruction list.

    Args:
      enhancement_factor: Instance of enhancement_factors.EnhancementFactor
        class, the input enhancement factor.
      num_instructions: Integer, the number of instructions in the randomized
        instruction list. If not specified, the new instruction list will
        have the same length with existing instruction list.

    Returns:
      new_instruction_list: List of instructions.Instruction instances, the
        new instruction list after change of argument.
      instruction_index: Integer, the index of altered instruction.
      change: Tuple of 2 instructions.Instruction instances, old and new
        instructions.
      message: String, the log of the mutation.

    Raises:
      ValueError: if self.num_fixed_instructions is nonzero.
    """
    if self.num_fixed_instructions:
      raise ValueError('randomize_instruction_list cannot be applied with '
                       'fixed instructions')

    num_instructions = num_instructions or enhancement_factor.num_instructions
    new_instruction_list = []
    bound_parameters = set()

    for _ in range(num_instructions):
      instruction = self.get_random_instruction(
          enhancement_factor,
          existing_bound_parameters=bound_parameters)
      for bound_parameter in instruction.get_bound_parameters():
        bound_parameters.add(bound_parameter)
      new_instruction_list.append(instruction)

    return (new_instruction_list, None, (None, None),
            'EnhancementFactorMutator: randomized instruction list. '
            f'New instruction list: {new_instruction_list}')


class XCFunctionalMutator:
  """Mutator for exchange-correlation functional.

  XCFunctionalMutator contains three EnhancementFactorMutator instances.
  The XCFunctionalMutator.mutate method will randomly call one of the mutate
  method of the three EnhancementFactorMutator instances.

  The three EnhancementFactor instances used by EnhancementFactorMutator are
  kept throughout the entire mutation process, with their instruction_list
  changed by mutators. Every mutation yield a new XCFunctional instance based on
  the same three EnhancementFactor instances. This design ensures that when
  Evaluator evaluates the functional, the jitted methods will be recompiled.
  """

  _default_component_mutation_probabilities = [1 / 3, 1 / 3, 1 / 3]

  def __init__(self,
               mutator_x,
               mutator_css,
               mutator_cos,
               component_mutation_probabilities=None,
               seed=None):
    """Initializes XCFunctionalMutator.

    Args:
      mutator_x: Instance of mutators.EnhancementFactorMutator, the mutator
        for exchange enhancement factor. If mutator_x is not specified, it
        will be constructed using instruction_pool, mutation_pool and
        max_num_instructions.
      mutator_css: Instance of mutators.EnhancementFactorMutator, the mutator
        for same-spin correlation enhancement factor. If mutator_css is not
        specified, it will be constructed using instruction_pool, mutation_pool
        and max_num_instructions.
      mutator_cos: Instance of mutators.EnhancementFactorMutator, the mutator
        for opposite-spin correlation enhancement factor. If mutator_cos is not
        specified, it will be constructed using instruction_pool, mutation_pool
        and max_num_instructions.
      component_mutation_probabilities: Sequence of 3 floats, the probabilities
        for mutating exchange, same-spin or opposite-spin component of the
        functional.
      seed: Integer, the random seed.

    Raises:
      ValueError, if enhancement factors in xc_functional do not correspond
        to enhancemenet factors of input enhancement factor mutators,
        or input component_mutation_probabilities has wrong shape, contains
        negative values or not normalized to 1.
    """
    self.mutator_x = mutator_x
    self.mutator_css = mutator_css
    self.mutator_cos = mutator_cos

    self.component_mutation_probabilities = (
        component_mutation_probabilities
        or self._default_component_mutation_probabilities)
    if len(self.component_mutation_probabilities) != 3:
      raise ValueError(
          'Wrong length for component_mutation_probabilities. '
          f'Expected 3, got {len(self.component_mutation_probabilities)}')
    if abs(sum(self.component_mutation_probabilities) - 1.) > 1e-8:
      raise ValueError(
          'component_mutation_probabilities not normalized to 1')
    if any(probability < 0.
           for probability in self.component_mutation_probabilities):
      raise ValueError(
          'component_mutation_probabilities contains negative probabilities')

    self.random_state = np.random.RandomState(seed=seed)

  def get_random_component(self):
    """Gets a random component (f_x, f_css or f_cos) of functional for mutation.

    Returns:
      String, the chosen component. 'f_x', 'f_css' or 'f_cos'.
    """
    return self.random_state.choice(
        ['f_x', 'f_css', 'f_cos'],
        p=self.component_mutation_probabilities)

  def get_mutator_for_component(self, component):
    """Gets the corresponding mutator for a given component of functional.

    Args:
      component: String, the functional compoenent. 'f_x', 'f_css' or 'f_cos'.

    Returns:
      Instance of EnhancementFactorMutator, the corresponding mutator for the
        given component. self.mutator_x, self.mutator_css or self.mutator_cos.
    """
    return getattr(self, {
        'f_x': 'mutator_x',
        'f_css': 'mutator_css',
        'f_cos': 'mutator_cos'
    }[component])

  def mutate(self, functional, verbose=True):
    """Mutates a random component (f_x, f_css or f_cos) of the functional.

    Args:
      functional: Instance of xc_functionals.XCFunctional, the exchange-
        correlation functional to be mutated.
      verbose: Boolean, if True, prints the log of mutation.

    Returns:
      new_functional: Instance of xc_functionals.XCFunctional, the new
        exchange-correlation functional after mutation.
      component: String, the mutated functional component. f_x, f_css or f_cos.
      mutation_type: String, the type of mutation performed.
      instruction_index: Integer, the index of mutated instruction.
      change: Tuple of two instances of instructions.Instruction.
        * (None, new_instruction) for insert_instruction
        * (old_instruction, None) for remove_instruction
        * (old_instruction, new_instruction) for replace_instruction or
          change_argument
    """
    functional_components = {
        'f_x': functional.f_x,
        'f_css': functional.f_css,
        'f_cos': functional.f_cos
    }
    component = self.get_random_component()
    if verbose:
      logging.info('XCFunctionalMutator: component %s is chosen.', component)

    new_enhancement_factor, mutation_type, instruction_index, change = (
        self.get_mutator_for_component(component).mutate(
            functional_components[component], verbose=verbose))

    functional_components[component] = new_enhancement_factor
    new_functional = xc_functionals.XCFunctional(**functional_components)

    return new_functional, component, mutation_type, instruction_index, change
