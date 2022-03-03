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

"""Enhancement factors."""

import itertools

import json
import jax.numpy as jnp
import numpy as onp
import sympy

from symbolic_functionals.syfes.symbolic import graphviz
from symbolic_functionals.syfes.symbolic import instructions


class EnhancementFactor:
  """Enhancement factor.

  An enhancement factor is defined as a list of instructions. An enhancement
  factor can be applied to a workspace, which is a dictionary {quantity_name:
  quantity_value} of quantities. Quantities include features, parameters and
  variables. Features are constant input 1D arrays defined on grids (e.g.
  density, density gradient, etc.); parameters are scalar values subject to
  optimization; variables are temporary intermediate quantities. A special
  variable named 'enhancement_factor' denotes the resulting enhancement factor
  values on grids.
  """

  _isomorphic_copy_shared_parameter_prefix = 'c'
  _isomorphic_copy_variable_prefix = 'v'

  def __init__(self,
               feature_names=None,
               shared_parameter_names=None,
               variable_names=None,
               instruction_list=None):
    """Initializes an enhancement factor.

    Args:
      feature_names: List of strings, the names for features. Features
        are 1D float numpy arrays.
      shared_parameter_names: List of strings, the names for shared parameters.
        Shared parameters are scalars that can be shared by multiple
        instructions.
      variable_names: List of strings, the names for variables. The variable
        names should include an element called 'enhancement_factor', which will
        be taken as the resulting enhancement factor after all instructions are
        applied to a workspace.
      instruction_list: List of instructions.Instruction instances, the sequence
        of instructions that defines the enhancement factor.
    """
    self.feature_names = feature_names or []
    self.shared_parameter_names = shared_parameter_names or []
    self.variable_names = variable_names or []
    self.allowed_input_names = (
        self.feature_names + self.shared_parameter_names + self.variable_names)
    self.instruction_list = instruction_list or []

    self.validate()

    self.bound_parameter_names = list(set(itertools.chain(
        *[instruction.get_bound_parameters()
          for instruction in self.instruction_list])))
    self.parameter_names = (
        self.shared_parameter_names + self.bound_parameter_names)

    shared_parameter_is_used = {
        name: False for name in self.shared_parameter_names}
    for instruction in self.instruction_list:
      for arg in instruction.args:
        if arg in shared_parameter_is_used:
          shared_parameter_is_used[arg] = True
    self.used_shared_parameter_names = [
        name for name in self.shared_parameter_names
        if shared_parameter_is_used[name]]
    self.used_parameter_names = (
        self.used_shared_parameter_names + self.bound_parameter_names)

  def validate(self):
    """Validates names and instructions.

    Raises:
      TypeError: if instruction is not an instance of instructions.Instruction.
      ValueError: if enhancement_factor not in variable names,
        or repeated entries found in feature, parameter or variable names,
        or instruction contains invalid input or output names.
    """
    # validate instruction list
    for instruction in self.instruction_list:
      if not isinstance(instruction, instructions.Instruction):
        raise TypeError(f'{instruction} is of type {type(instruction)}, not an '
                        'instance of instructions.Instruction')

    # validate names
    if 'enhancement_factor' not in self.variable_names:
      raise ValueError('"enhancement_factor" not found in variable_names.')
    if len(self.allowed_input_names) != len(set(self.allowed_input_names)):
      raise ValueError('Repeated names found in input.')

    # validates instruction arguments
    for instruction in self.instruction_list:
      if not isinstance(instruction, instructions.Instruction):
        raise TypeError(f'{instruction} is of type {type(instruction)}, not an '
                        'instance of instructions.Instruction')
      if instruction.output not in self.variable_names:
        raise ValueError(f'Instruction {instruction} contains invalid output '
                         f'argument {instruction.output}')
      for arg in instruction.inputs:
        if arg not in self.allowed_input_names:
          raise ValueError(f'Instruction {instruction} contains invalid input '
                           f'argument {arg}')

  @property
  def num_features(self):
    """Number of features."""
    return len(self.feature_names)

  @property
  def num_shared_parameters(self):
    """Number of shared parameters."""
    return len(self.shared_parameter_names)

  @property
  def num_bound_parameters(self):
    """Number of bound parameters."""
    return len(self.bound_parameter_names)

  @property
  def num_parameters(self):
    """Number of parameters."""
    return len(self.parameter_names)

  @property
  def num_used_parameters(self):
    """Number of used parameters."""
    return len(self.used_parameter_names)

  @property
  def num_variables(self):
    """Number of variables including 'enhancement_factor'."""
    return len(self.variable_names)

  @property
  def num_instructions(self):
    """Number of instructions."""
    return len(self.instruction_list)

  def eval(self, features, parameters, use_jax=True):
    """Evaluates the enhancement factor on grids.

    Args:
      features: Dict {feature_name: feature_value}, the input features.
      parameters: Dict {parameter_name: parameter_values}, the scalar parameters
        in the functional form.
      use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
        numpy.

    Returns:
      Float numpy array with shape (num_grids_all,), the enhancement factor
        on grids.
    """
    np = jnp if use_jax else onp
    workspace = {
        **features,
        **{parameter_name: np.array(parameter_value)
           for parameter_name, parameter_value in parameters.items()},
        **{variable_name: np.array(0.)
           for variable_name in self.variable_names}
    }

    for instruction in self.instruction_list:
      instruction.apply(workspace, use_jax=use_jax)

    return workspace['enhancement_factor']

  def to_dict(self):
    """Converts the enhancement factor to dictionary.

    Returns:
      Dict, the dictionary representation of enhancement factor.
    """
    return {
        'feature_names': list(self.feature_names),
        'shared_parameter_names': list(self.shared_parameter_names),
        'variable_names': list(self.variable_names),
        'instructions': [
            instruction.to_list() for instruction in self.instruction_list]
    }

  @staticmethod
  def from_dict(dictionary):
    """Loads enhancement factor from dictionary.

    Args:
      dictionary: Dict, the dictionary representation of enhancement factor.

    Returns:
      Instance of EnhancementFactor, the loaded enhancement factor.
    """
    return EnhancementFactor(
        feature_names=dictionary['feature_names'],
        shared_parameter_names=dictionary['shared_parameter_names'],
        variable_names=dictionary['variable_names'],
        instruction_list=[
            instructions.Instruction.from_list(lst)
            for lst in dictionary['instructions']])

  def make_isomorphic_copy(self,
                           feature_names=None,
                           num_shared_parameters=None,
                           num_variables=None,):
    """Makes an isomorphic copy of the EnhancementFactor instance.

    Here isomorphic copy denotes that the new EnhancementFactor instance will
    have identical instruction list with the currect instance, while all shared
    parameters and variables will be renamed based on class attributes
    _isomorphic_copy_shared_parameter_prefix and
    _isomorphic_copy_variable_prefix.

    The number of shared parameters and variables of the new instance can be
    specified, provided they are greater or equal to those of current instance.
    The feature names of the new instance can also be specified, as long as they
    constitute a superset of features names of current instance.

    Empty EnhancementFactors with desired features and number of shared
    parameters and variables can be constructed by calling this method of the
    f_empty object defined in this module.

    Args:
      feature_names: List of strings, if present, defines the feature names of
        the copy. Must be a superset of self.feature_names.
      num_shared_parameters: Integer, if present, specifies the number of
        shared parameters of the copy. Must be greater or equal to
        self.num_shared_parameters.
      num_variables: Integer, if present, specifies the number of variables of
        the copy. Must be greater or equal to self.num_variables.

    Returns:
      Instance of EnhancementFactor, the isomorphic copy.

    Raises:
      ValueError, if feature_names contains repeated elements,
        or if feature_names is not a superset of self.feature_names,
        or num_shared_parameters or num_variables is smaller than those of
        current instance.
    """
    if feature_names is None:
      feature_names = self.feature_names
    else:
      if len(feature_names) != len(set(feature_names)):
        raise ValueError('Repeated feature names')
      if not set(feature_names).issuperset(set(self.feature_names)):
        raise ValueError(
            f'feature_names {feature_names} is not a superset of '
            f'feature_names of current instance {self.feature_names}')

    if num_shared_parameters is None:
      num_shared_parameters = self.num_shared_parameters
    else:
      if num_shared_parameters < self.num_shared_parameters:
        raise ValueError(
            f'num_shared_parameters {num_shared_parameters} is smaller than '
            f'that of current instance {self.num_shared_parameters}')

    if num_variables is None:
      num_variables = self.num_variables
    else:
      if num_variables < self.num_variables:
        raise ValueError(
            f'num_variables {num_variables} is smaller than '
            f'that of current instance {self.num_variables}')

    name_mapping = {
        feature_name: feature_name for feature_name in self.feature_names}
    name_mapping['enhancement_factor'] = 'enhancement_factor'
    for index, shared_parameter_name in enumerate(self.shared_parameter_names):
      name_mapping[shared_parameter_name] = (
          self._isomorphic_copy_shared_parameter_prefix + str(index))

    index = 0
    for variable_name in self.variable_names:
      if variable_name == 'enhancement_factor':
        continue
      name_mapping[variable_name] = (
          self._isomorphic_copy_variable_prefix + str(index))
      index += 1
    assert len(name_mapping) == len(self.allowed_input_names)

    return EnhancementFactor(
        feature_names=feature_names,
        shared_parameter_names=[
            self._isomorphic_copy_shared_parameter_prefix + str(index)
            for index in range(num_shared_parameters)],
        variable_names=[
            self._isomorphic_copy_variable_prefix + str(index)
            for index in range(num_variables - 1)] + ['enhancement_factor'],
        instruction_list=[
            instruction.__class__(
                *[name_mapping[arg] for arg in instruction.args])
            for instruction in self.instruction_list])

  def get_graph(self):
    """Gets graph representation of enhancement factor."""
    return graphviz.create_graph(
        feature_names=self.feature_names,
        shared_parameter_names=self.shared_parameter_names,
        bound_parameter_names=self.bound_parameter_names,
        variable_names=self.variable_names,
        instruction_list=self.to_dict()['instructions']
    )

  def get_symbolic_expression(self, latex=True, simplify=False):
    """Gets symbolic expression of enhancement factor.

    Args:
      latex: Boolean, if True, the symbolic representation will be returned
        as a string in the latex format.
      simplify: Boolean, whether to simplify the expression.

    Returns:
      Sympy expression or string, the symbolic representation of enhancement
        factor.
    """
    workspace = {
        **{feature_name: sympy.Symbol(feature_name)
           for feature_name in self.feature_names},
        **{parameter_name: sympy.Symbol(parameter_name)
           for parameter_name in self.parameter_names},
        **{variable_name: 0. for variable_name in self.variable_names}
    }

    if 'x2' in workspace:
      workspace['x2'] = sympy.Symbol('x') ** 2

    for instruction in self.instruction_list:
      instruction.sympy_apply(workspace)

    enhancement_factor = workspace['enhancement_factor']
    if simplify:
      enhancement_factor = sympy.simplify(enhancement_factor)

    if latex:
      # replace utransform by u to simplify the expression
      return sympy.latex(enhancement_factor).replace('utransform', 'u')
    else:
      return enhancement_factor

  def __eq__(self, other):
    return all([self.feature_names == other.feature_names,
                self.shared_parameter_names == other.shared_parameter_names,
                self.variable_names == other.variable_names,
                self.instruction_list == other.instruction_list])

  def __str__(self):
    return json.dumps(self.to_dict(), indent=2)

  def __repr__(self):
    return self.__str__()


# empty enhancement factor
f_empty = EnhancementFactor(
    feature_names=[],
    shared_parameter_names=[],
    variable_names=['enhancement_factor'])

# LDA enhancement factor (identity)
f_lda = EnhancementFactor(
    feature_names=[],
    shared_parameter_names=[],
    variable_names=['enhancement_factor'],
    instruction_list=[
        instructions.AdditionBy1Instruction(
            'enhancement_factor', 'enhancement_factor'),])

# B97 enhancement factor as a function of u
f_b97_u = EnhancementFactor(
    feature_names=['u'],
    shared_parameter_names=['c0', 'c1', 'c2'],
    variable_names=['c1u', 'c2u2', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.MultiplicationInstruction('c1u', 'u', 'c1'),
        instructions.Power2Instruction('c2u2', 'u'),
        # power series
        instructions.MultiplicationInstruction('c2u2', 'c2', 'c2u2'),
        instructions.AdditionInstruction('enhancement_factor', 'c0', 'c1u'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'c2u2'),
    ])

# B97 enhancement factor as a function of u (short version)
f_b97_u_short = EnhancementFactor(
    feature_names=['u'],
    shared_parameter_names=['c0', 'c1', 'c2'],
    variable_names=['u2', 'enhancement_factor'],
    instruction_list=[
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'c0'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c1', 'u'),
        instructions.Power2Instruction('u2', 'u'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c2', 'u2'),
    ]
)

# B97 enhancement factor as a function of x^2
f_b97_x2 = EnhancementFactor(
    feature_names=['x2'],
    shared_parameter_names=['gamma', 'c0', 'c1', 'c2'],
    variable_names=['gx2', 'u', 'c1u', 'c2u2', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.MultiplicationInstruction('gx2', 'gamma', 'x2'),
        instructions.AdditionBy1Instruction('u', 'gx2'),
        instructions.DivisionInstruction('u', 'gx2', 'u'),
        # power series
        instructions.MultiplicationInstruction('c1u', 'u', 'c1'),
        instructions.Power2Instruction('c2u2', 'u'),
        instructions.MultiplicationInstruction('c2u2', 'c2', 'c2u2'),
        instructions.AdditionInstruction('enhancement_factor', 'c0', 'c1u'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'c2u2'),
    ]
)

# B97 enhancement factor as a function of x^2 (short version)
f_b97_x2_short = EnhancementFactor(
    feature_names=['x2'],
    shared_parameter_names=['c0', 'c1', 'c2'],
    variable_names=['u', 'u2', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.UTransformInstruction('u', 'x2'),
        # power series
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'c0'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c1', 'u'),
        instructions.Power2Instruction('u2', 'u'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c2', 'u2'),
    ]
)

f_x_wb97mv = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['gamma', 'c00', 'c10', 'c01'],
    variable_names=['gx2', 'u', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.MultiplicationInstruction('gx2', 'gamma', 'x2'),
        instructions.AdditionBy1Instruction('u', 'gx2'),
        instructions.DivisionInstruction('u', 'gx2', 'u'),
        # power series
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
)

f_css_wb97mv = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['gamma', 'c00', 'c10', 'c20', 'c43', 'c04'],
    variable_names=[
        'gx2', 'u', 'u2', 'w2', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.MultiplicationInstruction('gx2', 'gamma', 'x2'),
        instructions.AdditionBy1Instruction('u', 'gx2'),
        instructions.DivisionInstruction('u', 'gx2', 'u'),
        # calculation of w^2 and u^2
        instructions.Power2Instruction('w2', 'w'),
        instructions.Power2Instruction('u2', 'u'),
        # power series
        instructions.AdditionInstruction(
            'enhancement_factor', 'c00', 'enhancement_factor'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c10', 'w'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c20', 'w2'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.Power2Instruction(
            'linear_term', 'w2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'linear_term', 'u2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'linear_term', 'u'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c43', 'linear_term'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.Power2Instruction(
            'linear_term', 'u2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c04', 'linear_term'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
    ]
)

f_cos_wb97mv = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['gamma', 'c00', 'c10', 'c20', 'c60', 'c61', 'c21'],
    variable_names=[
        'gx2', 'u', 'u2', 'w2', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.MultiplicationInstruction('gx2', 'gamma', 'x2'),
        instructions.AdditionBy1Instruction('u', 'gx2'),
        instructions.DivisionInstruction('u', 'gx2', 'u'),
        # calculation of w^2 and u^2
        instructions.Power2Instruction('w2', 'w'),
        instructions.Power2Instruction('u2', 'u'),
        # power series
        instructions.AdditionInstruction(
            'enhancement_factor', 'c00', 'enhancement_factor'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c10', 'w'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c20', 'w2'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.Power2Instruction(
            'linear_term', 'w2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'linear_term', 'w2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c60', 'linear_term'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.Power2Instruction(
            'linear_term', 'w2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'linear_term', 'w2'),
        instructions.MultiplicationInstruction(
            'linear_term', 'linear_term', 'u'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c61', 'linear_term'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
        instructions.MultiplicationInstruction(
            'linear_term', 'w2', 'u'),
        instructions.MultiplicationInstruction(
            'linear_term', 'c21', 'linear_term'),
        instructions.AdditionInstruction(
            'enhancement_factor', 'enhancement_factor', 'linear_term'),
    ]
)

f_x_wb97mv_short = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['c00', 'c10', 'c01'],
    variable_names=['u', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.UTransformInstruction('u', 'x2'),
        # power series
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
)

f_css_wb97mv_short = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['c00', 'c10', 'c20', 'c43', 'c04'],
    variable_names=[
        'u', 'upower', 'wpower', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.UTransformInstruction('u', 'x2'),
        # power series
        instructions.AdditionInstruction(
            'enhancement_factor', 'c00', 'enhancement_factor'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c10', 'w'),
        instructions.Power2Instruction(
            'wpower', 'w'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c20', 'wpower'),
        instructions.Power4Instruction(
            'wpower', 'w'),
        instructions.Power3Instruction(
            'upower', 'u'),
        instructions.MultiplicationInstruction(
            'linear_term', 'wpower', 'upower'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c43', 'linear_term'),
        instructions.Power4Instruction(
            'upower', 'u'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c04', 'upower'),
    ]
)

f_cos_wb97mv_short = EnhancementFactor(
    feature_names=['w', 'x2'],
    shared_parameter_names=['c00', 'c10', 'c20', 'c60', 'c61', 'c21'],
    variable_names=[
        'u', 'upower', 'wpower', 'linear_term', 'enhancement_factor'],
    instruction_list=[
        # calculation of u
        instructions.UTransformInstruction('u', 'x2'),
        # power series
        instructions.AdditionInstruction(
            'enhancement_factor', 'c00', 'enhancement_factor'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c10', 'w'),
        instructions.Power2Instruction(
            'wpower', 'w'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c20', 'wpower'),
        instructions.MultiplicationInstruction(
            'linear_term', 'wpower', 'u'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c21', 'linear_term'),
        instructions.Power6Instruction(
            'wpower', 'w'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c60', 'wpower'),
        instructions.MultiplicationInstruction(
            'linear_term', 'wpower', 'u'),
        instructions.MultiplicationAdditionInstruction(
            'enhancement_factor', 'c61', 'linear_term'),
    ]
)
