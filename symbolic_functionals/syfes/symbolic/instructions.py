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

"""Mathematic instructions as building blocks of density functionals."""

import abc
import jax.numpy as jnp
import numpy as onp
import sympy
from symbolic_functionals.syfes.xc import gga


################
# Base classes #
################


class Instruction(abc.ABC):
  """Base class for instruction.

  An instruction represents a methamatical operation applied to certain
  quantities in a workspace (represented by a dictionary). An instruction
  is specified with input arguments, output arguments and bound parameters.
  Bound parameters are parameters associated with a certain instruction class.
  Different instances of the class share the same bound parameters.
  """

  # All subclasses should override the following class attributes
  _num_inputs = None
  _bound_parameters = None

  def __init__(self, *args):
    """Initializes instruction.

    Args:
      *args: List of strings, the name of the arguments. args[0] represents
        output argument; args[1:] represent input arguments.

    Raises:
      ValueError, if the length of args is not equal to the total number of
        arguments of the instruction class.
    """
    if len(args) != self.get_num_args():
      raise ValueError(
          f'{self.__class__.__name__}: wrong number of arguments. '
          f'Expected {self.get_num_args()}, got {len(args)}')
    self.args = list(args)
    self.output = self.args[0]
    self.inputs = self.args[1:]

  @classmethod
  def get_num_inputs(cls):
    """Gets the number of input arguments of the instruction class."""
    return cls._num_inputs

  @classmethod
  def get_bound_parameters(cls):
    """Gets the bound parameter names of the instruction class."""
    return cls._bound_parameters

  @classmethod
  def get_num_bound_parameters(cls):
    """Gets the number of bound parameters of the instruction class."""
    return len(cls._bound_parameters)

  @classmethod
  def get_num_args(cls):
    """Gets the number of arguments of the instruction class."""
    return 1 + cls._num_inputs  # 1 is output

  @classmethod
  def is_unary_instruction(cls):
    """Checks if the instruction class represents a unary instruction."""
    return cls._num_inputs == 1

  @classmethod
  def is_binary_instruction(cls):
    """Checks if the instruction class represents a binary instruction."""
    return cls._num_inputs == 2

  def __eq__(self, other):
    return type(self) is type(other) and self.args == other.args

  @abc.abstractmethod
  def apply(self, workspace, use_jax=True):
    """Applies the instruction to a given workspace.

    Args:
      workspace: Dict {quantity_name: quantity_value}, the workspace to which
        the instruction is applied. quantity_value can be float or 1D float
        numpy array.
      use_jax: Boolean, if True, use jax.numpy instead of original numpy. This
        flag has no effect on instructions not using numpy.
    """
    pass

  def sympy_apply(self, workspace):
    """Applies the instruction to a workspace of sympy symbols.

    This method should be overridden when the `apply` method cannot correctly
    handle sympy symbols.

    Args:
      workspace: Dict {quantity_name: quantity_sympy_symbol}.
    """
    self.apply(workspace, use_jax=False)

  @abc.abstractmethod
  def __str__(self):
    pass

  def __repr__(self):
    return self.__str__()

  def to_list(self):
    """Saves the current instruction to a list.

    Returns:
      List, the first element denotes the name of instruction class, the rest
        of elements denotes arguments.
    """
    return [self.__class__.__name__, *self.args]

  @staticmethod
  def from_list(lst):
    """Loads an instruction from a list.

    Args:
      lst: List, the first element denotes the name of instruction class,
        the rest of elements denotes arguments.

    Returns:
      Instance of Instruction, the loaded instruction.

    Raises:
      ValueError, if instruction class name is invalid.
    """
    if lst[0] not in INSTRUCTION_CLASSES:
      raise ValueError(f'Invalid instruction class name: {lst[0]}')
    return INSTRUCTION_CLASSES[lst[0]](*lst[1:])


######################
# Unary instructions #
######################


class AdditionBy1Instruction(Instruction):
  """Addition by 1."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.inputs[0]] + 1.

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} + 1'


class Power2Instruction(Instruction):
  """Square."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.inputs[0]] ** 2

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} ** 2'


class Power3Instruction(Instruction):
  """Cube."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.inputs[0]] ** 3

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} ** 3'


class Power4Instruction(Instruction):
  """4th power."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.inputs[0]] ** 4

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} ** 4'


class Power6Instruction(Instruction):
  """6th power."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.inputs[0]] ** 6

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} ** 6'


class SquareRootInstruction(Instruction):
  """Square root."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    np = jnp if use_jax else onp
    workspace[self.output] = np.sqrt(workspace[self.inputs[0]])

  def sympy_apply(self, workspace):
    workspace[self.output] = workspace[self.inputs[0]] ** (
        sympy.Integer(1) / sympy.Integer(2))

  def __str__(self):
    return f'{self.output} = sqrt({self.inputs[0]})'


class CubeRootInstruction(Instruction):
  """Cube root."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    np = jnp if use_jax else onp
    workspace[self.output] = np.cbrt(workspace[self.inputs[0]])

  def sympy_apply(self, workspace):
    workspace[self.output] = workspace[self.inputs[0]] ** (
        sympy.Integer(1) / sympy.Integer(3))

  def __str__(self):
    return f'{self.output} = cbrt({self.inputs[0]})'


class Log1PInstruction(Instruction):
  """Logarithm 1 plus: y = log(1 + x)."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    np = jnp if use_jax else onp
    workspace[self.output] = np.log1p(workspace[self.inputs[0]])

  def sympy_apply(self, workspace):
    workspace[self.output] = sympy.log(1. + workspace[self.inputs[0]])

  def __str__(self):
    return f'{self.output} = log(1 + {self.inputs[0]})'


class ExpInstruction(Instruction):
  """Exponential."""
  _num_inputs = 1
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    np = jnp if use_jax else onp
    workspace[self.output] = np.exp(workspace[self.inputs[0]])

  def sympy_apply(self, workspace):
    workspace[self.output] = sympy.exp(workspace[self.inputs[0]])

  def __str__(self):
    return f'{self.output} = exp({self.inputs[0]})'


#######################
# Binary instructions #
#######################


class AdditionInstruction(Instruction):
  """Addition."""
  _num_inputs = 2
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = (
        workspace[self.inputs[0]] + workspace[self.inputs[1]])

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} + {self.inputs[1]}'


class SubtractionInstruction(Instruction):
  """Subtraction."""
  _num_inputs = 2
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = (
        workspace[self.inputs[0]] - workspace[self.inputs[1]])

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} - {self.inputs[1]}'


class MultiplicationInstruction(Instruction):
  """Multiplication."""
  _num_inputs = 2
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = (
        workspace[self.inputs[0]] * workspace[self.inputs[1]])

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} * {self.inputs[1]}'


class MultiplicationAdditionInstruction(Instruction):
  """Multiplication."""
  _num_inputs = 2
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = workspace[self.output] + (
        workspace[self.inputs[0]] * workspace[self.inputs[1]])

  def __str__(self):
    return f'{self.output} += {self.inputs[0]} * {self.inputs[1]}'


class DivisionInstruction(Instruction):
  """Division."""
  _num_inputs = 2
  _bound_parameters = []

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    workspace[self.output] = (
        workspace[self.inputs[0]] / workspace[self.inputs[1]])

  def __str__(self):
    return f'{self.output} = {self.inputs[0]} / {self.inputs[1]}'


######################################
# Instructions with bound parameters #
######################################


class UTransformInstruction(Instruction):
  """Unary finite-domain transform y = gamma x / (1 + gamma x).

  This transform was used to define quantity u in B97 form:
    u = gamma * x^2 / (1 + gamma * x^2)
  and quantity v in MN12 form:
    v = omega * rho^(1/3) / (1 + omega * rho^(1/3))
  """
  _num_inputs = 1
  _bound_parameters = ['gamma_utransform']

  def apply(self, workspace, use_jax=True):
    del use_jax  # Unused.
    tmp = workspace[self._bound_parameters[0]] * workspace[self.inputs[0]]
    workspace[self.output] = tmp / (1. + tmp)

  def __str__(self):
    return (f'{self.output} = UTransform({self.inputs[0]}; '
            f'{self._bound_parameters[0]})')


class PBEXInstruction(Instruction):
  """PBE exchange enhancement factor."""
  _num_inputs = 1  # x
  _bound_parameters = ['kappa_pbex', 'mu_pbex']

  def apply(self, workspace, use_jax=True):
    workspace[self.output] = gga.f_x_pbe(
        x=workspace[self.inputs[0]],
        kappa=workspace[self._bound_parameters[0]],
        mu=workspace[self._bound_parameters[1]])

  def __str__(self):
    return (f'{self.output} = PBEXInstruction({self.inputs[0]}; '
            f'{self._bound_parameters[0], self._bound_parameters[1]})')


class RPBEXInstruction(Instruction):
  """RPBE exchange enhancement factor."""
  _num_inputs = 1  # x
  _bound_parameters = ['kappa_rpbex', 'mu_rpbex']

  def apply(self, workspace, use_jax=True):
    workspace[self.output] = gga.f_x_rpbe(
        x=workspace[self.inputs[0]],
        kappa=workspace[self._bound_parameters[0]],
        mu=workspace[self._bound_parameters[1]],
        use_jax=use_jax)

  def __str__(self):
    return (f'{self.output} = RPBEXInstruction({self.inputs[0]}; '
            f'{self._bound_parameters[0], self._bound_parameters[1]})')


class B88XInstruction(Instruction):
  """B88 exchange enhancement factor."""
  _num_inputs = 1  # x
  _bound_parameters = ['beta_b88x']

  def apply(self, workspace, use_jax=True):
    workspace[self.output] = gga.f_x_b88(
        x=workspace[self.inputs[0]],
        beta=workspace[self._bound_parameters[0]],
        use_jax=use_jax)

  def __str__(self):
    return (f'{self.output} = B88XInstruction({self.inputs[0]}; '
            f'{self._bound_parameters[0]})')


class PBECInstruction(Instruction):
  """PBE correlation energy density."""
  _num_inputs = 2  # rho, sigma
  _bound_parameters = ['beta_pbec', 'gamma_pbec']

  def apply(self, workspace, use_jax=True):
    workspace[self.output] = gga.e_c_pbe_unpolarized(
        rho=workspace[self.inputs[0]],
        sigma=workspace[self.inputs[1]],
        beta=workspace[self._bound_parameters[0]],
        gamma=workspace[self._bound_parameters[1]],
        use_jax=use_jax)

  def __str__(self):
    return (f'{self.output} = PBECInstruction({self.inputs[0]}, '
            f'{self.inputs[1]}; {self._bound_parameters[0]}, '
            f'{self._bound_parameters[1]})')


####################
# Helper functions #
####################

UNARY_INSTRUCTION_CLASSES = {
    instruction_class.__name__: instruction_class
    for instruction_class in Instruction.__subclasses__()
    if instruction_class.is_unary_instruction()}

BINARY_INSTRUCTION_CLASSES = {
    instruction_class.__name__: instruction_class
    for instruction_class in Instruction.__subclasses__()
    if instruction_class.is_binary_instruction()}

INSTRUCTION_CLASSES = {
    **UNARY_INSTRUCTION_CLASSES, **BINARY_INSTRUCTION_CLASSES}


def _get_bound_parameter_association():
  """Helper function to get a mapping from bound parameter to instruction."""
  bound_parameter_association = {}
  for instruction_class in INSTRUCTION_CLASSES.values():
    for bound_parameter in instruction_class.get_bound_parameters():
      bound_parameter_association[bound_parameter] = instruction_class.__name__
  return bound_parameter_association


BOUND_PARAMETER_ASSOCIATION = _get_bound_parameter_association()


def is_unary_instruction_name(name):
  """Checks if name corresponds to a unary instruction."""
  return name in UNARY_INSTRUCTION_CLASSES


def is_binary_instruction_name(name):
  """Checks if name corresponds to a binary instruction."""
  return name in BINARY_INSTRUCTION_CLASSES


def get_unary_instruction_names_from_list(names):
  """Gets unary instruction names from a list of strings."""
  return list(filter(is_unary_instruction_name, names))


def get_binary_instruction_names_from_list(names):
  """Gets binary instruction names from a list of strings."""
  return list(filter(is_binary_instruction_name, names))


def get_instruction_names_with_signature(
    num_inputs=None,
    num_bound_parameters=None,
    max_num_bound_parameters=None,
    instructions=None):
  """Gets instruction names with given signature.

  Args:
    num_inputs: Integer, if presents, specifies the number of input parameters.
    num_bound_parameters: Integer, if presents, specifies the number of bound
      parameters.
    max_num_bound_parameters: Integer, if presents, specifies the maximum number
      of bound parameters.
    instructions: Sequence of strings, the possible instruction names for
      search. Defaults to all subclasses of Instruction.

  Returns:
    List of strings, the names of instructions with given signature.
  """
  if instructions is None:
    instructions = INSTRUCTION_CLASSES

  instruction_names = []
  for instruction_name in instructions:
    instruction_class = INSTRUCTION_CLASSES[instruction_name]
    if (num_inputs is not None
        and instruction_class.get_num_inputs() != num_inputs):
      continue
    if (num_bound_parameters is not None and
        instruction_class.get_num_bound_parameters() != num_bound_parameters):
      continue
    if (max_num_bound_parameters is not None and
        instruction_class.get_num_bound_parameters()
        > max_num_bound_parameters):
      continue
    instruction_names.append(instruction_name)

  return instruction_names
