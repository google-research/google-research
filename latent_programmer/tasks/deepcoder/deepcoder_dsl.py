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

r"""DeepCoder DSL.

Features:
  * The operators and constant lambda functions are as described in Appendix F
    of the DeepCoder paper (https://arxiv.org/pdf/1611.01989.pdf)
  * The Program class stores a program and can do the following:
      * Run the program on inputs to produce an output or program state
      * Serialize the program into a sequence of tokens
      * Deserialize a sequence of tokens into a new Program
  * The Statement class represents partial programs (a single assignment)
  * The ProgramState class captures program state at partial executions
  * Utilities for random dataset generation:
      * Generate a new random program
          * Check to make sure the random program doesn't have dead code
      * Generate random inputs for a program
      * Ways of generating compositional generalization splits
  * A flag controlling whether we use mod-10 arithmetic or not

Helpful properties to know:

  * When converting from a list of string tokens to strings (representing
    programs or program states) or vice versa, tokens are all space separated.
    There is no need to parse parentheses. Use `' '.join(tokens)` or
    `string.split()` to convert between lists of string tokens and strings.

  * A program state is given as a string in the following form, which does not
    distinguish between inputs and previously-computed local variables:

    x0 = [ 3 , 1 , 2 ] | x1 = [ 4 , 2 , 3 ] | x2 = 9

  * A program is given as a string in the following form:

    x0 = INPUT | x1 = Map +1 x0 | x2 = Sum x1

  * In the context of running an entire program (e.g., to see if it satisfies
    the I/O examples), the program "output" is the result of its last line. In
    the context of running just one program line, we can think of the input and
    output as being entire program states.

  * Program variables (x0, x1, ...) only have types `int` or `list[int]`. Lambda
    functions may have types such as `int -> bool` or `(int, int) -> int`, but
    lambdas are only used as arguments to higher-order functions and can't be
    program variables themselves.
"""

import ast
import functools
import re
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from absl import flags

_DEEPCODER_MOD = flags.DEFINE_integer(
    'deepcoder_mod', 10,
    'The modulo we use for DeepCoder arithmetic, or 0 to not apply any mod.')


# Types for type specifications, e.g., `([int], bool)` has type LambdaType and
# specifies a lambda that takes 1 int and returns a bool.
InputsType = List[Union[Type[Any], 'LambdaType']]
OutputType = Type[Any]
LambdaType = Tuple[InputsType, OutputType]

# A type for the result of applying any operation.
ResultType = Union[int, List[int]]

MIN_INT = -256
MAX_INT = 255


class ParseError(Exception):
  """Could not parse from a string or tokens, similar to a syntax error."""
  # This could happen if a model predicts a bad token sequence.


class RunError(Exception):
  """Could not execute the operation or program, similar to a runtime error."""
  # This is raised for type mismatch, wrong arity given to an operation,
  # reference to an invalid variable, or running a program on the wrong number
  # of inputs.
  # This is NOT raised when an operation returns None, which results in run()
  # returning None.


class DeepCoderError(Exception):
  """Something happened that suggests there's a bug."""
  # This is more severe of an error than a program failing to execute. We should
  # not see this error during normal usage even considering model inaccuracy,
  # unless there's a bug.


class Example(object):
  """A DeepCoder specification in the form of an I/O example."""

  def __init__(self, inputs, output):
    self.inputs = inputs
    self.output = output


def join_token_lists(token_lists,
                     separator_token):
  return functools.reduce(lambda a, b: a + [separator_token] + b, token_lists)


def mod_result(result):
  """If desired, apply mod to ints."""
  if result is None:
    return result
  if _DEEPCODER_MOD.value > 0:
    result_type = type(result)
    if result_type == int:
      return result % _DEEPCODER_MOD.value
    elif result_type == list:
      return [x % _DEEPCODER_MOD.value for x in result]
    else:
      raise DeepCoderError(f'Unhandled result in mod_result: {result}')
  return result


def validate_int(i):
  """Checks that the integer is in range."""
  if _DEEPCODER_MOD.value > 0:
    return 0 <= i < _DEEPCODER_MOD.value
  return MIN_INT <= i <= MAX_INT


def validate_result(result):
  """Returns whether an object is valid or not."""
  # Distinguish between bool and int.
  # pylint: disable=unidiomatic-typecheck
  if type(result) == int:
    return validate_int(result)
  elif type(result) == list:
    return all(type(x) == int and validate_int(x) for x in result)
  else:
    return False
  # pylint: enable=unidiomatic-typecheck


def variable_token(index):
  return f'x{index}'


def variable_index_from_token(token):
  if not re.fullmatch(r'x\d+', token):
    raise ParseError(f'Invalid variable token: {token}')
  return int(token[1:])


def tokenize_result(result):
  """Returns a list of tokens for the result of an operation."""
  if isinstance(result, int):
    return [str(result)]
  elif isinstance(result, list):
    return ['['] + join_token_lists([tokenize_result(x) for x in result],
                                    separator_token=',') + [']']
  else:
    raise DeepCoderError(f'Unhandled type in tokenize_result({result})')


class ProgramState(object):
  """Holds a program state (for one example)."""

  def __init__(self, state):
    self.state = state

  def __len__(self):
    return len(self.state)

  def __getitem__(self, index):
    if index < 0 or index >= len(self.state):
      raise RunError(f'Invalid index: {index}')
    return self.state[index]

  def __eq__(self, other):
    return isinstance(other, ProgramState) and self.state == other.state

  def copy(self):
    return ProgramState(list(self.state))

  def add_result(self, result):
    self.state.append(result)

  def get_output(self):
    return self.state[-1]

  def tokenize(self):
    lines = [[variable_token(i), '='] + tokenize_result(result)
             for i, result in enumerate(self.state)]
    return join_token_lists(lines, separator_token='|')

  def __str__(self):
    return ' '.join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens):
    return cls.from_str(' '.join(tokens))

  @classmethod
  def from_str(cls, string):
    """Creates a ProgramState from its string representation."""
    lines = [line.strip() for line in string.split('|')]
    state = []
    for i, line in enumerate(lines):
      splitted = line.split('=')
      if len(splitted) != 2:
        raise ParseError(f"Expected exactly one '=': {line}")
      lhs, rhs = [part.strip() for part in splitted]
      if lhs != variable_token(i):
        raise ParseError(f'Found {lhs} but expected {variable_token(i)}')
      result = ast.literal_eval(rhs)
      if not validate_result(result):
        raise ParseError(f'Found invalid result: {result}')
      state.append(result)
    return cls(state)


class Function(object):
  """Base class for functionality in the DeepCoder DSL."""

  def __init__(self, token, func,
               inputs_type, output_type):
    self.token = token
    self.func = func
    self.inputs_type = inputs_type
    self.output_type = output_type
    self.arity = len(inputs_type)


class Lambda(Function):
  """A lambda function like `*2` or `+`."""


class Operation(Function):
  """Base class for first-order and higher-order operations."""

  def run(self, inputs):
    """Runs an operation on input arguments."""
    if len(inputs) != self.arity:
      raise RunError(f'Arity was {len(inputs)} but expected {self.arity}')
    try:
      result = self.func(*inputs)
    except TypeError as e:
      raise RunError(
          f'Encountered TypeError when running {self.func} on {inputs}') from e
    result = mod_result(result)
    if not validate_result(result):
      return None
    return result


class FirstOrderOperation(Operation):
  """A first-order function like `Head` or `Reverse`."""


class HigherOrderOperation(Operation):
  """A higher-order function like `Map` or `ZipWith`."""


class Statement(object):
  """One statement in a program."""

  def __init__(self, variable_index, operation,
               args):
    self.variable_index = variable_index
    self.operation = operation
    self.args = args

  def run(self, initial_state):
    """Runs the operation and assigns it to a variable."""
    if self.variable_index != len(initial_state):
      raise RunError(
          f'Statement has variable_index {self.variable_index} and cannot be '
          f'run on an initial state of length {len(initial_state)}')
    arg_values = []
    for arg in self.args:
      if isinstance(arg, int):
        arg_values.append(initial_state[arg])
      elif isinstance(arg, Lambda):
        arg_values.append(arg.func)
      else:
        raise DeepCoderError(f'Unhandled argument: {arg}')
    result = self.operation.run(arg_values)
    if result is None:
      return None
    result_state = initial_state.copy()
    result_state.add_result(result)
    return result_state

  def tokenize(self):
    tokens = [variable_token(self.variable_index), '=', self.operation.token]
    for arg in self.args:
      if isinstance(arg, int):
        tokens.append(variable_token(arg))
      elif isinstance(arg, Lambda):
        tokens.append(arg.token)
      else:
        raise DeepCoderError(f'Unhandled argument: {arg}')
    return tokens

  def __str__(self):
    return ' '.join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens):
    """Parses a Statement from a list of tokens."""
    # Parse LHS variable, =, and the operation.
    if len(tokens) < 4:
      raise ParseError(f'Too few tokens: {tokens}')
    variable_index = variable_index_from_token(tokens[0])
    if tokens[1] != '=':
      raise ParseError(f"Second token must be '=': {tokens}")
    operation_token = tokens[2]
    if operation_token not in TOKEN_TO_OPERATION:
      raise ParseError(f'Unknown operation {operation_token} in: {tokens}')
    operation = TOKEN_TO_OPERATION[operation_token]

    # Parse operation arguments.
    args = []
    for i, token in enumerate(tokens[3:]):
      if token in TOKEN_TO_LAMBDA:
        if isinstance(operation, FirstOrderOperation) or i > 0:
          raise ParseError(f'Did not expect lambda at token {i}: {tokens}')
        arg = TOKEN_TO_LAMBDA[token]
      else:
        if isinstance(operation, HigherOrderOperation) and i == 0:
          raise ParseError(f'Expected lambda for token {i}: {tokens}')
        arg = variable_index_from_token(token)
      args.append(arg)
    if len(args) != operation.arity:
      raise ParseError(
          f'Statement tokens have wrong arity for operation: {tokens}')
    return cls(variable_index, operation, args)

  @classmethod
  def from_str(cls, string):
    return cls.from_tokens(string.split(' '))


class Program(object):
  """A full DeepCoder program including input handling."""

  def __init__(self, num_inputs, statements):
    self.num_inputs = num_inputs
    self.statements = statements

  def run(self, inputs):
    if len(inputs) != self.num_inputs:
      raise RunError(f'Got {len(inputs)} inputs but expected {self.num_inputs}')
    state = ProgramState(inputs)
    for statement in self.statements:
      state = statement.run(state)
      if state is None:
        return None
    return state

  def tokenize(self):
    lines = []
    for i in range(self.num_inputs):
      lines.append([variable_token(i), '=', 'INPUT'])
    for statement in self.statements:
      lines.append(statement.tokenize())
    return join_token_lists(lines, separator_token='|')

  def __str__(self):
    return ' '.join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens):
    """Parses a Program from a list of tokens."""
    # Split tokens into lines at '|'.
    lines = []
    current_line = []
    for token in tokens:
      if token == '|':
        lines.append(current_line)
        current_line = []
      else:
        current_line.append(token)
    if current_line:
      lines.append(current_line)

    # Parse statements.
    num_inputs = 0
    statements = []
    for i, line in enumerate(lines):
      if line[-1] == 'INPUT':
        if i != num_inputs:
          raise ParseError(f'Misplaced INPUT on line {i}: {lines}')
        if line != [variable_token(i), '=', 'INPUT']:
          raise ParseError(f'Error while parsing INPUT statement: {line}')
        num_inputs += 1
      else:
        statements.append(Statement.from_tokens(line))
    return cls(num_inputs, statements)

  @classmethod
  def from_str(cls, string):
    return cls.from_tokens(string.split(' '))


def _scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys

# Use the Python code from Appendix F in the DeepCoder paper.
# pylint: disable=g-explicit-length-test, unnecessary-lambda
LAMBDAS = [
    Lambda('+1', lambda x: x + 1, [int], int),
    Lambda('-1', lambda x: x - 1, [int], int),
    Lambda('*2', lambda x: x * 2, [int], int),
    Lambda('/2', lambda x: x // 2, [int], int),
    Lambda('*(-1)', lambda x: -x, [int], int),
    Lambda('**2', lambda x: x ** 2, [int], int),
    Lambda('*3', lambda x: x * 3, [int], int),
    Lambda('/3', lambda x: x // 3, [int], int),
    Lambda('*4', lambda x: x * 4, [int], int),
    Lambda('/4', lambda x: x // 4, [int], int),
    Lambda('>0', lambda x: x > 0, [int], bool),
    Lambda('<0', lambda x: x < 0, [int], bool),
    Lambda('even', lambda x: x % 2 == 0, [int], bool),
    Lambda('odd', lambda x: x % 2 == 1, [int], bool),
    Lambda('+', lambda x, y: x + y, [int, int], int),
    Lambda('-', lambda x, y: x - y, [int, int], int),
    Lambda('*', lambda x, y: x * y, [int, int], int),
    Lambda('min', lambda x, y: min(x, y), [int, int], int),
    Lambda('max', lambda x, y: max(x, y), [int, int], int),
]

OPERATIONS = [
    FirstOrderOperation(
        'Head', lambda xs: xs[0] if len(xs) > 0 else None, [list], int),
    FirstOrderOperation(
        'Last', lambda xs: xs[-1] if len(xs) > 0 else None, [list], int),
    FirstOrderOperation(
        'Take', lambda n, xs: xs[:n], [int, list], list),
    FirstOrderOperation(
        'Drop', lambda n, xs: xs[n:], [int, list], list),
    FirstOrderOperation(
        'Access', lambda n, xs: xs[n] if 0 <= n < len(xs) else None,
        [int, list], int),
    FirstOrderOperation(
        'Minimum', lambda xs: min(xs) if len(xs) > 0 else None, [list], int),
    FirstOrderOperation(
        'Maximum', lambda xs: max(xs) if len(xs) > 0 else None, [list], int),
    FirstOrderOperation(
        'Reverse', lambda xs: list(reversed(xs)), [list], list),
    FirstOrderOperation(
        'Sort', lambda xs: sorted(xs), [list], list),
    FirstOrderOperation(
        'Sum', lambda xs: sum(xs), [list], int),

    HigherOrderOperation(
        'Map', lambda f, xs: [f(x) for x in xs], [([int], int), list], list),
    HigherOrderOperation(
        'Filter', lambda f, xs: [x for x in xs if f(x)],
        [([int], bool), list], list),
    HigherOrderOperation(
        'Count', lambda f, xs: len([x for x in xs if f(x)]),
        [([int], bool), list], int),
    HigherOrderOperation(
        'ZipWith', lambda f, xs, ys: [f(x, y) for (x, y) in zip(xs, ys)],
        [([int, int], int), list, list], list),
    HigherOrderOperation(
        'Scanl1', _scanl1, [([int, int], int), list], list),
]
# pylint: enable=g-explicit-length-test, unnecessary-lambda

TOKEN_TO_LAMBDA = {l.token: l for l in LAMBDAS}
TOKEN_TO_OPERATION = {op.token: op for op in OPERATIONS}
