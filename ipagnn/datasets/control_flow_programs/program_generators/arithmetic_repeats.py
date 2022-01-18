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

"""Generating and running arithmetic programs with repeat statements.

We use a list of statements to represent a program. Each statement is a list of
an operator and two operands. The standard ops in a program are +, -, *, and
a special "repeat" op ("r") that acts as a repeat block in the program.

The +, -, and * ops update a variable by modifying it. The first operand
indicates which variable is being updated. The second operand indicates
by how much to modify the variable.

In the repeat op, the first operand indicates the number of repetitions and the
second op indicates how many statements to repeat.
"""

import random

from ipagnn.datasets.control_flow_programs.program_generators import constants

REPEAT_OP = "r"
ASSIGN_OP = "="
PLACEHOLDER_OP = "_"


def generate_python_source(length, config):
  """Generates Python code according to the config."""
  num_variables = 10  # TODO(dbieber): Number of variables is hard coded.
  statements, unused_hole_statement_index = _generate_statements(length, config)
  return _to_python_source(statements, num_variables)


def generate_python_source_and_partial_python_source(length, config):
  """Generates Python code according to the config."""
  num_variables = 10  # TODO(dbieber): Number of variables is hard coded.
  statements, hole_statement_index = _generate_statements(length, config)
  partial_statements = statements.copy()
  partial_statements[hole_statement_index] = _placeholder_statement()
  return (_to_python_source(statements, num_variables),
          _to_python_source(partial_statements, num_variables))


def _placeholder_statement():
  return (PLACEHOLDER_OP, 0, 0)


def _generate_statements(length, config):
  """Generates a list of statements representing a control flow program.

  Args:
    length: The number of statements to generate.
    config: The ArithmeticRepeatsConfig specifying the properties of the program
      to generate.
  Returns:
    A list of statements, each statement being a 3-tuple (op, operand, operand).
  """
  max_value = config.base ** config.num_digits - 1

  statements = []
  nesting_lines_remaining = []
  num_repeats = 0
  hole_candidates = []
  if config.start_with_initialization:
    start_value = random.randint(0, max_value)
    statements.append((ASSIGN_OP, 0, start_value))
    length -= 1  # First line is initialization

  for statement_index in range(length):
    current_nesting = len(nesting_lines_remaining)
    repeat_permitted = not (
        # Too much nesting already.
        (not config.permit_nested_repeats and current_nesting > 0)
        # Too many repeats already.
        or num_repeats >= config.max_repeat_statements
        or statement_index == length - 1  # Last line of program.
        or 1 in nesting_lines_remaining  # Last line of another repeat.
    )
    is_repeat = repeat_permitted and (
        random.random() < config.repeat_probability)
    if is_repeat:
      num_repeats += 1
      repetitions = random.randint(2, config.max_repetitions)
      # statements_remaining* and num_statements includes current statement.
      statements_remaining_in_program = length - statement_index
      statements_remaining_in_block = min(
          [statements_remaining_in_program] + nesting_lines_remaining)
      if config.max_repeat_block_size:
        max_repeat_block_size = min(config.max_repeat_block_size,
                                    statements_remaining_in_block)
      else:
        max_repeat_block_size = statements_remaining_in_block
      num_statements = random.randint(2, max_repeat_block_size)
      nesting_lines_remaining.append(num_statements)
      # -1 is to not include current statement.
      statement = (REPEAT_OP, repetitions, num_statements - 1)
    else:
      op = random.choice(config.ops)
      variable_index = 0  # "v0"
      operand = random.randint(0, max_value)
      statement = (op, variable_index, operand)
      hole_candidates.append(statement_index)
    statements.append(statement)

    # Decrement nesting.
    for nesting_index in range(len(nesting_lines_remaining)):
      nesting_lines_remaining[nesting_index] -= 1
    while nesting_lines_remaining and nesting_lines_remaining[-1] == 0:
      nesting_lines_remaining.pop()
    assert 0 not in nesting_lines_remaining

  hole_statement_index = random.choice(hole_candidates)

  return statements, hole_statement_index


def _select_counter_variable(used_variables, num_variables):
  # TODO(dbieber): num_variables is hardcoded.
  max_counter_variable = num_variables - 1
  allowed_variables = (
      set(range(1, max_counter_variable + 1)) - set(used_variables))
  return random.choice(list(allowed_variables))


def _to_python_source(statements, num_variables):
  """Convert statements into Python source code.

  Repeat statements are rendered as while loops with a counter variable that
  tracks the number of iterations remaining.

  Args:
    statements: A list of statements. Each statement is a triple containing
        (op, operand, operand).
    num_variables: The number of variables permitted.
  Returns:
    Python source code representing the program.
  """
  lines = []
  nesting_lines_remaining = []
  used_variables = []
  for statement in statements:
    op, operand1, operand2 = statement
    indent = constants.INDENT_STRING * len(nesting_lines_remaining)
    if op is REPEAT_OP:
      repetitions, num_statements = operand1, operand2
      variable_index = _select_counter_variable(used_variables, num_variables)
      line1 = f"{indent}v{variable_index} = {repetitions}"
      line2 = f"{indent}while v{variable_index} > 0:"
      # +1 is for current statement.
      nesting_lines_remaining.append(num_statements + 1)
      used_variables.append(variable_index)
      line3_indent = constants.INDENT_STRING * len(nesting_lines_remaining)
      line3 = f"{line3_indent}v{variable_index} -= 1"
      lines.extend([line1, line2, line3])
    elif op is PLACEHOLDER_OP:
      lines.append(f"{indent}_ = 0")
    else:
      variable_index, operand = operand1, operand2
      line = f"{indent}v{variable_index} {op} {operand}"
      lines.append(line)

    # Decrement nesting.
    for nesting_index in range(len(nesting_lines_remaining)):
      nesting_lines_remaining[nesting_index] -= 1
    while nesting_lines_remaining and nesting_lines_remaining[-1] == 0:
      nesting_lines_remaining.pop()
      used_variables.pop()

  return "\n".join(lines)
