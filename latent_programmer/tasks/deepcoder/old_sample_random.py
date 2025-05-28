# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Generates random programs in Deepcoder DSL."""

import collections
import functools
import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module

# Multiple inputs for one example.
InputsList = List[Union[int, List[int]]]

# These are potential ranges to draw integers from, to fill a random input list.
# For a particular input, all examples will use the same range. These ranges
# include both endpoints.
# (Integer inputs don't use this. Instead they are constrained to
# [0, max_list_length] because int inputs are only useful as arguments to Take,
# Drop, and Access.)
LIST_INT_RANGES = [
    (0, 4),
    (0, 9),
    (-10, 10),
    (-50, 50),
]

# We need short lists with small numbers to use Scanl1 (*).
# Note that the result must only contain integers in range, for all examples,
# which is unlikely unless we choose short lists with small-magnitude integers.
LENGTH_RANGES = [
    (1, 3),
    (0, 5),
    (0, 10),
    (0, 20),
]


@functools.cache
def _get_length_range_options(upper_bound):
  """Choose options for list length range, respecting a length upper bound."""
  length_range_options = [
      (min_length, max_length)
      for min_length, max_length in LENGTH_RANGES
      if max_length <= upper_bound]
  if (0, upper_bound) not in length_range_options:
    length_range_options.append((0, upper_bound))
  return length_range_options


def random_inputs(num_examples, num_inputs,
                  types = None):
  """Randomly sample inputs for multiple examples."""
  if types is None:
    # At least one of inputs has to be a list.
    types = [list] + random.choices([int, list], k=num_inputs - 1)
    random.shuffle(types)
  elif len(types) != num_inputs:
    raise ValueError(f'Length of `types` ({len(types)}) should match '
                     f'`num_inputs` ({num_inputs}).')

  # Choose some constraints on input lists that will be used for all examples.
  # Ranges are inclusive.
  int_range_options = [
      (x, y) for x, y in LIST_INT_RANGES
      if dsl.deepcoder_min_int() < x and y < dsl.deepcoder_max_int()
  ]
  int_range_options.append((dsl.deepcoder_min_int(), dsl.deepcoder_max_int()))
  list_int_ranges = random.choices(int_range_options, k=num_inputs)
  length_range_options = _get_length_range_options(
      dsl.deepcoder_max_list_length())
  length_ranges = random.choices(length_range_options, k=num_inputs)

  # Create input lists for each example, making sure no two examples have
  # identical input lists.
  all_examples = []
  while len(all_examples) < num_examples:
    inputs_list = []
    for i, type_, in enumerate(types):
      if type_ == int:
        # Integer inputs are constrained to the range [0, max_list_length]
        # because int inputs are only useful as arguments to Take, Drop, and
        # Access. No other operation takes an integer input.
        max_int = dsl.deepcoder_max_list_length()
        random_input = random.randint(0, max_int)
      else:
        assert type_ == list
        min_int, max_int = list_int_ranges[i]
        min_length, max_length = length_ranges[i]
        random_input = random.choices(range(min_int, max_int + 1),
                                      k=random.randint(min_length, max_length))
      inputs_list.append(random_input)
    if inputs_list not in all_examples:
      all_examples.append(inputs_list)

  return all_examples


def random_new_variable(existing_variables,
                        ordered):
  """Returns a new variable token different from those in existing_variables."""
  if ordered:
    for i in range(dsl.MAX_NUM_VARIABLES):
      v = dsl.variable_token(i)
      if v not in existing_variables:
        return v
    raise ValueError('All variables were already used.')
  else:
    choices = sorted(list(dsl.ALL_VARIABLES - set(existing_variables)))
    if not choices:
      raise ValueError('All variables were already used.')
    return random.choice(choices)


def is_valid_operation(operation,
                       var_dict):
  """Checks if the operation is usable given the current variable types."""
  # First index is for Lambda if operation is higher-order.
  if isinstance(operation, dsl.FirstOrderOperation):
    start_idx = 0
  elif isinstance(operation, dsl.HigherOrderOperation):
    start_idx = 1
  else:
    raise dsl.DeepCoderError(f'Unhandled operation type: {type(operation)}')
  return all(var_dict[t] for t in operation.inputs_type[start_idx:])


def random_statement(program_state,
                     unused_variables,
                     is_train,
                     canonical_variable_order,
                     experiment,
                     operations,
                     lambdas,
                     last_statement = False):
  """Randomly sample new Statement given existing ProgramState."""
  variable = random_new_variable(program_state.variables,
                                 ordered=canonical_variable_order)

  # Maps from type to a list of variable names of that type.
  var_dict = collections.defaultdict(list)
  for i, v in enumerate(program_state.state):
    assert type(v) in (int, list)
    var_dict[type(v)].append(program_state.variables[i])

  valid_operations = [op for op in operations if
                      is_valid_operation(op, var_dict)]

  # If it's the last statement: force using all unused variables
  # If there is 1 unused variable:
  #   * 50% probability: force using that variable if possible
  #   * otherwise: do whatever
  # If there are 2 unused variables:
  #   * 50% probability: force using both of them if possible
  #   * otherwise: force using one of them if possible
  required_variables = []
  unused_variables_copy = list(unused_variables)
  if last_statement:
    # This could have length > 2, if it wasn't possible to use enough variables
    # in previous statements.
    required_variables = unused_variables_copy
  elif len(unused_variables) == 1:
    if random.random() < 0.5:
      required_variables = unused_variables_copy
  elif len(unused_variables) >= 2:
    required_variables = random.sample(unused_variables_copy,
                                       k=random.choice([1, 2]))

  # Find operations which make it possible to use the required variables.
  # In any situation where it isn't possible, try using one fewer variable,
  # until it becomes possible.
  while True:
    required_variable_types = [type(program_state.get_variable(v))
                               for v in required_variables]
    new_valid_operations = [
        op for op in valid_operations
        if all(required_variable_types.count(t) <= op.inputs_type.count(t)
               for t in required_variable_types)]
    if new_valid_operations:
      break
    else:
      required_variables.pop(random.choice(range(len(required_variables))))

  # Create a random statement, initially without constraints on which variables
  # are used.
  random_op = random.choice(new_valid_operations)
  args = []
  for t in random_op.inputs_type:
    if isinstance(t, tuple):  # Argument is a lambda.
      if (experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY and
          is_train and random_op.token == 'Scanl1'):
        valid_lambdas = dsl.LAMBDAS_ONLY_MINUS_MIN
      else:
        valid_lambdas = [
            l for l in lambdas if (l.inputs_type, l.output_type) == t]
      random_lambda = random.choice(valid_lambdas)
      args.append(random_lambda)
    else:  # Argument is either an int or list.
      random_arg = random.choice(var_dict[t])
      args.append(random_arg)

  # Override arguments to use required variables.
  used_indices = set()
  for required_variable, required_type in zip(required_variables,
                                              required_variable_types):
    required_index = random.choice([
        arg_index for arg_index, arg_type in enumerate(random_op.inputs_type)
        if arg_type == required_type and arg_index not in used_indices
    ])
    args[required_index] = required_variable
    used_indices.add(required_index)

  return dsl.Statement(variable, random_op, args)


def is_redundant(states):
  """Checks if the last result equals the same previous result for all examples."""
  common = set(range(len(states[0]) - 1))
  for program_state in states:
    latest_result = program_state.get_output()
    common_i = [i for i in range(len(program_state) - 1)
                if program_state.get_index(i) == latest_result]
    common = common.intersection(common_i)
    if not common:
      return False
  return bool(common)


def has_dead_code(program):
  """Checks whether a program has dead code, ignoring dead inputs."""
  # If a variable appears at all in a program, it must appear exactly once as
  # the LHS of an assignment. A variable is "dead" if it is never used again,
  # with the exception of the final "output" variable.

  # Dead inputs are ok, since the model can learn to ignore it, and it doesn't
  # cause the model to *learn* to predict dead code.

  program_tokens = program.tokenize()
  for statement in program.statements[:-1]:  # Ignoring the output statement.
    if program_tokens.count(statement.variable) == 1:
      return True
  return False


def has_duplicate_output(program,
                         example_inputs):
  """Checks if the program always produces the same output across examples."""
  states = [program.run(inputs) for inputs in example_inputs]
  outputs = [str(state.get_output()) for state in states if state is not None]
  if len(outputs) != len(example_inputs):
    raise dsl.DeepCoderError('Random program should execute on all inputs.')
  return len(set(outputs)) < len(outputs)


def has_constant_output(program,
                        example_inputs):
  """Checks if the program always produces the same output across examples."""
  if len(example_inputs) == 1:
    # If there's only one example, the output will always be "constant" but
    # that's ok.
    return False
  states = [program.run(inputs) for inputs in example_inputs]
  outputs = [state.get_output() for state in states if state is not None]
  if len(outputs) != len(example_inputs):
    raise dsl.DeepCoderError('Random program should execute on all inputs.')
  return all(output == outputs[0] for output in outputs)


def _sample_program(
    example_inputs,
    num_statements,
    is_train,
    canonical_variable_order,
    experiment,
    operations,
    lambdas,
    reject_redundant_code,
    input_variables,
):
  """Sample a new program that could be rejected later."""
  states = [dsl.ProgramState(inputs, input_variables)
            for inputs in example_inputs]
  statements = []
  unused_variables = []  # Track which variables have not been used yet.
  for i in range(num_statements):
    if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER:
      assert num_statements >= 2
      if is_train:
        operations = (dsl.HIGHER_ORDER_OPERATIONS if i < num_statements // 2
                      else dsl.FIRST_ORDER_OPERATIONS)
      else:
        operations = (dsl.FIRST_ORDER_OPERATIONS if i < num_statements // 2
                      else dsl.HIGHER_ORDER_OPERATIONS)

    # Sample a statement that executes successfully for all examples.
    num_statement_attempts = 0
    while True:
      if num_statement_attempts > 20:
        return None  # Reject the entire program.
      num_statement_attempts += 1
      statement = random_statement(
          states[0], unused_variables, is_train, canonical_variable_order,
          experiment=experiment,
          operations=operations,
          lambdas=lambdas,
          last_statement=len(statements) + 1 == num_statements)
      next_states = [statement.run(state) for state in states]
      if not all(next_states):
        continue  # The statement failed to run for at least 1 example.
      if reject_redundant_code and is_redundant(next_states):
        continue
      break
    statements.append(statement)
    states = next_states
    # Update unused variables.
    unused_variables.append(statement.variable)
    for arg in statement.args:
      if isinstance(arg, str) and arg in unused_variables:
        unused_variables.remove(arg)

  program = dsl.Program(input_variables, statements)
  assert len(program) == num_statements
  return program


def random_task(
    num_examples,
    num_inputs,
    num_statements,
    is_train,
    canonical_variable_order,
    experiment = None,
    operations = None,
    lambdas = None,
    reject_dead_code = True,
    reject_redundant_code = True,
    reject_duplicate_output = True,
    reject_constant_output = True):
  """Randomly sample a new program."""
  if operations is None:
    operations = dsl.OPERATIONS.copy()
  if lambdas is None:
    lambdas = dsl.LAMBDAS.copy()

  input_variables = []
  while len(input_variables) < num_inputs:
    # During training, generate unordered variables to teach the model about all
    # variable names. During test, generate variables in order (x0, x1, ...) to
    # simplify how statements combine to form programs in the end-to-end loop
    # (individual models only predict the RHS of statements, so we need to add
    # a variable name ourselves).
    input_variables.append(
        random_new_variable(input_variables, ordered=canonical_variable_order))

  while True:
    example_inputs = random_inputs(num_examples=num_examples,
                                   num_inputs=num_inputs)
    program = _sample_program(
        example_inputs=example_inputs,
        num_statements=num_statements,
        is_train=is_train,
        canonical_variable_order=canonical_variable_order,
        experiment=experiment,
        operations=operations,
        lambdas=lambdas,
        reject_redundant_code=reject_redundant_code,
        input_variables=input_variables,
    )

    should_reject_program = (
        program is None or
        (reject_dead_code and has_dead_code(program)) or
        (reject_duplicate_output and has_duplicate_output(program,
                                                          example_inputs)) or
        (reject_constant_output and has_constant_output(program,
                                                        example_inputs)))
    if not should_reject_program:
      break

  # `program` can't be None here.
  example_outputs = [
      program.run(inputs).get_output()  # pytype: disable=attribute-error
      for inputs in example_inputs]
  examples = [dsl.Example(inputs, output)
              for inputs, output in zip(example_inputs, example_outputs)]
  return dsl.ProgramTask(program, examples)
