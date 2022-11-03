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

"""Generates random programs in Deepcoder DSL."""

import collections
import random
from typing import Any, Dict, List, Optional, Type, Union

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module


def random_int():
  if dsl.deepcoder_mod() > 0:
    min_int, max_int = 0, dsl.deepcoder_mod() - 1
  else:
    min_int, max_int = dsl.MIN_INT, dsl.MAX_INT
  return random.randint(min_int, max_int)


def random_list():
  if dsl.deepcoder_mod() > 0:
    min_int, max_int = 0, dsl.deepcoder_mod() - 1
  else:
    min_int, max_int = dsl.MIN_INT, dsl.MAX_INT
  random_length = random.randint(1, dsl.MAX_LIST_LENGTH)
  return random.choices(range(min_int, max_int + 1), k=random_length)


def random_inputs(num_inputs):
  """Randomly sample inputs."""
  # At least one of inputs has to be a list.
  inputs = [random_list()]
  for _ in range(num_inputs - 1):
    if random.random() < 0.5:
      inputs.append(random_int())
    else:
      inputs.append(random_list())
  random.shuffle(inputs)
  return inputs


def random_inputs_like(
    inputs):
  new_inputs = []
  for inp in inputs:
    if isinstance(inp, list):
      new_inputs.append(random_list())
    else:
      assert type(inp) is int  # pylint: disable=unidiomatic-typecheck
      new_inputs.append(random_int())
  return new_inputs


def random_new_variable(existing_variables):
  return random.choice(list(dsl.ALL_VARIABLES - set(existing_variables)))


def is_valid_operation(operation,
                       var_dict):
  """Check is operation is valid given the current variable types."""
  # First index is for Lambda if operation is higher-order.
  if isinstance(operation, dsl.FirstOrderOperation):
    start_idx = 0
  elif isinstance(operation, dsl.HigherOrderOperation):
    start_idx = 1
  else:
    raise dsl.DeepCoderError(f'Unhandled operation type: {type(operation)}')
  return all(var_dict[t] for t in operation.inputs_type[start_idx:])


def random_statement(program_state,
                     is_train,
                     experiment,
                     operations,
                     lambdas):
  """Randomly sample new Statement given existing ProgramState."""
  variable = random_new_variable(program_state.variables)

  # Maps from type to a list of indices of variables of that type.
  var_dict = collections.defaultdict(list)
  for i, v in enumerate(program_state.state):
    assert type(v) in (int, list)
    var_dict[type(v)].append(i)

  valid_operations = [op for op in operations if
                      is_valid_operation(op, var_dict)]
  random_op = random.choice(valid_operations)

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
      # Make it more likely to sample a recent variable.
      weights = [index + 1 for index in var_dict[t]]
      variable_index = random.choices(var_dict[t], weights=weights)[0]
      args.append(program_state.variables[variable_index])
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


def random_program(
    example_inputs,
    num_statements,
    is_train,
    experiment = None,
    operations = None,
    lambdas = None):
  """Randomly sample a new program."""
  if operations is None:
    operations = dsl.OPERATIONS.copy()
  if lambdas is None:
    lambdas = dsl.LAMBDAS.copy()

  # All examples should have the same number of inputs.
  num_inputs = len(example_inputs[0])
  assert all(len(inputs) == num_inputs for inputs in example_inputs)

  input_variables = []
  while len(input_variables) < num_inputs:
    input_variables.append(random_new_variable(input_variables))

  states = [dsl.ProgramState(inputs, input_variables)
            for inputs in example_inputs]
  statements = []
  for i in range(num_statements):
    if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER:
      if is_train:
        operations = (dsl.FIRST_ORDER_OPERATIONS if i < num_statements // 2
                      else dsl.HIGHER_ORDER_OPERATIONS)
      else:
        operations = (dsl.HIGHER_ORDER_OPERATIONS if i < num_statements // 2
                      else dsl.FIRST_ORDER_OPERATIONS)

    # Sample a statement that executes successfully for all examples.
    while True:
      statement = random_statement(states[0], is_train, experiment=experiment,
                                   operations=operations, lambdas=lambdas)
      next_states = [statement.run(state) for state in states]
      if not all(next_states):
        continue
      elif not is_redundant(next_states):
        break
    statements.append(statement)
    states = next_states

  program = dsl.Program(input_variables, statements)
  assert len(program) == num_statements
  return program
