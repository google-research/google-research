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
import numpy as np
import string
from typing import Dict, List, Optional, Tuple

from absl import logging

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl


def random_int():
  if dsl._DEEPCODER_MOD == 0:
    min_int, max_int = dsl.MIN_INT, dsl.MAX_INT
  else:
    min_int, max_int = 0, dsl._DEEPCODER_MOD - 1
  return random.randint(min_int, max_int)


def random_list():
  if dsl._DEEPCODER_MOD == 0:
    min_int, max_int = dsl.MIN_INT, dsl.MAX_INT
  else:
    min_int, max_int = 0, dsl._DEEPCODER_MOD - 1
  random_length = np.random.randint(1, dsl.MAX_LENGTH)  # Random length.
  return random.sample(range(min_int, max_int + 1), k=random_length)


def random_inputs(num_inputs):
  """Randomly sample inputs."""
  inputs = []
  # At least one of inputs has to be a list.
  inputs.append(random_list())
  for _ in range(num_inputs - 1):
    if np.random.rand() < 0.5:
      inputs.append(random_int())
    else:
      inputs.append(random_list())
  random.shuffle(inputs)
  return inputs


def random_inputs_like(inputs):
  inputs = []
  for inp in inputs:
    if isinstance(inp, list):
      inputs.append(random_list())
    else:
      assert type(inp) == int
      inputs.append(random_int())
  return inputs


def is_valid_operation(operation, var_dict):
  """Check is operation is valid given the current variable types."""  
  # First index is for Lambda if operation is higher-order
  start_idx = (
      0 if isinstance(operation, dsl.FirstOrderOperation) else 1)
  return np.all(
      [len(var_dict[t]) > 0 for t in operation.inputs_type[start_idx:]])


def random_statement(program_state, operations, lambdas):
  """Randomly sample new Statement given existing ProgramState."""
  idx = len(program_state.state)  # Variable index should be line of program. 

  var_dict = defaultdict(list)
  for i, v in enumerate(program_state.state):
    if isinstance(v, list):
      var_dict[list].append(i)
    else:
      assert type(v) == int
      var_dict[int].append(i)
  
  valid_operations = [op for op in operations if 
                      is_valid_operation(op, var_dict)]
  random_op = random.choice(valid_operations)

  args = []
  for t in random_op.inputs_type:
    if isinstance(t, tuple):
      # Argument is a lambda.
      valid_lambdas = [
          l for l in lambdas if (l.inputs_type, l.output_type) == t]
      random_lambda = random.choice(valid_lambdas)
      args.append(random_lambda)
    else:
      # Argument is either an int or list.
      weights = (np.array(var_dict[t]) + 1)  # More likely to sample recent.
      args.append(random.choices(var_dict[t], weights=weights)[0])
  return dsl.Statement(idx, random_op, args)


def random_program(
    all_inputs, num_statements, operations=None, lambdas=None):
  """Randomly sample a new program."""
  if operations is None:
    operations = dsl.OPERATIONS
  if lambdas is None:
    lambas = dsl.LAMBAS

  states = [dsl.ProgramState(inputs) for inputs in all_inputs]
  statements = []
  for _ in range(num_statements):
    while True:
      statement = random_statement(states[0], operations=operations, lambdas=lambdas)
      next_states = [statement.run(state) for state in states]
      if all([next_state is not None for next_state in next_states]):
        break
    statements.append(statement)
    states = next_states
  return dsl.Program(len(all_inputs[0]), statements)


# TODO(jxihong): Can be collapsed into random_statement() function.
def random_statement_extend_op_functionality(program_state):
  """Randomly sample new Statement given existing ProgramState."""
  idx = len(program_state.state)  # Variable index should be line of program. 

  var_dict = defaultdict(list)
  for i, v in enumerate(program_state.state):
    if isinstance(v, list):
      var_dict[list].append(i)
    else:
      assert type(v) == int
      var_dict[int].append(i)
  
  valid_operations = [op for op in dsl.OPERATIONS if 
                      is_valid_operation(op, var_dict)]
  random_op = random.choice(valid_operations)

  args = []
  for t in random_op.inputs_type:
    if isinstance(t, tuple):
      # Argument is a lambda.
      if random_op.token == 'Scan1':
        valid_lambdas = [
            l for l in dsl.LAMBDAS_ONLY_MINUS_MIN if (l.inputs_type, l.output_type) == t]
      else:
        valid_lambdas = [
            l for l in dsl.LAMBDAS if (l.inputs_type, l.output_type) == t]
      random_lambda = random.choice(valid_lambdas)
      args.append(random_lambda)
    else:
      # Argument is either an int or list.
      weights = (np.array(var_dict[t]) + 1)  # More likely to sample recent.
      args.append(random.choices(var_dict[t], weights=weights)[0])
  return dsl.Statement(idx, random_op, args)


# TODO(jxihong): Can be collapsed into random_program() function.
def random_program_extend_op_functionality(all_inputs, num_statements):
  states = [dsl.ProgramState(inputs) for inputs in all_inputs]
  statements = []
  for _ in range(num_statements):
    while True:
      statement = random_statement_extend_op_functionality(states[0])
      next_states = [statement.run(state) for state in states]
      if all([next_state is not None for next_state in next_states]):
        break
    statements.append(statement)
    states = next_states
  return dsl.Program(len(all_inputs[0]), statements)


# TODO(jxihong): Can be collapsed into random_program() function.
def random_program_switch_concept_order(inputs, num_statements, is_train=True):
  states = [dsl.ProgramState(inputs) for inputs in all_inputs]
  statements = []
  for i in range(num_statements):
    if is_train:
      operations = (dsl.FIRST_ORDER_OPERATIONS if i < num_statements // 2 else
                    dsl.HIGHER_ORDER_OPERATIONS)
    else:
      operations = (dsl.FIRST_ORDER_OPERATIONS if i >= num_statements // 2 else
                    dsl.HIGHER_ORDER_OPERATIONS)

    while True:
      statement = random_statement_extend_op_functionality(states[0])
      next_states = [statement.run(state) for state in states]
      if all([next_state is not None for next_state in next_states]):
        break
    statements.append(statement)
    states = next_states
  return dsl.Program(len(all_inputs[0]), statements)
