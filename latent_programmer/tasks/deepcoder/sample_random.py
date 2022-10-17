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
import string
from typing import Dict, List, Optional, Tuple

from absl import logging

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl


def random_inputs(num_inputs):
  """Randomly sample inputs."""
  inputs = []
  # At least one of inputs has to be a list.
  random_length = np.random.randint(1, 20)  # Random length.
  inputs.append(
      np.random.randint(dsl.MIN_INT, dsl.MAX_INT + 1, size=random_length).tolist())
  for _ in range(num_inputs - 1):
    if np.random.rand() < 0.5:
      inputs.append(np.random.randint(dsl.MIN_INT, dsl.MAX_INT + 1))
    else:
      random_length = np.random.randint(1, dsl.MAX_LENGTH)
      inputs.append(
          np.random.randint(dsl.MIN_INT, dsl.MAX_INT + 1, size=random_length).tolist())
  return inputs


def random_inputs_like(inputs):
  inputs = []
  for inp in inputs:
    if isinstance(inp, list):
      random_length = np.random.randint(1, 20)  # Random length.
      inputs.append(
          np.random.randint(dsl.MIN_INT, dsl.MAX_INT + 1, size=random_length).tolist())
    else:
      assert isinstance(inp, int)
      random_length = np.random.randint(1, dsl.MAX_LENGTH)
      inputs.append(
          np.random.randint(dsl.MIN_INT, dsl.MAX_INT + 1, size=random_length).tolist())
  return inputs


def is_valid_operation(operation, var_dict):
  """Check is operation is valid given the current variable types."""  
  # First index is for Lambda if operation is higher-order
  start_idx = (
      0 if isinstance(operation, dsl.FirstOrderOperation) else 1)
  return np.all(
      [len(var_dict[str(t)]) > 0 for t in operation.inputs_type[start_idx:]])


def random_statement(
    program_state, operations=dsl.OPERATIONS, lambdas=dsl.LAMBDAS):
  """Randomly sample new Statement given existing ProgramState."""
  idx = len(program_state.state)  # Variable index should be line of program. 

  var_dict = defaultdict(list)
  for i, v in enumerate(program_state.state):
    if isinstance(v, list):
      var_dict[str(list)].append(i)
    else:
      assert isinstance(v, int)
      var_dict[str(int)].append(i)
  
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
      weights = (np.array(var_dict[str(t)]) + 1)  # More likely to sample recent.
      args.append(random.choices(var_dict[str(t)], weights=weights)[0])
  return dsl.Statement(idx, random_op, args)


def random_program(
    inputs, num_statements, operations=dsl.OPERATIONS, lambdas=dsl.LAMBDAS):
  """Randomly sample a new program."""
  state = dsl.ProgramState(inputs)
  statements = []
  for _ in range(num_statements):
    statement = random_statement(state, operations=operations, lambdas=lambdas)
    next_state = statement.run(state)
    # Make sure statement performs a valid computation (no empty lists, etc.). 
    while not isinstance(
        next_state.state[-1], statement.operation.output_type):
      statement = random_statement(state)
      next_state =  statement.run(state)
    statements.append(statement)
    state = next_state
  return dsl.Program(len(inputs), statements)


def random_statement_extend_op_functionality(program_state):
  """Randomly sample new Statement given existing ProgramState."""
  idx = len(program_state.state)  # Variable index should be line of program. 

  var_dict = defaultdict(list)
  for i, v in enumerate(program_state.state):
    if isinstance(v, list):
      var_dict[str(list)].append(i)
    else:
      assert isinstance(v, int)
      var_dict[str(int)].append(i)
  
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
      weights = (np.array(var_dict[str(t)]) + 1)  # More likely to sample recent.
      args.append(random.choices(var_dict[str(t)], weights=weights)[0])
  return dsl.Statement(idx, random_op, args)


def random_program_extend_op_functionality(inputs, num_statements):
  state = dsl.ProgramState(inputs)
  statements = []
  for _ in range(num_statements):
    statement = random_statement_extend_op_functionality(state)
    next_state = statement.run(state)
    # Make sure statement performs a valid computation (no empty lists, etc.). 
    while not isinstance(
        next_state.state[-1], statement.operation.output_type):
      statement = random_statement(state)
      next_state =  statement.run(state)
    statements.append(statement)
    state = next_state
  return dsl.Program(len(inputs), statements)


def random_program_switch_concept_order(inputs, num_statements, is_train=True):
  state = dsl.ProgramState(inputs)
  statements = []
  random_switch = np.random.randint(1, num_statements - 1)
  for i in range(num_statements):
    if is_train:
      operations = dsl.FIRST_ORDER_OPERATIONS if i < random_switch else dsl.HIGHER_ORDER_OPERATIONS
    else:
      operations = dsl.FIRST_ORDER_OPERATIONS if i >= random_switch else dsl.HIGHER_ORDER_OPERATIONS
      
    statement = random_statement(state, operations=operations, lambdas=dsl.LAMBDAS)
    next_state = statement.run(state)
    # Make sure statement performs a valid computation (no empty lists, etc.). 
    while not isinstance(
        next_state.state[-1], statement.operation.output_type):
      statement = random_statement(state)
      next_state =  statement.run(state)
    statements.append(statement)
    state = next_state
  return dsl.Program(len(inputs), statements)
