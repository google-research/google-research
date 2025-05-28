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

"""Samples random DeepCoder programs."""

import collections
import itertools
import random

from absl import logging

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder import old_sample_random


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


def has_constant_output(states):
  """Checks if the program always produces the same output across examples."""
  outputs = [state.get_output() for state in states if state is not None]
  if len(outputs) != len(states):
    raise dsl.DeepCoderError('Random program should execute on all inputs.')
  return all(output == outputs[0] for output in outputs)


def has_duplicate_output(states):
  """Checks if the program always produces the same output across examples."""
  outputs = [str(state.get_output()) for state in states if state is not None]
  if len(outputs) != len(states):
    raise dsl.DeepCoderError('Random program should execute on all inputs.')
  return len(set(outputs)) < len(outputs)


def _get_next_statements(program_state,
                         unused_variables,
                         is_train,
                         canonical_variable_order,
                         experiment=exp_module.Experiment.NONE,
                         operations=dsl.OPERATIONS,
                         lambdas=dsl.LAMBDAS,
                         last_statement=False):
  """Find list of possible statements given existing program state."""
  variable = old_sample_random.random_new_variable(
      program_state.variables, ordered=canonical_variable_order)

  # Maps from type to a list of variable names of that type.
  var_dict = collections.defaultdict(list)
  for i, v in enumerate(program_state.state):
    assert type(v) in (int, list)
    var_dict[type(v)].append(program_state.variables[i])

  valid_operations = [op for op in operations if
                      old_sample_random.is_valid_operation(op, var_dict)]
  random.shuffle(valid_operations)

  statements = []
  for op in valid_operations:
    product_args = []
    for t in op.inputs_type:
      if isinstance(t, tuple):  # Argument is a lambda.
        if (experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY and
            is_train and op.token == 'Scanl1'):
          valid_lambdas = dsl.LAMBDAS_ONLY_MINUS_MIN
        else:
          valid_lambdas = [
              l for l in lambdas if (l.inputs_type, l.output_type) == t]
        product_args.append(valid_lambdas)
      else:  # Argument is either an int or list.
        product_args.append(var_dict[t])
    all_args = itertools.product(*product_args)
    for args in all_args:
      used_variables = [arg for arg in args if isinstance(arg, str)]
      # If it's the last statement, force using all unused variables.
      if last_statement:
        if all(v in used_variables for v in unused_variables):
          statements.append(dsl.Statement(variable, op, args))  # pytype: disable=wrong-arg-types
      # If there are 2 unused variables, force using at least 1 unused variable.
      elif len(unused_variables) == 2:
        if any(v in used_variables for v in unused_variables):
          statements.append(dsl.Statement(variable, op, args))  # pytype: disable=wrong-arg-types
      # If there's only 1 unused variable, allow anything to enable "diamond"
      # computations. Or, on the first statement, all inputs are unused.
      elif len(unused_variables) <= 1:
        statements.append(dsl.Statement(variable, op, args))  # pytype: disable=wrong-arg-types
      else:
        raise dsl.DeepCoderError(
            f'Impossible number of unused variables! {unused_variables}')
  return statements


def _sample_program_helper(
    programs,
    example_inputs,
    input_variables,
    states,
    statements,
    unused_variables,
    num_statements,
    is_train,
    canonical_variable_order,
    experiment=exp_module.Experiment.NONE,
    operations=dsl.OPERATIONS,
    lambdas=dsl.LAMBDAS,
    reject_dead_code=True,
    reject_redundant_code=True,
    reject_duplicate_output=True,
    reject_constant_output=True,
    early_stopping_probability=0.95,
):
  """Recursive helper function for sampling programs up to a given length."""
  # Allow for early stopping on the last search step.
  if len(statements) >= 3 and random.random() < early_stopping_probability:
    return

  if len(statements) == num_statements:
    program = dsl.Program(input_variables, statements)
    should_reject_program = (
        program is None or
        (reject_dead_code and has_dead_code(program)) or
        (reject_duplicate_output and has_duplicate_output(states)) or
        (reject_constant_output and has_constant_output(states)))
    if not should_reject_program:
      programs.append(program)
    return

  if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER:
    assert num_statements >= 2
    if is_train:
      operations = (
          dsl.HIGHER_ORDER_NO_MAP if len(statements) < num_statements // 2
          else dsl.FIRST_ORDER_AND_MAP)
    else:
      operations = (
          dsl.FIRST_ORDER_AND_MAP if len(statements) < num_statements // 2
          else dsl.HIGHER_ORDER_NO_MAP)

  unique_outputs = set()
  for statement in _get_next_statements(
      program_state=states[0],
      unused_variables=unused_variables,
      is_train=is_train,
      canonical_variable_order=canonical_variable_order,
      experiment=experiment,
      operations=operations,
      lambdas=lambdas,
      last_statement=len(statements) + 1 == num_statements):

    next_states = [statement.run(state) for state in states]
    if not all(next_states):
      continue  # The statement failed to run for at least 1 example.
    if reject_redundant_code and old_sample_random.is_redundant(next_states):
      continue
    next_states_key = tuple(str(next_state.get_output())  # pytype: disable=attribute-error
                            for next_state in next_states)
    if next_states_key in unique_outputs:
      continue
    unique_outputs.add(next_states_key)

    next_statements = statements + [statement]
    # Update unused variables.
    unused_variables_copy = list(unused_variables)
    unused_variables_copy.append(statement.variable)
    for arg in statement.args:
      if isinstance(arg, str) and arg in unused_variables_copy:
        unused_variables_copy.remove(arg)

    _sample_program_helper(
        programs=programs,
        example_inputs=example_inputs,
        input_variables=input_variables,
        states=next_states,
        statements=next_statements,
        unused_variables=unused_variables_copy,
        num_statements=num_statements,
        is_train=is_train,
        canonical_variable_order=canonical_variable_order,
        experiment=experiment,
        operations=operations,
        lambdas=lambdas,
        reject_dead_code=reject_dead_code,
        reject_redundant_code=reject_redundant_code,
        reject_duplicate_output=reject_duplicate_output,
        reject_constant_output=reject_constant_output,
        early_stopping_probability=early_stopping_probability)


def _get_programs_of_length(
    programs_dict,
    length,
    is_train,
    canonical_variable_order,
    example_inputs,
    input_variables,
    experiment,
    reject_dead_code=True,
    reject_redundant_code=True,
    reject_duplicate_output=True,
    reject_constant_output=True,
    early_stopping_probability=0.95,
):
  """Collects programs of a given length into programs_dict."""
  # Condition the valid operations and lambdas on experiment and is_train.
  # Operations needs to be a list of operation pools to handle
  # COMPOSE_DIFFERENT_CONCEPTS for train.
  if is_train:
    keep_fn = None
    if experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS:
      operations = [dsl.FIRST_ORDER_AND_MAP, dsl.HIGHER_ORDER_NO_MAP]
      lambdas = dsl.LAMBDAS
    elif experiment == exp_module.Experiment.COMPOSE_NEW_OP:
      operations = [dsl.OPERATIONS_NO_SCAN]
      lambdas = dsl.LAMBDAS
    else:
      operations = [dsl.OPERATIONS]
      lambdas = dsl.LAMBDAS
  else:
    if experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS:
      operations = [dsl.OPERATIONS]
      lambdas = dsl.LAMBDAS
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(s.operation in dsl.FIRST_ORDER_AND_MAP
              for s in program.statements) and
          any(s.operation in dsl.HIGHER_ORDER_NO_MAP
              for s in program.statements))
    elif experiment == exp_module.Experiment.COMPOSE_NEW_OP:
      operations = [dsl.OPERATIONS]
      lambdas = dsl.LAMBDAS
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(s.operation.token == 'Scanl1' for s in program.statements))
    elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY:
      operations = [dsl.OPERATIONS]
      lambdas = dsl.LAMBDAS
      # In _get_next_statements, we make sure the Scanl1 operation only
      # uses the `-` or `min` lambdas during training.
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(f'Scanl1 {lambda_token}' in str(program)
              for lambda_token in ['(+)', '(*)', '(max)']))
    else:
      operations = [dsl.OPERATIONS]
      lambdas = dsl.LAMBDAS
      keep_fn = None

  programs = []
  for ops in operations:
    states = [dsl.ProgramState(inputs, input_variables)
              for inputs in example_inputs]
    _sample_program_helper(
        programs=programs,
        example_inputs=example_inputs,
        input_variables=input_variables,
        states=states,
        statements=[],
        unused_variables=[],
        num_statements=length,
        is_train=is_train,
        canonical_variable_order=canonical_variable_order,
        experiment=experiment,
        operations=ops,
        lambdas=lambdas,
        reject_dead_code=reject_dead_code,
        reject_redundant_code=reject_redundant_code,
        reject_duplicate_output=reject_duplicate_output,
        reject_constant_output=reject_constant_output,
        early_stopping_probability=early_stopping_probability)

  logging.info('Found %s length-%s programs in %s distribution (with early '
               'stopping %s)',
               len(programs), length, 'train' if is_train else 'test',
               early_stopping_probability)
  for program in programs:
    if keep_fn and not keep_fn(program):
      continue
    states = [program.run(inputs) for inputs in example_inputs]
    outputs_key = tuple(str(state.get_output()) for state in states)  # pytype: disable=attribute-error
    if outputs_key not in programs_dict:
      programs_dict[outputs_key] = program
  logging.info('  Functionally distinct programs so far: %s',
               len(programs_dict))


def _sample_list(options, num_samples):
  if len(options) <= num_samples:
    return options
  return random.sample(options, num_samples)


def sample_programs_experiment(
    example_inputs,
    input_variables,
    experiment,
    is_train,
    canonical_variable_order,
    reject_dead_code=True,
    reject_redundant_code=True,
    reject_duplicate_output=True,
    reject_constant_output=True,
    num_programs=1000,
):
  """Samples programs for an experiment and train/test split."""
  # Store output and shortest program that produces that output in train/test
  # distribution.
  programs_train_dict, programs_test_dict = {}, {}

  # Set program length boundaries depending on experiment.
  if experiment == exp_module.Experiment.NONE:
    train_range = [1, 2, 3, 4, 5] if is_train else []
    test_range = [] if is_train else [1, 2, 3, 4, 5]
  elif experiment == exp_module.Experiment.LENGTH_1_4_TO_5:
    train_range = [1, 2, 3, 4]
    test_range = [5]
  elif experiment == exp_module.Experiment.LENGTH_4_TO_1_5:
    train_range = [4]
    test_range = [1, 2, 3, 5]
  elif experiment in [exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS,
                      exp_module.Experiment.SWITCH_CONCEPT_ORDER,
                      exp_module.Experiment.COMPOSE_NEW_OP]:
    train_range = test_range = [2, 3, 4]
  elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY:
    train_range = test_range = [1, 2, 3, 4]
  else:
    raise ValueError(f'Unhandled experiment: {experiment}')

  # Collect programs in train and test distribution.
  for get_train, length_range, programs_dict in zip(
      (True, False),
      (train_range, test_range),
      (programs_train_dict, programs_test_dict)
  ):
    for length in length_range:
      if is_train and not get_train and length == max(test_range):
        # When generating train data, we don't need to exhaustively search the
        # max length of the test distribution.
        continue
      if length == 5:
        # Length 5 is only for length generalization where we use the full DSL.
        # Use aggressive early stopping out of necessity.
        early_stopping_probability = 0.9
      elif get_train == is_train and length == max(length_range):
        assert length == 4
        # Less aggressive early stopping, since it's a shorter length and/or
        # only part of the DSL.
        early_stopping_probability = 0.7
      else:
        early_stopping_probability = 0.0
      if len(programs_dict) < num_programs * 2 and len(length_range) > 1:
        # Don't use early stopping if we don't have many programs yet. And note
        # that some programs that we have now will be removed later for being
        # functionally equivalent to a shorter program in the opposite split.
        # But, in some length generalization cases, this will be the first and
        # only search, so we have 0 programs so far, but we can still do early
        # stopping regardless.
        early_stopping_probability = 0.0
      _get_programs_of_length(
          programs_dict=programs_dict,
          length=length,
          is_train=get_train,
          canonical_variable_order=canonical_variable_order,
          example_inputs=example_inputs,
          input_variables=input_variables,
          experiment=experiment,
          reject_dead_code=reject_dead_code,
          reject_redundant_code=reject_redundant_code,
          reject_duplicate_output=reject_duplicate_output,
          reject_constant_output=reject_constant_output,
          early_stopping_probability=early_stopping_probability,
      )

  logging.info('programs_train_dict len: %s', len(programs_train_dict))
  logging.info('programs_test_dict len: %s', len(programs_test_dict))

  # Programs are added to the train/test set if there doesn't exist a program
  # in the other set of shorter length that produces the same outputs.
  programs_train_length = collections.defaultdict(list)
  for example_outputs, program in programs_train_dict.items():
    if len(programs_test_dict.get(example_outputs, [0] * 10)) >= len(program):
      programs_train_length[len(program)].append(program)

  programs_test_length = collections.defaultdict(list)
  for example_outputs, program in programs_test_dict.items():
    if len(programs_train_dict.get(example_outputs, [0] * 10)) > len(program):
      programs_test_length[len(program)].append(program)

  # Sample num_programs from train and test programs (trying best to keep length
  # distribution even.
  programs_train, programs_test = [], []
  for i, length in enumerate(sorted(programs_train_length)):
    programs_with_length = programs_train_length[length]
    num_programs_k = ((num_programs - len(programs_train)) //
                      (len(programs_train_length.keys()) - i))

    if experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS:
      # Experiment needs to be handled differently to ensure that first-order
      # and higher-order functions are relatively evenly distributed.
      # Process these two groups in a random order to randomize which group gets
      # more programs due to rounding error, which actually matters for test
      # because we take very few programs per search.
      groups = [dsl.FIRST_ORDER_AND_MAP, dsl.HIGHER_ORDER_NO_MAP]
      random.shuffle(groups)
      first_group, second_group = groups

      first_programs = [program for program in programs_with_length
                        if all(s.operation in first_group
                               for s in program.statements)]
      programs_train.extend(_sample_list(first_programs, num_programs_k // 2))
      num_programs_taken = min(len(first_programs), num_programs_k // 2)

      second_programs = [program for program in programs_with_length
                         if all(s.operation in second_group
                                for s in program.statements)]
      programs_train.extend(_sample_list(second_programs,
                                         num_programs_k - num_programs_taken))
    else:
      programs_train.extend(_sample_list(programs_with_length, num_programs_k))

  for i, length in enumerate(sorted(programs_test_length)):
    programs_with_length = programs_test_length[length]
    num_programs_k = ((num_programs - len(programs_test)) //
                      (len(programs_test_length.keys()) - i))
    programs_test.extend(_sample_list(programs_with_length, num_programs_k))

  programs = programs_train if is_train else programs_test
  return programs
