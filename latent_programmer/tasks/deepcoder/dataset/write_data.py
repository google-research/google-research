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

"""Write supervised training tasks to TFRecord dataset."""

import copy
import hashlib
import os
import random
import typing
from typing import List, Union

from absl import app
from absl import flags

import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder import sample_random


gfile = tf.io.gfile

_SEED = flags.DEFINE_integer(
    'seed', None, 'Base random seed.')
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', '/tmp/decomposition/deepcoder', 'Directory to save results to.')
_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards', 1, 'Total number of shards for this TFRecords file.')
_SHARD_ID = flags.DEFINE_integer(
    'shard_id', 0, 'An index number for this shard.')

_EXPERIMENT = flags.DEFINE_enum(
    'experiment', 'NONE', [e.name for e in exp_module.Experiment],
    'Kind of experiment (see document for descriptions).')
_SPLIT = flags.DEFINE_enum(
    'split', None, ['train', 'valid', 'test', 'finetune'],
    'Which split of the dataset to generate.')
_NUM_PROGRAMS = flags.DEFINE_integer(
    'num_programs', 100000, 'Number of programs to generate.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 5, 'Number of examples per task.')
_MAX_PROGRAM_ARITY = flags.DEFINE_integer(
    'max_program_arity', 2, 'Maximum number of inputs.')


def _bytes_feature(strs):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[s if isinstance(s, bytes) else str.encode(s) for s in strs]))


def serialize_entire_program_example(task):
  """Creates a tf.Example message for the entire program."""
  input_variables = task.program.input_variables
  example_inputs_strs = [str(dsl.ProgramState(e.inputs, input_variables))
                         for e in task.examples]
  example_outputs_strs = [dsl.result_to_str(e.output) for e in task.examples]
  feature = {
      'inputs': _bytes_feature(example_inputs_strs),
      'outputs': _bytes_feature(example_outputs_strs),
      'program': _bytes_feature([str(task.program)]),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def _corrupt(next_parts, outputs):
  """Corrupts next_part so that the SynthesizerModel can be more robust."""
  results = [dsl.str_to_result(part) for part in next_parts]
  assert all(type(r) == type(results[0]) for r in results)  # pylint: disable=unidiomatic-typecheck
  num_examples = len(outputs)

  corrupted = next_parts
  while corrupted == next_parts:  # Reject corruption if nothing changed.
    technique = random.choice(['copy_output', 'perturb', 'new_random'])

    if technique == 'copy_output':
      corrupted = list(outputs)

    elif technique == 'perturb':

      if type(results[0]) == int:  # pylint: disable=unidiomatic-typecheck
        # Choose the number of examples to change, favoring fewer changes but
        # allowing up to `num_examples - 1` changes.
        max_changes = random.randint(1, num_examples - 1)
        num_changes = random.randint(1, max_changes)
        should_change = ([True] * num_changes +
                         [False] * (num_examples - num_changes))
        random.shuffle(should_change)
        random_new_inputs = sample_random.random_inputs(
            num_examples=num_examples, num_inputs=1, types=[int])
        changed = [random_new_inputs[i][0] if should_change[i] else results[i]
                   for i in range(num_examples)]

      else:
        assert type(results[0]) == list  # pylint: disable=unidiomatic-typecheck
        changed = copy.deepcopy(results)  # Will perform changes in place.
        for result in changed:
          result = typing.cast(List[int], result)  # Not an int or None.
          # Perturbed numbers should be chosen reasonably considering the
          # existing numbers.
          min_result = min(result) if result else dsl.min_int()
          max_result = max(result) if result else dsl.max_int()
          range_expansion = max(5, (max_result - min_result) // 2)
          min_perturb = max(dsl.min_int(), min_result - range_expansion)
          max_perturb = min(dsl.max_int(), max_result + range_expansion)

          max_num_perturbations = min(max(2, len(result) // 2), 5)
          num_perturbations = random.randint(0, max_num_perturbations)
          for _ in range(num_perturbations):
            # Note, it's possible for a perturbation to undo a previous one.
            kind_options = ['insert']
            if result:
              kind_options.extend(['delete', 'replace'])
            kind = random.choice(kind_options)
            if kind == 'insert':
              result.insert(random.randint(0, len(result)),
                            random.randint(min_perturb, max_perturb))
            elif kind == 'delete':
              del result[random.randint(0, len(result) - 1)]
            elif kind == 'replace':
              result[random.randint(0, len(result) - 1)] = random.randint(
                  min_perturb, max_perturb)
            else:
              raise ValueError(f'Unhandled perturbation kind: {kind}')

      corrupted = [dsl.result_to_str(r) for r in changed]

    elif technique == 'new_random':
      random_new_inputs = sample_random.random_inputs(
          num_examples=num_examples, num_inputs=1,
          types=[random.choice([int, list])])
      corrupted = [dsl.result_to_str(input_list[0])
                   for input_list in random_new_inputs]

    else:
      raise ValueError('Unhandled corruption technique: {}'.format(technique))

  return corrupted


def serialize_decomposition_examples(task):
  """Returns tf.Example messages for decomposition.

  The current features correspond to the following values:
    inputs: string representation of program state (including inputs and
      intermediate variables)
    outputs: string representation of desired outputs
    next_part: string representation of desired next intermediate outputs
    program_part: string representation of next statement in program that
      generates the next intermediate outputs

  Args:
    task: a dsl.ProgramTask to turn into multiple decomposed examples.
  """
  input_variables = task.program.input_variables
  example_outputs_strs = [dsl.result_to_str(e.output) for e in task.examples]
  states = [dsl.ProgramState(e.inputs, input_variables) for e in task.examples]
  results = []

  for statement in task.program.statements:
    example_inputs_strs = [str(state) for state in states]
    next_states = [statement.run(state) for state in states]
    next_part = [dsl.result_to_str(next_state.get_output())
                 for next_state in next_states]
    corrupted_next_part = _corrupt(next_part, outputs=example_outputs_strs)
    program_part_string = str(statement)
    feature = {
        'inputs': _bytes_feature(example_inputs_strs),
        'outputs': _bytes_feature(example_outputs_strs),
        'next_part': _bytes_feature(next_part),
        'corrupted_next_part': _bytes_feature(corrupted_next_part),
        'program_part': _bytes_feature([program_part_string]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    results.append(example_proto.SerializeToString())
    states = next_states

  assert ([state.get_output() for state in states]
          == [e.output for e in task.examples])

  return results


def generate_task_for_experiment(experiment,
                                 is_train):
  """Generates a random task for a given experiment and dataset split."""
  if isinstance(experiment, str):
    experiment = exp_module.Experiment[experiment]

  # Some tasks require a rejection sampling step to enforce some constraints.
  keep_fn = None

  # Generate program.
  if experiment == exp_module.Experiment.NONE:
    num_statements = random.randint(1, 5)
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.LENGTH_1_4_TO_5:
    num_statements = random.randint(1, 4) if is_train else 5
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.LENGTH_4_TO_1_5:
    if is_train:
      num_statements = 4
    else:
      num_statements = random.choice([1, 2, 3, 5])
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS:
    num_statements = random.randint(2, 4)
    if is_train:
      operations_pool = random.choice([dsl.FIRST_ORDER_OPERATIONS,
                                       dsl.HIGHER_ORDER_OPERATIONS])
    else:
      operations_pool = dsl.OPERATIONS
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(s.operation in dsl.FIRST_ORDER_OPERATIONS
              for s in program.statements) and
          any(s.operation in dsl.HIGHER_ORDER_OPERATIONS
              for s in program.statements))
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER:
    num_statements = random.randint(2, 4)
    operations_pool = None  # Will be set later in sample_random.random_program.
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.COMPOSE_NEW_OP:
    if is_train:
      if random.random() < 0.25:
        num_statements = 1
        operations_pool = dsl.OPERATIONS_ONLY_SCAN
      else:
        num_statements = random.randint(2, 4)
        operations_pool = dsl.OPERATIONS_NO_SCAN
    else:
      num_statements = random.randint(2, 4)
      operations_pool = dsl.OPERATIONS
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(s.operation.token == 'Scanl1' for s in program.statements))
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY:
    num_statements = random.randint(1, 4)
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS
    # In sample_random.random_statement, we make sure the Scanl1 operation only
    # uses the `-` or `min` lambdas during training.
    if not is_train:
      keep_fn = lambda program: (  # pylint: disable=g-long-lambda
          any(f'Scanl1 {lambda_token}' in str(program)
              for lambda_token in ['(+)', '(*)', '(max)']))
  else:
    raise ValueError(f'Unhandled experiment: {experiment}')

  program = None
  task = None
  while program is None or (keep_fn and not keep_fn(program)):
    task = sample_random.random_task(
        num_examples=_NUM_EXAMPLES.value,
        num_inputs=random.randint(1, _MAX_PROGRAM_ARITY.value),
        num_statements=num_statements,
        is_train=is_train,
        experiment=experiment,
        operations=operations_pool,
        lambdas=lambdas_pool)
    program = task.program

  return task


def main(_):
  if _SEED.value is not None:
    # By setting seeds this way, they are not dependent on the order jobs are
    # run in. This allows the flexibility to generate a part of the data without
    # affecting other parts.
    seed_phrase = (f'{_EXPERIMENT.value}-{_SPLIT.value}-{_SHARD_ID.value}-'
                   f'{_SEED.value}')  # Distinguishes this worker from others.
    seed = int(hashlib.md5(seed_phrase.encode('utf-8')).hexdigest()[:8], 16)
    random.seed(seed)

  experiment_save_dir = os.path.join(_SAVE_DIR.value,
                                     f'{_EXPERIMENT.value}_data')
  if not gfile.isdir(experiment_save_dir):
    gfile.makedirs(experiment_save_dir)

  entire_programs_fname = os.path.join(
      experiment_save_dir,
      'entire_programs_{}.tf_records-{:05d}-of-{:05d}'.format(
          _SPLIT.value, _SHARD_ID.value, _NUM_SHARDS.value))
  decomposition_data_fname = os.path.join(
      experiment_save_dir,
      'decomposition_data_{}.tf_records-{:05d}-of-{:05d}'.format(
          _SPLIT.value, _SHARD_ID.value, _NUM_SHARDS.value))

  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(entire_programs_fname) as entire_programs_writer, \
      tf.io.TFRecordWriter(decomposition_data_fname) as decomposition_writer:
    for i in range(_NUM_PROGRAMS.value):
      if _SPLIT.value in ['train', 'valid']:
        is_train = True
      elif _SPLIT.value == 'test':
        is_train = False
      elif _SPLIT.value == 'finetune':
        is_train = bool(i % 2)
      else:
        raise ValueError('Unhandled split: {}'.format(_SPLIT.value))
      task = generate_task_for_experiment(_EXPERIMENT.value, is_train)

      entire_programs_writer.write(serialize_entire_program_example(task))
      for example in serialize_decomposition_examples(task):
        decomposition_writer.write(example)

if __name__ == '__main__':
  app.run(main)
