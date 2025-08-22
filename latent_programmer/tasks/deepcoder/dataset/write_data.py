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

"""Write supervised training tasks to TFRecord dataset."""

import copy
import hashlib
import os
import random
import timeit
import typing
from typing import List, Union

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder import old_sample_random
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
    'split', None, ['train', 'valid', 'test'],
    'Which split of the dataset to generate.')
_NUM_PROGRAMS_PER_SEARCH = flags.DEFINE_integer(
    'num_programs_per_search', 1000,
    'Number of programs to generate per search.')
_NUM_SEARCHES = flags.DEFINE_integer(
    'num_searches', 100, 'Number of searches to perform.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 3, 'Number of examples per task.')
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
        # allowing up to `num_examples` changes.
        max_changes = random.randint(1, num_examples)
        num_changes = random.randint(1, max_changes)
        should_change = ([True] * num_changes +
                         [False] * (num_examples - num_changes))
        random.shuffle(should_change)
        random_new_inputs = old_sample_random.random_inputs(
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
          min_result = min(result) if result else dsl.deepcoder_min_int()
          max_result = max(result) if result else dsl.deepcoder_max_int()
          range_expansion = max(5, (max_result - min_result) // 2)
          min_perturb = max(dsl.deepcoder_min_int(),
                            min_result - range_expansion)
          max_perturb = min(dsl.deepcoder_max_int(),
                            max_result + range_expansion)

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
      random_new_inputs = old_sample_random.random_inputs(
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
    corrupted_next_part: a corrupted version of next_part, used to train the
      SynthesizerModel to be more robust
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


def generate_tasks_for_experiment(
    experiment,
    is_train,
    canonical_variable_order,
    num_programs=1000):
  """Generates a random task for a given experiment and dataset split."""
  if isinstance(experiment, str):
    experiment = exp_module.Experiment[experiment]

  num_examples = _NUM_EXAMPLES.value
  num_inputs = random.randint(1, _MAX_PROGRAM_ARITY.value)

  input_variables = []
  while len(input_variables) < num_inputs:
    # During training, generate unordered variables to teach the model about all
    # variable names. During test, generate variables in order (x0, x1, ...) to
    # simplify how statements combine to form programs in the end-to-end loop
    # (individual models only predict the RHS of statements, so we need to add
    # a variable name ourselves).
    input_variables.append(
        old_sample_random.random_new_variable(input_variables,
                                              ordered=canonical_variable_order))

  example_inputs = old_sample_random.random_inputs(num_examples=num_examples,
                                                   num_inputs=num_inputs)

  programs = sample_random.sample_programs_experiment(
      example_inputs=example_inputs,
      input_variables=input_variables,
      experiment=experiment,
      is_train=is_train,
      canonical_variable_order=canonical_variable_order,
      reject_dead_code=True,
      reject_redundant_code=True,
      reject_duplicate_output=True,
      reject_constant_output=True,
      num_programs=num_programs)

  tasks = []
  for program in programs:
    example_outputs = [
        program.run(inputs).get_output()  # pytype: disable=attribute-error
        for inputs in example_inputs]
    examples = [dsl.Example(inputs, output)
                for inputs, output in zip(example_inputs, example_outputs)]
    tasks.append(dsl.ProgramTask(program, examples))

  if is_train and experiment == exp_module.Experiment.COMPOSE_NEW_OP:
    num_new = num_programs // 4
    num_keep = num_programs - num_new
    if len(tasks) > num_keep:
      tasks = random.sample(tasks, num_keep)

    # Generate more length-1 programs of new operation.
    for _ in range(num_new):
      task = old_sample_random.random_task(
          num_examples=num_examples,
          num_inputs=num_inputs,
          num_statements=1,
          is_train=True,
          canonical_variable_order=canonical_variable_order,
          experiment=experiment,
          operations=dsl.OPERATIONS_ONLY_SCAN,
          lambdas=dsl.LAMBDAS,
      )
      tasks.append(task)

  if len(tasks) < num_programs:
    logging.warning('Too few programs! Wanted %s, got %s. Inputs were: %s',
                    num_programs, len(tasks), example_inputs)

  random.shuffle(tasks)
  return tasks


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

  is_train = _SPLIT.value in ['train', 'valid']
  canonical_variable_order = _SPLIT.value in ['valid', 'test']
  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(entire_programs_fname) as entire_programs_writer, \
      tf.io.TFRecordWriter(decomposition_data_fname) as decomposition_writer:
    for i in range(_NUM_SEARCHES.value):

      # Sometimes the inputs are weird/degenerate and we don't find enough
      # programs. In that case, do another search with new inputs.
      tasks = []
      while len(tasks) < _NUM_PROGRAMS_PER_SEARCH.value:
        logging.info('Starting search #%s for %s, %s split',
                     i + 1, _EXPERIMENT.value, _SPLIT.value)
        start_time = timeit.default_timer()
        tasks = generate_tasks_for_experiment(
            experiment=_EXPERIMENT.value,
            is_train=is_train,
            canonical_variable_order=canonical_variable_order,
            num_programs=_NUM_PROGRAMS_PER_SEARCH.value)
        elapsed_time = timeit.default_timer() - start_time
        logging.info('Obtained %s tasks in %.1f seconds.',
                     len(tasks), elapsed_time)
      assert len(tasks) == _NUM_PROGRAMS_PER_SEARCH.value

      for task in tasks:
        entire_programs_writer.write(serialize_entire_program_example(task))
        for example in serialize_decomposition_examples(task):
          decomposition_writer.write(example)


if __name__ == '__main__':
  app.run(main)
