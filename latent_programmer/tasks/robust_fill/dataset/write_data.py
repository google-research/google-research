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

import functools
import hashlib
import os
import random

from absl import app
from absl import flags
import tensorflow as tf

from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import sample_random
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.robust_fill.dataset import experiment as exp_module


gfile = tf.io.gfile


_SEED = flags.DEFINE_integer(
    'seed', None, 'Base random seed.')
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', '/tmp/decomposition/robustfill',
    'Directory to save results to.')
_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards', 1, 'Total number of shards for this TFRecords file.')
_SHARD_ID = flags.DEFINE_integer(
    'shard_id', 0, 'An index number for this shard.')

_EXPERIMENT = flags.DEFINE_enum(
    'experiment', 'NONE', [e.name for e in exp_module.Experiment],
    'Kind of experiment.')
_SPLIT = flags.DEFINE_enum(
    'split', None, ['train', 'valid', 'test'],
    'Which split of the dataset to generate.')
_NUM_PROGRAMS = flags.DEFINE_integer(
    'num_programs', 100000, 'Number of programs to generate.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 5, 'Number of examples per task.')
_MAX_INPUT_LENGTH = flags.DEFINE_integer(
    'max_input_length', 20,
    'Maximum number of characters in input strings.')


def _bytes_feature(strs):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[s if isinstance(s, bytes) else str.encode(s) for s in strs]))


def serialize_entire_program_example(task, token_id_table):
  """Creates a tf.Example message for the entire program."""
  program_string = '|'.join(' '.join(map(str, expr.encode(token_id_table)))
                            for expr in task.program.expressions)
  feature = {
      'inputs': _bytes_feature(task.inputs),
      'outputs': _bytes_feature(task.outputs),
      'program': _bytes_feature([program_string]),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def _corrupt(next_parts, outputs, remaining_parts):
  """Corrupts next_part so that the SynthesizerModel can be more robust."""
  num_examples = len(outputs)
  corrupted = next_parts
  while corrupted == next_parts:  # Reject corruption if nothing changed.
    technique = random.choice(['multiple_parts', 'perturb'])

    if technique == 'multiple_parts':
      num_remaining_parts = len(remaining_parts[0])
      choices = []
      for num_parts in range(2, num_remaining_parts):
        max_parts_len = max(len(''.join(parts[:num_parts]))
                            for parts in remaining_parts)
        if max_parts_len <= _MAX_INPUT_LENGTH.value:
          choices.append(num_parts)
      if not choices:
        # Can't use this corruption technique because there are no more parts to
        # use, or using more parts leads to too-long strings.
        continue
      num_parts_to_use = random.choice(choices)
      corrupted = [''.join(parts[:num_parts_to_use])
                   for parts in remaining_parts]

    elif technique == 'perturb':
      num_to_perturb = random.randint(1, num_examples)
      example_indices_to_perturb = random.sample(range(num_examples),
                                                 num_to_perturb)
      corrupted = list(next_parts)
      for example_index in example_indices_to_perturb:
        new_len = random.randint(0, _MAX_INPUT_LENGTH.value)
        corrupted[example_index] = outputs[example_index][:new_len]

    else:
      raise ValueError('Unhandled corruption technique: {}'.format(technique))

  assert all(len(c) <= _MAX_INPUT_LENGTH.value for c in corrupted)
  return corrupted


def serialize_decomposition_examples(task, token_id_table):
  """Creates tf.Example messages for decomposition."""
  # TODO(kshi): If we want to include length-2 programs in the subprogram
  # synthesizer's training data, we'll need to create a separate dataset for
  # that, since we don't want such data in the spec decomposer model's training
  # data.
  output_parts = [[expr(inp) for expr in task.program.expressions]
                  for inp in task.inputs]
  assert all(''.join(parts) == out
             for parts, out in zip(output_parts, task.outputs))
  results = []

  for i, expr in enumerate(task.program.expressions):
    remaining_parts = [parts[i:] for parts in output_parts]
    outputs = [''.join(parts) for parts in remaining_parts]
    next_part = [parts[i] for parts in output_parts]
    corrupted_next_part = _corrupt(next_part, outputs=outputs,
                                   remaining_parts=remaining_parts)
    program_part_string = ' '.join(map(str, expr.encode(token_id_table)))
    feature = {
        'inputs': _bytes_feature(task.inputs),
        'outputs': _bytes_feature(outputs),
        'next_part': _bytes_feature(next_part),
        'corrupted_next_part': _bytes_feature(corrupted_next_part),
        'program_part': _bytes_feature([program_part_string]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    results.append(example_proto.SerializeToString())

  return results


def generate_task_for_experiment(experiment, is_train):
  """Generates a random task for a given experiment and dataset split."""
  if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER.name:
    # Handle this case separately because it's the most different from the rest.
    return sample_random.random_task_switch_concept_order(
        max_k=3,
        max_input_tokens=5,
        max_input_length=_MAX_INPUT_LENGTH.value,
        num_examples=_NUM_EXAMPLES.value,
        min_expressions=2,
        max_expressions=6,
        is_train=is_train)

  # Still pass in max_expressions, min_expressions, sampler_pool,
  # valid_length_fn, and keep_fn.
  random_task_partial = functools.partial(
      sample_random.random_task,
      max_k=3,
      max_input_tokens=5,
      max_input_length=_MAX_INPUT_LENGTH.value,
      num_examples=_NUM_EXAMPLES.value)

  valid_num_expressions_fn = None
  keep_fn = None

  if experiment == exp_module.Experiment.NONE.name:
    min_expressions = 1
    max_expressions = 10
    sampler_pool = sample_random.SAMPLER_POOL_ALL

  elif experiment == exp_module.Experiment.LENGTH_1_6_TO_7_10.name:
    min_expressions = 1 if is_train else 7
    max_expressions = 6 if is_train else 10
    sampler_pool = sample_random.SAMPLER_POOL_ALL

  elif experiment == exp_module.Experiment.LENGTH_6_TO_1_10.name:
    min_expressions = 6 if is_train else 1
    max_expressions = 6 if is_train else 10
    sampler_pool = sample_random.SAMPLER_POOL_ALL
    if not is_train:
      valid_num_expressions_fn = lambda n: n != 6

  elif experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS.name:
    min_expressions = 2
    max_expressions = 6
    if is_train:
      sampler_pool = random.choice([sample_random.ALL_SUBSTRING,
                                    sample_random.SAMPLER_POOL_MODIFY_OR_CONST])
    else:
      sampler_pool = [sample_random.ALL_SUBSTRING,
                      sample_random.SAMPLER_POOL_MODIFY_OR_CONST]
      keep_fn = lambda c: (  # pylint: disable=g-long-lambda
          any(isinstance(e, dsl.Substring) for e in c.expressions) and
          any(isinstance(e, (dsl.Modification, dsl.ConstStr))
              for e in c.expressions))

  elif experiment == exp_module.Experiment.COMPOSE_NEW_OP.name:
    if is_train:
      if random.random() < 0.25:
        min_expressions = 1
        max_expressions = 1
        sampler_pool = sample_random.SAMPLER_POOL_ONLY_COMPOSE
      else:
        min_expressions = 2
        max_expressions = 6
        sampler_pool = sample_random.SAMPLER_POOL_NO_COMPOSE
    else:
      min_expressions = 2
      max_expressions = 6
      sampler_pool = sample_random.SAMPLER_POOL_ALL
      keep_fn = lambda c: any(isinstance(e, dsl.Compose) for e in c.expressions)

  elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY.name:
    min_expressions = 1
    max_expressions = 6
    sampler_pool = (sample_random.SAMPLER_POOL_NO_COMPOSE_SUBSTRING if is_train
                    else sample_random.SAMPLER_POOL_ALL)
    if not is_train:
      keep_fn = lambda c: any(  # pylint: disable=g-long-lambda
          isinstance(e, dsl.Compose) and
          isinstance(e.modification_or_substring, dsl.Substring)
          for e in c.expressions)

  else:
    raise ValueError('Unhandled experiment name: {}'.format(experiment))

  if is_train:
    # These are only used for test.
    assert valid_num_expressions_fn is None and keep_fn is None

  return random_task_partial(
      max_expressions=max_expressions,
      min_expressions=min_expressions,
      sampler_pool=sampler_pool,
      valid_num_expressions_fn=valid_num_expressions_fn,
      keep_fn=keep_fn)


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
  _, token_id_table = dsl_tokens.build_token_tables()

  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(entire_programs_fname) as entire_programs_writer, \
      tf.io.TFRecordWriter(decomposition_data_fname) as decomposition_writer:
    for _ in range(_NUM_PROGRAMS.value):
      is_train = _SPLIT.value in ['train', 'valid']
      task = generate_task_for_experiment(
          _EXPERIMENT.value,
          is_train=is_train)

      entire_programs_writer.write(
          serialize_entire_program_example(task, token_id_table))
      for example in serialize_decomposition_examples(task, token_id_table):
        decomposition_writer.write(example)

if __name__ == '__main__':
  app.run(main)
