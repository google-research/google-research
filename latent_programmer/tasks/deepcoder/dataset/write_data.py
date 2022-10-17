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

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os
import random
import sys
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl
from latent_programmer.tasks.deepcoder import sample_random
from latent_programmer.tasks.deepcoder.dataset import experiment as exp_module

sys.path.append('../../../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_work_units', 1, 'Total number of work units.')
flags.DEFINE_integer('seed', None, 'Fixed random seed.')

flags.DEFINE_integer('num_tasks', 100000, 'Number of tasks to write.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_expressions', 4,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('min_expressions', 1,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_input_length', 20,
                     'Maximum number of characters in input strings.')

flags.DEFINE_string('save_dir', '/tmp/decomposition/deepcoder',
                    'Directory to save results to.')

flags.DEFINE_boolean('split_program', False,
                     'Whether to split program by parial program.')

flags.DEFINE_enum('split', None, ['train', 'valid', 'test', 'finetune'],
                  'Which split of the dataset to generate.')
flags.DEFINE_enum('experiment', 'NONE', [e.name for e in exp_module.Experiment],
                  'Kind of experiment (see document for descriptions).')



Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@jxihong 
jxihong
/
google-research
Public
forked from google-research/google-research
Code
Pull requests
Actions
Projects
Security
Insights
Settings
google-research/latent_programmer/tasks/robust_fill/dataset/write_data.py /
@kensens
kensens generates decomposition data
…
Latest commit fbba41b on Jun 6
 History
 5 contributors
@kensens@jxihong@yilei@joel-shor@andrewluchen
279 lines (232 sloc)  10.1 KB

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

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os
import random
import sys
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import sample_random
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.robust_fill.dataset import experiment as exp_module


sys.path.append('../../../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_work_units', 1, 'Total number of work units.')
flags.DEFINE_integer('seed', None, 'Fixed random seed.')

flags.DEFINE_integer('num_tasks', 100000, 'Number of tasks to write.')
flags.DEFINE_integer('num_concurrent_inputs', 4,
                     'Number of concurrent inputs per task.')
flags.DEFINE_integer('max_expressions', 4,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('min_expressions', 1,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_inputs', 2,
                     'Maximum number of inputs.')

flags.DEFINE_string('save_dir', '/tmp/decomposition/robust_fill',
                    'Directory to save results to.')

flags.DEFINE_boolean('split_program', False,
                     'Whether to split program by parial program.')

flags.DEFINE_enum('split', None, ['train', 'valid', 'test', 'finetune'],
                  'Which split of the dataset to generate.')
flags.DEFINE_enum('experiment', 'NONE', [e.name for e in exp_module.Experiment],
                  'Kind of experiment (see document for descriptions).')


def _bytes_feature(strs):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[str.encode(s) for s in strs]))


def serialize_entire_program_example(task, token_id_table):
  """Creates a tf.Example message for the entire program."""
  program_string = ''
  if FLAGS.split_program:
    for expr in task.program.expressions:
      program_string += ' '.join(map(str, expr.encode(token_id_table)))
      program_string += '|'
    program_string = program_string[:-1]
  else:
    program_string = ' '.join(
        map(str, task.program.encode(token_id_table)[:-1]))

  feature = {
      'inputs': _bytes_feature(task.inputs),
      'outputs': _bytes_feature(task.outputs),
      'program_encoding': _bytes_feature([program_string]),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


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
    outputs = [''.join(parts[i:]) for parts in output_parts]
    next_part = [parts[i] for parts in output_parts]
    program_part_string = ' '.join(map(str, expr.encode(token_id_table)))
    feature = {
        'inputs': _bytes_feature(task.inputs),
        'outputs': _bytes_feature(outputs),
        'next_part': _bytes_feature(next_part),
        'program_part': _bytes_feature([program_part_string]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    results.append(example_proto.SerializeToString())

  return results


def generate_task_for_experiment(experiment, is_train):
  """Generates a random task for a given experiment and dataset split."""
  num_inputs = np.random.randint(1, FLAGS.max_inputs + 1)
  inputs = sample_random.random_inputs(num_inputs)
  
  if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER.name:
    num_statements = np.random.randint(1, 5)
    # Handle this case separately because it's the most different from the rest.
    return inputs, sample_random.random_program_switch_concept_order(
        inputs,
        num_statements,
        is_train=is_train)

  if experiment == exp_module.Experiment.LENGTH_1_4_TO_5.name:
    num_statements = np.random.randint(1, 5) if is_train else 5
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.LENGTH_4_TO_1_5.name:
    if is_train:
        num_statements = 4
    else:
        num_statements = np.random.randint(1, 5)
        while num_statements == 4:
            num_statements = np.random.randint(1, 5)
    operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS.name:
    num_statements = np.random.randint(1, 5)
    if is_train:
      operations_pool = random.choice([dsl.FIRST_ORDER_OPERATIONS,
                                       dsl.HIGHER_ORDER_OPERATIONS])
    else:
      operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.COMPOSE_NEW_OP.name:
    if is_train:
      if random.random() < 0.25:
        num_statements = 1
        operations_pool = dsl.OPERATIONS_ONLY_SCAN
      else:
        num_statements = np.random.randint(2, 5)
        operations_pool = dsl.OPERATIONS_NO_SCAN
    else:
      operations_pool = dsl.OPERATIONS
    lambdas_pool = dsl.LAMBDAS

  elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY.name:
    num_statements = np.random.randint(1, 5)
    if is_train:
        return inputs, sample_random.random_program_extend_op_functionality(
            inputs, num_statements)
    else:
        operations_pool = dsl.OPERATIONS
        lambdas_pool = dsl.LAMBDAS
  else:
    raise ValueError('Unhandled experiment name: {}'.format(experiment))

  return inputs, samepl_random.random_program(
      inputs, num_statements, operations=operations_pool, lambdas=lambdas_pool)


def main(_):
  tf.enable_v2_behavior()

  if FLAGS.seed is not None:
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  shard_id = 0
  total_shards = 1

  entire_programs_fname = os.path.join(
      FLAGS.save_dir,
      'entire_programs_{}.tf_records-{:05d}-of-{:05d}'.format(
          FLAGS.split, shard_id, total_shards))
  decomposition_data_fname = os.path.join(
      FLAGS.save_dir,
      'decomposition_data_{}.tf_records-{:05d}-of-{:05d}'.format(
          FLAGS.split, shard_id, total_shards))

  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(entire_programs_fname) as entire_programs_writer, \
      tf.io.TFRecordWriter(decomposition_data_fname) as decomposition_data_writer:
    for i in range(FLAGS.num_tasks):
      if FLAGS.experiment == exp_module.Experiment.NONE.name:
          pass
      else:
        if FLAGS.split in ['train', 'valid']:
          is_train = True
        elif FLAGS.split == 'test':
          is_train = False
        elif FLAGS.split == 'finetune':
          is_train = bool(i % 2)
        else:
          raise ValueError('Unhandled split: {}'.format(FLAGS.split))
        task = generate_task_for_experiment(FLAGS.experiment, is_train)

      entire_programs_writer.write(
          serialize_entire_program_example(task, token_id_table))
      for example in serialize_decomposition_examples(task, token_id_table):
        decomposition_data_writer.write(example)

if __name__ == '__main__':
  app.run(main)
