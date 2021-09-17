# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# python3
"""Write supervised training tasks to TFRecord dataset."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import random
import sys
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import sample_random
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens


sys.path.append('../../../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_work_units', 1, 'Total number of work units.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed.')

flags.DEFINE_integer('num_tasks', 100000, 'Number of tasks to write.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_expressions', 10,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('min_expressions', 1,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_characters', 100,
                     'Maximum number of characters in input/output strings.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')

flags.DEFINE_boolean('split_program', False,
                     'Whether to split program by parial program.')
flags.DEFINE_boolean('split_outputs', False,
                     'Whether to split outputs by partial program.')


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(task,
                      token_id_table):
  """Creates a tf.Example message to be written to a file."""
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  io_string = ''
  if FLAGS.split_outputs:
    for inp in task.inputs:
      io_string += inp + '<'
      for expr in task.program.expressions:
        io_string += expr(inp) + '|'
      io_string = io_string[:-1] + '>'
    io_string = io_string[:-1]
  else:
    for inp, out in zip(task.inputs, task.outputs):
      io_string += inp + '<' + out + '>'
    io_string = io_string[:-1]

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
      'i/o': _bytes_feature(str.encode(io_string)),
      'program_encoding': _bytes_feature(str.encode(program_string)),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(_):
  tf.enable_v2_behavior()

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  _, token_id_table = dsl_tokens.build_token_tables()

  if not gfile.isdir(FLAGS.save_dir):
    gfile.mkdir(FLAGS.save_dir)

  worker_fname = os.path.join(FLAGS.save_dir,
                              'program_tasks.tf_records-00000-of-00001')

  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(worker_fname) as writer:
    for _ in range(FLAGS.num_tasks):
      task = sample_random.random_task(
          max_expressions=FLAGS.max_expressions,
          min_expressions=FLAGS.min_expressions,
          max_k=5,
          max_input_tokens=10,
          max_input_length=FLAGS.max_characters,
          max_output_length=FLAGS.max_characters,
          num_examples=FLAGS.num_strings_per_task,
      )
      example = serialize_example(task, token_id_table)
      writer.write(example)


if __name__ == '__main__':
  app.run(main)
