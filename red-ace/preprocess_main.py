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

"""Convert a JSON dataset into the TFRecord format.

The resulting TFRecord file will be used when training a RED-ACE model.
"""

import random

from absl import app
from absl import flags
from absl import logging
import example_builder
import redace_flags  # pylint: disable=unused-import
import tensorflow as tf
import tokenization
import utils


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples to be converted to tf.Examples.'
)

flags.DEFINE_bool(
    'store_debug_features', False,
    'Debugging information, i.e. source tokens, are stored in the tf.Examples.')

flags.DEFINE_bool(
    'write_tfrecord_to_file', True,
    'If False no tfrecord is written to file, instead only joint length'
    'information is written to a file.')

flags.DEFINE_integer('max_input_lines', None, 'Number of samples.')


def _write_example_count(count, example_path):
  """Saves the number of converted examples to a file.

  This count is used when determining the number of training steps.

  Args:
    count: The number of converted examples.
    example_path: Path to the file where the examples are saved.

  Returns:
    The path to which the example count is saved
      (example_path + '.num_examples.txt').
  """
  count_fname = example_path + '.num_examples.txt'
  with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
    count_writer.write(str(count))
  return count_fname


def _write_length(length, example_path):
  """Saves the 99 percentile joint insertion length to a file.

  This count is used when determining the number of decoding steps.

  Args:
    length: The 99 percentile length.
    example_path: Path to the file where the length is saved.

  Returns:
    The path to which the length is saved
      (example_path + '.length.txt').
  """
  count_fname = example_path + '.length.txt'
  with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
    count_writer.write(str(length))
  return count_fname


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.output_file.count('@') != 0:
    raise app.UsageError('Output-file sharding is not supported.')

  builder = example_builder.RedAceExampleBuilder(
      tokenization.FullTokenizer(FLAGS.vocab_file), FLAGS.max_seq_length)

  # Number of examples successfully converted to a tagging TF example.
  num_converted = 0
  random.seed(42)

  output_file = FLAGS.output_file
  with tf.io.TFRecordWriter(output_file) as writer:
    for i, (source, target, confidence_scores,
            utterance_id) in enumerate(utils.read_input(FLAGS.input_file)):
      logging.log_every_n(
          logging.INFO,
          f'{i} examples processed, {num_converted} converted to tf.Example.',
          10000,
      )
      example = builder.build_redace_example(source, confidence_scores, target)
      if example is not None:
        example.debug_features['utterance_id'] = utterance_id
        writer.write(example.to_tf_example().SerializeToString())
        num_converted += 1

  logging.info('Done. %d tagging examples converted to tf.Example.',
               num_converted)
  count_fname = _write_example_count(num_converted, FLAGS.output_file)
  logging.info('\n'.join(['Wrote:', FLAGS.output_file, count_fname]))


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('output_file')
  flags.mark_flag_as_required('vocab_file')
  app.run(main)
