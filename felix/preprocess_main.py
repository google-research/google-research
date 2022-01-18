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

"""Convert a dataset into the TFRecord format.

The resulting TFRecord file will be used when training a Felix model.
"""

import random

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from felix import felix_flags  # pylint: disable=unused-import
from felix import preprocess
from felix import utils
FLAGS = flags.FLAGS
# Preprocessing specific flags are listed below.
# See felix_flags.py for the other flags.
flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples to be converted to '
    'tf.Examples.')
flags.DEFINE_string('output_tfrecord', None,
                    'Path to the resulting TFRecord file.')
flags.DEFINE_integer('max_input_lines', None, 'Number of samples.')


# Suffix added to insertion output data filenames.
_INSERTION_FILENAME_SUFFIX = '.ins'


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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.output_file.count('@') != 0:
    raise app.UsageError('Output-file sharding is not supported.')
  if not FLAGS.use_open_vocab:
    raise  app.UsageError('Currently only use_open_vocab=True is supported')
  builder = preprocess.initialize_builder(
      FLAGS.use_pointing, FLAGS.use_open_vocab, FLAGS.label_map_file,
      FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, FLAGS.vocab_file,
      FLAGS.do_lower_case,
      FLAGS.special_glue_string_for_joining_sources,
      max_mask=FLAGS.max_mask,
      insert_after_token=FLAGS.insert_after_token)
  # Number of examples successfully converted to a tagging TF example.
  num_converted = 0
  # Number of examples successfully converted to an insertion TF example.
  num_converted_insertion = 0
  random.seed(42)

  indexes = None

  if FLAGS.max_input_lines:
    input_len = sum(1 for _ in utils.yield_sources_and_targets(
        FLAGS.input_file, FLAGS.input_format, FLAGS.source_key,
        FLAGS.target_key))
    max_len = min(input_len, FLAGS.max_input_lines)
    indexes = set(random.sample(range(input_len), max_len))

  insertion_path = FLAGS.output_file + _INSERTION_FILENAME_SUFFIX
  with tf.io.TFRecordWriter(FLAGS.output_file) as writer:
    with tf.io.TFRecordWriter(insertion_path) as writer_insertion:
      for i, (sources, target) in enumerate(
          utils.yield_sources_and_targets(FLAGS.input_file,
                                          FLAGS.input_format)):
        if indexes and i not in indexes:
          continue
        if target is None or not target.strip():
          continue
        if FLAGS.use_open_vocab:
          # '<::::>' is not a valid bert token.
          target = target.replace('<::::>', '[SEP]')
        logging.log_every_n(
            logging.INFO,
            f'{i} examples processed, {num_converted} converted to tf.Example.',
            10000)
        example, insertion_example = builder.build_bert_example(sources, target)
        if example is not None:
          writer.write(example.to_tf_example().SerializeToString())
          num_converted += 1
        if insertion_example is not None:
          writer_insertion.write(
              utils.feed_dict_to_tf_example(
                  insertion_example,
                  source=FLAGS.special_glue_string_for_joining_sources.join(
                      sources),
                  target=target).SerializeToString())
          num_converted_insertion += 1

  logging.info('Done. %d tagging and %d insertion examples converted to '
               'tf.Example.', num_converted, num_converted_insertion)
  count_fname = _write_example_count(num_converted, FLAGS.output_file)
  insertion_count_fname = _write_example_count(num_converted_insertion,
                                               insertion_path)
  logging.info('\n'.join([
      'Wrote:',
      FLAGS.output_file,
      count_fname,
      insertion_path,
      insertion_count_fname,
  ]))


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  flags.mark_flag_as_required('use_pointing')
  flags.mark_flag_as_required('use_open_vocab')
  app.run(main)
