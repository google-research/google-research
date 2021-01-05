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

r"""A Beam pipeline to generate OpenKP examples for ETC."""

import collections
import json
import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options
import tensorflow.compat.v1 as tf

from etcmodel.models.openkp import beam_utils
from etcmodel.models.openkp import generate_examples_lib as lib

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_patterns', None,
    'Comma-separated list of jsonl files with OpenKP examples.')

flags.DEFINE_string('output_dir', None,
                    'The output directory to write results to.')

flags.DEFINE_integer(
    'long_max_length', 4096,
    'The maximum total long sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')

flags.DEFINE_integer(
    'global_max_length', 512,
    'The maximum total global sequence length. Sequences longer than this '
    'will be truncated, and sequences shorter than this will be padded.')

flags.DEFINE_integer(
    'url_max_code_points', 1000,
    'Maximum number of Unicode code points in an example URL.')

flags.DEFINE_string(
    'bert_vocab_path', None,
    'The path to the BERT vocabulary file to use. Leave empty when '
    '`spm_model_path` is provided.')

flags.DEFINE_string(
    'spm_model_path', None,
    'Path to a SentencePiece model file to use instead of a BERT vocabulary '
    'file. If given, we use the tokenization code from ALBERT instead of BERT. '
    '`bert_vocab_path` must be set to an empty string if `spm_model_path` is '
    'given.')

flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lowercase all text before tokenizing. Must match assumption '
    'in `bert_vocab_path`. Ignored when `spm_model_path` is provided.')

flags.DEFINE_integer('output_num_shards', 100,
                     'Number of shards to output TF Examples into.')

flags.DEFINE_integer(
    'fixed_block_len', None,
    'If set, then the VDOM structure is discarded, and 1 global token is '
    'created per fixed_block_len long tokens. Also, no visual features are '
    'created.')

flags.DEFINE_integer(
    'direct_num_workers', 1,
    'Number of workers to use for the Beam DirectRunner. '
    'Increasing this should speed up example generation, '
    'but DirectRunner appears to run out of memory quickly '
    'when using more workers.')


def pipeline(root):
  """Beam pipeline to run."""
  file_patterns = FLAGS.input_patterns.split(',')
  featurization_config = lib.EtcFeaturizationConfig(
      long_max_length=FLAGS.long_max_length,
      global_max_length=FLAGS.global_max_length,
      url_max_code_points=FLAGS.url_max_code_points,
      bert_vocab_path=FLAGS.bert_vocab_path,
      spm_model_path=FLAGS.spm_model_path,
      do_lower_case=FLAGS.do_lower_case,
      fixed_block_len=FLAGS.fixed_block_len)

  stats = collections.OrderedDict()
  for i, pattern in enumerate(file_patterns):
    prefix_str = 'Pattern' + str(i)
    outputs = (
        root
        | f'{prefix_str}Read' >> beam.io.textio.ReadFromText(pattern)
        | f'{prefix_str}Reshuffle' >> beam.transforms.Reshuffle()
        | f'{prefix_str}Parse' >> beam.ParDo(
            beam_utils.ParseExampleFn(featurization_config)).with_outputs())

    output_name_prefix = os.path.basename(pattern)
    period_idx = output_name_prefix.rfind('.')
    if period_idx != -1:
      output_name_prefix = output_name_prefix[:period_idx]

    # Write TF Examples.
    _ = (
        outputs[None]
        | f'{prefix_str}WriteTfExamples' >> beam.io.WriteToTFRecord(
            os.path.join(FLAGS.output_dir, f'{output_name_prefix}.tfrecord'),
            coder=beam.coders.ProtoCoder(tf.train.Example),
            num_shards=FLAGS.output_num_shards))

    # Write text examples.
    _ = (
        outputs.text_examples
        | f'{prefix_str}WriteTextExamples' >> beam.io.WriteToText(
            os.path.join(FLAGS.output_dir,
                         f'{output_name_prefix}_text_examples.jsonl'),
            shard_name_template='',  # To force unsharded output.
        ))

    # Write failure cases.
    _ = (
        outputs.parse_failures
        | f'{prefix_str}WriteFailures' >> beam.io.WriteToText(
            os.path.join(FLAGS.output_dir,
                         f'{output_name_prefix}_parse_failures.jsonl'),
            shard_name_template='',  # To force unsharded output.
        ))

    # Collect statistics.
    counts = collections.OrderedDict()
    counts['parse_success_count'] = (
        outputs[None]  # Undeclared main output.
        | f'{prefix_str}SuccessCount' >> beam.combiners.Count.Globally())
    counts['parse_fail_count'] = (
        outputs.parse_failures
        | f'{prefix_str}FailureCount' >> beam.combiners.Count.Globally())

    stats[pattern] = beam_utils.singletons_to_dict(
        beam_label=f'{prefix_str}Stats', **counts)

  _ = (
      beam_utils.singletons_to_dict(**stats)
      | 'StatsToJson' >> beam.Map(lambda x: json.dumps(x, indent=2))
      | 'WriteStats' >> beam.io.WriteToText(
          os.path.join(FLAGS.output_dir, 'example_gen_stats.txt'),
          shard_name_template='',  # To force unsharded output.
      ))


def main(unused_argv):
  # Run the pipeline.
  options = pipeline_options.PipelineOptions(
      runner='DirectRunner',
      direct_running_mode='multi_processing',
      direct_num_workers=FLAGS.direct_num_workers)
  p = beam.Pipeline(options=options)
  pipeline(p)
  p.run().wait_until_finish()


if __name__ == '__main__':
  app.run(main)
