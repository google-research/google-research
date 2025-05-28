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
r"""A FlumePython program to generate tf.Examples with a sentencepiece model."""

import difflib
import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam import metrics
from apache_beam.io import tfrecordio
from apache_beam.options import pipeline_options
import sentencepiece as spm
from tensor2tensor.data_generators.generator_utils import pack_examples
from tensor2tensor.data_generators.generator_utils import to_example
import tensorflow.compat.v2 as tf

_PAD_ID = 1
_DEFAULT_EOS_ID = 2  # Only used if spm_path is None.

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'Input RecordIO with TSV lines.')
flags.DEFINE_integer('tsv_source_column', 0, 'Source sentence TSV column.')
flags.DEFINE_integer(
    'tsv_target_column', 1, 'Target sentence TSV column. If '
    'negative, use all available columns except source_column')
flags.DEFINE_string('spm_path', None, 'Path to the SPM model.')
flags.DEFINE_integer(
    'packed_length', 256, 'Length of packed examples. Set to zero to disable '
    'packing.')
flags.DEFINE_integer('pad_length', 256,
                     'If positive, pad all features to this length.')
flags.DEFINE_integer(
    'num_guide_shards', 4,
    'Number of shards for the output of the TFRecords for guide and dev splits.'
)
flags.DEFINE_integer(
    'num_train_shards', 64,
    'Number of shards for the output of the TFRecords for train split.')

flags.DEFINE_float(
    'min_edit_distance', 0.3,
    'Minimum edit distance. Examples for which source and target are more similar will be omitted.'
)


class SelectTSVColumns(beam.DoFn):
  """Selects two columns in TSV lines."""

  def __init__(self, source_column=0, target_column=1):
    self._source_column = source_column
    self._target_column = target_column

  def process(self, tsv_line):
    columns = tsv_line.split('\t')
    try:
      source_sentence = columns[self._source_column]
    except IndexError:
      metrics.Metrics.counter('err_malformed_tsv_source', 'count').inc()
      return
    if self._target_column >= 0:
      try:
        yield source_sentence, columns[self._target_column]
      except IndexError:
        metrics.Metrics.counter('err_malformed_tsv_target', 'count').inc()
    else:
      for column_index, target_sentence in enumerate(columns):
        if column_index != self._source_column:
          yield source_sentence, target_sentence


class PrepareTfExamples(beam.DoFn):
  """Prepare (packed) TFExamples from a list of source/target sentence pairs."""

  def __init__(self, spm_path, packed_length=256, pad_length=256):
    self._spm_path = spm_path
    self._spm = None
    self._packed_length = packed_length
    self._packed_examples = packed_length > 0
    self._pad_length = pad_length

  def start_bundle(self):
    if self._spm_path:
      with tf.io.gfile.GFile(self._spm_path, 'rb') as f:
        spm_model = f.read()
      self._spm = spm.SentencePieceProcessor()
      self._spm.LoadFromSerializedProto(spm_model)

  def _make_spm_example_dict(self, source_text, target_text):
    return {
        'inputs': self._encode_with_spm(source_text),
        'targets': self._encode_with_spm(target_text)
    }

  def _encode_with_spm(self, text):
    if self._spm is not None:
      return self._spm.EncodeAsIds(text) + [self._spm.eos_id()]
    return [int(t) for t in text.strip().split()] + [_DEFAULT_EOS_ID]

  def _pad_example_dict(self, example_dict):
    if self._pad_length <= 0:
      return example_dict

    padded_example_dict = {}
    for key, sequence in example_dict.items():
      num_pads = self._pad_length - len(sequence)
      if num_pads < 0:
        raise ValueError('Feature %r too long' % key)

      padded_example_dict[key] = sequence + [_PAD_ID] * num_pads
    return padded_example_dict

  def process(self, source_target_list):
    example_dicts = [
        self._make_spm_example_dict(source_text, target_text)
        for source_text, target_text in source_target_list
    ]
    if self._packed_examples:
      example_dicts = pack_examples(
          example_dicts, has_inputs=True, packed_length=self._packed_length)
    for example_dict in example_dicts:
      try:
        padded_example_dict = self._pad_example_dict(example_dict)
      except ValueError:
        metrics.Metrics.counter('err_too_long', 'count').inc()
      else:
        yield to_example(padded_example_dict).SerializeToString()


class ValidateSentencePair(beam.DoFn):
  """String transformation applied to a sequence pair to validate the quality of a sample."""

  def __init__(self, min_distance):
    self.min_distance = min_distance

  def validate_similarity(self, source, target):
    """Discard sentence pair if their distance is less or equal than the threshold."""
    distance = difflib.SequenceMatcher(None, source, target).ratio()
    return distance > self.min_distance

  def process(self, sequence_pair):
    stripped_sequence_pair = sequence_pair.strip()

    try:
      source_sequence, target_sequence = stripped_sequence_pair.split('\t')
    except ValueError:
      metrics.Metrics.counter('erroneous_tab_split', 'count').inc()
      return

    source_tokens = source_sequence.split()
    target_tokens = target_sequence.split()

    if not self.validate_similarity(source_tokens, target_tokens):
      logging.info('Discarded due to high similarity: %r, %r', source_sequence,
                   target_sequence)
      metrics.Metrics.counter('high_similarity', 'count').inc()
      return

    metrics.Metrics.counter('samples_processed', 'count').inc()
    yield stripped_sequence_pair


def pipeline(root):
  """Method to pass into flume runner."""
  for i, tsv_in in enumerate(
      tf.io.gfile.glob(os.path.join(FLAGS.input_path, '*.tsv'))):
    print('Processing tsv input: %s' % tsv_in)
    tfr_out = tsv_in.replace('.tsv', '.tfr')
    num_output_shards = FLAGS.num_train_shards if 'train' in tsv_in else FLAGS.num_guide_shards
    _ = (
        root
        | 'Read RecordIO TSV__%s' % i >> beam.io.ReadFromText(tsv_in)
        | 'Validate sentence pair__%s' % i >> beam.ParDo(
            ValidateSentencePair(FLAGS.min_edit_distance))
        | 'Select TSV columns__%s' % i >> beam.ParDo(
            SelectTSVColumns(
                source_column=FLAGS.tsv_source_column,
                target_column=FLAGS.tsv_target_column))
        | 'Reshuffle__%s' % i >> beam.Reshuffle()
        | 'Batch elements__%s' % i >> beam.BatchElements(
            min_batch_size=1024, max_batch_size=1024)
        | 'Make tf.Examples__%s' % i >> beam.ParDo(
            PrepareTfExamples(
                spm_path=FLAGS.spm_path,
                packed_length=FLAGS.packed_length,
                pad_length=FLAGS.pad_length))
        | 'Write to tf.Record__%s' % i >> tfrecordio.WriteToTFRecord(
            tfr_out, num_shards=num_output_shards))


def main(unused_args):
  """Runs the Beam pipeline."""
  options = pipeline_options.PipelineOptions()
  p = beam.Pipeline(options=options)
  pipeline(p)
  p.run().wait_until_finish()


if __name__ == '__main__':
  app.run(main)
