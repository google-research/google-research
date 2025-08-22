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

"""Beam pipeline for creating tf records for train and eval."""
import json
import re
from typing import Iterator, Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import ml_collections
from ml_collections import config_flags
import tensorflow as tf

from mave.benchmark.data import data_utils

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default=(
        'experimental/table_talk/product_attribute_data/benchmark/configs.py'),
    help_string='Training configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config'])

_MODEL_TYPE = flags.DEFINE_string(
    'model_type', default=None, help='The model type.', required=True)

_INPUT_JSON_LINES_FILEPATTERN = flags.DEFINE_string(
    'input_json_lines_filepattern',
    default=None,
    help='The input JSON Lines file pattern of the examples.',
    required=True)

_OUTPUT_TF_RECORDS_FILEPATTERN = flags.DEFINE_string(
    'output_tf_records_filepattern',
    default=None,
    help=('The output TF Records filepattern. If not set, the output files will'
          ' be in the same dir as the input JSON Lines files, with filepattern:'
          ' "input_filename_<model_type>_tfrecords@*".'))

_DEBUG = flags.DEFINE_boolean(
    'debug',
    default=True,
    help='Whether the ooutput TF Records contain debug info.')


def _get_config():
  """Returns a frozen config dict by updating dynamic fields."""
  config = ml_collections.ConfigDict(_CONFIG.value)
  if _MODEL_TYPE.value == 'bert':
    config.model_type = 'bert'
    config.data.use_category = True
    config.data.use_attribute_key = True
    config.data.use_cls = True
    config.data.use_sep = True
  elif _MODEL_TYPE.value == 'bilstm_crf':
    config.model_type = 'bilstm_crf'
    config.data.use_category = False
    config.data.use_attribute_key = False
    config.data.use_cls = False
    config.data.use_sep = False
  elif _MODEL_TYPE.value == 'etc':
    config.model_type = 'etc'
  else:
    raise ValueError(
        f'Invalid model_type for this binary {config.model_type!r}')
  config.data.debug = _DEBUG.value
  logging.info(config)
  return ml_collections.FrozenConfigDict(config)


class CreateTFRecordFn(beam.DoFn):
  """DoFn to convert a Json data point to a TF Reccord."""

  def __init__(self, config, *unused_args,
               **unused_kwargs):
    self._converter = data_utils.get_tf_record_converter(config)

  def process(self, element, *args,
              **kwargs):
    json_example = json.loads(element.strip())
    yield from self._converter.convert(json_example)


def pipeline(root):
  """Beam pipeline to run."""

  config = _get_config()

  input_json_lines_filepaths = tf.io.gfile.glob(
      _INPUT_JSON_LINES_FILEPATTERN.value)

  logging.info('Num input file paths: %s', len(input_json_lines_filepaths))
  logging.info('%s', '\n'.join(input_json_lines_filepaths))

  for index, input_json_lines_filepath in enumerate(input_json_lines_filepaths):
    if _OUTPUT_TF_RECORDS_FILEPATTERN.value:
      output_tfrecord_filepattern = _OUTPUT_TF_RECORDS_FILEPATTERN.value
    else:
      output_tfrecord_filepattern = re.sub(
          r'(.jsonl)?$',
          f'_{config.model_type}_tfrecord@*',
          input_json_lines_filepath,
          count=1)
    output_counts_filename = re.sub(
        r'@?(\*?|\d*)$', '_counts', output_tfrecord_filepattern, count=1)

    tf_records = (
        root
        | f'{index}_ReadJsonLines' >>
        beam.io.textio.ReadFromText(input_json_lines_filepath)
        | f'{index}_CreateTFRecord' >> beam.ParDo(CreateTFRecordFn(config)))

    _ = (
        tf_records
        | f'{index}_WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
            output_tfrecord_filepattern,
            coder=beam.coders.ProtoCoder(tf.train.Example)))

    _ = (
        tf_records
        | f'{index}_CountTFRecords' >> beam.combiners.Count.Globally()
        | f'{index}_JsonDumps' >> beam.Map(lambda x: json.dumps(x, indent=2))
        | f'{index}_WriteCounts' >> beam.io.WriteToText(
            output_counts_filename,
            shard_name_template='',  # To force unsharded output.
        ))


def main(unused_argv):
  # To enable distributed workflows, follow instructions at
  # https://beam.apache.org/documentation/programming-guide/
  # to set pipeline options.
  with beam.Pipeline() as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)
