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

# pylint:disable=line-too-long
r"""Beam job for model conversion.

"""
# pylint:enable=line-too-long

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf
# Import from main to force ourselves to use the same flags.
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_main  # pylint:disable=unused-import
from non_semantic_speech_benchmark.export_model import model_conversion_beam_utils as utils

flags.DEFINE_list('xids', None, 'List of job IDs to run.')
flags.DEFINE_string('base_experiment_dir', None, 'Base experiment dir.')
flags.DEFINE_string('output_dir', None, 'Base output dir.')
flags.DEFINE_bool('include_frontend', False, 'Whether to export with frontend.')
flags.DEFINE_list('conversion_types', ['tflite', 'savedmodel'],
                  'Type of conversions.')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Get metadata for conversion.
  metadata = utils.get_pipeline_metadata(FLAGS.base_experiment_dir, FLAGS.xids,
                                         FLAGS.output_dir,
                                         FLAGS.conversion_types)
  if not metadata:
    raise ValueError(
        f'No data found: {FLAGS.base_experiment_dir}, {FLAGS.xids}')
  logging.info('%i models in %i xids.', len(metadata), len(FLAGS.xids))

  # Check that models don't already exist, and create directories if necessary.
  for m in metadata:
    utils.sanity_check_output_filename(m.output_filename)

  logging.info('Starting to create flume pipeline...')
  # Make and run beam pipeline.
  beam_options = None
  def _convert_and_write_model(m):
    utils.convert_and_write_model(
        m, include_frontend=FLAGS.include_frontend, sanity_check=True)
    return m

  with beam.Pipeline(beam_options) as root:
    _ = (
        root
        | 'MakeCollection' >> beam.Create(metadata)
        | 'ConvertAndWriteModelsToDisk' >> beam.Map(_convert_and_write_model))


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  flags.mark_flags_as_required(['xids', 'base_experiment_dir', 'output_dir'])
  app.run(main)
