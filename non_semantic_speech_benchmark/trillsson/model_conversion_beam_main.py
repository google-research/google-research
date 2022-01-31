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

# pylint:disable=line-too-long
r"""Beam job for model conversion for TRILLsson jobs.

"""
# pylint:enable=line-too-long

import os
from typing import Any, Dict, Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
# Import from main to force ourselves to use the same flags.
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_main  # pylint:disable=unused-import
from non_semantic_speech_benchmark.export_model import model_conversion_beam_utils as utils
from non_semantic_speech_benchmark.trillsson import models

flags.DEFINE_list('xids', None, 'List of job IDs to run.')
flags.DEFINE_string('base_experiment_dir', None, 'Base experiment dir.')
flags.DEFINE_string('output_dir', None, 'Base output dir.')
flags.DEFINE_string('output_suffix', None,
                    'Output dir is {output_dir}/{xid}/{output_suffix}.')
flags.DEFINE_bool('sanity_check', True, 'Whether to run sanity check.')

FLAGS = flags.FLAGS


def _get_model(params,
               checkpoint_folder_path):
  model_type = params['mt']
  static_model = models.get_keras_model(
      model_type=model_type,
      manually_average=True if 'ast' in model_type else False)
  checkpoint = tf.train.Checkpoint(model=static_model)
  checkpoint_to_load = tf.train.latest_checkpoint(checkpoint_folder_path)
  checkpoint.restore(checkpoint_to_load).expect_partial()
  return static_model


def convert_and_write_model(m):
  """Convert model and write to disk for data prep."""
  if not tf.io.gfile.exists(os.path.dirname(m.output_filename)):
    raise ValueError(
        f'Existing dir didn\'t exist: {os.path.dirname(m.output_filename)}')

  logging.info('Working on experiment dir: %s', m.experiment_dir)

  model = _get_model(params=m.params, checkpoint_folder_path=m.experiment_dir)
  tf.keras.models.save_model(model, m.output_filename)

  logging.info('Sanity checking...')
  model = tf.saved_model.load(m.output_filename)
  model_input = tf.zeros([2, 64000])
  output = model(model_input)['embedding'].numpy()
  np.testing.assert_array_equal(output.shape, (2, 1024))
  logging.info('Model "%s" worked.', m.output_filename)


def main(unused_argv):
  beam_options = None

  # Get metadata for conversion.
  metadata = utils.get_pipeline_metadata(
      FLAGS.base_experiment_dir,
      FLAGS.xids,
      FLAGS.output_dir,
      conversion_types=[utils.SAVEDMODEL_],
      output_suffix=FLAGS.output_suffix)
  if not metadata:
    raise ValueError(
        f'No data found: {FLAGS.base_experiment_dir}, {FLAGS.xids}')
  logging.info('%i models in %i xids.', len(metadata), len(FLAGS.xids))

  # Check that models don't already exist, and create directories if necessary.
  for m in metadata:
    utils.sanity_check_output_filename(m.output_filename)

  logging.info('Starting to create flume pipeline...')

  # Make and run beam pipeline.
  with beam.Pipeline(beam_options) as root:
    _ = (
        root
        | 'MakeMetadataCollection' >> beam.Create(metadata)
        | 'ConvertAndWriteModelsToDisk' >> beam.Map(convert_and_write_model))


if __name__ == '__main__':
  flags.mark_flags_as_required(['xids', 'base_experiment_dir', 'output_dir'])
  app.run(main)
