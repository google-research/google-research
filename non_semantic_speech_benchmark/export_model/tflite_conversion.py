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
r"""Converts distilled models to TFLite by iterating over experiment folders.

The aim of this file is:

1. To get TFLite models corresponding to the trained models, but only returning
the embedding (and not the target output used during training).

"""
# pylint:enable=line-too-long

import collections
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from non_semantic_speech_benchmark.export_model import model_export_utils

flags.DEFINE_string(
    'experiment_dir', None,
    '(CNS) Directory containing directories with parametrized names like '
    '"1-al=1.0,ap=False,cop=False,lr=0.0001,ms=small,qat=False,tbs=512". '
    'Note that only the mentioned hyper-params are supported right now.')
flags.DEFINE_string('output_dir', None, 'Place to write models to.')
flags.DEFINE_string('checkpoint_number', None, 'Optional checkpoint number to '
                    'use, instead of most recent.')
flags.DEFINE_boolean('quantize', False,
                     'Whether to quantize converted models if possible.')
flags.DEFINE_boolean('include_frontend', False, 'Whether to include frontend.')
flags.DEFINE_boolean('sanity_checks', True, 'Whether to run inference checks.')

FLAGS = flags.FLAGS

Metadata = collections.namedtuple(
    'Metadata', ['param_str', 'params', 'experiment_dir', 'output_filename'])


def main(_):
  if tf.io.gfile.glob(os.path.join(FLAGS.output_dir, 'model_*.tflite')):
    existing_files = tf.io.gfile.glob(os.path.join(
        FLAGS.output_dir, 'model_*.tflite'))
    raise ValueError(f'Models cant already exist: {existing_files}')
  else:
    tf.io.gfile.makedirs(FLAGS.output_dir)

  # Get experiment dirs names, params, and output location.
  metadata = []
  exp_names = model_export_utils.get_experiment_dirs(FLAGS.experiment_dir)
  if not exp_names:
    raise ValueError(f'No experiments found: {FLAGS.experiment_dir}')
  for i, exp_name in enumerate(exp_names):
    cur_metadata = Metadata(
        exp_name,
        model_export_utils.get_params(exp_name),
        os.path.join(FLAGS.experiment_dir, exp_name),
        os.path.join(FLAGS.output_dir, f'model_{i}.tflite'))
    metadata.append(cur_metadata)
  logging.info('Number of metadata: %i', len(metadata))

  for m in metadata:
    logging.info('Working on experiment dir: %s', m.param_str)

    # Export SavedModel & convert to TFLite
    # Note that we keep over-writing the SavedModel while converting experiments
    # to TFLite, since we only care about the final flatbuffer models.
    static_model = model_export_utils.get_model(
        checkpoint_folder_path=m.experiment_dir,
        params=m.params,
        tflite_friendly=True,
        checkpoint_number=FLAGS.checkpoint_number,
        include_frontend=FLAGS.include_frontend)

    model_export_utils.convert_tflite_model(
        static_model, quantize=m.params['qat'], model_path=m.output_filename)

    if FLAGS.sanity_checks:
      logging.info('Sanity checking...')
      model_export_utils.sanity_check(
          FLAGS.include_frontend,
          m.output_filename,
          embedding_dim=1024,
          tflite=True)

  logging.info('Total TFLite models generated: %i', len(metadata))


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  flags.mark_flags_as_required(['experiment_dir', 'output_dir'])
  app.run(main)
