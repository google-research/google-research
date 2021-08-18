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

r"""Loads a distilled Keras model and writes to disk as a savedmodel.

"""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from non_semantic_speech_benchmark.export_model import model_export_utils


flags.DEFINE_string('logdir', None, 'Dataset location.')
flags.DEFINE_string('output_dir', None, 'Place to write models to.')
flags.DEFINE_string('checkpoint_number', None, 'Optional checkpoint number to '
                    'use, instead of most recent.')
flags.DEFINE_bool('frontend', False, 'Whether to add the frontend.')
flags.DEFINE_bool('tflite', False, 'Whether to make a TFLite model.')

# Controls the model.
flags.DEFINE_integer('bottleneck_dimension', None, 'Dimension of bottleneck.')
flags.DEFINE_float('alpha', 1.0, 'Alpha controlling model size.')
flags.DEFINE_enum('mobilenet_size', 'small', ['tiny', 'small', 'large'],
                  'Size of mobilenet')
flags.DEFINE_bool('avg_pool', False, 'Whether to use average pool.')
flags.DEFINE_string('compressor', None,
                    'Whether to use bottleneck compression.')
flags.DEFINE_bool('qat', False, 'Whether to use quantization-aware training.')

FLAGS = flags.FLAGS


def main(unused_argv):
  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  model = model_export_utils.get_model(
      FLAGS.logdir,
      params={
          'bd': FLAGS.bottleneck_dimension,
          'al': FLAGS.alpha,
          'ms': FLAGS.mobilenet_size,
          'ap': FLAGS.avg_pool,
          'cop': FLAGS.compressor or None,
          'qat': FLAGS.qat,
      },
      tflite_friendly=False,
      checkpoint_number=FLAGS.checkpoint_number,
      include_frontend=FLAGS.frontend)
  tf.keras.models.save_model(model, FLAGS.output_dir)
  assert tf.io.gfile.exists(FLAGS.output_dir)
  logging.info('Successfully wrote to: %s', FLAGS.output_dir)

  # Sanity check the resulting model.
  logging.info('Sanity checking...')
  model_export_utils.sanity_check(
      FLAGS.include_frontend,
      FLAGS.output_dir,
      embedding_dim=FLAGS.bottleneck_dimension,
      tflite=False)

if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  flags.mark_flags_as_required(['output_dir', 'logdir'])
  app.run(main)
