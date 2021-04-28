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

r"""Loads a graph and checkpoint, and writes to disk as a savedmodel.

"""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import models


flags.DEFINE_string('logdir', None, 'Dataset location.')
flags.DEFINE_string('checkpoint_filename', None, 'Optional.')
flags.DEFINE_string('output_directory', None, 'Place to write savedmodel.')
flags.DEFINE_bool('frontend', False, 'Whether to add the frontend.')
flags.DEFINE_bool('tflite', False, 'Whether to make a TFLite model.')

# Controls the model.
flags.DEFINE_integer('bottleneck_dimension', None, 'Dimension of bottleneck.')
flags.DEFINE_float('alpha', 1.0, 'Alpha controlling model size.')
flags.DEFINE_string('mobilenet_size', 'small', 'Size of mobilenet')
flags.DEFINE_bool('avg_pool', False, 'Whether to use average pool.')
flags.DEFINE_string('compressor', None,
                    'Whether to use bottleneck compression.')
flags.DEFINE_bool('qat', False, 'Whether to use quantization-aware training.')

FLAGS = flags.FLAGS


def load_and_write_model(keras_model_args, checkpoint_to_load,
                         output_directory):
  model = models.get_keras_model(**keras_model_args)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_to_load).expect_partial()
  tf.keras.models.save_model(model, output_directory)


def main(unused_argv):
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  if FLAGS.checkpoint_filename:
    assert tf.train.load_checkpoint(FLAGS.checkpoint_filename)
    checkpoint_to_load = FLAGS.checkpoint_filename
  else:
    assert FLAGS.logdir
    checkpoint_to_load = tf.train.latest_checkpoint(FLAGS.logdir)
  keras_model_args = {
      'bottleneck_dimension': FLAGS.bottleneck_dimension,
      'output_dimension': None,
      'alpha': FLAGS.alpha,
      'mobilenet_size': FLAGS.mobilenet_size,
      'frontend': FLAGS.frontend,
      'avg_pool': FLAGS.avg_pool,
      'compressor': FLAGS.compressor,
      'quantize_aware_training': FLAGS.qat,
      'tflite': FLAGS.tflite,
  }
  load_and_write_model(
      keras_model_args, checkpoint_to_load, FLAGS.output_directory)
  assert tf.io.gfile.exists(FLAGS.output_directory)
  logging.info('Successfully wrote to: %s', FLAGS.output_directory)


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_directory'])
  flags.mark_flags_as_mutual_exclusive(['logdir', 'checkpoint_filename'])
  app.run(main)
