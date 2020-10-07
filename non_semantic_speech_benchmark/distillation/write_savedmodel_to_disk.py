# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import models


flags.DEFINE_string('logdir', None, 'Dataset location.')
flags.DEFINE_string('checkpoint_filename', None, 'Optional.')
flags.DEFINE_string('output_directory', None, 'Place to write savedmodel.')

# Controls the model.
flags.DEFINE_integer('bottleneck_dimension', None, 'Dimension of bottleneck.')
flags.DEFINE_integer('output_dimension', None, 'Dimension of targets.')
flags.DEFINE_float('alpha', 1.0, 'Alpha controlling model size.')

FLAGS = flags.FLAGS


def load_and_write_model(logdir, checkpoint_filename, output_directory):
  model = models.get_keras_model(
      FLAGS.bottleneck_dimension, FLAGS.output_dimension, alpha=FLAGS.alpha)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_to_load = tf.train.latest_checkpoint(logdir, checkpoint_filename)
  checkpoint.restore(checkpoint_to_load)
  tf.keras.models.save_model(model, output_directory)


def main(unused_argv):
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  if FLAGS.checkpoint_filename:
    raise ValueError('Implement me.')
  assert FLAGS.logdir
  load_and_write_model(
      FLAGS.logdir,
      FLAGS.checkpoint_filename,
      FLAGS.output_directory)


if __name__ == '__main__':
  flags.mark_flags_as_required(['logdir', 'output_directory'])
  app.run(main)
