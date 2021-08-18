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

# Lint as: python3
"""Eval a Keras model on embeddings."""

import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub  # pylint:disable=g-bad-import-order

from non_semantic_speech_benchmark.distillation import get_data
from non_semantic_speech_benchmark.distillation import models

FLAGS = flags.FLAGS

flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_string('sk', None, 'Samples name.')
flags.DEFINE_alias('samples_key', 'sk')
flags.DEFINE_integer('ml', 16000, 'Minimum length.')
flags.DEFINE_alias('min_length', 'ml')

flags.DEFINE_boolean(
    'precomputed_targets', False,
    'Flag to enable training with precomputed targets. '
    'If True, `file_pattern` must point to precomputed targets, and '
    '`target_key` must be supplied.')
flags.DEFINE_string(
    'target_key', None, 'Teacher embedding key in precomputed tf.Examples. '
    'This flag is ignored if `precomputed_targets` is False.')


# Teacher / student network flags.
flags.DEFINE_string('teacher_model_hub', None, 'Hub teacher model.')
flags.DEFINE_string('output_key', None, 'Teacher model output_key.')
flags.DEFINE_integer('output_dimension', None, 'Dimension of targets.')
flags.DEFINE_integer('bd', None, 'Dimension of bottleneck.')
flags.DEFINE_alias('bottleneck_dimension', 'bd')
flags.DEFINE_string(
    'model_type', 'mobilenet_debug_1.0_False',
    'Specification for student model. For mobilenet, includes')
flags.DEFINE_alias('mt', 'model_type')

flags.DEFINE_integer('batch_size', None, 'The number of images in each batch.')
flags.DEFINE_integer('tbs', None, 'not used')

flags.DEFINE_integer('nc', None, 'num_clusters')
flags.DEFINE_float('alpha_init', None, 'Initial autopool alpha.')
flags.DEFINE_alias('ai', 'alpha_init')
flags.DEFINE_boolean('ubn', None, 'Whether to normalize')
flags.DEFINE_float('lr', None, 'not used')

# Not used.
flags.DEFINE_float('qat', None, 'not used')
flags.DEFINE_float('cop', None, 'not used')

flags.DEFINE_string('logdir', None,
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')
flags.DEFINE_integer('take_fixed_data', None,
                     'If not `None`, take a fixed number of data elements.')
flags.DEFINE_integer('timeout', 7200, 'Wait-for-checkpoint timeout.')


def eval_and_report():
  """Eval on voxceleb."""
  tf.logging.info('samples_key: %s', FLAGS.samples_key)
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.batch_size)

  writer = tf.summary.create_file_writer(FLAGS.eval_dir)
  model = models.get_keras_model(
      model_type=FLAGS.model_type,
      bottleneck_dimension=FLAGS.bottleneck_dimension,
      output_dimension=FLAGS.output_dimension,
      frontend=True)
  checkpoint = tf.train.Checkpoint(model=model)

  for ckpt in tf.train.checkpoints_iterator(
      FLAGS.logdir, timeout=FLAGS.timeout):
    assert 'ckpt-' in ckpt, ckpt
    step = ckpt.split('ckpt-')[-1]
    logging.info('Starting to evaluate step: %s.', step)

    checkpoint.restore(ckpt)

    logging.info('Loaded weights for eval step: %s.', step)

    reader = tf.data.TFRecordDataset
    target_key = FLAGS.target_key
    if FLAGS.precomputed_targets:
      teacher_fn = None
      assert target_key is not None
    else:
      teacher_fn = get_data.savedmodel_to_func(
          hub.load(FLAGS.teacher_model_hub), FLAGS.output_key)
      assert target_key is None
    ds = get_data.get_data(
        file_pattern=FLAGS.file_pattern,
        output_dimension=FLAGS.output_dimension,
        reader=reader,
        samples_key=FLAGS.samples_key,
        min_length=FLAGS.min_length,
        batch_size=FLAGS.batch_size,
        loop_forever=False,
        shuffle=False,
        teacher_fn=teacher_fn,
        target_key=target_key)
    logging.info('Got dataset for eval step: %s.', step)
    if FLAGS.take_fixed_data:
      ds = ds.take(FLAGS.take_fixed_data)

    mse_m = tf.keras.metrics.MeanSquaredError()
    mae_m = tf.keras.metrics.MeanAbsoluteError()

    logging.info('Starting the ds loop...')
    count, ex_count = 0, 0
    s = time.time()
    for wav_samples, targets in ds:
      wav_samples.shape.assert_is_compatible_with([None, FLAGS.min_length])
      targets.shape.assert_is_compatible_with([None, FLAGS.output_dimension])

      logits = model(wav_samples, training=False)['embedding_to_target']
      logits.shape.assert_is_compatible_with(targets.shape)

      mse_m.update_state(y_true=targets, y_pred=logits)
      mae_m.update_state(y_true=targets, y_pred=logits)
      ex_count += logits.shape[0]
      count += 1
      logging.info('Saw %i examples after %i iterations as %.2f secs...',
                   ex_count, count,
                   time.time() - s)
    with writer.as_default():
      tf.summary.scalar('mse', mse_m.result().numpy(), step=int(step))
      tf.summary.scalar('mae', mae_m.result().numpy(), step=int(step))
    logging.info('Done with eval step: %s in %.2f secs.', step, time.time() - s)


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.output_dimension
  assert FLAGS.bottleneck_dimension >= 0
  assert FLAGS.logdir
  assert FLAGS.samples_key

  if FLAGS.precomputed_targets:
    assert FLAGS.teacher_model_hub is None
    assert FLAGS.output_key is None
    assert FLAGS.target_key
  else:
    assert FLAGS.teacher_model_hub
    assert FLAGS.output_key
    assert FLAGS.target_key is None

  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  eval_and_report()


if __name__ == '__main__':
  app.run(main)
