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
"""Trains on embeddings using Keras.
"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub  # pylint:disable=g-bad-import-order


from non_semantic_speech_benchmark.distillation import get_data
from non_semantic_speech_benchmark.distillation import models

from non_semantic_speech_benchmark.distillation.compression_lib import compression_op as compression
from non_semantic_speech_benchmark.distillation.compression_lib import compression_wrapper

FLAGS = flags.FLAGS

# Data config flags.
flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_integer('output_dimension', None, 'Dimension of targets.')

flags.DEFINE_boolean(
    'precomputed_targets', False,
    'Flag to enable training with precomputed targets. '
    'If True, `file_pattern` must point to precomputed targets, and '
    '`target_key` must be supplied.')
flags.DEFINE_string(
    'target_key', None, 'Teacher embedding key in precomputed tf.Examples. '
    'This flag is ignored if `precomputed_targets` is False.')

flags.DEFINE_string('teacher_model_hub', None, 'Hub teacher model.')
flags.DEFINE_string('output_key', None, 'Teacher model output_key.')
flags.DEFINE_string('samples_key', None, 'Samples name.')
flags.DEFINE_integer('min_length', 16000, 'Minimum audio sample length.')
flags.DEFINE_alias('ml', 'min_length')

# Student network config flags.
flags.DEFINE_integer(
    'bottleneck_dimension', None, 'Dimension of bottleneck. '
    'If 0, bottleneck layer is excluded.')
flags.DEFINE_alias('bd', 'bottleneck_dimension')
flags.DEFINE_string(
    'model_type', 'mobilenet_debug_1.0_False',
    'Specification for student model.')
flags.DEFINE_alias('mt', 'model_type')

# Training config flags.
flags.DEFINE_integer('train_batch_size', 1, 'Hyperparameter: batch size.')
flags.DEFINE_alias('tbs', 'train_batch_size')
flags.DEFINE_integer('shuffle_buffer_size', None, 'shuffle_buffer_size')
flags.DEFINE_float('lr', 0.001, 'Hyperparameter: learning rate.')
flags.DEFINE_string('logdir', None,
                    'Path to directory where to store summaries.')
flags.DEFINE_integer('training_steps', 1000,
                     'The number of steps to run training for.')
flags.DEFINE_integer('measurement_store_interval', 10,
                     'The number of steps between storing objective value in '
                     'measurements.')
flags.DEFINE_integer(
    'checkpoint_max_to_keep', None,
    'Number of previous checkpoints to save to disk.'
    'Default (None) is to store all checkpoints.')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train for.')
flags.DEFINE_alias('e', 'num_epochs')


# Quantization
flags.DEFINE_boolean(
    'quantize_aware_training', False, 'Dynamically '
    'quantize final layer weights during training.')
flags.DEFINE_alias('qat', 'quantize_aware_training')

# Compression
flags.DEFINE_boolean(
    'compression_op', False, 'Compress pre-bottleneck '
    'layer dynamically during training using SVD.')
flags.DEFINE_alias('cop', 'compression_op')
flags.DEFINE_integer('comp_rank', 100, 'Inner dimension of '
                     'compressed matrices.')
flags.DEFINE_integer('comp_freq', 10, 'Compressed matrix update frequency')
flags.DEFINE_integer('comp_begin_step', 0,
                     'Training step to begin compression.')
flags.DEFINE_integer(
    'comp_end_step', -1, 'Training step to end compression. '
    '-1 means compression never ends.')
flags.DEFINE_float(
    'alpha_step_size', 0.01, 'Amount to decrement alpha each time '
    'a compression op takes place.')


def train_and_report(debug=False):
  """Trains the classifier."""
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.train_batch_size)

  reader = tf.data.TFRecordDataset
  target_key = FLAGS.target_key
  if FLAGS.precomputed_targets:
    teacher_fn = None
    assert target_key is not None
    assert FLAGS.output_key is None
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
      batch_size=FLAGS.train_batch_size,
      loop_forever=True,
      shuffle=True,
      teacher_fn=teacher_fn,
      target_key=target_key,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)
  assert len(ds.element_spec) == 2, ds.element_spec
  ds.element_spec[0].shape.assert_has_rank(2)  # audio samples
  ds.element_spec[1].shape.assert_has_rank(2)  # teacher embeddings
  output_dimension = ds.element_spec[1].shape[1]
  assert output_dimension == FLAGS.output_dimension

  # Define loss and optimizer hyparameters.
  loss_obj = tf.keras.losses.MeanSquaredError(name='mse_loss')
  opt = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
  global_step = opt.iterations
  # Create model, loss, and other objects.
  compressor = None
  if FLAGS.compression_op:
    custom_params = ','.join([
        'compression_frequency=%d',
        'rank=%d',
        'begin_compression_step=%d',
        'end_compression_step=%d',
        'alpha_decrement_value=%d',
    ]) % (FLAGS.comp_freq, FLAGS.comp_rank, FLAGS.comp_begin_step,
          FLAGS.comp_end_step, FLAGS.alpha_step_size)
    compression_params = compression.CompressionOp.get_default_hparams().parse(
        custom_params)
    compressor = compression_wrapper.get_apply_compression(
        compression_params, global_step=global_step)
  model = models.get_keras_model(
      model_type=FLAGS.model_type,
      bottleneck_dimension=FLAGS.bottleneck_dimension,
      output_dimension=output_dimension,
      frontend=True,
      compressor=compressor,
      quantize_aware_training=FLAGS.quantize_aware_training)
  model.summary()
  # Add additional metrics to track.
  train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
  train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
  summary_writer = tf.summary.create_file_writer(FLAGS.logdir)
  train_step = get_train_step(
      model, loss_obj, opt, train_loss, train_mae, summary_writer)
  checkpoint = tf.train.Checkpoint(model=model, global_step=global_step)
  manager = tf.train.CheckpointManager(
      checkpoint, FLAGS.logdir, max_to_keep=FLAGS.checkpoint_max_to_keep)
  logging.info('Checkpoint prefix: %s', FLAGS.logdir)
  checkpoint.restore(manager.latest_checkpoint)

  if debug: return
  for inputs, targets in ds:
    # Inputs are audio vectors.
    inputs.shape.assert_has_rank(2)
    inputs.shape.assert_is_compatible_with(
        [FLAGS.train_batch_size, FLAGS.min_length])
    targets.shape.assert_has_rank(2)
    targets.shape.assert_is_compatible_with(
        [FLAGS.train_batch_size, FLAGS.output_dimension])
    train_step(inputs, targets, global_step)
    # Optional print output and save model.
    if global_step % 10 == 0:
      logging.info('step: %i, train loss: %f, train mean abs error: %f',
                   global_step, train_loss.result(), train_mae.result())
    if global_step % FLAGS.measurement_store_interval == 0:
      manager.save(checkpoint_number=global_step)

  manager.save(checkpoint_number=global_step)
  logging.info('Finished training.')


def get_train_step(model, loss_obj, opt, train_loss, train_mae, summary_writer):
  """Returns a function for train step."""
  def train_step(wav_samples, targets, step):
    with tf.GradientTape() as tape:
      logits = model(wav_samples, training=True)['embedding_to_target']
      assert model.trainable_variables
      logits.shape.assert_is_compatible_with(targets.shape)
      loss_value = loss_obj(y_true=targets, y_pred=logits)
    # Grads and optimizer.
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # Record loss.
    train_loss.update_state(y_pred=targets, y_true=logits)
    train_mae.update_state(y_pred=targets, y_true=logits)

    # Summaries.
    with summary_writer.as_default():
      tf.summary.scalar('mse_loss', loss_value, step=step)
      tf.summary.scalar('mse_loss_smoothed', train_loss.result(), step=step)
      tf.summary.scalar('mae', train_mae.result(), step=step)
  return train_step


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.output_dimension
  assert FLAGS.bottleneck_dimension >= 0
  assert FLAGS.shuffle_buffer_size
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

  # Incompatible tools.
  assert not (FLAGS.quantize_aware_training and FLAGS.compression_op)

  tf.enable_v2_behavior()
  assert tf.executing_eagerly()
  train_and_report()


if __name__ == '__main__':
  app.run(main)
