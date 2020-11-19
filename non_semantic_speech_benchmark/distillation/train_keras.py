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

# Lint as: python3
"""Trains on embeddings using Keras."""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub  # pylint:disable=g-bad-import-order

from non_semantic_speech_benchmark.distillation import get_data
from non_semantic_speech_benchmark.distillation import models

FLAGS = flags.FLAGS

# Data config flags.
flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_integer('output_dimension', None, 'Dimension of targets.')

flags.DEFINE_boolean(
    'precomputed_frontend_and_targets', False,
    'Flag to enable training with precomputed frontend and targets. '
    'If True, `file_pattern` must point to tf_records of tf.Examples. '
    'See get_data.get_precomputed_data for details about tf.Example formatting. '
    'If True, `teacher_model_hub`, `output_key`, `samples_key` '
    'and `min_length` flags are ignored.')
flags.DEFINE_string(
    'frontend_key', None, 'Frontend feature key in precomputed tf.Examples. '
    'This flag is ignored if `precomputed_frontend_and_targets` is False.')
flags.DEFINE_string(
    'target_key', None, 'Teacher embedding key in precomputed tf.Examples. '
    'This flag is ignored if `precomputed_frontend_and_targets` is False.')

flags.DEFINE_string('teacher_model_hub', None, 'Hub teacher model.')
flags.DEFINE_string('output_key', None, 'Teacher model output_key.')
flags.DEFINE_string('samples_key', None, 'Samples name.')
flags.DEFINE_integer('min_length', 16000, 'Minimum audio sample length.')
flags.DEFINE_alias('ml', 'min_length')

# Student network config flags.
flags.DEFINE_integer(
    'bottleneck_dimension', None, 'Dimension of bottleneck. '
    'If 0, bottleneck layer is excluded.')
flags.DEFINE_float('alpha', 1.0, 'Alpha controlling MobileNet width.')
flags.DEFINE_boolean('average_pool', False, 'Average pool MobileNet output.')
flags.DEFINE_string(
    'mobilenet_size', 'small',
    'Size specification for MobileNet in student model. '
    'valid entries are `tiny`, `small`, and `large`.')
flags.DEFINE_alias('bd', 'bottleneck_dimension')
flags.DEFINE_alias('al', 'alpha')
flags.DEFINE_alias('ap', 'average_pool')
flags.DEFINE_alias('mnet', 'mobilenet_size')

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


def train_and_report(debug=False):
  """Trains the classifier."""
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.train_batch_size)

  reader = tf.data.TFRecordDataset
  if FLAGS.precomputed_frontend_and_targets:
    ds = get_data.get_precomputed_data(
        file_pattern=FLAGS.file_pattern,
        output_dimension=FLAGS.output_dimension,
        frontend_key=FLAGS.frontend_key,
        target_key=FLAGS.target_key,
        batch_size=FLAGS.train_batch_size,
        num_epochs=FLAGS.num_epochs,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    ds.element_spec[0].shape.assert_has_rank(3)  # log Mel spectrograms
    ds.element_spec[1].shape.assert_has_rank(2)  # teacher embeddings
  else:
    ds = get_data.get_data(
        file_pattern=FLAGS.file_pattern,
        teacher_fn=get_data.savedmodel_to_func(
            hub.load(FLAGS.teacher_model_hub), FLAGS.output_key),
        output_dimension=FLAGS.output_dimension,
        reader=reader,
        samples_key=FLAGS.samples_key,
        min_length=FLAGS.min_length,
        batch_size=FLAGS.train_batch_size,
        loop_forever=True,
        shuffle=True,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    assert len(ds.element_spec) == 2, ds.element_spec
    ds.element_spec[0].shape.assert_has_rank(2)  # audio samples
    ds.element_spec[1].shape.assert_has_rank(2)  # teacher embeddings
  output_dimension = ds.element_spec[1].shape[1]
  assert output_dimension == FLAGS.output_dimension

  # Create model, loss, and other objects.
  model = models.get_keras_model(
      bottleneck_dimension=FLAGS.bottleneck_dimension,
      output_dimension=output_dimension,
      alpha=FLAGS.alpha,
      mobilenet_size=FLAGS.mobilenet_size,
      frontend=not FLAGS.precomputed_frontend_and_targets,
      avg_pool=FLAGS.average_pool)
  # Define loss and optimizer hyparameters.
  loss_obj = tf.keras.losses.MeanSquaredError(name='mse_loss')
  opt = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
  # Add additional metrics to track.
  train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
  train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
  summary_writer = tf.summary.create_file_writer(FLAGS.logdir)
  train_step = get_train_step(
      model, loss_obj, opt, train_loss, train_mae, summary_writer)
  global_step = opt.iterations
  checkpoint = tf.train.Checkpoint(model=model, global_step=global_step)
  manager = tf.train.CheckpointManager(
      checkpoint, FLAGS.logdir, max_to_keep=FLAGS.checkpoint_max_to_keep)
  logging.info('Checkpoint prefix: %s', FLAGS.logdir)
  checkpoint.restore(manager.latest_checkpoint)

  if debug: return
  for inputs, targets in ds:
    if FLAGS.precomputed_frontend_and_targets:  # inputs are spectrograms
      inputs.shape.assert_has_rank(3)
      inputs.shape.assert_is_compatible_with([FLAGS.train_batch_size, 96, 64])
    else:  # inputs are audio vectors
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
  @tf.function
  def train_step(wav_samples, targets, step):
    with tf.GradientTape() as tape:
      logits = model(wav_samples, training=True)
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

  if FLAGS.precomputed_frontend_and_targets:
    assert FLAGS.frontend_key
    assert FLAGS.target_key
  else:
    assert FLAGS.teacher_model_hub
    assert FLAGS.output_key
    assert FLAGS.samples_key

  tf.enable_v2_behavior()
  assert tf.executing_eagerly()
  train_and_report()


if __name__ == '__main__':
  app.run(main)
