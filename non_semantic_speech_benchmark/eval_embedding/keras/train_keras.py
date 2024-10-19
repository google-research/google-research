# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Trains on embeddings using Keras."""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from non_semantic_speech_benchmark.eval_embedding.keras import get_data
from non_semantic_speech_benchmark.eval_embedding.keras import models

FLAGS = flags.FLAGS

flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_string('embedding_name', None, 'Embedding name.')
flags.DEFINE_alias('en', 'embedding_name')
flags.DEFINE_integer('embedding_dimension', None, 'Embedding dimension.')
flags.DEFINE_alias('ed', 'embedding_dimension')
flags.DEFINE_string('label_name', None, 'Name of label to use.')
flags.DEFINE_list('label_list', None, 'List of possible label values.')
flags.DEFINE_list('bucket_boundaries', ['99999'],
                  'bucket_boundaries for data. Default is all one bucket.')

flags.DEFINE_integer('train_batch_size', 1, 'Hyperparameter: batch size.')
flags.DEFINE_alias('tbs', 'train_batch_size')
flags.DEFINE_integer('shuffle_buffer_size', None, 'shuffle_buffer_size')

flags.DEFINE_integer('num_clusters', None, 'num_clusters')
flags.DEFINE_alias('nc', 'num_clusters')
flags.DEFINE_float('alpha_init', None, 'Initial autopool alpha.')
flags.DEFINE_alias('ai', 'alpha_init')
flags.DEFINE_boolean('use_batch_normalization', None,
                     'Whether to use batch normalization.')
flags.DEFINE_alias('ubn', 'use_batch_normalization')
flags.DEFINE_float('lr', 0.001, 'Hyperparameter: learning rate.')

flags.DEFINE_string('logdir', None,
                    'Path to directory where to store summaries.')

flags.DEFINE_integer('training_steps', 1000,
                     'The number of steps to run training for.')
flags.DEFINE_integer('measurement_store_interval', 10,
                     'The number of steps between storing objective value in '
                     'measurements.')
flags.DEFINE_boolean('preaverage', False, 'Whether to preaverage.')


def train_and_report(debug=False):
  """Trains the classifier."""
  logging.info('embedding_name: %s', FLAGS.embedding_name)
  logging.info('embedding_dimension: %s', FLAGS.embedding_dimension)
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.train_batch_size)

  reader = tf.data.TFRecordDataset
  ds = get_data.get_data(
      file_pattern=FLAGS.file_pattern,
      reader=reader,
      embedding_name=FLAGS.embedding_name,
      embedding_dim=FLAGS.embedding_dimension,
      label_name=FLAGS.label_name,
      label_list=FLAGS.label_list,
      bucket_boundaries=FLAGS.bucket_boundaries,
      bucket_batch_sizes=[FLAGS.train_batch_size] * (len(FLAGS.bucket_boundaries) + 1),  # pylint:disable=line-too-long
      loop_forever=True,
      shuffle=True,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      preaverage=FLAGS.preaverage)

  # Create model, loss, and other objects.
  y_onehot_spec = ds.element_spec[1]
  assert len(y_onehot_spec.shape) == 2, y_onehot_spec.shape
  num_classes = y_onehot_spec.shape[1]
  model = models.get_keras_model(
      num_classes, FLAGS.use_batch_normalization,
      num_clusters=FLAGS.num_clusters, alpha_init=FLAGS.alpha_init)
  # Define loss and optimizer hyparameters.
  loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  opt = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
  # Add additional metrics to track.
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  summary_writer = tf.summary.create_file_writer(FLAGS.logdir)
  train_step = get_train_step(
      model, loss_obj, opt, train_loss, train_accuracy, summary_writer)
  global_step = opt.iterations
  checkpoint = tf.train.Checkpoint(model=model, global_step=global_step)
  manager = tf.train.CheckpointManager(
      checkpoint, FLAGS.logdir, max_to_keep=None)
  logging.info('Checkpoint prefix: %s', FLAGS.logdir)
  checkpoint.restore(manager.latest_checkpoint)

  if debug: return
  for emb, y_onehot in ds:
    emb.shape.assert_has_rank(3)
    assert emb.shape[2] == FLAGS.embedding_dimension
    y_onehot.shape.assert_has_rank(2)
    assert y_onehot.shape[1] == len(FLAGS.label_list)

    train_step(emb, y_onehot, global_step)

    # Optional print output and save model.
    if global_step % 10 == 0:
      logging.info('step: %i, train loss: %f, train accuracy: %f',
                   global_step, train_loss.result(), train_accuracy.result())
    if global_step % FLAGS.measurement_store_interval == 0:
      manager.save(checkpoint_number=global_step)

  manager.save(checkpoint_number=global_step)
  logging.info('Finished training.')


def get_train_step(model, loss_obj, opt, train_loss, train_accuracy,
                   summary_writer):
  """Returns a function for train step."""
  @tf.function
  def train_step(emb, y_onehot, step):
    with tf.GradientTape() as tape:
      logits = model(emb, training=True)
      assert model.trainable_variables
      logits.shape.assert_is_compatible_with(y_onehot.shape)
      loss_value = loss_obj(y_true=y_onehot, y_pred=logits)
    # Grads and optimizer.
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # Record loss.
    train_loss.update_state(loss_value)
    train_accuracy.update_state(tf.math.argmax(y_onehot, axis=1), logits)

    # Summaries.
    with summary_writer.as_default():
      tf.summary.scalar('train_time_length', emb.shape[1], step=step)
      tf.summary.scalar('xent_loss', loss_value, step=step)
      tf.summary.scalar('xent_loss_smoothed', train_loss.result(), step=step)
      tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
      _maybe_add_autopool_summary(model, step)

  return train_step


def _maybe_add_autopool_summary(model, step):
  autopool_layer_names = [l.name for l in model.layers if 'auto_pool' in l.name]
  if autopool_layer_names:
    assert len(autopool_layer_names) == 1, autopool_layer_names
    avg_alpha = model.get_layer(autopool_layer_names[0]).average_alpha
    tf.summary.scalar('average_alpha', avg_alpha, step=step)


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.shuffle_buffer_size
  assert FLAGS.embedding_name
  assert FLAGS.embedding_dimension
  assert FLAGS.label_name
  assert FLAGS.label_list
  assert FLAGS.bucket_boundaries
  assert FLAGS.train_batch_size
  assert FLAGS.logdir

  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  train_and_report()


if __name__ == '__main__':
  app.run(main)
