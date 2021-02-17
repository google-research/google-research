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

"""Train a Felix model in TF2.0."""
import os

from absl import app
from absl import flags
from absl import logging
from official.common import distribute_utils
from official.nlp import optimization
from official.nlp.bert import configs
from official.utils.misc import keras_utils
import tensorflow as tf

from felix import felix_flags  # pylint: disable=unused-import
from felix import felix_input_pipeline
from felix import felix_models

FLAGS = flags.FLAGS
_CHECKPOINT_FILE_NAME = 'felix_model.ckpt'


def _get_input_data_fn(input_file_pattern,
                       seq_length,
                       max_predictions_per_seq,
                       batch_size,
                       is_training,
                       use_insertion=True,
                       use_pointing=True,
                       use_weighted_labels=True):
  """Returns input dataset from input file string."""
  input_files = []
  for input_pattern in input_file_pattern.split(','):
    input_files.extend(tf.io.gfile.glob(input_pattern))
  train_dataset = felix_input_pipeline.create_felix_dataset(
      input_files,
      seq_length,
      max_predictions_per_seq,
      batch_size,
      is_training,
      use_insertion=use_insertion,
      use_pointing=use_pointing,
      use_weighted_labels=use_weighted_labels)
  return train_dataset


def _get_loss_fn(loss_scale=1.0):
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses) * loss_scale

  return _bert_pretrain_loss_fn


class CheckPointSaver(tf.keras.callbacks.Callback):
  """TensorFlow callback, which saves the model with the highest exact match."""

  def __init__(self, checkpoint_manager, current_epoch=0):
    super(CheckPointSaver, self).__init__()
    self._manager = checkpoint_manager
    self._best_exact_match = -1
    self._current_epoch = current_epoch

  def on_test_end(self, logs=None):
    current_exact_match = logs['exact_match']
    self._current_epoch += 1
    if current_exact_match > self._best_exact_match:
      self._best_exact_match = current_exact_match
      self._manager.save(checkpoint_number=self._current_epoch)


def run_train(bert_config,
              seq_length,
              max_predictions_per_seq,
              model_dir,
              epochs,
              initial_lr,
              warmup_steps,
              loss_scale,
              train_file,
              eval_file,
              train_batch_size,
              eval_batch_size,
              use_insertion = True,
              use_pointing = True,
              pointing_weight = 1.0,
              mini_epochs_per_epoch = 1):
  """Runs BERT pre-training using Keras `fit()` API."""

  mini_epochs_per_epoch = max(1, mini_epochs_per_epoch)

  if use_insertion:
    pretrain_model, bert_encoder = felix_models.get_insertion_model(
        bert_config, seq_length, max_predictions_per_seq)
  else:
    pretrain_model, bert_encoder = felix_models.get_tagging_model(
        bert_config,
        seq_length,
        use_pointing=use_pointing,
        pointing_weight=pointing_weight)
  # The original BERT model does not scale the loss by 1/num_replicas_in_sync.
  # It could be an accident. So, in order to use the same hyper parameter,
  # we do the same thing here.
  loss_fn = _get_loss_fn(loss_scale=loss_scale)

  steps_per_mini_epoch = int(FLAGS.num_train_examples / train_batch_size /
                             mini_epochs_per_epoch)
  eval_steps = max(1, int(FLAGS.num_eval_examples / eval_batch_size))

  optimizer = optimization.create_optimizer(
      init_lr=initial_lr,
      num_train_steps=steps_per_mini_epoch * mini_epochs_per_epoch * epochs,
      num_warmup_steps=warmup_steps)

  pretrain_model.compile(
      optimizer=optimizer,
      loss=loss_fn,
      experimental_steps_per_execution=FLAGS.steps_per_loop)
  train_dataset = _get_input_data_fn(
      train_file,
      seq_length,
      max_predictions_per_seq,
      train_batch_size,
      is_training=True,
      use_insertion=use_insertion,
      use_pointing=use_pointing,
      use_weighted_labels=FLAGS.use_weighted_labels)
  eval_dataset = _get_input_data_fn(
      eval_file,
      seq_length,
      max_predictions_per_seq,
      eval_batch_size,
      is_training=False,
      use_insertion=use_insertion,
      use_pointing=use_pointing,
      use_weighted_labels=FLAGS.use_weighted_labels)

  latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
  if latest_checkpoint_file is not None:
    checkpoint = tf.train.Checkpoint(model=pretrain_model, optimizer=optimizer)
    # Since some model components(e.g. optimizer slot variables)
    # are loaded lazily for some components, we do not add any asserts
    # before model.call() is invoked.
    checkpoint.restore(latest_checkpoint_file)
    checkpoint_iteration = tf.keras.backend.get_value(
        pretrain_model.optimizer.iterations)
    current_mini_epoch = checkpoint_iteration // steps_per_mini_epoch
  else:
    # No latest checkpoint found so load a pre-trained checkpoint.
    if FLAGS.init_checkpoint:
      if _CHECKPOINT_FILE_NAME not in FLAGS.init_checkpoint:
        logging.info('Initializing from a BERT checkpoint...')
        checkpoint = tf.train.Checkpoint(model=bert_encoder)
        checkpoint.restore(
            FLAGS.init_checkpoint).assert_existing_objects_matched()
      else:
        logging.info('Initializing from a Felix checkpoint...')
        # Initialize from a previously trained checkpoint.
        checkpoint = tf.train.Checkpoint(model=pretrain_model)
        checkpoint.restore(
            FLAGS.init_checkpoint).assert_existing_objects_matched()
        # Reset the iteration number to have the learning rate adapt correctly.
        tf.keras.backend.set_value(pretrain_model.optimizer.iterations, 0)

    checkpoint = tf.train.Checkpoint(model=pretrain_model, optimizer=optimizer)
    checkpoint_iteration = 0
    current_mini_epoch = 0

  logging.info('Starting training from iteration: %s.', checkpoint_iteration)
  summary_dir = os.path.join(model_dir, 'summaries')
  summary_cb = tf.keras.callbacks.TensorBoard(summary_dir, update_freq=1000)

  manager = tf.train.CheckpointManager(
      checkpoint, directory=model_dir, max_to_keep=FLAGS.keep_checkpoint_max,
      checkpoint_name=_CHECKPOINT_FILE_NAME)
  checkpoint_cb = CheckPointSaver(manager, current_mini_epoch)
  time_history_cb = keras_utils.TimeHistory(FLAGS.train_batch_size,
                                            FLAGS.log_steps)
  training_callbacks = [summary_cb, checkpoint_cb, time_history_cb]
  pretrain_model.fit(
      train_dataset,
      initial_epoch=current_mini_epoch,
      epochs=mini_epochs_per_epoch * epochs,
      verbose=1,
      steps_per_epoch=steps_per_mini_epoch,
      validation_data=eval_dataset,
      validation_steps=eval_steps,
      callbacks=training_callbacks)


def main(_):
  if not FLAGS.use_open_vocab:
    raise  app.UsageError('Currently only use_open_vocab=True is supported')
  if FLAGS.train_insertion:
    model_dir = FLAGS.model_dir_insertion
    bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_insertion)
  else:
    model_dir = FLAGS.model_dir_tagging
    bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_tagging)
  if FLAGS.tpu is not None:
    cluster_resolver = distribute_utils.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    with strategy.scope():
      return run_train(bert_config, FLAGS.max_seq_length,
                       FLAGS.max_predictions_per_seq, model_dir,
                       FLAGS.num_train_epochs, FLAGS.learning_rate,
                       FLAGS.warmup_steps, 1.0, FLAGS.train_file,
                       FLAGS.eval_file, FLAGS.train_batch_size,
                       FLAGS.eval_batch_size, FLAGS.train_insertion,
                       FLAGS.use_pointing, FLAGS.pointing_weight)
  else:
    return run_train(bert_config, FLAGS.max_seq_length,
                     FLAGS.max_predictions_per_seq, model_dir,
                     FLAGS.num_train_epochs, FLAGS.learning_rate,
                     FLAGS.warmup_steps, 1.0, FLAGS.train_file,
                     FLAGS.eval_file, FLAGS.train_batch_size,
                     FLAGS.eval_batch_size, FLAGS.train_insertion,
                     FLAGS.use_pointing, FLAGS.pointing_weight,
                     FLAGS.mini_epochs_per_epoch)


if __name__ == '__main__':
  app.run(main)
