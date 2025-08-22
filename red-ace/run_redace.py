# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Train a RED-ACE model in TF2.0."""

import os

from absl import app
from absl import flags
from absl import logging
from official.nlp import optimization
from official.utils.misc import keras_utils
import redace_config as configs
import redace_flags  # pylint: disable=unused-import
import redace_input_pipeline
import redace_models
import tensorflow as tf
import train_callbacks

FLAGS = flags.FLAGS
_CHECKPOINT_FILE_NAME = 'redace_model.ckpt'


def _get_loss_fn(loss_scale=1.0):
  """Returns loss function for training."""
  del loss_scale

  def train_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses)

  return train_loss_fn


def restore_checkpoint(path, bert_encoder):
  """Loads the checkpoint weights to the given model.

  Args:
    path: a folder, a file, possible without the counter suffix (e.g. "-XX").
    bert_encoder: Bert model to use.
  """
  logging.info('Initializing from a BERT checkpoint...')
  status = tf.train.Checkpoint(model=bert_encoder).restore(path)
  status.expect_partial()


def run_train(
    redace_config,
    seq_length,
    model_dir,
    epochs,
    initial_lr,
    warmup_steps,
    loss_scale,
    train_file,
    eval_file,
    train_batch_size,
    eval_batch_size,
    test_file=None,
    validation_checkpoint_metric=None,
    truncate_dataset_size=None,
    init_checkpoint=None,
    num_train_examples=None,
    num_eval_examples=None,
    steps_per_loop=1,
    use_weighted_labels=False,
    keep_checkpoint_max=3,
    log_steps=1000,
):
  """Runs BERT pre-training using Keras `fit()` API."""
  pretrain_model, bert_encoder = redace_models.get_model(
      redace_config, seq_length)

  # The original BERT model does not scale the loss by 1/num_replicas_in_sync.
  # It could be an accident. So, in order to use the same hyper parameter,
  # we do the same thing here.
  loss_fn = _get_loss_fn(loss_scale=loss_scale)
  steps_per_mini_epoch = int(num_train_examples / train_batch_size)
  logging.info('Steps per mini epoch: %d', steps_per_mini_epoch)
  eval_steps = max(1, int(num_eval_examples / eval_batch_size))

  optimizer = optimization.AdamWeightDecay(
      initial_lr,
      weight_decay_rate=0.0,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias', 'scale'],
  )

  pretrain_model.compile(
      optimizer=optimizer, loss=loss_fn, steps_per_execution=steps_per_loop)

  train_dataset = redace_input_pipeline.create_redace_dataset(
      train_file,
      seq_length,
      train_batch_size,
      True,  # is_training
      use_weighted_labels=use_weighted_labels,
  )
  eval_dataset = redace_input_pipeline.create_redace_dataset(
      eval_file,
      seq_length,
      eval_batch_size,
      False,  # is_training
      use_weighted_labels=use_weighted_labels,
  )
  latest_checkpoint_directory = os.path.join(model_dir, 'latest')
  latest_checkpoint_file = tf.train.latest_checkpoint(
      latest_checkpoint_directory)
  if latest_checkpoint_file is None:
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
    if init_checkpoint:
      # No latest checkpoint found so load a pre-trained checkpoint.
      restore_checkpoint(init_checkpoint, bert_encoder)
    checkpoint = tf.train.Checkpoint(model=pretrain_model, optimizer=optimizer)
    checkpoint_iteration = 0
    current_mini_epoch = 0

  logging.info('Starting training from iteration: %s.', checkpoint_iteration)
  summary_dir = os.path.join(model_dir, 'summaries')
  summary_cb = tf.keras.callbacks.TensorBoard(summary_dir, update_freq=1000)

  manager_latest = tf.train.CheckpointManager(
      checkpoint,
      directory=latest_checkpoint_directory,
      max_to_keep=keep_checkpoint_max,
      checkpoint_name=_CHECKPOINT_FILE_NAME)
  manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=keep_checkpoint_max,
      checkpoint_name=_CHECKPOINT_FILE_NAME)
  checkpoint_cb = train_callbacks.CheckPointSaver(
      manager,
      manager_latest,
      latest_checkpoint_directory,
      current_mini_epoch,
      eval_file,
      test_file,
      truncate_dataset_size,
      pretrain_model,
      eval_batch_size,
      validation_checkpoint_metric,
      warmup_steps=warmup_steps,
      initial_lr=initial_lr,
      steps_per_loop=steps_per_loop,
      enable_async_checkpoint=redace_config.enable_async_checkpoint,
  )
  time_history_cb = keras_utils.TimeHistory(train_batch_size, log_steps)
  training_callbacks = [checkpoint_cb, time_history_cb, summary_cb]

  pretrain_model.fit(
      train_dataset,
      initial_epoch=current_mini_epoch,
      epochs=epochs,
      verbose=2,
      steps_per_epoch=steps_per_mini_epoch,
      validation_data=eval_dataset,
      validation_steps=eval_steps,
      callbacks=training_callbacks,
  )


def main(_):
  logging.info('Training using keras fit with a non-distributed strategy.')
  loss_scale = 1
  run_train(
      redace_config=configs.RedAceConfig(),
      seq_length=FLAGS.max_seq_length,
      model_dir=FLAGS.model_dir,
      epochs=FLAGS.num_train_epochs,
      initial_lr=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      loss_scale=loss_scale,
      train_file=FLAGS.train_file,
      eval_file=FLAGS.eval_file,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      truncate_dataset_size=FLAGS.num_eval_examples,
      test_file=FLAGS.test_file,
      validation_checkpoint_metric=FLAGS.validation_checkpoint_metric,
      init_checkpoint=FLAGS.init_checkpoint,
      num_train_examples=FLAGS.num_train_examples,
      num_eval_examples=FLAGS.num_eval_examples,
      steps_per_loop=FLAGS.steps_per_loop,
      use_weighted_labels=FLAGS.use_weighted_labels,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      log_steps=FLAGS.log_steps,
  )


if __name__ == '__main__':
  app.run(main)
