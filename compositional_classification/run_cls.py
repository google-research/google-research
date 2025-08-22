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

#!/usr/bin/python
"""Main Python file to run the classification experiments."""

import os

from absl import app
from absl import flags
from absl import logging
from modules import datasets
from modules import hyperparameters
from modules import losses
from modules import models
from modules import optimizers
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', 'lstm',
                  ['lstm', 'transformer', 'relative_transformer'],
                  'Model architecture.')
flags.DEFINE_string('data_dir', 'data', 'Path to the dataset root.')
flags.DEFINE_bool('do_train', False, 'Whether to run training.')
flags.DEFINE_enum('dataset', 'test', ['train', 'dev', 'test'],
                  'Dataset split to use when do_train=False.')
flags.DEFINE_string('output_dir', 'exp_lstm', 'Output directory to save '
                    'log and checkpoints.')
flags.DEFINE_integer('train_steps', 10000, 'Number of steps to train.')
flags.DEFINE_integer('checkpoint_iter', 1000, 'Steps per checkpoint save.')
flags.DEFINE_integer('eval_iter', 500, 'Steps per validation.')
flags.DEFINE_integer('display_iter', 100, 'Steps per print.')

flags.register_validator('data_dir', os.path.exists, 'Dataset not found.')


def get_epoch_result(model, dataset,
                     loss_fn):
  """Evaluates loss and accuracy of the model on the dataset."""
  total_loss, total_correct = [], []
  for batch_inputs, batch_labels in dataset:
    batch_logits = model(batch_inputs)
    loss = loss_fn(batch_labels, batch_logits)
    correct = batch_labels == tf.cast(batch_logits > 0.0, tf.int32)
    total_loss.append(loss)
    total_correct.append(correct)
  loss = tf.reduce_mean(tf.concat(total_loss, -1))
  acc = tf.reduce_mean(tf.cast(tf.concat(total_correct, -1), tf.float32))
  return loss, acc


def main(argv):
  del argv  # unused

  # Loads hyper-parameters.
  if FLAGS.model == 'lstm':
    hparams = hyperparameters.lstm_model_hparams()
  elif FLAGS.model == 'transformer':
    hparams = hyperparameters.transformer_model_hparams()
  elif FLAGS.model == 'relative_transformer':
    hparams = hyperparameters.relative_transformer_model_hparams()

  # Loads datasets
  if FLAGS.parse_tree_input:
    dataset_fn = datasets.load_cls_mask_dataset
  elif FLAGS.model != 'relative_transformer':
    dataset_fn = datasets.load_cls_dataset
  else:
    dataset_fn = datasets.load_cls_nomask_dataset

  if FLAGS.do_train:
    dataset_train = dataset_fn(hparams, FLAGS.data_dir, name='train')
    dataset_val = dataset_fn(hparams, FLAGS.data_dir, name='dev')
    data_iter_train = iter(dataset_train)
  else:
    dataset_test = dataset_fn(hparams, FLAGS.data_dir, name=FLAGS.dataset)
  with open(os.path.join(FLAGS.data_dir, 'vocab.txt'), 'r') as fd:
    hparams.vocab_size = len(fd.readlines())

  # Builds model.
  if FLAGS.model == 'lstm':
    model = models.build_lstm_model(hparams)
  elif FLAGS.model == 'transformer':
    hparams.pad_index = datasets.PAD_TOKEN_ID
    hparams.sep_index = datasets.SEP_TOKEN_ID
    model = models.build_transformer_model(hparams)
  elif FLAGS.model == 'relative_transformer':
    model = models.build_relative_transformer_model(hparams)

  # Define loss and optimizer.
  loss_fn = losses.get_weighted_binary_cross_entropy_fn(hparams)
  optimizer = optimizers.get_optimizer(hparams)

  # Create checkpoint manager and initialize.
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  init_step = 0
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint, directory=FLAGS.output_dir, max_to_keep=5)
  if ckpt_manager.latest_checkpoint:
    ckpt_path = ckpt_manager.latest_checkpoint
    logging.info('Load model checkpoint from %s', ckpt_path)
    checkpoint.restore(ckpt_path)
    init_step = int(ckpt_path[ckpt_path.rfind('-') + 1:])

  if FLAGS.do_train:
    # Training model
    for step in range(init_step, FLAGS.train_steps):
      batch_inputs, batch_labels = next(data_iter_train)
      with tf.GradientTape() as tape:
        batch_logits = model(batch_inputs, training=True)
        loss = tf.reduce_mean(loss_fn(batch_labels, batch_logits))
      acc = tf.reduce_mean(
          tf.cast(batch_labels == tf.cast(batch_logits > 0.0, tf.int32),
                  tf.float32))
      # Retrives lr before the optimizer step increases.
      lr = optimizers.get_lr(optimizer)
      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step % FLAGS.display_iter == 0 or step == FLAGS.train_steps - 1:
        logging.info('(Train) step %6i, loss=%.6f, acc=%.4f, lr=%.5f', step,
                     loss, acc, lr)

      # Testing model on validation set
      if step % FLAGS.eval_iter == 0 or step == FLAGS.train_steps - 1:
        loss, acc = get_epoch_result(model, dataset_val, loss_fn)
        logging.info('(Eval)  step %6i, loss %.6f, acc=%.4f', step, loss, acc)

      # Saves checkpoint.
      if step % FLAGS.checkpoint_iter == 0 or step == FLAGS.train_steps - 1:
        ckpt_path = ckpt_manager.save(step)
        logging.info('Saved checkpoint to %s', ckpt_path)
  else:
    # Testing model
    loss, acc = get_epoch_result(model, dataset_test, loss_fn)
    logging.info('(Test)  step %6i, loss %.6f, acc=%.4f', init_step, loss, acc)


if __name__ == '__main__':
  app.run(main)
