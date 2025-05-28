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

"""Callbacks for RED-ACE."""

import os

from absl import logging
import numpy as np
import tensorflow as tf


def load_best_score_file(file_name):
  """Read from a checkpoint file to determine the score of the best model.

  Args:
    file_name: The name of the file which stores the best score. This file
      contains one line, containing a float.

  Returns:
    If the file doesn't exist or is ill-formatted return None, else return the
    score.
  """
  if not tf.io.gfile.exists(file_name):
    return None
  try:
    with tf.io.gfile.GFile(file_name, 'r') as f:
      return float(f.readline())
  except ValueError:
    return None


def load_lr_file(file_name):
  """Read from a file information used to set the learning rate.

  Args:
    file_name: The name of the file containing learning rate information. This
      file consists of three lines, the first contains the learning rate (a
      float), the second the number of bad epochs (int), and the third, whether
      warmup is finished (bool).

  Returns:
    If the file doesn't exist or is ill-formatted return None, else return a
    tuple consists of the learning rate, the number of bad epochs, and if
    warm-up is completed.
    score.
  """
  if not tf.io.gfile.exists(file_name):
    return None
  try:
    with tf.io.gfile.GFile(file_name, 'r') as f:
      line = f.readlines()
      lr = float(line[0])
      bad_epochs = int(line[1])
      warmup_done = line[2].strip().lower() == 'true'
      return (lr, bad_epochs, warmup_done)
  except (ValueError, IndexError):
    return None


class CheckPointSaver(tf.keras.callbacks.Callback):
  """TensorFlow callback, which saves the model with the highest exact match.

  Additionaly performs LR warmup and reduces the LR if the validation score has
  not improved for self._patience epochs.
  """

  def __init__(self,
               checkpoint_manager,
               checkpoint_manager_latest,
               latest_checkpoint_directory,
               current_epoch=0,
               eval_input_file=None,
               test_input_file=None,
               truncate_dataset_size=None,
               model=None,
               batch_size=None,
               validation_metric=None,
               initial_lr=1.0,
               warmup_steps=0,
               steps_per_loop=1,
               enable_async_checkpoint=False,
               learning_rate_patience=10):
    self._manager = checkpoint_manager
    self._manager_latest = checkpoint_manager_latest
    tf.io.gfile.makedirs(latest_checkpoint_directory)
    checkpoint_directory, _ = os.path.split(latest_checkpoint_directory)
    self._best_checkpoint_score_file = os.path.join(checkpoint_directory,
                                                    'score.txt')
    self._best_score = None
    loaded_best_score = load_best_score_file(self._best_checkpoint_score_file)
    if loaded_best_score is not None:
      self._best_score = loaded_best_score

    self._initial_lr = initial_lr
    self._steps_per_loop = max(1, steps_per_loop)
    self._lr_file = os.path.join(checkpoint_directory, 'lr.txt')

    self._lr = initial_lr
    self._bad_epochs = 0
    self._warmup_done = False
    loaded_lr_tuple = load_lr_file(self._lr_file)
    if loaded_lr_tuple is not None:
      self._lr, self._bad_epochs, self._warmup_done = loaded_lr_tuple
    self._current_step = 1
    self._current_epoch = current_epoch
    self._eval_input_file = eval_input_file
    self._test_input_file = test_input_file
    self._truncate_dataset_size = truncate_dataset_size
    self._model = model
    self._batch_size = batch_size
    self._predict_dev_output_file = (
        self._manager._directory + '/dev_pred_{}.tsv')
    self._predict_test_output_file = self._manager._directory + '/test_pred.tsv'
    if validation_metric is None:
      validation_metric = 'bleu'
    self._validation_metric = validation_metric
    self._warmup_steps = warmup_steps
    self._learning_rate_patience = learning_rate_patience
    self._enable_async_checkpoint = enable_async_checkpoint

  def on_train_batch_begin(self, batch, logs=None):
    """Used to perform linear LR warmup."""
    self._current_step += self._steps_per_loop
    if self._warmup_done:
      self._set_lr(logs)
    elif self._current_step >= self._warmup_steps:
      self._lr = self._initial_lr
      self._set_lr(logs)
      self._warmup_done = True
    else:
      warmup_fraction_done = min(1.0, self._current_step / self._warmup_steps)
      self._lr = self._initial_lr * warmup_fraction_done
      self._set_lr(logs)

  def _set_lr(self, logs, write_to_file=False):
    """Updated the LR for the model and logs it."""
    logs = logs or {}
    logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
    tf.keras.backend.set_value(self.model.optimizer.lr, self._lr)
    if write_to_file:
      with tf.io.gfile.GFile(self._lr_file, 'w') as f:
        f.write(str(self._lr) + '\n')
        f.write(str(self._bad_epochs) + '\n')
        f.write(str(self._warmup_done))

  def _reduce_lr_on_plateau(self, logs):
    """Reduce LR if validation scores have not improved in patience epochs."""
    if (self._learning_rate_patience > 0 and
        self._bad_epochs > self._learning_rate_patience and self._warmup_done):
      # Reduce LR by 10%.
      self._lr = self._lr * 0.9
      if self._lr < 0.000001:
        self.model.stop_training = True
      self._bad_epochs = 0
    self._set_lr(logs, write_to_file=True)

  def _save_best_model(self, score, logs):
    checkpoint_options = tf.train.CheckpointOptions(
        experimental_enable_async_checkpoint=self._enable_async_checkpoint)
    self._manager_latest.save(
        checkpoint_number=self._current_epoch, options=checkpoint_options)
    if self._best_score is None or score > self._best_score:
      self._best_score = score
      self._manager.save(
          checkpoint_number=self._current_epoch, options=checkpoint_options)
      with tf.io.gfile.GFile(self._best_checkpoint_score_file, 'w') as f:
        f.write(str(self._best_score))
      self._bad_epochs = 0
    # Only count bad epochs after warmup.
    elif self._warmup_done:
      self._bad_epochs += 1
    self._reduce_lr_on_plateau(logs)

    logging.info('Main_metric: %f vs best main_metric: %f', score,
                 self._best_score)

  def on_test_end(self, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    if np.isnan(loss) or np.isinf(loss):
      raise ValueError('Invalid loss, terminating training.')
    if self._validation_metric == 'latest':
      current_main_metric = self._current_epoch
    elif 'exact_match' in logs:
      current_main_metric = logs['exact_match']
    elif 'estimated_exact_match' in logs:
      current_main_metric = logs['estimated_exact_match']
    else:
      current_main_metric = -logs['loss']
    self._save_best_model(current_main_metric, logs)

  def on_epoch_end(self, epoch, logs=None):
    self._current_epoch += 1

  def on_train_end(self, logs=None):
    return
