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

"""Defines utility functions."""

import tensorflow as tf


def assign_moving_average_vars(model, ema_model, optimizer):
  """Assigns moving average variables to the model using moving average.

  Args:
    model: An original model.
    ema_model: A model using moving average.
    optimizer: An optimizer which stores moving average variables.
  """
  non_avg_weights = model.get_weights()
  optimizer.assign_average_vars(model.variables)
  ema_model.set_weights(model.get_weights())
  model.set_weights(non_avg_weights)


def create_checkpoint(ckpt_dir_path,
                      max_to_keep=10,
                      keep_checkpoint_every_n_hours=0.5,
                      **ckpt_objects):
  """Creates and restores checkpoint (if one exists on the path).

  Args:
    ckpt_dir_path: A string for the directory path where the checkpoints are
      created and restored.
    max_to_keep: An integer for the maximume number of checkpoints to store.
    keep_checkpoint_every_n_hours: A float indicating the time interval to keep
      checkpoints since the last preserved one.
    **ckpt_objects: A dictionary for attributes saved with the checkpoint.
      Values must be trackable objects.

  Returns:
    ckpt_manager: A tf.train.CheckpointManager object.
    status: A load status object.
    checkpoint: The restored tf.train.Checkpoint object.
  """
  checkpoint = tf.train.Checkpoint(**ckpt_objects)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=ckpt_dir_path,
      max_to_keep=max_to_keep,
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
  status = checkpoint.restore(ckpt_manager.latest_checkpoint)
  return ckpt_manager, status, checkpoint
