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

"""Default config for training Spin-Weighted Spherical CNNs."""

import ml_collections


def get_config():
  """Config to train on dummy data; it should be specialized for other tasks."""
  config = ml_collections.ConfigDict()

  config.model_name = "tiny_classifier"
  config.dataset = "tiny_dummy"
  # This combines train and validation sets during training and evaluates on the
  # test set after (use sparingly).
  config.combine_train_val_and_eval_on_test = False

  # This is the learning rate for batch size 32. The code scales it linearly
  # with the batch size.
  config.learning_rate = 1e-3
  config.learning_rate_schedule = "cosine"
  config.warmup_epochs = 1
  config.weight_decay = 0.
  config.num_epochs = 12
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = -1
  # Evaluates for a full epoch if num_eval_steps==-1. Set to a smaller value for
  # fast iteration when running train.train_and_eval() from a Colab.
  config.num_eval_steps = -1
  config.per_device_batch_size = 32
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = True

  config.log_loss_every_steps = 100
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 1000
  config.shuffle_buffer_size = 1000

  config.seed = 42

  config.trial = 0  # Dummy for repeated runs.
  config.lock()
  return config


