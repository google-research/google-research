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

"""A config for training ResNet-50 on CelebA."""

import ml_collections


def get_config():
  """Config for training on imagenet. Should be used on a 4x4 TPU slice."""
  config = ml_collections.ConfigDict()

  config.dataset_name = "celeb_a"
  config.model_name = "resnet50"
  # This is the learning rate for batch size 256. The code scales it linearly
  # with the batch size. This is common for ImageNet and SGD.
  config.loss_fn = "cross_entropy"
  config.learning_rate = 0.0001
  # config.learning_rate_schedule = "cosine"
  # config.warmup_epochs = 5
  config.sgd_momentum = 0.9
  config.ema_decay = 0.99
  config.batch_norm_decay = 0.99
  config.weight_decay = 0.0001
  config.num_epochs = 50
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = -1
  # Evaluates for a full epoch if num_eval_steps==-1. Set to a smaller value for
  # fast iteration when running train.train_and_eval() from a Colab.
  config.num_eval_steps = -1
  config.per_device_batch_size = 128
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = True

  config.log_loss_every_steps = 50
  config.eval_every_steps = 50
  config.checkpoint_every_steps = 100
  config.shuffle_buffer_size = 10000

  config.seed = 2

  config.trial = 0  # Dummy for repeated runs.
  return config


def get_hyper(h):
  return h.product([
      h.sweep("config.trial", range(1)),
  ])
