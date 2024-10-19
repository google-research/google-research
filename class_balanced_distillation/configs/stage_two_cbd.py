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

"""Configuration and hyperparameter sweeps."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 1

  config.dataset = "imagenet-lt"
  config.model_name = "resnet50"
  config.sampling = "class_balanced"

  config.add_color_jitter = False
  config.loss = "ce"

  config.learning_rate = 0.1
  config.learning_rate_schedule = "cosine"
  config.warmup_epochs = 0
  config.sgd_momentum = 0.9

  if config.dataset == "imagenet-lt":
    config.weight_decay = 0.0005
    config.num_epochs = 90
  elif config.dataset == "inaturalist18":
    config.weight_decay = 0.0002
    config.num_epochs = 200
  else:
    raise ValueError(f"Dataset {config.dataset} not supported.")

  config.global_batch_size = 128
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = -1
  config.num_eval_steps = -1

  config.log_loss_every_steps = 500
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 1000
  config.shuffle_buffer_size = 1000

  config.trial = 0  # dummy for repeated runs.

  # Distillation parameters
  model_dir = "/home/user/class_balanced_distillation/data/models/"
  model_ids = [
      "vanilla_seed1", "vanilla_seed2", "data_aug_seed1", "data_aug_seed2"
  ]

  num_teachers = len(model_ids)
  config.distill_teacher = ""
  for i in range(num_teachers):
    if i == 0:
      config.distill_teacher = "{}/{}".format(model_dir,
                                              model_ids[i])
    else:
      config.distill_teacher = "{},{}/{}".format(config.distill_teacher,
                                                 model_dir, model_ids[i])

  config.proj_dim = 2048 * num_teachers
  config.distill_alpha = 0.4
  config.distill_fd_beta = 100.0

  return config
