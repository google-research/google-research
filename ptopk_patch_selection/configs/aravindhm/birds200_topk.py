# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
# pylint: disable=line-too-long

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  # Use "caltech_birds2011_train-test" for train-test split.
  config.dataset = "caltech_birds2011_train-test"
  config.batch_size = 16

  # From Learning to Navigate for Fine-grained Classification
  # https://arxiv.org/pdf/1809.00287.pdf
  # Code: https://github.com/yangze0930/NTS-Net/blob/master/core/dataset.py#L41
  # config.train_preprocess_str = "to_float_0_1|resize((600, 600))|random_crop((448,448))|random_left_right_flip|normalize(mu=(0.485, 0.456, 0.406), sigma=(0.229, 0.224, 0.225))"
  # config.eval_preprocess_str = "to_float_0_1|resize((600, 600))|central_crop((448,448))|normalize(mu=(0.485, 0.456, 0.406), sigma=(0.229, 0.224, 0.225))"
  config.train_preprocess_str = ("to_float_0_1"
                                 "|resize((600, 600))"
                                 "|random_crop((448,448))"
                                 "|random_left_right_flip"
                                 "|value_range(-1,1)")
  config.eval_preprocess_str = ("to_float_0_1"
                                "|resize((600, 600))"
                                "|central_crop((448,448))"
                                "|value_range(-1,1)")

  config.k = 4
  config.ptopk_sigma = 0.05
  config.ptopk_num_samples = 500
  config.selection_method = "perturbed-topk"
  config.linear_decrease_perturbed_sigma = True

  config.entropy_regularizer = -0.05
  config.part_dropout = False
  config.communication = "squeeze_excite_d"

  # Insert link to a checkpoint file.
  # contact original authors if the checkpoint from the paper is needed.
  config.pretrained_checkpoint = ""
  config.pretrained_prefix = ""

  # == optimization
  config.num_train_steps = 31300
  config.optimizer = "sgd"
  config.learning_rate = 1e-3
  config.momentum = .9
  config.weight_decay_coupled = 1e-4
  config.cosine_decay = False
  config.learning_rate_step_decay = 0.1
  config.learning_rate_decay_at_steps = [18780, 25040]
  config.warmup_ratio = 0.05

  config.log_loss_every_steps = 100
  config.eval_every_steps = 5000
  config.checkpoint_every_steps = 1000
  config.debug = False

  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  sweeps = []
  sweeps.append(h.sweep("config.k", [2, 4]))
  sweeps.append(h.sweep("config.seed", [3, 5, 7, 9, 11]))
  return h.product(sweeps)
