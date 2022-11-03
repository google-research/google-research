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

"""Configuration for trafficsigns_topk."""
# pylint: disable=line-too-long

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  config.dataset = "trafficsigns"
  config.train_preprocess_str = "to_float_0_1|pad(ensure_small=(1160, 1480))|random_crop(resolution=(960, 1280))|random_linear_transform((.8,1.2),(-.1,.1),.8)"
  config.eval_preprocess_str = "to_float_0_1"

  # Top-k extraction.
  config.model = "patchnet"
  config.k = 5
  config.patch_size = 100
  config.downscale = 3

  config.append_position_to_input = False
  config.feature_network = "ats-traffic"

  config.aggregation_method = "meanpooling"

  config.scorer_has_se = False

  config.selection_method = "perturbed-topk"
  config.selection_method_inference = "hard-topk"
  config.entropy_regularizer = -0.01
  config.entropy_before_normalization = True

  # parameters for perturbed_topk
  config.perturbed_topk_kwargs = ml_collections.ConfigDict()
  config.perturbed_topk_kwargs.num_samples = 500
  config.perturbed_topk_kwargs.sigma = 0.05
  config.linear_decrease_perturbed_sigma = True

  # parameters for sinkhorn topk
  config.sinkhorn_topk_kwargs = ml_collections.ConfigDict()
  config.sinkhorn_topk_kwargs.epsilon = 1e-4
  config.sinkhorn_topk_kwargs.num_iterations = 2000

  config.normalization_str = "zerooneeps(1e-5)"
  config.use_iterative_extraction = True

  config.optimizer = "adam"
  config.learning_rate = 1e-4
  config.gradient_value_clip = 0.1
  config.momentum = .9

  config.weight_decay = 1e-4
  config.cosine_decay = True
  config.warmup_ratio = 0.1
  config.batch_size = 32
  config.num_train_steps = 70_000

  config.log_loss_every_steps = 50
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 5000

  config.log_images = True
  config.log_histograms = False
  config.skip_nan_updates = True
  config.do_eval_only = False

  config.trial = 0  # Dummy for repeated runs.

  return config


