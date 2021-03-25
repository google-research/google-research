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

# Lint as: python3
"""Configuration and hyperparameter sweeps."""
# pylint: disable=line-too-long

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  config.dataset = "billiard-max-left-right-test"
  config.train_preprocess_str = "to_float_0_1|pad(ensure_small=(1100, 1100))|random_crop(resolution=(1000, 1000))"
  config.eval_preprocess_str = "to_float_0_1"

  config.append_position_to_input = False
  config.downsample_input_factor = 1

  # Top-k extraction.
  config.model = "patchnet"
  config.k = 10
  config.patch_size = 100
  config.downscale = 4
  config.feature_network = "ResNet18"

  config.aggregation_method = "transformer"
  config.aggregation_method_kwargs = ml_collections.ConfigDict()
  config.aggregation_method_kwargs.num_layers = 3
  config.aggregation_method_kwargs.num_heads = 8
  config.aggregation_method_kwargs.dim_hidden = 256
  config.aggregation_method_kwargs.pooling = "sum"

  config.selection_method = "perturbed-topk"
  config.selection_method_inference = "hard-topk"
  config.entropy_regularizer = -0.01
  config.entropy_before_normalization = True

  config.scorer_has_se = False

  # parameters for sinkhorn topk
  config.sinkhorn_topk_kwargs = ml_collections.ConfigDict()
  config.sinkhorn_topk_kwargs.epsilon = 1e-4
  config.sinkhorn_topk_kwargs.num_iterations = 2000

  # parameters for perturbed_topk
  config.perturbed_topk_kwargs = ml_collections.ConfigDict()
  config.perturbed_topk_kwargs.num_samples = 500
  config.perturbed_topk_kwargs.sigma = 0.05
  config.linear_decrease_perturbed_sigma = True

  config.normalization_str = "zerooneeps(1e-5)"
  config.use_iterative_extraction = True

  # Same set up as usual.
  config.optimizer = "adam"
  config.learning_rate = 1e-4
  config.gradient_value_clip = 1.
  config.momentum = .9
  config.weight_decay = 1e-4
  config.cosine_decay = True
  config.warmup_ratio = 0.05
  config.batch_size = 64
  config.num_train_steps = 30_000

  config.log_loss_every_steps = 100
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 5000
  config.log_images = True
  config.log_histograms = False
  config.do_eval_only = False

  config.trial = 0  # Dummy for repeated runs.

  return config



