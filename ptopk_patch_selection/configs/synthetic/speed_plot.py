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

# Lint as: python3
"""Configuration file."""
# pylint: disable=line-too-long

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  config.dataset = "billiardleft-color-min-max"  # -- overwritten in sweep
  config.train_preprocess_str = "to_float_0_1"
  config.eval_preprocess_str = "to_float_0_1"

  # Top-k extraction.
  config.model = "patchnet"
  config.k = 10
  config.patch_size = 100
  config.downscale = 4
  config.feature_network = "ResNet50"

  config.aggregation_method = "maxpooling"

  config.scorer_has_se = True

  config.selection_method = "perturbed-topk"
  config.selection_method_inference = "hard-topk"

  # parameters for perturbed_topk
  config.perturbed_topk_kwargs = ml_collections.ConfigDict()
  config.perturbed_topk_kwargs.num_samples = 500
  config.perturbed_topk_kwargs.sigma = 0.05

  config.normalization_str = "zerooneeps(1e-5)"
  config.use_iterative_extraction = True

  # Same set up as usual.
  config.optimizer = "adam"
  config.learning_rate = 1e-10
  config.batch_size = 2
  config.num_train_steps = 201

  config.log_loss_every_steps = 100
  config.eval_every_steps = 100
  config.checkpoint_every_steps = 5000
  config.log_images = False
  config.log_histograms = False
  config.do_eval_only = True

  config.trial = 0  # Dummy for repeated runs.

  return config


