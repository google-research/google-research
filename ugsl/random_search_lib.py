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

# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Running a random_search on the UGSL components."""

import copy
import os
import random
from absl import flags

from ml_collections import config_dict
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(__file__), "config.py"),
    "Path to file containing configuration hyperparameters. "
    "File must define method `get_config()` to return an instance of "
    "`config_dict.ConfigDict`",
)
_DATASET = flags.DEFINE_string(
    "dataset", None, "The name of the dataset.", required=True
)
_NFEATS = flags.DEFINE_integer(
    "nfeats", None, "Number of node features.", required=True
)
_NRUNS = flags.DEFINE_integer("nruns", 5, "Number of runs.", required=False)
_EXPERIMENT_DIR = flags.DEFINE_string(
    "experiment_dir", None, "The model directory.", required=True
)


def sample_random_configs():
  """Returns list of random configurations based on flags in this file."""
  default_config: config_dict.ConfigDict = _CONFIG.value
  dataset: str = _DATASET.value
  num_runs: int = _NRUNS.value
  num_feats: int = _NFEATS.value
  experiment_dir: str = _EXPERIMENT_DIR.value
  configs = []
  for run in range(num_runs):
    config = copy.deepcopy(default_config)
    config.dataset.name = dataset
    config.run.learning_rate = random.uniform(1e-3, 1e-1)
    config.run.weight_decay = random.uniform(5e-4, 5e-2)
    config.run.model_dir = os.path.join(experiment_dir, str(run))
    config = set_edge_scorer_config(config, num_feats)
    config = set_sparsifier_config(config)
    config = set_processor_config(config)
    config = set_encoder_config(config)
    config = set_regularizer_config(config)
    config = get_unsupervised_config(config)
    config = get_positional_encoding_config(config)
    configs.append(config)

  return configs


def set_edge_scorer_config(config, num_feats):
  """Sets the edge scorer config."""
  config.model.edge_scorer_cfg.name = random.choice(["mlp", "fp", "attentive"])
  config.model.edge_scorer_cfg.nlayers = random.choice([1, 2])
  # if es is mlp, then:
  config.model.edge_scorer_cfg.nheads = random.choice([1, 2, 4])
  config.model.edge_scorer_cfg.activation = random.choice(["relu", "tanh"])
  config.model.edge_scorer_cfg.dropout_rate = random.uniform(0.0, 75e-2)
  config.model.edge_scorer_cfg.initialization = random.choice(
      ["method1", "method2"]
  )
  config.model.edge_scorer_cfg.hidden_size = random.choice([500, num_feats])
  config.model.edge_scorer_cfg.output_size = random.choice([500, num_feats])
  # if es is attentive
  config.model.edge_scorer_cfg.nheads = random.choice([1, 2, 4])
  return config


def set_sparsifier_config(config):
  """Sets the sparsifier config."""
  config.model.sparsifier_cfg.name = random.choice(["knn", "dilated-knn"])
  config.model.sparsifier_cfg.k = random.choice([20, 25, 30])
  config.model.sparsifier_cfg.d = random.choice([2, 3])
  config.model.sparsifier_cfg.random_dilation = bool(random.getrandbits(1))
  return config


def set_processor_config(config):
  """Sets the preprocessor config."""
  config.model.processor_cfg.name = random.choice(
      ["none", "symmetrize", "activation", "activation-symmetrize"]
  )
  config.model.processor_cfg.activation = random.choice(["relu", "elu"])
  config.model.merger_cfg.dropout_rate = random.uniform(0.0, 75e-2)
  return config


def set_encoder_config(config):
  """Sets the encoder config."""
  config.model.encoder_cfg.name = random.choice(["gcn", "gin"])
  config.model.encoder_cfg.hidden_units = random.choice([16, 32, 64, 128])
  config.model.encoder_cfg.activation = random.choice(["relu", "tanh"])
  config.model.encoder_cfg.dropout_rate = random.uniform(0.0, 75e-2)
  return config


def set_regularizer_config(config):
  """Sets the regularizer config."""
  config.model.regularizer_cfg.closeness_enable = bool(random.getrandbits(1))
  config.model.regularizer_cfg.smoothness_enable = bool(random.getrandbits(1))
  config.model.regularizer_cfg.sparseconnect_enable = bool(
      random.getrandbits(1)
  )
  config.model.regularizer_cfg.logbarrier_enable = bool(random.getrandbits(1))
  config.model.regularizer_cfg.information_enable = bool(random.getrandbits(1))
  config.model.regularizer_cfg.closeness_w = random.uniform(0.0, 20.0)
  config.model.regularizer_cfg.smoothness_w = random.uniform(0.0, 20.0)
  config.model.regularizer_cfg.sparseconnect_w = random.uniform(0.0, 20.0)
  config.model.regularizer_cfg.logbarrier_w = random.uniform(0.0, 20.0)
  config.model.regularizer_cfg.information_w = random.uniform(0.0, 20.0)
  return config


def get_unsupervised_config(config):
  """Sets the unsupervised loss config."""
  # Contrastive loss
  config.model.unsupervised_cfg.contrastive_cfg.enable = bool(
      random.getrandbits(1))
  config.model.unsupervised_cfg.contrastive_cfg.w = random.uniform(0.0, 20.0)
  config.model.unsupervised_cfg.contrastive_cfg.feature_mask_rate = (
      random.uniform(1e-2, 75e-2))
  config.model.unsupervised_cfg.contrastive_cfg.temperature = random.uniform(
      0.1, 1.0)
  config.model.unsupervised_cfg.contrastive_cfg.tau = random.uniform(0.0, 0.2)

  # Denoising loss
  config.model.unsupervised_cfg.denoising_cfg.enable = bool(
      random.getrandbits(1))
  config.model.unsupervised_cfg.denoising_cfg.w = random.uniform(0.0, 20.0)
  config.model.unsupervised_cfg.denoising_cfg.dropout_rate = random.uniform(
      0.0, 0.75)
  config.model.unsupervised_cfg.denoising_cfg.hidden_units = random.choice(
      [512, 1024])
  config.model.unsupervised_cfg.denoising_cfg.ones_ratio = random.choice(
      [1, 5, 10])
  config.model.unsupervised_cfg.denoising_cfg.negative_ratio = random.choice(
      [1, 5])

  return config


def get_positional_encoding_config(config):
  """Sets the positional encoding config."""
  config.dataset.add_wl_position_encoding = bool(random.getrandbits(1))
  config.dataset.add_spectral_encoding = bool(random.getrandbits(1))
  return config
