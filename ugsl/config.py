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

"""Training configurations."""
import os
import time
from ml_collections import config_dict


def get_config():
  """The default configuration."""
  cfg = config_dict.ConfigDict()

  # The dataset name.
  cfg.dataset = config_dict.ConfigDict()
  cfg.dataset.name = "cora"
  cfg.dataset.remove_noise_ratio = 0.0
  cfg.dataset.add_noise_ratio = 0.0
  cfg.dataset.add_wl_position_encoding = False
  cfg.dataset.add_spectral_encoding = False

  cfg.run = config_dict.ConfigDict()
  # Number of epochs.
  cfg.run.num_epochs = 1000
  # Evaluate on validation every num_epochs.
  cfg.run.eval_every = 10
  cfg.run.print_loss_every = 10

  # Optimizer hyperparameters.
  cfg.run.learning_rate = 1e-3
  cfg.run.weight_decay = 5e-4
  cfg.run.model_dir = os.path.expanduser(
      os.path.join("~", "ugsl", str(int(time.time()*1000))))

  cfg.model = config_dict.ConfigDict()
  # GSL layer(s).
  cfg.model.depth = 2
  cfg.model.adjacency_learning_mode = "shared_adjacency_matrix"
  # Accepted values: 'WL', 'spectral' (see position_encoders.py).
  cfg.model.position_encoders = []

  # Edge scorer (see edge_scorer.py).
  cfg.model.edge_scorer_cfg = config_dict.ConfigDict()
  cfg.model.edge_scorer_cfg.name = "mlp"
  cfg.model.edge_scorer_cfg.hidden_size = 1433
  cfg.model.edge_scorer_cfg.output_size = 1433
  cfg.model.edge_scorer_cfg.nlayers = 2
  cfg.model.edge_scorer_cfg.nheads = 2
  cfg.model.edge_scorer_cfg.activation = "relu"
  cfg.model.edge_scorer_cfg.initialization = "method1"
  cfg.model.edge_scorer_cfg.dropout_rate = 0.5

  # Sparsifier (see sparsifier.py).
  cfg.model.sparsifier_cfg = config_dict.ConfigDict()
  cfg.model.sparsifier_cfg.name = "knn"
  cfg.model.sparsifier_cfg.k = 20
  cfg.model.sparsifier_cfg.epsilon = 0.5
  # Parameters for dilated knn.
  cfg.model.sparsifier_cfg.d = 2
  cfg.model.sparsifier_cfg.random_dilation = False
  # Parameters for Bernoulli.
  cfg.model.sparsifier_cfg.do_sigmoid = False
  cfg.model.sparsifier_cfg.soft_version = True
  cfg.model.sparsifier_cfg.temperature = 1.0

  # Processor (see processor.py).
  cfg.model.processor_cfg = config_dict.ConfigDict()
  cfg.model.processor_cfg.name = "none"
  cfg.model.processor_cfg.activation = "relu"

  # Merger (see merger.py).
  cfg.model.merger_cfg = config_dict.ConfigDict()
  cfg.model.merger_cfg.name = "none"
  cfg.model.merger_cfg.dropout_rate = 0.5
  cfg.model.merger_cfg.given_adjacency_weight = 0.5

  # Encoder (see encoder.py).
  cfg.model.encoder_cfg = config_dict.ConfigDict()
  cfg.model.encoder_cfg.name = "gcn"
  cfg.model.encoder_cfg.hidden_units = 32
  cfg.model.encoder_cfg.activation = "relu"
  cfg.model.encoder_cfg.dropout_rate = 0.5

  # Regularizers (see regularizers.py).
  cfg.model.regularizer_cfg = config_dict.ConfigDict()
  cfg.model.regularizer_cfg.closeness_enable = False
  cfg.model.regularizer_cfg.smoothness_enable = False
  cfg.model.regularizer_cfg.sparseconnect_enable = False
  cfg.model.regularizer_cfg.logbarrier_enable = False
  cfg.model.regularizer_cfg.information_enable = False
  cfg.model.regularizer_cfg.closeness_w = 0.0
  cfg.model.regularizer_cfg.smoothness_w = 0.0
  cfg.model.regularizer_cfg.sparseconnect_w = 0.0
  cfg.model.regularizer_cfg.logbarrier_w = 0.0
  cfg.model.regularizer_cfg.information_w = 0.0
  cfg.model.regularizer_cfg.information_r = 0.5
  cfg.model.regularizer_cfg.information_do_sigmoid = True

  # Unsupervised losses (see unsupervised_losses.py)
  # Denoising loss.
  cfg.model.unsupervised_cfg = config_dict.ConfigDict()
  cfg.model.unsupervised_cfg.denoising_cfg = config_dict.ConfigDict()
  cfg.model.unsupervised_cfg.denoising_cfg.enable = False
  cfg.model.unsupervised_cfg.denoising_cfg.w = 0.0
  cfg.model.unsupervised_cfg.denoising_cfg.ones_ratio = 5
  cfg.model.unsupervised_cfg.denoising_cfg.negative_ratio = 10
  cfg.model.unsupervised_cfg.denoising_cfg.hidden_units = 512
  cfg.model.unsupervised_cfg.denoising_cfg.depth = 2
  cfg.model.unsupervised_cfg.denoising_cfg.dropout_rate = 0.5
  cfg.model.unsupervised_cfg.denoising_cfg.activation = "relu"
  # Contrastive loss.
  cfg.model.unsupervised_cfg.contrastive_cfg = config_dict.ConfigDict()
  cfg.model.unsupervised_cfg.contrastive_cfg.enable = False
  cfg.model.unsupervised_cfg.contrastive_cfg.w = 0.0
  cfg.model.unsupervised_cfg.contrastive_cfg.tau = 0.0
  cfg.model.unsupervised_cfg.contrastive_cfg.feature_mask_rate = 0.6
  cfg.model.unsupervised_cfg.contrastive_cfg.temperature = 1.0

  return cfg.lock()
