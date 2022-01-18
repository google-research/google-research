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

"""Minimally-sized unquantized model.

The minimal model has the smallest legal values for all model size parameters.

Useful for testing that training the model doesn't generate errors.
"""

from aqt.jax.wmt_mlperf.hparams_configs import base_config


def get_config(quant_target=base_config.QuantTarget.none):
  """Returns configuration for a minimal transformer model."""
  config = base_config.get_config(quant_target=quant_target, n_layers=1)
  config.num_train_steps = 1
  model = config.model_hparams
  model.emb_dim = 1
  model.num_heads = 1
  model.qkv_dim = 1
  model.mlp_dim = 1
  return config
