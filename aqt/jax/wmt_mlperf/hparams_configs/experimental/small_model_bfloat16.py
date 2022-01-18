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

"""Small-sized unquantized model.

A 'small size' halves each model size parameter of the full model:
* 3 layers in the encoder and decoder
* 512 emd_dim
* 8 heads
* 512 qkv_dim
* 2048 mlp_dim
"""

from aqt.jax.wmt_mlperf.hparams_configs import base_config


def get_config(quant_target=base_config.QuantTarget.none):
  """Returns configuration for a small transformer model."""
  config = base_config.get_config(n_layers=3, quant_target=quant_target)
  model = config.model_hparams
  model.emb_dim = 512
  model.num_heads = 8
  model.qkv_dim = 512
  model.mlp_dim = 2048
  return config
