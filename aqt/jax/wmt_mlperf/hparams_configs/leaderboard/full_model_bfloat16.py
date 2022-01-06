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

"""Full-sized unquantized model.

A 'full size' means
* 6 layers in the encoder and decoder
* 1024 emd_dim
* 16 heads
* 1024 qkv_dim
* 4096 mlp_dim
"""

from aqt.jax.wmt_mlperf.hparams_configs import base_config


def get_config(quant_target=base_config.QuantTarget.none):
  config = base_config.get_config(n_layers=6, quant_target=quant_target)
  config.metadata.hyper_str = 'full_bfloat16'
  return config
