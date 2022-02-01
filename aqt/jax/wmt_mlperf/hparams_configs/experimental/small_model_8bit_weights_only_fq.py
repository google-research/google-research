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

"""Full-sized model with 8-bit quantization for weights only."""

from aqt.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax.wmt_mlperf.hparams_configs.experimental import small_model_bfloat16


def get_config():
  config = small_model_bfloat16.get_config(
      quant_target=base_config.QuantTarget.weights_only)
  config.quant_type = "fake_quant"
  config.weight_prec = 8
  config.metadata.hyper_str = "small_8bit_weights_only_fq"
  return config
