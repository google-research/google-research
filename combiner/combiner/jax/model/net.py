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

# pylint: skip-file
from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import lax

from combiner.jax.model.transformer_base import AddPositionEmbs, TransformerConfig
from functools import partial


class AutoregTransformer(nn.Module):
  config: TransformerConfig
  transformer_layer: Callable
  pred_dim: int

  @nn.compact
  def __call__(self, inputs):
    config = self.config
    assert inputs.ndim == 3  # (batch, len, embed)
    y = AddPositionEmbs(config=config)(inputs)
    y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)
    assert issubclass(type(self.transformer_layer), partial)
    for l in range(config.num_layers):
      y = self.transformer_layer(name='transformer_layer_%d' % l)(y)
    y = nn.LayerNorm(dtype=config.dtype)(y)

    out = nn.Dense(self.pred_dim,
                   dtype=config.dtype,
                   kernel_init=config.kernel_init,
                   bias_init=config.bias_init)(y)
    return out
