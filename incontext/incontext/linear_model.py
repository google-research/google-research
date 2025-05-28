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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Linear model in flax."""
from typing import Any, Callable

from flax import linen as nn
from flax import struct
from incontext import transformer_lib_flax as tl
from incontext import utils
import jax.numpy as jnp

Array = utils.Array


@struct.dataclass
class LinearConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  dtype: Any = jnp.float32
  kernel_init: Callable[[Any, Any, tl.Dtype], Array] = tl.uniform_scaling()
  alpha: float = 0.0


def l2_loss(x, alpha):
  return alpha * (x**2).mean()


class LinearModel(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
  """

  config: LinearConfig

  @nn.compact
  def __call__(self, inputs, train):
    """Applies PositionEmbeddings module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.
    Args:
      inputs: input data.
      train: training mode vs eval mode.

    Returns:
      output: `(bs, in_dim)`
    """
    del train
    config = self.config
    # inputs.shape is (batch_size, x_dim)
    return nn.Dense(
        1, use_bias=False, dtype=config.dtype, kernel_init=config.kernel_init)(
            inputs)
