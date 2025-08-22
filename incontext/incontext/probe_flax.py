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
"""Probe model to decode linear weights."""
from typing import Any, Callable

from flax import linen as nn
from flax import struct
from incontext import transformer_lib_flax as tl
from incontext import utils
import jax.numpy as jnp

Array = utils.Array
Dtype = utils.Dtype


@struct.dataclass
class ProbeConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  dtype: Any = jnp.float32
  kernel_init: Callable[Ellipsis, Array] = tl.uniform_scaling()
  hidden_size: int = 0
  num_layers: int = 1
  x_dim: int = 2
  max_len: int = 128


class ProbeModel(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
  """

  config: ProbeConfig

  @nn.compact
  def __call__(self, *, seq_hiddens, coefficients,
               train):
    """Applies PositionEmbeddings module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.
    Args:
      seq_hiddens: input hidden states
      coefficients: target coefficients to decode
      train: training mode vs eval mode

    Returns:
      output: `(bs, in_dim)`
    """
    del train
    config = self.config
    # inputs.shape is (batch_size, x_dim)
    probe_errors = []
    for i in range(seq_hiddens.shape[1]):
      layer_probe = nn.Dense(
          config.x_dim,
          use_bias=True,
          dtype=config.dtype,
          kernel_init=config.kernel_init)
      probe_pred = layer_probe(seq_hiddens[:, i, -1:, :])
      probe_err = (probe_pred[:, -1:, :] - coefficients[:, None, :])**2
      probe_errors.append(probe_err.sum(axis=2))
    return jnp.stack(probe_errors, axis=1)
