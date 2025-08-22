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
"""Transformer predictor."""
from typing import Optional, Tuple, Union

from absl import flags
import flax.linen as nn
from incontext import transformer_lib_flax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]

flags.DEFINE_bool("loss_on_x_steps", default=False, help="Take loss on x steps")


def extract_y(seq, offset = 0):
  """Extracts the y vector from the input tensor.

  Args:
          seq (torch.Tensor): tensor with shape (batch_size, seq_length,
            hidden_size)
          offset (int, optional): optional offset for where ys start. Defaults
            to 0.

  Returns:
          torch.Tensor: tensor with shape (batch_size, num_exemplars,
          hidden_size)
  """
  return seq[:, jnp.arange(offset, seq.shape[1], 2), :1]


class CausalLM(nn.Module):
  """CausalLM model that predicts next vector or onyl the next y."""
  config: transformer_lib_flax.TransformerConfig

  @nn.compact
  def __call__(
      self,
      *,
      inputs,
      train,
      return_attention = False,
  ):
    """CausalLM is a transformer based auto-regressive model.

    It predicts the next continuos vector based on the previous
    vectors.
    Args:
      inputs (Array): input tensor
      train (bool): training mode
      return_attention (bool): whether to return attentions

    Returns:
      Tuple[Array, Tuple[Array, ...]]: Tuple of loss and extras
    """
    config = self.config
    seq_from = inputs[:, :-1, :]
    mask = nn.attention.make_causal_mask(seq_from[:, :, 0])
    seq_enc, seq_hiddens, attn_weights = transformer_lib_flax.Transformer(
        config)(
            inputs=seq_from,
            mask=mask,
            train=train,
            return_attention=return_attention)

    if config.loss_on_x_steps:
      output_shape = seq_from.shape[-1]
    else:
      output_shape = 1

    seq_pred = nn.Dense(
        output_shape,
        kernel_init=config.linear_w_init,
        bias_init=config.linear_bias_init)(
            seq_enc)

    y_pred = extract_y(seq_pred, offset=0)
    seq_target = inputs[:, 1:, :]
    y_target = extract_y(seq_target, offset=0)
    y_errors = ((y_pred - y_target)**2).sum(axis=-1)
    if config.loss_on_x_steps:
      errors = ((seq_pred - seq_target)**2).sum(axis=-1)
    else:
      errors = y_errors

    if return_attention:
      return errors, (y_errors, y_pred, seq_pred, seq_hiddens, attn_weights)
    else:
      return errors, (y_errors, y_pred, seq_pred, seq_hiddens)

  def extract_y(self, seq, offset = 0):
    return extract_y(seq, offset=offset)
