# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Layers used in a Transformer."""

from flax import nn
import jax.numpy as jnp


LAYER_NORM_EPSILON = 1e-6


class PositionalEncoding(nn.Module):
  """Learned positional embeddings for the Transformer."""

  def apply(self,
            inputs, *,
            max_len = 2048,
            posemb_init=nn.initializers.xavier_normal()):
    """Applies PositionalEncoding module."""
    assert inputs.ndim == 3, (
        f"Number of dimension should be 3, but it is: {inputs.ndim}")
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    pos_embedding = self.param("embedding", pos_emb_shape, posemb_init)
    return pos_embedding[:, :length, :]


class FeedForward(nn.Module):
  """Feed-forward layer for a Transformer model."""
  # TODO(kitaev): support chunking

  def apply(self,
            hidden_states, *,
            d_ff,
            dropout_rate = 0.0,
            intermediate_activation=nn.gelu,
            # TODO(kitaev): chunk_size hparam for chunking
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic = False):
    """Applies FeedForward module."""
    d_model = hidden_states.shape[-1]
    hidden_states = nn.Dense(
        hidden_states,
        d_ff,
        kernel_init=kernel_init,
        name="intermediate")
    hidden_states = intermediate_activation(hidden_states)
    hidden_states = nn.Dense(
        hidden_states,
        d_model,
        kernel_init=kernel_init,
        name="output")
    hidden_states = nn.dropout(
        hidden_states, rate=dropout_rate, deterministic=deterministic)
    return hidden_states


class TransformerBlock(nn.Module):
  """Post-norm transformer block.."""

  def apply(self,
            hidden_states, mask=None, *,
            feed_forward,
            attention,
            deterministic = False):
    """Applies TransformerBlock module."""
    attention_output = attention(hidden_states, mask,
                                 deterministic=deterministic,
                                 name="self_attention")
    hidden_states = nn.LayerNorm(hidden_states + attention_output,
                                 epsilon=LAYER_NORM_EPSILON,
                                 name="self_attention_layer_norm")
    feed_forward_output = feed_forward(hidden_states,
                                       deterministic=deterministic,
                                       name="feed_forward")
    hidden_states = nn.LayerNorm(hidden_states + feed_forward_output,
                                 epsilon=LAYER_NORM_EPSILON,
                                 name="output_layer_norm")

    return hidden_states


class OutputProjection(nn.Module):
  """A dense projection layer for computing output logits."""

  def apply(self,
            inputs, kernel=None, *,
            n_out=None,
            bias=True,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros):
    """Applies OutputProjection module."""
    if kernel is None:
      assert n_out is not None, (
          "n_out argument is required when not re-using an embedding matrix")
      kernel = self.param("kernel", (n_out, inputs.shape[-1]), kernel_init)
    y = jnp.matmul(inputs, jnp.transpose(kernel, (1, 0)))
    if bias:
      bias = self.param("bias", (y.shape[-1],), bias_init)
      y = y + bias
    return y
