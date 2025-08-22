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

"""Transformer modules using Jax/Flax."""

import dataclasses
from typing import Any, Type

import flax.linen as nn
import flax.typing as flax_typing
import jax
from jax import numpy as jnp

from imp.max.core import constants
from imp.max.core import transforms
from imp.max.modeling import attention
from imp.max.modeling import linear
from imp.max.modeling import moe
from imp.max.modeling import normalization
from imp.max.modeling import stacking
from imp.max.utils import sharding
from imp.max.utils import typing


ScanIn = flax_typing.In
PARAMS = constants.FlaxCollection.PARAMS


def _verify_broadcastable_batch_instance(
    attention_batch_size, inputs_batch_size,
    attention_instance, inputs_instance):
  batch_nonbroadcastable = (attention_batch_size != inputs_batch_size
                            and attention_batch_size != 1)
  instance_nonbroadcastable = (attention_instance != inputs_instance
                               and attention_instance != 1)
  if batch_nonbroadcastable or instance_nonbroadcastable:
    raise ValueError(
        'Attention mask should have broadcastable `batch` and `instance` '
        f'dimensions. Instead, received {attention_batch_size=}, '
        f'{inputs_batch_size=} and {attention_instance=}, {inputs_instance=}.')


def _verify_attention_mask(inputs,
                           attention_mask):
  """Assert attention mask has a proper shape."""
  if attention_mask is None:
    return
  # attetion_mask should have a shape broadcastable to
  # shape: [batch, instance, heads, length, length]
  inputs_shape = inputs.shape
  attention_shape = attention_mask.shape
  if len(attention_shape) != 5:
    raise ValueError('Attention mask should have a rank of 5. Instead, '
                     f'received an array with rank {len(attention_shape)}.')

  (attention_batch_size, attention_instance, _,
   attention_q_len, attention_k_len) = attention_shape
  inputs_batch_size, inputs_instance, inputs_length = inputs_shape[:-1]

  if attention_q_len != attention_k_len:
    raise ValueError(
        'Attention mask should have equal q_len and k_len. Instead, received '
        f'{attention_q_len=} and {attention_k_len=}.')

  if attention_q_len != inputs_length:
    raise ValueError(
        'Attention mask and the inputs should have the same `length`. Instead, '
        f'received {attention_q_len=}, {inputs_length=}.')

  _verify_broadcastable_batch_instance(attention_batch_size,
                                       inputs_batch_size,
                                       attention_instance,
                                       inputs_instance)


def _verify_cross_attention_mask(
    inputs,
    cross_inputs,
    cross_attention_mask):
  """Assert cross attention mask has a proper shape."""
  if cross_attention_mask is None or cross_inputs is None:
    return
  # cross_attention_mask should have a shape broadcastable to
  # shape: [batch, instance, heads, q_length, k_length]
  inputs_shape = inputs.shape
  cross_inputs_shape = cross_inputs.shape
  attention_shape = cross_attention_mask.shape

  if len(attention_shape) != 5:
    raise ValueError('Cross attention mask should have a rank of 5. Instead, '
                     f'received an array with rank {len(attention_shape)}.')

  (attention_batch_size, attention_instance, _,
   attention_q_len, attention_k_len) = attention_shape
  inputs_batch_size, inputs_instance, inputs_q_length = inputs_shape[:-1]
  inputs_k_length = cross_inputs_shape[2]

  if attention_q_len != inputs_q_length or attention_k_len != inputs_k_length:
    raise ValueError(
        'Cross attention mask should reflect the same `q_length` and '
        'as the inputs and cross-inputs. Instead, received'
        f' {attention_q_len=}, {inputs_q_length=} and {attention_k_len=}, '
        f'{inputs_k_length=}.')

  _verify_broadcastable_batch_instance(attention_batch_size,
                                       inputs_batch_size,
                                       attention_instance,
                                       inputs_instance)


def _verify_attention_bias(inputs,
                           attention_bias,
                           model_heads):
  """Assert attention bias has a proper shape."""
  if attention_bias is None:
    return
  # attetion_mask should have a shape broadcastable to
  # shape: [batch, instance, heads, length, length]
  inputs_shape = inputs.shape
  attention_shape = attention_bias.shape
  if len(attention_shape) != 3:
    raise ValueError('Attention bias should have a rank of 3. Instead, '
                     f'received an array with rank {len(attention_shape)}.')

  attention_heads, attention_q_len, attention_k_len = attention_shape
  inputs_length = inputs_shape[2]

  if attention_q_len != attention_k_len:
    raise ValueError(
        'Attention bias should have equal q_len and k_len. Instead, received '
        f'{attention_q_len=} and {attention_k_len=}.')

  if attention_q_len != inputs_length:
    raise ValueError(
        'Attention bias and the inputs should have the same `length`. Instead, '
        f'received {attention_q_len=}, {inputs_length=}.')

  if attention_heads != model_heads:
    raise ValueError(
        'Attention bias should have the same number of heads as the model. '
        f'Instead, {attention_heads=}, {model_heads=}.')


class FeedForward(nn.Module):
  """FFN layer in Transformer architecture."""

  d_ff: int
  d_model: int
  use_bias: bool
  dropout_rate: float
  approximate_gelu: bool
  dtype: jax.typing.DTypeLike
  dot_general: typing.DotGeneral
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  inner_kernel_shardings: typing.ShardingAxes
  outer_kernel_shardings: typing.ShardingAxes
  intermediate_shardings: typing.ShardingAxes

  def setup(self):
    self.wi = linear.DenseGeneral(
        features=self.d_ff,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_shardings=self.inner_kernel_shardings,
        dot_general=self.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='wi',
    )
    self.dropout_mlp = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))
    self.wo = linear.DenseGeneral(
        features=self.d_model,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_shardings=self.outer_kernel_shardings,
        dot_general=self.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='wo'
    )

  def __call__(self,
               inputs,
               deterministic = True,
               metadata = None):
    x = self.wi(inputs)
    x = nn.gelu(x, approximate=self.approximate_gelu)
    x = self.dropout_mlp(x, deterministic)
    x = sharding.shard_array(x, self.intermediate_shardings)
    output = self.wo(x)

    return output


@dataclasses.dataclass
class SetupFFN:
  """Helper class for constructing a dense FFN under nn.Module.setup()."""

  d_model: int
  d_ff: int
  use_bias: bool
  dropout_rate: float
  dtype: jax.typing.DTypeLike
  approximate_gelu: bool
  ffn_dot_general: typing.DotGeneral
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  precision: typing.Precision
  lora_rank: int
  lora_scale: float

  def setup_feed_forward(self):
    self.feed_forward = FeedForward(
        d_ff=self.d_ff,
        d_model=self.d_model,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        approximate_gelu=self.approximate_gelu,
        dot_general=self.ffn_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        inner_kernel_shardings=self.ffn_inner_kernel_shardings,
        outer_kernel_shardings=self.ffn_outer_kernel_shardings,
        intermediate_shardings=self.ffn_intermediate_shardings,
        name='feed_forward',
    )


class TransformerEncoderLayer(SetupFFN, nn.Module):
  """A single-layer Transformer encoder."""

  num_heads: int
  qk_layernorm: bool
  mha_qkv_dot_general: typing.DotGeneral
  mha_out_dot_general: typing.DotGeneral
  mha_einsum_dot_general: typing.DotGeneral
  mha_qkv_kernel_shardings: typing.ShardingAxes
  mha_out_kernel_shardings: typing.ShardingAxes
  mha_activation_shardings: typing.ShardingAxes
  layernorm_shardings: typing.ShardingAxes

  def setup(self):
    d_head = self.d_model // self.num_heads

    # self-attention modules
    self.layer_norm_self = normalization.LayerNorm(
        use_bias=self.use_bias,
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm_sa'
    )
    self.mha_self = attention.MultiHeadAttention(
        d_head=d_head,
        d_model=self.d_model,
        num_heads=self.num_heads,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        qk_layernorm=self.qk_layernorm,
        qkv_kernel_shardings=self.mha_qkv_kernel_shardings,
        out_kernel_shardings=self.mha_out_kernel_shardings,
        activation_shardings=self.mha_activation_shardings,
        layernorm_shardings=self.layernorm_shardings,
        qkv_dot_general=self.mha_qkv_dot_general,
        out_dot_general=self.mha_out_dot_general,
        einsum_dot_general=self.mha_einsum_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='self_attention',
    )
    self.dropout_self = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

    # post-attention modules
    self.layer_norm_ffn = normalization.LayerNorm(
        use_bias=self.use_bias,
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm_ffn'
    )
    self.setup_feed_forward()
    self.dropout_ffn = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

  def __call__(self,
               inputs,
               deterministic = True,
               attention_mask = None,
               attention_bias = None,
               metadata = None):
    # TODO(b/230341025): add per-layer relative biases

    # self-attention
    x = self.layer_norm_self(inputs)
    x = self.mha_self(query=x, key=x, value=x,
                      attention_mask=attention_mask,
                      attention_bias=attention_bias,
                      deterministic=deterministic)
    x = self.dropout_self(x, deterministic)

    # skip connection
    x = inputs + x

    # feedforward projection
    y = self.layer_norm_ffn(x)
    y = self.feed_forward(y, deterministic, metadata)
    y = self.dropout_ffn(y, deterministic)

    # skip connection
    output = x + y

    return output


class TransformerDecoderLayer(SetupFFN, nn.Module):
  """A single-layer Transformer decoder."""

  num_heads: int
  qk_layernorm: bool
  mha_qkv_dot_general: typing.DotGeneral
  mha_out_dot_general: typing.DotGeneral
  mha_einsum_dot_general: typing.DotGeneral
  mha_qkv_kernel_shardings: typing.ShardingAxes
  mha_out_kernel_shardings: typing.ShardingAxes
  mha_activation_shardings: typing.ShardingAxes
  layernorm_shardings: typing.ShardingAxes

  def setup(self):
    d_head = self.d_model // self.num_heads

    # self-attention modules
    self.layer_norm_self = normalization.LayerNorm(
        use_bias=self.use_bias,
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm_sa'
    )
    self.mha_self = attention.MultiHeadAttention(
        d_head=d_head,
        num_heads=self.num_heads,
        d_model=self.d_model,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        qk_layernorm=self.qk_layernorm,
        qkv_kernel_shardings=self.mha_qkv_kernel_shardings,
        out_kernel_shardings=self.mha_out_kernel_shardings,
        activation_shardings=self.mha_activation_shardings,
        layernorm_shardings=self.layernorm_shardings,
        qkv_dot_general=self.mha_qkv_dot_general,
        out_dot_general=self.mha_out_dot_general,
        einsum_dot_general=self.mha_einsum_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='self_attention',
    )
    self.dropout_self = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

    # cross-attention modules
    self.layer_norm_cross = normalization.LayerNorm(
        use_bias=self.use_bias,
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm_ca'
    )
    self.mha_cross = attention.MultiHeadAttention(
        d_head=d_head,
        num_heads=self.num_heads,
        d_model=self.d_model,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        qk_layernorm=self.qk_layernorm,
        qkv_kernel_shardings=self.mha_qkv_kernel_shardings,
        out_kernel_shardings=self.mha_out_kernel_shardings,
        activation_shardings=self.mha_activation_shardings,
        layernorm_shardings=self.layernorm_shardings,
        qkv_dot_general=self.mha_qkv_dot_general,
        out_dot_general=self.mha_out_dot_general,
        einsum_dot_general=self.mha_einsum_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='cross_attention',
    )
    self.dropout_cross = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

    # post-attention modules
    self.layer_norm_ffn = normalization.LayerNorm(
        use_bias=self.use_bias,
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm_ffn'
    )
    self.setup_feed_forward()
    self.dropout_ffn = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

  def __call__(self,
               inputs,
               cross_inputs,
               deterministic = True,
               attention_mask = None,
               attention_bias = None,
               cross_attention_mask = None,
               max_decode_length = None,
               decode = False,
               metadata = None):
    """Applies multi-head dot product attention on the input data.

    Args:
      inputs: input sequence of shape `[batch, instance, self_length, dim]`
      cross_inputs: input sequence for the decoder to be conditioned on. This
        is usually the output of the encoder tower, with shape
        `[batch, instance, cross_length, dim]`
      deterministic: bool, whether to apply non-deterministic transforms
      attention_mask: attention mask of shape
        `[batch, instance, 1, dec_length, dec_length]`.
      attention_bias: attention bias of shape
        `[1, 1, num_heads, dec_length, dec_length]`.
      cross_attention_mask: cross-attention mask of shape
        `[1, 1, num_heads, dec_length, cross_length]`.
      max_decode_length: maximum length of the decoding sequence
      decode: bool, whether the decoder is in decoding mode or not. if True,
        it caches key/values and fetches the attention masks and biases
        for the current step.
      metadata: additional metadata, for instance, to track input modality.

    Returns:
      output: output sequence of shape `[batch, instance, dec_length, dim]`
    """

    # TODO(b/230341025): add per-layer relative biases
    del max_decode_length

    # self-attention
    x = self.layer_norm_self(inputs)
    x = self.mha_self(query=x, key=x, value=x, decode=decode,
                      attention_mask=attention_mask,
                      attention_bias=attention_bias,
                      deterministic=deterministic)
    x = self.dropout_self(x, deterministic)

    # skip connection
    x = inputs + x

    if cross_inputs is not None:
      # cross-attention
      y = self.layer_norm_cross(x)
      y = self.mha_cross(query=y, key=cross_inputs, value=cross_inputs,
                         attention_mask=cross_attention_mask,
                         deterministic=deterministic)
      y = self.dropout_cross(y, deterministic)

      # skip connection
      y = x + y

    else:
      y = x

    # feedforward projection
    z = self.layer_norm_ffn(y)
    z = self.feed_forward(z, deterministic, metadata)
    z = self.dropout_ffn(z, deterministic)

    # skip connection
    output = y + z

    return output


@dataclasses.dataclass
class SetupLayerStack:
  """Helper class for construction of the stack of a MHA-FFN layer."""

  d_model: int
  d_ff: int
  num_heads: int
  num_layers: int
  use_bias: bool
  dropout_rate: float
  remat: str
  scanned_layers: bool
  scan_axis: int
  dtype: jax.typing.DTypeLike
  qk_layernorm: bool
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  approximate_gelu: bool
  # Sharding annotations
  scan_sharding_axis: str | None
  layernorm_shardings: typing.ShardingAxes
  mha_qkv_kernel_shardings: typing.ShardingAxes
  mha_out_kernel_shardings: typing.ShardingAxes
  mha_activation_shardings: typing.ShardingAxes
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  # DotGenerals
  mha_qkv_dot_general: typing.DotGeneral
  mha_out_dot_general: typing.DotGeneral
  mha_einsum_dot_general: typing.DotGeneral
  ffn_dot_general: typing.DotGeneral

  def setup_layer_stack(
      self,
      layer_module,
      static_argnums,
      scan_in_axes):
    """Constructs the stack of a given MHA-FFN layer."""

    RematLayer = transforms.remat(  # pylint: disable=invalid-name
        module=layer_module,
        level=self.remat,
        scanned=self.scanned_layers,
        static_argnums=static_argnums,
    )

    layer_config = {
        'd_model': self.d_model,
        'd_ff': self.d_ff,
        'num_heads': self.num_heads,
        'use_bias': self.use_bias,
        'dropout_rate': self.dropout_rate,
        'dtype': self.dtype,
        'qk_layernorm': self.qk_layernorm,
        'layernorm_shardings': self.layernorm_shardings,
        'mha_qkv_kernel_shardings': self.mha_qkv_kernel_shardings,
        'mha_out_kernel_shardings': self.mha_out_kernel_shardings,
        'mha_activation_shardings': self.mha_activation_shardings,
        'ffn_inner_kernel_shardings': self.ffn_inner_kernel_shardings,
        'ffn_outer_kernel_shardings': self.ffn_outer_kernel_shardings,
        'ffn_intermediate_shardings': self.ffn_intermediate_shardings,
        'mha_qkv_dot_general': self.mha_qkv_dot_general,
        'mha_out_dot_general': self.mha_out_dot_general,
        'mha_einsum_dot_general': self.mha_einsum_dot_general,
        'ffn_dot_general': self.ffn_dot_general,
        'precision': self.precision,
        'lora_rank': self.lora_rank,
        'lora_scale': self.lora_scale,
        'approximate_gelu': self.approximate_gelu,
    }

    if self.scanned_layers:
      initializing = self.is_mutable_collection(PARAMS)  # type: ignore
      scan_axis = self.scan_axis if initializing else ScanIn(self.scan_axis)
      self.layer_stack = transforms.scan(
          module=RematLayer,
          length=self.num_layers,
          scan_axis=scan_axis,
          in_axes=scan_in_axes,
          out_axes=0,
          rng_keys=('dropout',),
          sharding_axis=self.scan_sharding_axis,
      )(name='layer_scan', **layer_config)

    else:
      self.layer_stack = stacking.SequentialStackCall([
          RematLayer(
              name='layer_{}'.format(n), **layer_config,
              ) for n in range(self.num_layers)
      ])

    self.final_layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='final_layer_norm'
    )
    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))


class TransformerEncoder(SetupLayerStack, nn.Module):
  """Transformer Encoder."""

  def setup(self):
    self.setup_layer_stack(
        layer_module=TransformerEncoderLayer,
        static_argnums=(1, 4),
        scan_in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast)
    )

  def __call__(self,
               inputs,
               deterministic = True,
               attention_mask = None,
               attention_bias = None,
               metadata = None):

    _verify_attention_mask(inputs, attention_mask)
    _verify_attention_bias(inputs, attention_bias, self.num_heads)

    if attention_bias is not None:
      # make sure bias is broadcastable to attention scores
      # with shape: [batch, instance, heads, length, length]
      attention_bias = attention_bias[jnp.newaxis, jnp.newaxis, :]

    inputs = self.layer_stack(
        inputs, deterministic, attention_mask, attention_bias, metadata
    )

    outputs = self.final_layer_norm(inputs)
    outputs = self.dropout(outputs, deterministic)

    return outputs


class TransformerDecoder(SetupLayerStack, nn.Module):
  """Transformer Decoder."""

  def setup(self):
    self.setup_layer_stack(
        layer_module=TransformerDecoderLayer,
        static_argnums=(2, 6, 7, 8),
        scan_in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
    )

  def __call__(self,
               inputs,
               cross_inputs,
               deterministic = True,
               attention_mask = None,
               attention_bias = None,
               cross_attention_mask = None,
               max_decode_length = None,
               decode = False,
               metadata = None):
    """Applies a stack of multi-head dot product attentions on the input data.

    Args:
      inputs: input sequence of shape `[batch, instance, dec_length, dim]`
      cross_inputs: input sequence for the decoder to be conditioned on. This
        is usually the output of the encoder tower, with shape
        `[batch, instance, cross_length, dim]`
      deterministic: bool, whether to apply non-deterministic transforms
      attention_mask: attention mask of shape
        `[batch, instance, 1, dec_length, dec_length]`.
      attention_bias: attention bias of shape
        `[1, 1, num_heads, dec_length, dec_length]`.
      cross_attention_mask: cross-attention mask of shape
        `[1, 1, num_heads, dec_length, cross_length]`.
      max_decode_length: maximum length of the decoding sequence
      decode: bool, whether the decoder is in decoding mode or not. if True,
        it caches key/values and fetches the attention masks and biases
        for the current step.
      metadata: additional metadata, for instance, to track input modality.

    Returns:
      outputs: output sequence of shape `[batch, instance, dec_length, dim]`
    """

    _verify_attention_mask(inputs, attention_mask)
    _verify_attention_bias(inputs, attention_bias, self.num_heads)
    _verify_cross_attention_mask(inputs, cross_inputs, cross_attention_mask)

    if attention_bias is not None:
      # make sure bias is broadcastable to attention scores
      # with shape: [batch, instance, heads, dec_length, dec_length]
      attention_bias = attention_bias[jnp.newaxis, jnp.newaxis, :]

    inputs = self.layer_stack(
        inputs, cross_inputs, deterministic, attention_mask,
        attention_bias, cross_attention_mask,
        max_decode_length, decode, metadata,
    )

    inputs = self.final_layer_norm(inputs)
    outputs = self.dropout(inputs, deterministic)

    return outputs


class SparseMixtureOfFeedforward(moe.BaseSparseMoE):
  """Sparse Mixture-of-Experts with FeedForward as Experts."""

  d_ff: int
  d_model: int
  use_bias: bool
  dropout_rate: float
  approximate_gelu: bool
  dot_general: typing.DotGeneral
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  dtype: jax.typing.DTypeLike
  inner_kernel_shardings: typing.ShardingAxes
  outer_kernel_shardings: typing.ShardingAxes
  intermediate_shardings: typing.ShardingAxes

  def setup(self):
    expert = FeedForward(d_ff=self.d_ff,
                         d_model=self.d_model,
                         use_bias=self.use_bias,
                         dropout_rate=self.dropout_rate,
                         approximate_gelu=self.approximate_gelu,
                         dtype=self.dtype,
                         dot_general=self.dot_general,
                         precision=self.precision,
                         lora_rank=self.lora_rank,
                         lora_scale=self.lora_scale,
                         inner_kernel_shardings=self.inner_kernel_shardings,
                         outer_kernel_shardings=self.outer_kernel_shardings,
                         intermediate_shardings=self.intermediate_shardings,
                         name='experts')
    rng_keys = ('dropout',)
    self.assign_expert(expert, rng_keys)

  def __call__(self,
               inputs,
               deterministic = True,
               metadata = None):
    return super().__call__(inputs=inputs,
                            deterministic=deterministic,
                            metadata=metadata)


class SoftMixtureOfFeedforward(moe.BaseSoftMoE):
  """Soft Mixture-of-Experts with FeedForward as Experts."""

  d_ff: int
  d_model: int
  use_bias: bool
  dropout_rate: float
  approximate_gelu: bool
  dot_general: typing.DotGeneral
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  dtype: jax.typing.DTypeLike
  inner_kernel_shardings: typing.ShardingAxes
  outer_kernel_shardings: typing.ShardingAxes
  intermediate_shardings: typing.ShardingAxes

  def setup(self):
    expert = FeedForward(d_ff=self.d_ff,
                         d_model=self.d_model,
                         use_bias=self.use_bias,
                         dropout_rate=self.dropout_rate,
                         approximate_gelu=self.approximate_gelu,
                         dtype=self.dtype,
                         dot_general=self.dot_general,
                         precision=self.precision,
                         lora_rank=self.lora_rank,
                         lora_scale=self.lora_scale,
                         inner_kernel_shardings=self.inner_kernel_shardings,
                         outer_kernel_shardings=self.outer_kernel_shardings,
                         intermediate_shardings=self.intermediate_shardings,
                         name='experts')
    rng_keys = ('dropout',)
    self.assign_expert(expert, rng_keys)

  def __call__(self,
               inputs,
               deterministic = True,
               metadata = None):
    return super().__call__(inputs=inputs,
                            deterministic=deterministic,
                            metadata=metadata)


@dataclasses.dataclass
class SetupSparseMixtureOfFFN:
  """Helper class with method for configuration of a Sparse MoE FFN."""

  d_model: int
  d_ff: int
  num_heads: int
  use_bias: bool
  dropout_rate: float
  dtype: jax.typing.DTypeLike
  ffn_dot_general: typing.DotGeneral
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  router_kernel_shardings: typing.ShardingAxes
  tokens_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str
  approximate_gelu: bool
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  max_group_size: int
  capacity_factor: float
  min_expert_capacity: int
  router_type: str    # TODO(b/267474477): Merge it into `router_kwargs`.
  router_bias: bool
  strict_group_size: bool
  num_selected_experts: int
  batch_prioritized_routing: bool

  def setup_feed_forward(self):
    self.feed_forward = SparseMixtureOfFeedforward(
        d_ff=self.d_ff,
        d_model=self.d_model,
        use_bias=self.use_bias,
        dot_general=self.ffn_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        dropout_rate=self.dropout_rate,
        inner_kernel_shardings=self.ffn_inner_kernel_shardings,
        outer_kernel_shardings=self.ffn_outer_kernel_shardings,
        intermediate_shardings=self.ffn_intermediate_shardings,
        router_kernel_shardings=self.router_kernel_shardings,
        tokens_shardings=self.tokens_shardings,
        model_axis_size=self.model_axis_size,
        model_axis_name=self.model_axis_name,
        approximate_gelu=self.approximate_gelu,
        num_experts=self.num_experts,
        max_group_size=self.max_group_size,
        capacity_factor=self.capacity_factor,
        min_expert_capacity=self.min_expert_capacity,
        router_type=self.router_type,
        router_bias=self.router_bias,
        jitter_noise=self.jitter_noise,
        comm_dtype=self.comm_dtype,
        split_params=self.split_params,
        optimize_parallel_comms=self.optimize_parallel_comms,
        strict_group_size=self.strict_group_size,
        num_selected_experts=self.num_selected_experts,
        batch_prioritized_routing=self.batch_prioritized_routing,
        ignore_padding_tokens=self.ignore_padding_tokens,
        router_kwargs=self.router_kwargs,
        dtype=self.dtype,  # TODO(b/232452238): investigate self.comm_dtype
        name='feed_forward',
    )


@dataclasses.dataclass
class SetupSoftMixtureOfFFN:
  """Helper class with method for configuration of a Soft MoE FFN."""

  d_model: int
  d_ff: int
  use_bias: bool
  dropout_rate: float
  dtype: jax.typing.DTypeLike
  ffn_dot_general: typing.DotGeneral
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  router_kernel_shardings: typing.ShardingAxes
  tokens_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str
  approximate_gelu: bool
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  expert_capacity: int

  def setup_feed_forward(self):
    self.feed_forward = SoftMixtureOfFeedforward(
        d_ff=self.d_ff,
        d_model=self.d_model,
        use_bias=self.use_bias,
        dot_general=self.ffn_dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        dropout_rate=self.dropout_rate,
        inner_kernel_shardings=self.ffn_inner_kernel_shardings,
        outer_kernel_shardings=self.ffn_outer_kernel_shardings,
        intermediate_shardings=self.ffn_intermediate_shardings,
        router_kernel_shardings=self.router_kernel_shardings,
        tokens_shardings=self.tokens_shardings,
        model_axis_size=self.model_axis_size,
        model_axis_name=self.model_axis_name,
        approximate_gelu=self.approximate_gelu,
        num_experts=self.num_experts,
        expert_capacity=self.expert_capacity,
        ignore_padding_tokens=self.ignore_padding_tokens,
        jitter_noise=self.jitter_noise,
        comm_dtype=self.comm_dtype,
        split_params=self.split_params,
        optimize_parallel_comms=self.optimize_parallel_comms,
        router_kwargs=self.router_kwargs,
        dtype=self.dtype,  # TODO(b/232452238): investigate self.comm_dtype
        name='feed_forward',
    )


class SparseMoeTransformerEncoderLayer(SetupSparseMixtureOfFFN,
                                       TransformerEncoderLayer):
  """A single-layer Transformer encoder with Sparse-MoE-FFN."""


class SoftMoeTransformerEncoderLayer(SetupSoftMixtureOfFFN,
                                     TransformerEncoderLayer):
  """A single-layer Transformer encoder with Soft-MoE-FFN."""


class SparseMoeTransformerDecoderLayer(SetupSparseMixtureOfFFN,
                                       TransformerDecoderLayer):
  """A single-layer Transformer decoder with Sparse-MoE-FFN."""


class SoftMoeTransformerDecoderLayer(SetupSoftMixtureOfFFN,
                                     TransformerDecoderLayer):
  """A single-layer Transformer decoder with Soft-MoE-FFN."""


@dataclasses.dataclass
class SetupMoeLayerStack:
  """Helper class for construction of the stack of an MoE-MHA-FFN layer."""

  d_model: int
  d_ff: int
  num_heads: int
  num_layers: int
  use_bias: bool
  dropout_rate: float
  remat: str
  scanned_layers: bool
  scan_axis: int
  dtype: jax.typing.DTypeLike
  qk_layernorm: bool
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  approximate_gelu: bool
  # DotGenerals
  mha_qkv_dot_general: typing.DotGeneral
  mha_out_dot_general: typing.DotGeneral
  mha_einsum_dot_general: typing.DotGeneral
  ffn_dot_general: typing.DotGeneral
  # Sharding annotations
  layernorm_shardings: typing.ShardingAxes
  mha_qkv_kernel_shardings: typing.ShardingAxes
  mha_out_kernel_shardings: typing.ShardingAxes
  mha_activation_shardings: typing.ShardingAxes
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  # Common MoE args
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  # MoE sharding annotations
  tokens_shardings: typing.ShardingAxes
  router_kernel_shardings: typing.ShardingAxes
  routed_ffn_intermediate_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str
  scan_sharding_axis: str | None
  # MoE stack args
  num_moe_layers: int
  moe_layers_distribution: str

  def setup_layer_stack(
      self,
      moe_layer_module,
      moe_layer_config,
      dense_layer_module,
      static_argnums,
      scan_in_axes):
    """Constructs the stack of a given MoE-MHA-FFN layer."""
    if self.moe_layers_distribution not in ('last', 'uniform'):
      raise ValueError('Wrong value for moe_layers_distribution = '
                       f'{self.moe_layers_distribution!r}. It must be '
                       'either `last` or `uniform`.')

    if (not isinstance(self.num_moe_layers, int) or
        self.num_moe_layers < 1):
      raise ValueError('Wrong value for num_moe_layers = '
                       f'{self.num_moe_layers!r}. It must be equal to or '
                       'larger than 1.')

    if self.num_moe_layers > self.num_layers:
      raise ValueError(f'You have num_layers = {self.num_layers}, you cannot '
                       f'have num_moe_layers = {self.num_moe_layers} > '
                       f'{self.num_layers}.')

    if (self.moe_layers_distribution == 'uniform' and
        self.num_layers % self.num_moe_layers != 0):
      raise ValueError('When using moe_layers_distribution = `uniform`, '
                       'num_moe_layers must be a divisor of num_layers, but '
                       f'you have num_layers = {self.num_layers} and '
                       f'num_moe_layers = {self.num_moe_layers}.')

    if (self.scanned_layers and self.moe_layers_distribution == 'uniform' and
        self.num_moe_layers != self.num_layers):
      raise ValueError('Scanned layers cannot be used with alternated moe '
                       'and dense layers, use scanned_layers=False.')

    DenseRematLayer = transforms.remat(  # pylint: disable=invalid-name
        module=dense_layer_module,
        level=self.remat,
        scanned=self.scanned_layers,
        static_argnums=static_argnums,
    )
    MoeRematLayer = transforms.remat(  # pylint: disable=invalid-name
        module=moe_layer_module,
        level=self.remat,
        scanned=self.scanned_layers,
        static_argnums=static_argnums,
    )

    dense_layer_config = {
        'd_model': self.d_model,
        'd_ff': self.d_ff,
        'num_heads': self.num_heads,
        'use_bias': self.use_bias,
        'dropout_rate': self.dropout_rate,
        'dtype': self.dtype,
        'qk_layernorm': self.qk_layernorm,
        'layernorm_shardings': self.layernorm_shardings,
        'mha_qkv_kernel_shardings': self.mha_qkv_kernel_shardings,
        'mha_out_kernel_shardings': self.mha_out_kernel_shardings,
        'mha_activation_shardings': self.mha_activation_shardings,
        'ffn_inner_kernel_shardings': self.ffn_inner_kernel_shardings,
        'ffn_outer_kernel_shardings': self.ffn_outer_kernel_shardings,
        'ffn_intermediate_shardings': self.ffn_intermediate_shardings,
        'mha_qkv_dot_general': self.mha_qkv_dot_general,
        'mha_out_dot_general': self.mha_out_dot_general,
        'mha_einsum_dot_general': self.mha_einsum_dot_general,
        'ffn_dot_general': self.ffn_dot_general,
        'precision': self.precision,
        'lora_rank': self.lora_rank,
        'lora_scale': self.lora_scale,
        'approximate_gelu': self.approximate_gelu,
    }
    moe_layer_config.update(dense_layer_config)
    moe_layer_config['ffn_intermediate_shardings'] = (
        self.routed_ffn_intermediate_shardings
    )

    num_dense_layers = self.num_layers - self.num_moe_layers
    sequential_stack = []

    if self.scanned_layers:
      # Note: If we enter this block, then distribution = 'last' and/or we have
      # num_layers == num_moe_layers.
      initializing = self.is_mutable_collection(PARAMS)  # type: ignore
      scan_axis = self.scan_axis if initializing else ScanIn(self.scan_axis)
      if num_dense_layers > 0:
        sequential_stack.append(
            transforms.scan(
                module=DenseRematLayer,
                length=num_dense_layers,
                scan_axis=scan_axis,
                in_axes=scan_in_axes,
                out_axes=0,
                rng_keys=('dropout',),
                sharding_axis=self.scan_sharding_axis,
            )(name='layer_scan_dense', **dense_layer_config))
      sequential_stack.append(
          transforms.scan(
              module=MoeRematLayer,
              length=self.num_moe_layers,
              scan_axis=scan_axis,
              in_axes=scan_in_axes,
              out_axes=0,
              rng_keys=('dropout', 'jitter'),
              sharding_axis=self.scan_sharding_axis,
          )(name='layer_scan_moe', **moe_layer_config))

    else:
      if self.moe_layers_distribution == 'uniform':
        # MoE layers are distributed uniformly within the stack of layers.
        every_n = self.num_layers // self.num_moe_layers
        is_moe = tuple(n % every_n == 0 for n in range(self.num_layers))
        is_moe = tuple(reversed(is_moe))
      elif self.moe_layers_distribution == 'last':
        # MoE layers are located at the end of the stack of layers.
        is_moe = tuple(n >= num_dense_layers for n in range(self.num_layers))
      else:
        raise ValueError('Unknown moe_layers_distribution = '
                         f'{self.moe_layers_distribution}')

      for n in range(self.num_layers):
        if is_moe[n]:
          sequential_stack.append(
              MoeRematLayer(
                  name='layer_{}'.format(n), **moe_layer_config))
        else:
          sequential_stack.append(
              DenseRematLayer(name='layer_{}'.format(n), **dense_layer_config))

    if not sequential_stack:
      raise ValueError(
          'An empty stack is configured. Something went really wrong!')

    self.layer_stack = stacking.SequentialStackCall(sequential_stack)
    self.final_layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='final_layer_norm'
    )
    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))


class SparseMoeTransformerEncoder(SetupMoeLayerStack, TransformerEncoder):
  """Transformer Encoder with Sparse-MoE."""

  max_group_size: int
  capacity_factor: float
  min_expert_capacity: int
  router_type: str    # TODO(b/267474477): Merge it into `router_kwargs`.
  router_bias: bool
  strict_group_size: bool
  num_selected_experts: int
  batch_prioritized_routing: bool

  def setup(self):
    moe_layer_config = {
        'num_experts': self.num_experts,
        'max_group_size': self.max_group_size,
        'capacity_factor': self.capacity_factor,
        'min_expert_capacity': self.min_expert_capacity,
        'router_type': self.router_type,
        'router_bias': self.router_bias,
        'jitter_noise': self.jitter_noise,
        'comm_dtype': self.comm_dtype,
        'split_params': self.split_params,
        'optimize_parallel_comms': self.optimize_parallel_comms,
        'strict_group_size': self.strict_group_size,
        'num_selected_experts': self.num_selected_experts,
        'batch_prioritized_routing': self.batch_prioritized_routing,
        'ignore_padding_tokens': self.ignore_padding_tokens,
        'router_kwargs': self.router_kwargs,
        'router_kernel_shardings': self.router_kernel_shardings,
        'tokens_shardings': self.tokens_shardings,
        'model_axis_size': self.model_axis_size,
        'model_axis_name': self.model_axis_name,
    }
    self.setup_layer_stack(
        moe_layer_module=SparseMoeTransformerEncoderLayer,
        moe_layer_config=moe_layer_config,
        dense_layer_module=TransformerEncoderLayer,
        static_argnums=(1, 4),
        scan_in_axes=(nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast),
    )


class SoftMoeTransformerEncoder(SetupMoeLayerStack, TransformerEncoder):
  """Transformer Encoder with Soft-MoE."""

  expert_capacity: int

  def setup(self):
    moe_layer_config = {
        'num_experts': self.num_experts,
        'expert_capacity': self.expert_capacity,
        'ignore_padding_tokens': self.ignore_padding_tokens,
        'jitter_noise': self.jitter_noise,
        'comm_dtype': self.comm_dtype,
        'split_params': self.split_params,
        'optimize_parallel_comms': self.optimize_parallel_comms,
        'router_kwargs': self.router_kwargs,
        'router_kernel_shardings': self.router_kernel_shardings,
        'tokens_shardings': self.tokens_shardings,
        'model_axis_size': self.model_axis_size,
        'model_axis_name': self.model_axis_name,
    }
    self.setup_layer_stack(
        moe_layer_module=SoftMoeTransformerEncoderLayer,
        moe_layer_config=moe_layer_config,
        dense_layer_module=TransformerEncoderLayer,
        static_argnums=(1, 4),
        scan_in_axes=(nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast),
    )


class SparseMoeTransformerDecoder(SetupMoeLayerStack, TransformerDecoder):
  """Transformer Decoder with Sparse-MoE."""

  max_group_size: int
  capacity_factor: float
  min_expert_capacity: int
  router_type: str    # TODO(b/267474477): Merge it into `router_kwargs`.
  router_bias: bool
  strict_group_size: bool
  num_selected_experts: int
  batch_prioritized_routing: bool

  def setup(self):
    moe_layer_config = {
        'num_experts': self.num_experts,
        'max_group_size': self.max_group_size,
        'capacity_factor': self.capacity_factor,
        'min_expert_capacity': self.min_expert_capacity,
        'router_type': self.router_type,
        'router_bias': self.router_bias,
        'jitter_noise': self.jitter_noise,
        'comm_dtype': self.comm_dtype,
        'split_params': self.split_params,
        'optimize_parallel_comms': self.optimize_parallel_comms,
        'strict_group_size': self.strict_group_size,
        'num_selected_experts': self.num_selected_experts,
        'batch_prioritized_routing': self.batch_prioritized_routing,
        'ignore_padding_tokens': self.ignore_padding_tokens,
        'router_kwargs': self.router_kwargs,
        'router_kernel_shardings': self.router_kernel_shardings,
        'tokens_shardings': self.tokens_shardings,
        'model_axis_size': self.model_axis_size,
        'model_axis_name': self.model_axis_name,
    }
    self.setup_layer_stack(
        moe_layer_module=SparseMoeTransformerDecoderLayer,
        moe_layer_config=moe_layer_config,
        dense_layer_module=TransformerDecoderLayer,
        static_argnums=(2, 6, 7, 8),
        scan_in_axes=(nn.broadcast, nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast),
    )


class SoftMoeTransformerDecoder(SetupMoeLayerStack, TransformerDecoder):
  """Transformer Decoder with Soft-MoE."""

  expert_capacity: int

  def setup(self):
    moe_layer_config = {
        'num_experts': self.num_experts,
        'expert_capacity': self.expert_capacity,
        'ignore_padding_tokens': self.ignore_padding_tokens,
        'jitter_noise': self.jitter_noise,
        'comm_dtype': self.comm_dtype,
        'split_params': self.split_params,
        'optimize_parallel_comms': self.optimize_parallel_comms,
        'router_kwargs': self.router_kwargs,
        'router_kernel_shardings': self.router_kernel_shardings,
        'tokens_shardings': self.tokens_shardings,
        'model_axis_size': self.model_axis_size,
        'model_axis_name': self.model_axis_name,
    }
    self.setup_layer_stack(
        moe_layer_module=SoftMoeTransformerDecoderLayer,
        moe_layer_config=moe_layer_config,
        dense_layer_module=TransformerDecoderLayer,
        static_argnums=(2, 6, 7, 8),
        scan_in_axes=(nn.broadcast, nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast, nn.broadcast,
                      nn.broadcast, nn.broadcast),
    )
