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

"""Flax implementation of the transformer encoder.
"""
from functools import partial

from . import util

import flax
from flax.deprecated import nn
import jax
import jax.numpy as jnp
import jax.random


class TransformerEncoderLayer(nn.Module):

  def apply(self,
            inputs,
            mask,
            activation_fn=flax.deprecated.nn.relu,
            num_heads=8,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies one transformer encoder layer.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, data_dim] tensor of outputs.
    """
    value_dim = inputs.shape[-1]

    attn_outs = flax.deprecated.nn.SelfAttention(
        inputs_q=inputs,
        num_heads=num_heads,
        qkv_features=value_dim,
        padding_mask=mask,
        kernel_init=weight_init)

    attn_outs = inputs + attn_outs

    out1 = activation_fn(
        flax.deprecated.nn.Dense(
            attn_outs, features=value_dim, kernel_init=weight_init))
    out2 = flax.deprecated.nn.Dense(
        out1, features=value_dim, kernel_init=weight_init)
    return attn_outs + out2


class TransformerEncoderStack(nn.Module):

  def apply(self,
            inputs,
            mask,
            num_encoders=6,
            num_heads=8,
            value_dim=128,
            activation_fn=flax.deprecated.nn.relu,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer encoder layers.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      num_encoders: The number of encoder layers in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, value_dim] tensor of outputs.
    """
    inputs = flax.deprecated.nn.Dense(
        inputs, features=value_dim, kernel_init=weight_init)
    for _ in range(num_encoders):
      inputs = TransformerEncoderLayer(inputs,
                                       mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       weight_init=weight_init)
    return inputs


class TransformerDecoderLayer(nn.Module):

  def apply(self,
            target_inputs,
            target_mask,
            encoder_inputs,
            encoder_mask,
            activation_fn=flax.deprecated.nn.relu,
            num_heads=8,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies one transformer decoder layer.

    Args:
      target_inputs: The inputs derived from the transformer outputs, a
        [batch_size, max_k, value_dim] tensor.
      target_mask: The mask for the targets indicating which elements are
        padding. A tensor of shape [batch_size, max_k].
      encoder_inputs: The inputs derived from the transformer inputs, a
        [batch_size, max_num_data_points, value_dim] tensor.
      encoder_mask: The mask for the inputs indicating which elements are
        padding. A tensor of shape [batch_size, num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_k, value_dim] tensor of outputs.
    """
    value_dim = target_inputs.shape[-1]
    target_inputs_attn = flax.deprecated.nn.SelfAttention(
        inputs_q=target_inputs,
        num_heads=num_heads,
        causal_mask=True,
        padding_mask=target_mask,
        qkv_features=value_dim,
        kernel_init=weight_init)

    target_inputs_out = target_inputs_attn + target_inputs

    enc_dec_attn_out = flax.deprecated.nn.MultiHeadDotProductAttention(
        inputs_q=target_inputs_attn,
        inputs_kv=encoder_inputs,
        padding_mask=target_mask,
        key_padding_mask=encoder_mask,
        num_heads=num_heads,
        qkv_features=value_dim,
        kernel_init=weight_init)

    enc_dec_attn_out += target_inputs_out

    out_layer1 = activation_fn(
        flax.deprecated.nn.Dense(
            enc_dec_attn_out, features=value_dim, kernel_init=weight_init))
    out_layer2 = flax.deprecated.nn.Dense(
        out_layer1, features=value_dim, kernel_init=weight_init)

    return out_layer2 + enc_dec_attn_out


class TransformerDecoderStack(nn.Module):

  def apply(self,
            target_inputs,
            target_mask,
            encoder_inputs,
            encoder_mask,
            activation_fn=flax.deprecated.nn.relu,
            num_decoders=6,
            num_heads=8,
            value_dim=128,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer decoder layers.

    Args:
      target_inputs: The inputs derived from the transformer outputs, a
        [batch_size, max_k, data_dim] tensor.
      target_mask: The mask for the targets indicating which elements are
        padding. A tensor of shape [batch_size, max_k].
      encoder_inputs: The inputs derived from the transformer inputs, a
        [batch_size, max_num_data_points, value_dim] tensor.
      encoder_mask: The mask for the inputs indicating which elements are
        padding. A tensor of shape [batch_size, num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_decoders: The number of decoders in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_k, value_dim] tensor of outputs.
    """
    inputs = flax.deprecated.nn.Dense(
        target_inputs, features=value_dim, kernel_init=weight_init)

    for _ in range(num_decoders):
      inputs = TransformerDecoderLayer(inputs,
                                       target_mask,
                                       encoder_inputs,
                                       encoder_mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       weight_init=weight_init)
    return inputs


class EncoderDecoderTransformer(nn.Module):

  def apply(self,
            inputs,
            input_lengths,
            target_lengths,
            targets=None,
            target_dim=32,
            max_input_length=100,
            max_target_length=100,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            activation_fn=flax.deprecated.nn.relu,
            weight_init=jax.nn.initializers.xavier_uniform()):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data, [batch_size, max_num_data_points, data_dim].
      input_lengths: A [batch_size] vector containing the number of samples
        in each batch element.
      target_lengths: A [batch_size] vector containing the length of each
        target sequence.
      targets: The outputs to be produced by the transformer. Supplied only
        during training. If None, then the transformer's own outputs are fed
        back in.
      target_dim: The length of each output vector.
      max_input_length: An int at least as large as the largest element of
        num_data_points, used for determining output shapes.
      max_target_length: An int at least as large as the largest element of
        target_lengths, used for determining output shapes.
      num_heads: The number of heads for the self attention.
      num_encoders: The number of transformer encoder layers.
      num_decoders: The number of transformer decoder layers.
      qkv_dim: The dimension of the query/key/value.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: The transformer output, a tensor of shape
        [batch_size, max_target_length, target_dim].
    """
    input_mask = util.make_mask(input_lengths, max_input_length)
    target_mask = util.make_mask(target_lengths, max_target_length)

    encoder_hs = TransformerEncoderStack(inputs,
                                         input_mask,
                                         num_encoders=num_encoders,
                                         num_heads=num_heads,
                                         value_dim=qkv_dim,
                                         weight_init=weight_init)
    batch_size = inputs.shape[0]
    if targets is not None:
      sampling = False
    else:
      sampling = True
      targets = jnp.zeros([batch_size, max_target_length, target_dim])

    target_inputs = jnp.zeros([batch_size, max_target_length, target_dim])
    target_inputs = target_inputs.at[:, 0, 0].set(target_lengths)

    def decode_body(target_inputs, i):
      # decoder_out is [batch_size, max_target_length, value_dim]
      decoder_out = TransformerDecoderStack(
          target_inputs,
          target_mask,
          encoder_hs,
          input_mask,
          activation_fn=flax.deprecated.nn.relu,
          num_decoders=num_decoders,
          num_heads=num_heads,
          value_dim=qkv_dim,
          weight_init=weight_init)
      # out is [batch_size, qkv_dim]
      out = activation_fn(
          flax.deprecated.nn.Dense(
              decoder_out[:, i], features=qkv_dim, kernel_init=weight_init))
      # dense layer to arrive at [batch_size, target_dim]
      out = flax.deprecated.nn.Dense(
          out, features=target_dim, kernel_init=weight_init)

      if sampling:
        target_inputs = target_inputs.at[:, i + 1].set(out)
      else:
        target_inputs = target_inputs.at[:, i + 1].set(targets[:, i])

      return target_inputs, out

    if self.is_initializing():
      decode_body(target_inputs, 0)

    _, outs = jax.lax.scan(
        decode_body,
        target_inputs,
        jnp.arange(max_target_length),
    )
    # outs is currently [max_target_length, batch_size, target_dim],
    # transpose to put the batch dimension first.
    return jnp.transpose(outs, axes=(1, 0, 2))

  @classmethod
  def wasserstein_distance_loss(
      cls,
      params,
      inputs,
      input_lengths,
      targets,
      target_lengths,
      key):
    batch_size = inputs.shape[0]
    max_target_length = targets.shape[1]
    # [batch_size, max_target_length, target_dim]
    predicted = cls.call(params,
                         inputs,
                         input_lengths,
                         target_lengths,
                         targets=targets)
    ranges = jnp.tile(
        jnp.arange(max_target_length)[jnp.newaxis, :],
        [batch_size, 1])
    weights = jnp.where(ranges < target_lengths[:, jnp.newaxis],
                        jnp.zeros([batch_size, max_target_length]),
                        jnp.full([batch_size, max_target_length], -jnp.inf))

    wdists, _ = jax.vmap(util.atomic_sinkhorn)(
        predicted, weights, targets, weights,
        jax.random.split(key, num=batch_size)
    )
    return wdists
