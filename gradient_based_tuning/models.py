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
"""Transformer-based machine translation model."""
from typing import Mapping, Union
# pylint: disable=g-bad-import-order, g-multiple-import
from flax import optim
from flax.deprecated import nn
from flax.deprecated.nn.linear import DenseGeneral, default_kernel_init
from flax.linen.initializers import zeros
from flax.linen.activation import softmax
from flax.deprecated.nn.stochastic import make_rng
from flax.deprecated.nn.attention import Cache, _make_causal_mask, make_padding_mask
from collections.abc import Iterable  # pylint: disable=g-importing-member
import jax
from absl import logging
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


def create_model(key, input_shape, target_shape, model_kwargs):
  """Instantiate transformer model and associated autoregressive cache def."""
  model_def = Transformer.partial(**model_kwargs)
  with nn.attention.Cache().mutate() as cache_def:
    _, initial_params = model_def.init_by_shape(
        key, [(input_shape, jnp.float32), (target_shape, jnp.float32)],
        cache=cache_def)
    model = nn.Model(model_def, initial_params)
  return model, cache_def


def create_lamb_optimizer(model, learning_rate):
  """Given a model, return a LAMB optimizer."""
  optimizer_def = optim.LAMB(learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer


def init_guided_optimizer(
    guided_param_subdict):
  """Initializes an optimizer as specified by a guided-parameters dictionary.

  Args:
    guided_param_subdict: dictionary specifying optimizer_type and
      initialization values for guided parameters

  Returns:
    a Flax Optimizer object, with type and initialization as specified by input
    dictionary
  """
  optimizer_type = guided_param_subdict['optimizer_type']
  if optimizer_type == 'adafactor':
    opt_def = optim.Adafactor(
        learning_rate=0.1, decay_rate=0.8, beta1=None, min_dim_size_to_factor=0)
  elif optimizer_type == 'gradient_descent' or optimizer_type == 'gd':
    opt_def = optim.GradientDescent(learning_rate=0.001)
  elif optimizer_type == 'adam':
    opt_def = optim.Adam(learning_rate=0.001)
  elif optimizer_type == 'lamb':
    opt_def = optim.LAMB(learning_rate=0.001)
  else:
    raise ValueError('Unrecognized optimizer_type: %s' % optimizer_type)
  optimizer = opt_def.create(guided_param_subdict['raw_guided_vars'])
  return optimizer


def init_optimizer_by_type(
    model,
    optimizer_type,
    optimizer_hparams=None,
):
  """Initializes an optimizer of optimizer_type with params of model.

  Args:
    model: params to be optimized
    optimizer_type: kind of optimizer
    optimizer_hparams: hparams for the optimizer

  Returns:
    a Flax Optimizer object, with type optimizer_type and initialization model
  """
  if not optimizer_hparams:
    optimizer_hparams = {}
  logging.info('Using model optimizer_type: %s', optimizer_type)
  if optimizer_type == 'adafactor':
    opt_def = optim.Adafactor(**optimizer_hparams)
  elif optimizer_type in ['sgd', 'gd', 'gradient_descent']:
    opt_def = optim.GradientDescent(**optimizer_hparams)
  elif optimizer_type == 'adam':
    opt_def = optim.Adam(**optimizer_hparams)
  elif optimizer_type == 'mom':
    opt_def = optim.Momentum(**optimizer_hparams)
  elif optimizer_type == 'adagrad':
    opt_def = optim.Adagrad(**optimizer_hparams)
  elif optimizer_type == 'lamb':
    opt_def = optim.LAMB(**optimizer_hparams)
  else:
    raise ValueError('Unrecognized optimizer_type: %s' % optimizer_type)

  optimizer = opt_def.create(model)
  return optimizer


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2:2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=512,
            posemb_init=None,
            cache=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      max_len: maximum possible length for the input.
      posemb_init: positional embedding initializer, if None, then use a fixed
        (non-learned) sinusoidal embedding table.
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    if posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=max_len)(None, pos_emb_shape,
                                                       None)
    else:
      pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(
        inputs,
        mlp_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    x = nn.relu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x,
        actual_out_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer decoder layer."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      num_heads: number of heads.
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      deterministic: bool, deterministic or not (to apply dropout).

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs, dtype=dtype)
    x = SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)

    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer."""

  def apply(self,
            targets,
            encoded,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            targets_segmentation=None,
            padding_mask=None,
            key_padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax attention cache for fast decoding.

    Returns:
      output after transformer encoder-decoder block.
    """

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(targets, dtype=dtype)
    x = SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        padding_mask=padding_mask,
        segmentation=targets_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)

    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = SelfAttention(
        y,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=encoded,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=targets_segmentation,
        key_segmentation=inputs_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    y = nn.dropout(y, rate=dropout_rate, deterministic=deterministic)

    y = y + x

    # MLP block.
    z = nn.LayerNorm(y, dtype=dtype)
    z = MlpBlock(
        z,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def apply(
      self,
      inputs,
      vocab_size,
      inputs_positions=None,
      inputs_segmentation=None,
      shared_embedding=None,
      use_bfloat16=False,
      emb_dim=512,
      num_heads=8,
      num_layers=6,
      qkv_dim=512,
      mlp_dim=2048,
      max_len=512,
      train=True,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[Ellipsis, None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        max_len=max_len,
        name='posembed_input')
    if train:
      x = nn.dropout(x, rate=dropout_rate, deterministic=False)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(num_layers):
      x = Encoder1DBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}')
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""

  def apply(
      self,
      encoded,
      src_padding_mask,
      targets,
      output_vocab_size,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
      tgt_padding_mask=None,
      shared_embedding=None,
      logits_via_embedding=False,
      shift=True,
      use_bfloat16=False,
      emb_dim=512,
      num_heads=8,
      num_layers=6,
      qkv_dim=512,
      mlp_dim=2048,
      max_len=512,
      train=True,
      cache=None,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
  ):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      output_vocab_size: size of the vocabulary.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      shared_embedding: a shared embedding layer to use.
      logits_via_embedding: bool: whether final logit transform shares embedding
        weights.
      shift: whether to shift or not (for fast decoding).
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      cache: flax attention cache for fast decoding.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.

    Returns:
      output of a transformer decoder.
    """
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[Ellipsis, None]

    # Target Embedding
    if shared_embedding is None:
      output_embed = nn.Embed.partial(
          num_embeddings=output_vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = shared_embedding

    y = targets.astype('int32')
    if shift:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        y,
        inputs_positions=targets_positions,
        max_len=max_len,
        cache=cache,
        name='posembed_output')
    if train:
      y = nn.dropout(y, rate=dropout_rate, deterministic=False)

    if use_bfloat16:
      y = y.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Target-Input Decoder
    for lyr in range(num_layers):
      y = EncoderDecoder1DBlock(
          y,
          encoded,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=tgt_padding_mask,
          key_padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          name=f'encoderdecoderblock_{lyr}')
    y = nn.LayerNorm(y, dtype=dtype, name='encoderdecoder_norm')

    # Decoded Logits
    if logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          y,
          output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')
    return logits


# The following final model is simple but looks verbose due to all the
# repetitive keyword argument plumbing.  It just sticks the Encoder and
# Decoder in series for training, but allows running them separately for
# inference.


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation."""

  def apply(self,
            inputs,
            targets,
            vocab_size=None,
            output_vocab_size=None,
            inputs_positions=None,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            share_embeddings=False,
            logits_via_embedding=False,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      vocab_size: size of the input vocabulary.
      output_vocab_size: size of the output vocabulary. If None, the output
        vocabulary size is assumed to be the same as vocab_size.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      share_embeddings: bool: share embedding layer for inputs and targets.
      logits_via_embedding: bool: whether final logit transform shares embedding
        weights.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      shift: whether to right-shift targets.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output of a transformer decoder.
    """
    src_padding_mask = (inputs > 0)[Ellipsis, None]

    if share_embeddings:
      if output_vocab_size is not None:
        assert output_vocab_size == vocab_size, (
            "can't share embedding with different vocab sizes.")
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        name='decoder')
    return logits.astype(jnp.float32) if use_bfloat16 else logits

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while providing a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  @nn.module_method
  def encode(self,
             inputs,
             vocab_size=None,
             output_vocab_size=None,
             inputs_positions=None,
             inputs_segmentation=None,
             targets_positions=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None):
    del (output_vocab_size, shift, targets_positions, targets_segmentation,
         tgt_padding_mask, logits_via_embedding, cache)

    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')

    return encoded

  @nn.module_method
  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             inputs_positions=None,
             vocab_size=None,
             output_vocab_size=None,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None):
    del inputs_positions

    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        shift=shift,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        name='decoder')

    return logits


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # apply dropout
  # GRADSAFE
  if not deterministic:
    # GRADSAFE
    if dropout_rng is None:
      dropout_rng = make_rng()
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = attn_weights.shape[-(2 * len(axis)):]
      dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      keep_prob = jnp.mean(keep)

    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
      keep_prob = jnp.mean(keep)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=zeros,
            bias=True,
            attention_fn=dot_product_attention):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]` or
        None for self-attention, inn which case key/values will be derived from
        inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token w/ False.
      key_padding_mask: boolean specifying key-value tokens that are pad token
        w/ False.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    is_self_attention = inputs_kv is inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(np.array((key.ndim,) + key.shape[-2:], dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        # if not isinstance(cache_entry, _CacheEntry):
        #   raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = np.prod(np.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(
            i=cache_entry.i + one, key=key, value=value)
        cache.store(cache_entry)

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(np.take(key.shape, attention_axis))
        attn_size = np.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if (padding_mask is not None or key_padding_mask is not None) and not cache:
      if key_padding_mask is None:
        if is_self_attention:
          key_padding_mask = padding_mask
        else:
          key_padding_shape = [inputs_kv.shape[dim] for dim in attention_axis]
          key_padding_mask = jnp.full(key_padding_shape, True)
      if padding_mask is None:
        if is_self_attention:
          padding_mask = key_padding_mask
        else:
          padding_shape = [inputs_q.shape[dim] for dim in attention_axis]
          padding_mask = jnp.full(padding_shape, True)

      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        assert is_self_attention
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0,
          jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = attention_fn(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SelfAttention = MultiHeadDotProductAttention.partial(inputs_kv=None)
