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

"""Transformer-based language model.

Reusing decoder only model from examples/wmt.
"""


from typing import Any, Callable, Optional

from absl import logging
from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np
from autoregressive_diffusion.model.architecture_components import time_embedding
from autoregressive_diffusion.utils import util_fns


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  max_time: float
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  context_length: int = 0
  is_causal: bool = True
  kernel_init: Callable[Ellipsis, Any] = nn.initializers.kaiming_uniform()
  bias_init: Callable[Ellipsis, Any] = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable[Ellipsis, Any]] = None
  condition_time: bool = False


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  return shifted


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
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
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               permutations):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: Input data.
      permutations: Permutations for data generation order.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    total_length = cfg.context_length + cfg.max_len

    if permutations is None:
      pos_emb_shape = (1, total_length, inputs.shape[-1])
    else:
      assert inputs.shape[-1] % 2 == 0
      # Only use half of the number of dims, because the positional embeddings
      # will be concatenated with shifted positional embeddings.
      pos_emb_shape = (1, total_length, inputs.shape[-1] // 2)

    # Use a fixed (non-learned) sinusoidal position embedding.
    pos_embedding = sinusoidal_init(max_len=total_length)(
        None, pos_emb_shape, None)

    # Here the pos_embedding for the inputs (and not the context) need to be
    # permuted. In addition, here the positional embeddings also need
    # to be shifted. Before there wasn't really an issue when positions were
    # not shifted: They were very close in representation to their
    # neighbours. Now however, the permutation really randomly distributes the
    # positions, and they need to be matched with the inputs/targets correctly.
    # For that reason we concatenate shifted (for inputs) and non-shifted (for
    # targets) positional embeddings.
    if permutations is not None:
      # Normally, pos_embedding has an empty first axis. But since for each
      # example the model may take a different permutation, we now have to
      # repeat the pos_embeddings over the batch dimension.
      pos_embedding = pos_embedding.repeat(inputs.shape[0], axis=0)

      if cfg.context_length == 0:
        pos_embedding = util_fns.batch_permute(pos_embedding, permutations)
        pos_embedding_shifted = shift_inputs(pos_embedding)
        pos_embedding = jnp.concatenate(
            [pos_embedding, pos_embedding_shifted], axis=2)
      elif cfg.context_length > 0:
        # If we utilize a context, only the embeddings for the data need to be
        # permuted.
        context_emb = pos_embedding[:, :cfg.context_length]
        data_emb = pos_embedding[:, cfg.context_length:]

        data_emb = util_fns.batch_permute(data_emb, permutations)
        shifted_data_emb = shift_inputs(data_emb)
        data_emb = jnp.concatenate([data_emb, shifted_data_emb], axis=2)
        context_emb = context_emb.repeat(2, axis=2)
        pos_embedding = jnp.concatenate((context_emb, data_emb), axis=1)
      else:
        raise ValueError

    pe = pos_embedding

    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, temb, deterministic):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)

    # Add in the time embedding if applicable.
    if temb is not None:
      x += nn.Dense(cfg.mlp_dim,
                    dtype=cfg.dtype,
                    kernel_init=cfg.kernel_init,
                    bias_init=cfg.bias_init)(temb)[:, None, :]
    x = nn.gelu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=deterministic)
    return output


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               temb,
               deterministic,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: Input data for decoder.
      temb: Time embedding representation.
      deterministic: Should be deterministic in dropout?
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: Encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=deterministic,
        decode=False)(x, decoder_mask)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(x)
    z = MlpBlock(config=cfg)(z, temb, deterministic)

    return x + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               temb,
               train,
               context,
               permutations):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Input data.
      temb: The time embedding.
      train: Is the model training?
      context: A context to condition on.
      permutations: A batch of permutations that specifies generation order.

    Returns:
      Output of a transformer decoder.
    """
    cfg = self.config
    assert inputs.ndim == 2  # (batch, len)
    deterministic = not train

    # Permutations give the permutation order, for XLNet style training only. It
    # is important that permutations are applied _before shifting_. For this
    # reason, we also have to deal with the positional embeddings seperately
    # at a later point.
    if permutations is not None:
      assert cfg.is_causal
      assert permutations.shape == inputs.shape

      # Use the permutations to act on the inputs.
      inputs = util_fns.batch_permute(inputs, permutations)

    # Target Embedding
    embedding_layer = nn.Embed(
        num_embeddings=cfg.output_vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    # Concatenate context if available.
    if context is not None:
      assert cfg.context_length == context.shape[
          1], f'{cfg.context_length} != {context.shape[1]} for {context.shape}'
      inputs = jnp.concatenate([context, inputs], axis=1)

    y = inputs.astype('int32')

    if cfg.is_causal:
      logging.info('Using causal Transformer')
      decoder_mask = nn.make_causal_mask(inputs, dtype=cfg.dtype)
    else:
      logging.info('Using fully connected (non-causal) Transformer')
      decoder_mask = None

    if cfg.is_causal:
      y = shift_inputs(y)
    y = embedding_layer(y)

    y = AddPositionEmbs(
        config=cfg, name='add_posemb')(y, permutations)

    y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=deterministic)

    y = y.astype(cfg.dtype)

    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg, name=f'encoderdecoderblock_{lyr}')(
              y, temb, deterministic,
              decoder_mask=decoder_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    logits = nn.Dense(
        cfg.output_vocab_size,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        name='logitdense')(y)

    if context is not None:
      # Take only predictions for inputs, not context.
      logits = logits[:, cfg.context_length:]

    if permutations is not None:
      assert cfg.is_causal
      # Apply the inverse permutation to the logits.
      inv_permutations = util_fns.compute_batch_inverse_permute(permutations)
      logits = util_fns.batch_permute(logits, inv_permutations)

    return logits


class TransformerLM(nn.Module):
  """Transformer pure decoder stack for language modelling.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs, t, mask, train, context=None, permutations=None):
    """Applies TransformerLM on the inputs.

    Args:
      inputs: The inputs to the model.
      t: The timestep that the model has to predict.
      mask: The mask of which variables are already filled in.
      train: Is the model training?
      context: Used to give sequence before the current sequence.
      permutations: Permutations to permute the sequence with, used for
        XLNet style training.

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.config

    # Should contain a single channel axis, for compatibility reasons.
    assert inputs.shape[2] == 1, f' for {inputs.shape}'
    inputs = jnp.squeeze(inputs, axis=2)

    # Mask is used implicitly via the additional class that the network can
    # take as input. No need to use the explicit binary mask. This is because
    # the inputs are _nominal_ categorical variables.
    del mask

    if permutations is not None:
      # We first invert the permutations, this is important because the inverse
      # permutation action corresponds to a generation order. This is best seen
      # via an example:
      # Suppose we want to generate in the order [1, 2, 0],
      # then we want to permute it to [0, 1, 2]. To do this permutation we
      # want to find the inverse permutation of [1, 2, 0], which is
      # [2, 0, 1]. So we want to permute using [2, 0, 1] as an index tensor,
      # which then gives a generation order of [1, 2, 0].
      # In short this corresponds to `orders == inv(permutations)`, and so
      # also inv(orders) = permutations.
      permutations = util_fns.compute_batch_inverse_permute(permutations)

    if cfg.max_time > 0 and cfg.condition_time:
      temb = time_embedding.TimeEmbedding(cfg.emb_dim, cfg.max_time)(t)
    else:
      logging.info('Not using time embedding.')
      temb = None

    logging.info('Compiling Transformer')

    logits = Decoder(config=cfg, name='decoder')(
        inputs, temb, train, context, permutations)

    return logits.astype(self.config.dtype)
