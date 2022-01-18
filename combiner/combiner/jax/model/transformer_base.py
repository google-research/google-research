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

# pylint: skip-file
from typing import Callable, Any, Optional

from jax import lax
import jax.numpy as jnp
import numpy as np
import jax

from flax import linen as nn
from flax import struct
from functools import partial

from flax.linen.attention import dot_product_attention, PRNGKey, Shape, Dtype, Array
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.initializers import zeros
from flax.linen.module import merge_param
import sys


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None
  architecture: str = 'vanilla'
  max_seg_len: int = 32
  share_param: bool = True
  use_layernorm: bool = True
  seq_summary: str = 'pool-max'
  window_size: int = 3


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

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class MultiDimMultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  num_repeat: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False

  def setup(self):
    self.to_tile_shape = jnp.array([1, 1, self.num_repeat, 1, 1])

  @nn.compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               mask = None,
               deterministic = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    assert inputs_q.ndim == 3 and inputs_kv.ndim == 3
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = merge_param('deterministic', self.deterministic, deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = partial(DenseGeneral,
                    axis=-1,
                    features=(self.num_heads, head_dim),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    use_bias=self.use_bias,
                    precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query', features=(self.num_repeat, self.num_heads, head_dim))(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))
    key = jnp.expand_dims(key, -3)
    value = jnp.expand_dims(value, -3)
    key = jnp.tile(key, self.to_tile_shape)
    value = jnp.tile(value, self.to_tile_shape)
    query = jnp.swapaxes(query, -3, -4)
    key = jnp.swapaxes(key, -3, -4)
    value = jnp.swapaxes(value, -3, -4)
    '''
    query shape: (batch_size, num_repeat, query_seq_len, num_head, emb_dim)
    kv shape: (batch_size, num_repeat, kv_seq_len, num_head, emb_dim)
    '''

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=self.dtype,
                       precision=self.precision,
                       name='out')(x)

    out = jnp.swapaxes(out, -2, -3)
    '''
    swap out from (batch_size, num_repeat, seq_len, emb_dim) to (batch_size, seq_len, num_repeat, emb_dim)
    '''
    return out


class MultiDimSelfAttention(MultiDimMultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @nn.compact
  def __call__(self, inputs_q, mask = None,
               deterministic = None):
    return super().__call__(inputs_q, inputs_q, mask, deterministic=deterministic)


class MultiDimEncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig
  num_repeat: int
  is_self_att: bool = True
  out_features: Optional[int] = None

  def setup(self):
    self.to_tile_shape = [1, 1, self.num_repeat, 1]

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None,
               encoder_decoder_mask=None,
               inputs_kv=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert inputs.ndim == 3
    input_shape = list(inputs.shape)
    input_shape[-1] *= self.num_repeat

    if cfg.use_layernorm:
      x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    else:
      x = inputs
    if self.is_self_att:
      x = MultiDimSelfAttention(
          num_heads=cfg.num_heads,
          num_repeat=self.num_repeat,
          out_features=self.out_features,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          decode=cfg.decode)(x, decoder_mask)
    else:
      if cfg.use_layernorm:
        x_kv = nn.LayerNorm(dtype=cfg.dtype)(inputs_kv)
      else:
        x_kv = inputs_kv
      x = MultiDimMultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          num_repeat=self.num_repeat,
          out_features=self.out_features,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          decode=cfg.decode)(x, x_kv, mask=decoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    inputs = jnp.expand_dims(inputs, -2)
    inputs = jnp.tile(inputs, self.to_tile_shape)
    x = x + inputs

    # MLP block.
    if cfg.use_layernorm:
      z = nn.LayerNorm(dtype=cfg.dtype)(x)
    else:
      z = x
    z = MlpBlock(config=cfg)(z)

    return jnp.reshape(x + z, input_shape)


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig
  is_self_att: bool = True
  out_features: Optional[int] = None
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None,
               encoder_decoder_mask=None,
               inputs_kv=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert inputs.ndim == 3
    # assert decoder_mask.ndim == 4
    if cfg.use_layernorm:
      x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    else:
      x = inputs
    if self.is_self_att:
      x = nn.SelfAttention(
          num_heads=cfg.num_heads,
          out_features=self.out_features,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          attention_fn=self.attention_fn,
          decode=cfg.decode)(x, decoder_mask)
    else:
      if cfg.use_layernorm:
        x_kv = nn.LayerNorm(dtype=cfg.dtype)(inputs_kv)
      else:
        x_kv = inputs_kv
      x = nn.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          out_features=self.out_features,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          attention_fn=self.attention_fn,
          decode=cfg.decode)(x, x_kv, mask=decoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    if cfg.use_layernorm:
      z = nn.LayerNorm(dtype=cfg.dtype)(x)
    else:
      z = x
    z = MlpBlock(config=cfg)(z)

    return x + z

if __name__ == '__main__':
  key = jax.random.PRNGKey(1)

  cfg = TransformerConfig(
    vocab_size=1,
    output_vocab_size=1,
    num_heads=2,
    max_len=8,
    max_seg_len=4,
    dropout_rate=0,
    attention_dropout_rate=0,
    seq_summary='pool-max',
  )

  num_repeat = int(sys.argv[1])
  model = EncoderDecoder1DBlock(cfg,
                               )
  print(cfg.qkv_dim)

  key, model_key = jax.random.split(key)
  a = jax.random.uniform(key, (2, 8, 3))
  params = model.init(model_key, a)
  out = model.apply(params, a)
  print(out.shape)

