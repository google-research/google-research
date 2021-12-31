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

"""Model layers."""

from typing import Any, Callable, Optional, Iterable

from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp

import ml_collections

# Type Stubs
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
MixingLayer = Any

default_kernel_init = nn.initializers.normal(stddev=2e-2)
# TODO(b/181607810): Doubt this will make a difference, but BERT uses zeros for
#  initial bias.
default_bias_init = nn.initializers.normal(stddev=2e-2)

LAYER_NORM_EPSILON = 1e-12


class FeedForwardLayer(nn.Module):
  """Feed-forward layer - position independent, dense, nonlinear transformation.

  Attributes:
    d_ff: Dimension of feed-forward layer.
    dropout_rate: The dropout probability.
    intermediate_activation: (Nonlinear) transform applied in layer.
    kernel_init: Initialization scheme for kernel.
    bias_init: Initialization scheme for bias.
  """
  d_ff: int
  dropout_rate: float = 0.0
  intermediate_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], jnp.ndarray] = default_bias_init

  @nn.compact
  def __call__(self,
               inputs,
               deterministic = False):
    """Applies FeedForwardLayer module.

    Args:
      inputs: Batch of input embeddings, typically of shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
      deterministic: Whether or not to apply dropout to input.

    Returns:
      Transformed inputs with shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
    """
    d_model = inputs.shape[-1]
    x = nn.Dense(
        self.d_ff,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name="intermediate")(
            inputs)
    x = self.intermediate_activation(x)
    x = nn.Dense(d_model, kernel_init=self.kernel_init, name="output")(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    return x


class FourierTransform(nn.Module):
  """Fourier Transform layer.

  Applies 2D Fourier Transform over final two dimensions of inputs - typically
  the sequence and hidden dimensions.

  Attributes:
    fourier_transform: Discrete multi-dimensional Fourier Transform function.
  """
  fourier_transform: Callable[[jnp.ndarray], jnp.ndarray]

  @nn.compact
  def __call__(self,
               inputs,
               padding_mask = None,
               deterministic = False):
    """Applies FourierTransform module.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
      padding_mask: Ignored. Mask only used by self-attention sublayers.
      deterministic: Ignored. Whether or not to apply dropout to input.

    Returns:
      Real part of discrete Fourier Transform of inputs with shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
    """
    del padding_mask  # Only used by self-attention sublayer.
    del deterministic  # Fourier Transform is always deterministic.
    return jax.vmap(self.fourier_transform)(inputs).real


class IdentityTransform(nn.Module):
  """No op layer."""

  @nn.compact
  def __call__(self,
               inputs,
               padding_mask = None,
               deterministic = False):
    """Returns input unchanged.

    Args:
      inputs: Batch of input embeddings.
      padding_mask: Ignored. Mask only used by self-attention sublayers.
      deterministic: Ignored. Whether or not to apply dropout to input.

    Returns:
      Inputs unchanged.
    """
    del padding_mask  # Only used by self-attention sublayer.
    del deterministic  # Identity is always deterministic.
    return inputs


class LinearTransform(nn.Module):
  """Dense, linear transformation layer.

  Applies matrix multiplications over sequence and hidden dimensions.

  Attributes:
    precision: XLA precision for matrix multiplication computation.
    kernel_init: Initializer scheme for (matrix) kernel parameters.
  """
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        jnp.ndarray] = default_kernel_init

  @nn.compact
  def __call__(self,
               inputs,
               padding_mask = None,
               deterministic = False):
    """Applies LinearTransform module.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
      padding_mask: Ignored. Mask only used by self-attention sublayers.
      deterministic: Ignored. Whether or not to apply dropout to input.

    Returns:
      Linearly transformed inputs with shape
        <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
    """
    del padding_mask  # Only used by self-attention sublayer.
    del deterministic  # LinearTransform is always deterministic.

    mat_hidden = self.param("hidden_kernel", self.kernel_init,
                            (inputs.shape[-1], inputs.shape[-1]))
    mat_seq = self.param("seq_kernel", self.kernel_init,
                         (inputs.shape[-2], inputs.shape[-2]))

    return jnp.einsum(
        "bij,jk,ni->bnk",
        inputs,
        mat_hidden,
        mat_seq,
        optimize=True,
        precision=self.precision)


class RandomTransform(nn.Module):
  """Dense, random matrix transformation layer.

  Applies fixed, random matrix multiplications over sequence and model hidden
  dimensions.

  Attributes:
    max_seq_length: The maximum total input sequence length after tokenization.
    d_model: Hidden dimension of model.
    key: Random number generator key.
    kernel_init: Initializer scheme for the fixed, random matrices.
    precision: XLA precision for matrix multiplication computation.
  """
  max_seq_length: int
  d_model: int
  key: PRNGKey
  kernel_init: Callable[[PRNGKey, Shape],
                        jnp.ndarray] = nn.initializers.lecun_normal()
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  def setup(self):
    hidden_key, seq_key = random.split(self.key)
    self.mat_hidden = self.kernel_init(hidden_key, (self.d_model, self.d_model))
    self.mat_seq = self.kernel_init(seq_key,
                                    (self.max_seq_length, self.max_seq_length))

  def __call__(self,
               inputs,
               padding_mask = None,
               deterministic = False):
    """Applies RandomTransform module.

    Args:
      inputs: Batch of input embeddings of shape <float>[BATCH_SIZE,
        max_seq_length, d_model].
      padding_mask: Ignored. Mask only used by self-attention sublayers.
      deterministic: Ignored. Whether or not to apply dropout to input.

    Returns:
      Linearly transformed inputs with shape
        <float>[BATCH_SIZE, max_seq_length, d_model].
    """
    del padding_mask  # Only used by self-attention sublayer.
    del deterministic  # RandomTransform uses fixed, random matrices.

    return jnp.einsum(
        "bij,jk,ni->bnk",
        inputs,
        self.mat_hidden,
        self.mat_seq,
        optimize=True,
        precision=self.precision)


class EncoderBlock(nn.Module):
  """Post-norm encoder model block.

  An EncoderBlock consists of applying the following submodules:
    (1) mixing_sublayer
    (2) Residual connection
    (3) Layer norm
    (4) feed_forward_sublayer
    (5) Residual connection
    (6) Layer norm

  Attributes:
    feed_forward_sublayer: Feed-forward module.
    mixing_sublayer: Mixing module.
  """
  mixing_sublayer: MixingLayer
  feed_forward_sublayer: FeedForwardLayer

  @nn.compact
  def __call__(self,
               inputs,
               padding_mask,
               deterministic = False):
    """Applies EncoderBlock module.

    Args:
      inputs: Batch of input embeddings of shape <float>[BATCH_SIZE,
        MAX_SEQ_LENGTH, HIDDEN_DIM].
      padding_mask: <bool>[BATCH_SIZE, MAX_SEQ_LENGTH] specifying with False
        which tokens in the inputs are pad tokens and should be ignored.
      deterministic: Whether or not to apply dropout.

    Returns:
      Encoder outputs of shape <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
    """
    mixing_output = self.mixing_sublayer(
        inputs, padding_mask, deterministic=deterministic)

    x = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, name="mixing_layer_norm")(
            inputs + mixing_output)

    feed_forward_output = self.feed_forward_sublayer(
        x, deterministic=deterministic)

    return nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, name="output_layer_norm")(
            x + feed_forward_output)


class OutputProjection(nn.Module):
  """A dense projection layer for computing output logits.

  Attributes:
    kernel: Pre-computed kernel parameters of shape <float>[n_out, HIDDEN_DIM].
    n_out: Number of output dimensions. Required if kernel is None.
    bias: Whether or not to apply a bias term.
    kernel_init: Initializer scheme for kernel parameters.
    bias_init: Initializer scheme for bias parameters.
  """
  kernel: Optional[jnp.ndarray] = None
  n_out: Optional[int] = None  # Required if kernel is None.
  bias: bool = True
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], jnp.ndarray] = default_bias_init

  @nn.compact
  def __call__(self, inputs):
    """Applies OutputProjection module.

    Args:
      inputs: Batch of input embeddings of shape <float>[BATCH_SIZE, ...,
        HIDDEN_DIM].

    Returns:
      Output projected logits of shape <float>[BATCH_SIZE, ..., n_out]

    Raises:
      ValueError: If self.kernel and self.n_out are both None.
    """
    if self.kernel is None:
      if self.n_out is None:
        raise ValueError(
            "OutputProjection must be initialized with n_out attribute when "
            "not re-using an existing kernel, such as an embedding matrix.")
      kernel = self.param("output_kernel", self.kernel_init,
                          (self.n_out, inputs.shape[-1]))
    else:
      kernel = self.kernel
    y = jnp.matmul(inputs, jnp.transpose(kernel, (1, 0)))
    if self.bias:
      bias = self.param("output_bias", self.bias_init, (y.shape[-1],))
      y = y + bias
    return y


class EmbeddingLayer(nn.Module):
  """Sums word, position and type embeddings.

  Attributes:
    config: Model configuration.
  """
  config: ml_collections.FrozenConfigDict

  @nn.compact
  def __call__(self,
               input_ids,
               type_ids,
               deterministic = False):
    """Applies EmbeddingLayer module.

    Args:
      input_ids: Batch of tokenized inputs of shape <int>[BATCH_SIZE,
        MAX_SEQ_LENGTH].
      type_ids: Ids partitioning input into different types.
      deterministic: Whether or not to apply dropout to output embeddings.

    Returns:
      Embedded tokens of shape <float>[BATCH_SIZE, MAX_SEQ_LENGTH, EMB_DIM].
    """
    word_embeddings = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.d_emb,
        embedding_init=default_kernel_init,
        name="word")(
            input_ids)
    position_embeddings = PositionalEncoding(
        max_seq_length=self.config.max_seq_length,
        posemb_init=default_kernel_init,
        name="position")(
            word_embeddings)
    type_embeddings = nn.Embed(
        num_embeddings=self.config.type_vocab_size,
        features=self.config.d_emb,
        embedding_init=default_kernel_init,
        name="type")(
            type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, name="layer_norm")(
            embeddings)
    embeddings = nn.Dense(
        self.config.d_model, name="hidden_mapping_in")(
            embeddings)
    return nn.Dropout(rate=self.config.dropout_rate)(
        embeddings, deterministic=deterministic)


class PositionalEncoding(nn.Module):
  """Learned positional embeddings.

  Attributes:
    max_seq_length: Maximum sequence length.
    posemb_init: Initializer scheme for positional embedding parameters.
  """
  max_seq_length: int
  posemb_init: Callable[[PRNGKey, Shape, Dtype],
                        jnp.ndarray] = default_kernel_init

  @nn.compact
  def __call__(self, word_embeddings):
    """Applies PositionalEncoding module.

    Args:
      word_embeddings: Embeddings of input tokens of shape <float>[BATCH_SIZE,
        MAX_SEQ_LENGTH, EMB_DIM].

    Returns:
      Positional embeddings <float>[BATCH_SIZE, MAX_SEQ_LENGTH, EMB_DIM]
        associated with input word embeddings.

    Raises:
      ValueError: If word_embeddings dimension is not 3.
    """
    if word_embeddings.ndim != 3:
      raise ValueError(
          "Input word_embeddings dimension should be 3, but it is: %d" %
          word_embeddings.ndim)

    length = word_embeddings.shape[1]
    pos_emb_shape = (1, self.max_seq_length, word_embeddings.shape[-1])
    pos_embedding = self.param("embedding", self.posemb_init, pos_emb_shape)
    return pos_embedding[:, :length, :]


def gather(sequence, indices):
  """Gathers sequence at the specified indices.

  Args:
    sequence: Sequence of shape <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM].
    indices: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] indices of tokens in
      sequence to gather.

  Returns:
    <float>[BATCH_SIZE * MAX_PREDICTIONS_PER_SEQ, HIDDEN_DIM] elements of input
    sequence at specified indices.

  Raises:
    ValueError: If input sequence and indices have different batch sizes or
    MAX_PREDICTIONS_PER_SEQ > MAX_SEQ_LENGTH.
  """
  if sequence.shape[0] != indices.shape[0]:
    raise ValueError(
        "Input sequence and indices must have the same batch size: "
        "sequence.shape[0] = %d whereas indices.shape[0] = %d." %
        (sequence.shape[0], indices.shape[0]))

  if indices.shape[1] > sequence.shape[1]:
    raise ValueError(
        "The maximum number of predictions per sequence cannot be greater "
        "than the maximum sequence length. indices.shape[1] = %d and "
        "sequence.shape[1] = %d." % (indices.shape[1], sequence.shape[1]))

  batch_size, max_seq_length, hidden_dim = sequence.shape
  flat_offsets = jnp.reshape(jnp.arange(batch_size) * max_seq_length, [-1, 1])
  flat_indices = jnp.reshape(indices + flat_offsets, [-1])
  flat_sequence = jnp.reshape(sequence,
                              [batch_size * max_seq_length, hidden_dim])
  return jnp.take(flat_sequence, flat_indices, axis=0)
