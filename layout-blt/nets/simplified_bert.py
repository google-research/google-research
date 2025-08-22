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

# pytype: skip-file

"""Simplified BERT without NSP (next setence prediction) and segment ids."""

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

# BERT layer norm
TF_LAYERNORM_EPSILON = 1e-12

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


def bert_masking(inputs,
                 nospecial_tokens,
                 mask_token,
                 pad_token = -1,
                 mask_rate = 0.15,
                 mask_token_proportion = 0.8,
                 random_token_proportion = 0.1):
  """Preprocess inputs following BERT.

  First chooses mask_rate of the token positions at random for prediction.
  Among them, mask_token_proportion of them will be replaced with mask_token.
  random_token_proportion of them will be replaced with randomly sampled tokens.
  Others will remain the same.

  Args:
    inputs: [batch, sequence_length] input tokens.
    nospecial_tokens: jnp.ndarray. Set of tokens usable for replacing (not
      include special tokens such as pad, mask, bos, etc.)
    mask_token: Int ID for mask token.
    pad_token: Int ID for PAD token.
    mask_rate: Proportion of tokens to mask out.
    mask_token_proportion: Replace this proportion of chosen positions with
      MASK.
    random_token_proportion: Replace this proportion of chosen positions with
      randomly sampled tokens

  Returns:
    Tuple of [batch, sequence_length] masked_inputs, inputs, mask_weights.
  """
  mask_rate = jnp.maximum(1e-3, mask_rate)
  total = random_token_proportion + mask_token_proportion
  if total < 0 or total > 1:
    raise ValueError('Sum of random proportion and mask proportion must be'
                     ' in [0, 1] range.')
  targets = inputs

  rng = jax.random.PRNGKey(jnp.sum(inputs))

  # Gets positions to leave untouched.
  is_pad = inputs == pad_token
  lens = jnp.sum(~is_pad, axis=-1)
  # Obtains the ceiling of the lens to make sure we can mask at least one token.
  mask_lens = jax.lax.ceil(lens * mask_rate)
  # Positions to mask.
  rng, subrng = jax.random.split(rng)
  # Randomly generates the mask score uniformly.
  should_mask = jax.random.uniform(subrng, shape=inputs.shape)
  # Doesn't mask out padding.
  should_mask = jnp.where(is_pad, 2., should_mask)
  sorted_should_mask = jnp.sort(should_mask, axis=-1)

  # Obtains the cutoff score for the mask lens.
  cut_off = jnp.take_along_axis(
      sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1)
  cut_off = jnp.repeat(cut_off, inputs.shape[1], axis=1)

  # Scores smaller than the cutoff will be masked.
  should_mask = jnp.where(should_mask <= cut_off, 1., 0.)

  # Generate full array of random tokens.
  rng, subrng = jax.random.split(rng)
  random_ids = jax.random.randint(
      subrng, inputs.shape, minval=0, maxval=len(nospecial_tokens))

  fullrandom = nospecial_tokens[random_ids]
  # Full array of MASK tokens
  fullmask = jnp.full_like(inputs, mask_token)

  # Build up masked array by selecting from inputs/fullmask/fullrandom.
  rand = jax.random.uniform(rng, shape=inputs.shape)
  masked_inputs = inputs
  # Remaining probability mass stays original values after MASK and RANDOM.
  # MASK tokens.
  masked_inputs = jnp.where(rand < mask_token_proportion, fullmask,
                            masked_inputs)
  # Random tokens.
  masked_inputs = jnp.where(
      jnp.logical_and(rand >= mask_token_proportion,
                      rand < mask_token_proportion + random_token_proportion),
      fullrandom, masked_inputs)

  # Only replace positions where `should_mask`
  masked_inputs = jnp.where(should_mask, masked_inputs, inputs)
  weights = should_mask

  return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)


def truncated_normal(stddev, dtype=jnp.float32):
  def init(key, shape, dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
  return init


class Bias(nn.Module):
  """Adds a (learned) bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class BertAttention(nn.Module):
  """BERT attention layer that is part of each BERT layer."""
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input, input_mask,
               deterministic):
    attention_mask = nn.make_attention_mask(input_mask, input_mask)
    attention_output = nn.attention.SelfAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(layer_input, attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic)
    attention_output = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON,
        name='attention_output_ln')(
            attention_output + layer_input)

    return attention_output


class BertMlp(nn.Module):
  """BERT MLP layer that is part of each BERT layer."""
  hidden_size: int
  hidden_dropout_prob: float
  intermediate_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, attention_output,
               deterministic):
    # Bert intermediate layer.
    intermediate_output = nn.Dense(
        features=self.intermediate_size,
        kernel_init=self.initializer_fn,
        name='intermediate_output')(
            attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    # Bert output layer.
    layer_output = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='layer_output')(
            intermediate_output)
    layer_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        layer_output, deterministic=deterministic)
    layer_output = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON,
        name='layer_output_ln')(
            layer_output + attention_output)

    return layer_output


class BertLayer(nn.Module):
  """A single Bert layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input, input_mask,
               deterministic):
    attention_output = BertAttention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            layer_input=layer_input,
            input_mask=input_mask,
            deterministic=deterministic)

    layer_output = BertMlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output,
            deterministic=deterministic)

    return layer_output


class BertEmbed(nn.Module):
  """Embeds Bert-style."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self,
               input_ids,
               deterministic):
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(position_ids)

    input_embeddings = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON,
        name='embeddings_ln')(
            word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class BertMlmLayer(nn.Module):
  """Bert layer for masked token prediction."""
  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, last_layer,
               embeddings):
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense')(last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON,
        name='mlm_ln')(
            mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Bert(nn.Module):
  """BERT as a Flax module."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  initializer_range: float = 0.02
  pad_token_id: int = -1

  @nn.compact
  def __call__(self,
               input_ids,
               deterministic = True):
    # We assume that all pad tokens should be masked out.
    input_ids = input_ids.astype('int32')
    input_mask = jnp.asarray(input_ids != self.pad_token_id, dtype=jnp.int32)

    input_embeddings = BertEmbed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids,
            deterministic=deterministic)

    # Stack BERT layers.
    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = BertLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=input_mask,
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['BertEmbed_0'][
        'word_embeddings']['embedding']
    logits = BertMlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits
