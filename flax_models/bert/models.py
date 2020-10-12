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

"""Transformer models."""
from flax import nn
from flax.training.common_utils import onehot
import jax.numpy as jnp

from flax_models.bert import efficient_attention
from flax_models.bert import layers

NEG_INFINITY = -10000.0
LAYER_NORM_EPSILON = 1e-12


# TODO(marcvanzee): Use config.hidden_activation and config.kernel_initializer
# instead of these two once it is re-enabled (see comment on config file).
hidden_activation = nn.gelu
kernel_initializer = nn.initializers.xavier_uniform()


class BertModel(nn.Module):
  """BERT model without any task-specific heads."""

  def apply(self,
            input_ids, input_mask, type_ids, *,
            config,
            deterministic=False):
    """Applies BERT model on the inputs."""

    word_embeddings = nn.Embed(
        input_ids,
        num_embeddings=config.vocab_size,
        features=config.d_emb,
        embedding_init=kernel_initializer,
        name="word_embeddings")
    position_embeddings = layers.PositionalEncoding(
        word_embeddings,
        max_len=config.max_len,
        posemb_init=kernel_initializer,
        name="position_embeddings")
    type_embeddings = nn.Embed(
        type_ids,
        num_embeddings=config.type_vocab_size,
        features=config.d_emb,
        embedding_init=kernel_initializer,
        name="type_embeddings")

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = nn.LayerNorm(
        embeddings, epsilon=LAYER_NORM_EPSILON, name="embeddings_layer_norm")
    embeddings = nn.Dense(
        embeddings, config.d_model, name="embedding_hidden_mapping_in")
    embeddings = nn.dropout(
        embeddings, rate=config.dropout_rate, deterministic=deterministic)

    # Transformer blocks
    feed_forward = layers.FeedForward.partial(
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
        intermediate_activation=hidden_activation,
        kernel_init=kernel_initializer)

    self_attention = efficient_attention.BertSelfAttention.partial(
        num_heads=config.num_heads,
        num_parallel_heads=config.num_parallel_heads,
        d_qkv=config.d_model // config.num_heads,
        attention_dropout_rate=config.attention_dropout_rate,
        output_dropout_rate=config.dropout_rate,
        kernel_init=kernel_initializer,
        output_kernel_init=kernel_initializer)

    hidden_states = embeddings
    mask = input_mask.astype(jnp.int32)
    shared_encoder_layer = layers.TransformerBlock.shared(
        feed_forward=feed_forward,
        attention=self_attention,
        deterministic=deterministic,
        name="encoder_layer_0")
    for _ in range(config.num_layers):
      hidden_states = shared_encoder_layer(hidden_states, mask)

    pooled_output = nn.Dense(
        hidden_states[:, 0],
        config.d_model,
        kernel_init=kernel_initializer,
        name="pooler")
    pooled_output = jnp.tanh(pooled_output)

    return hidden_states, pooled_output

  @nn.base.module_method
  def get_embedding_table(self, **unused_kwargs):
    return self.get_param("word_embeddings")["embedding"]


class GatherIndexes(nn.Module):
  """Gathers the vectors at the specific positions."""

  def apply(self,
            sequence_tensor,
            positions):
    """Applies gather indexes layer.

    Args:
      sequence_tensor: Sequence output of `BertModel` layer of shape
        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
        hidden units of `BertModel` layer.
      positions: Positions ids of tokens in sequence to mask for pretraining
        of with dimension (batch_size, num_predictions) where
        `num_predictions` is maximum number of tokens to mask out and predict
        per each sequence.

    Returns:
      Masked out sequence tensor of shape (batch_size * num_predictions,
      num_hidden).
    """
    batch_size, seq_length, width = sequence_tensor.shape
    flat_offsets = jnp.reshape(jnp.arange(batch_size) * seq_length, [-1, 1])
    flat_positions = jnp.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = jnp.reshape(sequence_tensor,
                                       [batch_size * seq_length, width])
    output_tensor = jnp.take(flat_sequence_tensor, flat_positions, axis=0)

    return output_tensor


class BertForSequenceClassification(nn.Module):
  """Bert model for sequence classification."""

  def apply(self,
            input_ids, input_mask, type_ids, labels=None, *,
            config, n_classes, deterministic=False):
    """Applies BERT for sequence classification."""
    unused_sequence_output, pooled_output = BertModel(
        input_ids, input_mask, type_ids,
        config=config, deterministic=deterministic, name="bert")
    # TODO(kitaev): I think I'm missing dropout here
    logits = layers.OutputProjection(
        pooled_output,
        n_out=n_classes,
        kernel_init=kernel_initializer,
        name="classification")

    if labels is None:
      return logits
    elif logits.shape[-1] == 1:
      # Regression task
      loss = jnp.mean((logits[Ellipsis, 0] - labels) ** 2)
      return {"loss": loss}
    else:
      # Classification task
      logits = nn.log_softmax(logits)
      loss = -jnp.mean(jnp.sum(
          onehot(labels, logits.shape[-1]) * logits, axis=-1))
      return {"loss": loss}


class BertForPreTraining(nn.Module):
  """Bert model for pre-training."""

  def apply(self,
            input_ids, input_mask, type_ids,
            masked_lm_positions=None,
            masked_lm_labels=None,
            masked_lm_weights=None,
            next_sentence_labels=None,
            *,
            config, deterministic=False):
    """Applies BERT for pre-training."""
    bert = BertModel.shared(config=config, name="bert")
    sequence_output, pooled_output = bert(
        input_ids, input_mask, type_ids, deterministic=deterministic)
    if masked_lm_positions is None:
      return sequence_output, pooled_output

    # Masked LM
    masked_lm_input = GatherIndexes(sequence_output, masked_lm_positions)
    masked_lm_input = nn.Dense(
        masked_lm_input,
        config.d_emb,
        kernel_init=kernel_initializer,
        name="predictions_transform_dense")
    masked_lm_input = hidden_activation(masked_lm_input)
    masked_lm_input = nn.LayerNorm(
        masked_lm_input,
        epsilon=LAYER_NORM_EPSILON,
        name="predictions_transform_layernorm")
    masked_lm_logits = layers.OutputProjection(
        masked_lm_input, kernel=bert.get_embedding_table(),
        name="predictions_output")

    # Next-sentence prediction
    next_sentence_logits = layers.OutputProjection(
        pooled_output,
        n_out=2,
        kernel_init=kernel_initializer,
        name="classification")

    if masked_lm_labels is None or next_sentence_labels is None:
      return masked_lm_logits, next_sentence_logits
    else:
      return self._compute_metrics(
          masked_lm_logits, next_sentence_logits,
          masked_lm_labels, masked_lm_weights, next_sentence_labels)

  def _compute_metrics(self,
                       masked_lm_logits,
                       next_sentence_logits,
                       masked_lm_labels,
                       masked_lm_weights,
                       next_sentence_labels,
                       **unused_kwargs):
    """Computes the pre-training loss and its components."""
    masked_lm_logits = nn.log_softmax(masked_lm_logits)
    masked_lm_labels = onehot(
        masked_lm_labels.reshape((-1,)), masked_lm_logits.shape[-1])
    masked_lm_weights = masked_lm_weights.reshape((-1,))
    masked_lm_loss = -jnp.sum(
        jnp.sum(masked_lm_logits * masked_lm_labels,
                axis=-1) * masked_lm_weights) / jnp.sum(masked_lm_weights)

    next_sentence_logits = nn.log_softmax(next_sentence_logits)
    next_sentence_labels = next_sentence_labels.reshape((-1,))
    next_sentence_loss = -jnp.mean(jnp.sum(
        onehot(
            next_sentence_labels, next_sentence_logits.shape[-1]
            ) * next_sentence_logits, axis=-1))
    return {
        "loss": masked_lm_loss + next_sentence_loss,
        "masked_lm_loss": masked_lm_loss,
        "next_sentence_loss": next_sentence_loss,
    }

  compute_metrics = nn.base.module_method(_compute_metrics)
