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

"""Encoder models."""

import functools
import math
from typing import Any, Dict, Optional, Tuple, Union

from flax import linen as nn
from flax.training.common_utils import onehot
from jax import lax
from jax import random
import jax.numpy as jnp
import ml_collections
from scipy import linalg

from f_net import fourier
from f_net import layers
from f_net.configs.base import HybridAttentionLayout
from f_net.configs.base import ModelArchitecture

# Type Stubs
MixingLayer = Any
PRNGKey = Any

default_kernel_init = nn.initializers.normal(stddev=2e-2)
# TODO(b/181607810): Doubt this will make a difference, but BERT uses zeros for
#  initial bias.
default_bias_init = nn.initializers.normal(stddev=2e-2)

LAYER_NORM_EPSILON = 1e-12


class EncoderModel(nn.Module):
  """Encoder model without any task-specific heads.

  Attributes:
    config: Model specifications.
    random_seed: Random number generator seed. Only used by
      ModelArchitecture.RANDOM architecture.
  """
  config: ml_collections.FrozenConfigDict
  random_seed: int = 0

  def setup(self):
    """Initializes encoder with config-dependent mixing layer."""
    if self.config.model_arch == ModelArchitecture.F_NET:
      self._init_fourier_transform()

    # Random number generator key for RANDOM model architecture.
    key = random.PRNGKey(self.random_seed)

    encoder_blocks = []  # Attributes are immutable so use temporary list
    for layer in range(self.config.num_layers):
      key, mixing_key = random.split(key)
      mixing_arch = ModelArchitecture.BERT if self._is_attention_layer(
          layer) else self.config.model_arch
      mixing_layer = self._init_mixing_sublayer(layer, mixing_arch, mixing_key)
      feed_forward_layer = layers.FeedForwardLayer(
          d_ff=self.config.d_ff,
          dropout_rate=self.config.dropout_rate,
          name=f"feed_forward_{layer}")
      encoder_blocks.append(
          layers.EncoderBlock(
              mixing_sublayer=mixing_layer,
              feed_forward_sublayer=feed_forward_layer,
              name=f"encoder_{layer}"))
    self.encoder_blocks = encoder_blocks

    self.embedder = layers.EmbeddingLayer(config=self.config, name="embedder")

    self.pooler = nn.Dense(
        self.config.d_model, kernel_init=default_kernel_init, name="pooler")

  def __call__(self,
               input_ids,
               input_mask,
               type_ids,
               deterministic = False):
    """Applies model on the inputs.

    Args:
      input_ids: Tokenized inputs of shape <int>[BATCH_SIZE, MAX_SEQ_LENGTH].
      input_mask: <bool>[BATCH_SIZE, MAX_SEQ_LENGTH] mask separating actual
        inputs from padding. Only used by BERT.
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] ids partitioning input into
        different types.
      deterministic: Whether or not to apply dropout in each layer.

    Returns:
      Hidden states of shape <float>[BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM],
        and pooled output <float>[BATCH_SIZE, HIDDEN_DIM] scaled to (-1, 1).
    """
    hidden_states = self.embedder(
        input_ids, type_ids, deterministic=deterministic)

    # Only used by (BERT) self-attention sublayer.
    padding_mask = input_mask.astype(jnp.int32)
    padding_mask = nn.make_attention_mask(
        query_input=padding_mask, key_input=padding_mask)

    for encoder_block in self.encoder_blocks:
      hidden_states = encoder_block(
          hidden_states, padding_mask, deterministic=deterministic)

    pooled_output = self.pooler(hidden_states[:, 0])
    pooled_output = jnp.tanh(pooled_output)

    return hidden_states, pooled_output

  def _init_fourier_transform(self):
    """Initializes Fourier Transform.

    On GPUs/CPUs: The native FFT implementation is optimal for all sequence
    lengths.

    On TPUs: For relatively shorter sequences, it is faster to pre-compute the
    DFT matrix and then compute Fourier Transform using matrix multiplications.
    For longer sequences, the FFT is faster, provided the MAX_SEQ_LENGTH is a
    power of 2.
    """
    if self.config.use_fft:
      if (self.config.max_seq_length > 4096 and
          not math.log2(self.config.max_seq_length).is_integer()):
        raise ValueError(
            "For large input sequence lengths (>4096), the maximum input "
            "sequence length must be a power of 2 to take advantage of FFT "
            "optimizations. We encourage the same for the model hidden "
            "dimension. config.max_seq_length: %d. config.d_model: $d" %
            self.config.max_seq_length, self.config.d_model)

      self.fourier_transform = jnp.fft.fftn
    else:
      dft_mat_hidden = linalg.dft(self.config.d_model)
      dft_mat_seq = linalg.dft(self.config.max_seq_length)

      self.fourier_transform = functools.partial(
          fourier.two_dim_matmul,
          matrix_dim_one=jnp.asarray(dft_mat_seq),
          matrix_dim_two=jnp.asarray(dft_mat_hidden),
          precision=lax.Precision.DEFAULT)

  def _is_attention_layer(self, layer):
    """Returns true if the current layer should be an attention layer."""
    num_attention_layers = self.config.num_attention_layers
    num_layers = self.config.num_layers

    if self.config.attention_layout == HybridAttentionLayout.BOTTOM:
      return layer < num_attention_layers
    elif self.config.attention_layout == HybridAttentionLayout.MIDDLE:
      return (num_layers - num_attention_layers <= 2 * layer <
              num_layers + num_attention_layers)
    elif self.config.attention_layout == HybridAttentionLayout.MIXED:
      return layer % (num_layers // num_attention_layers) == 0
    elif self.config.attention_layout == HybridAttentionLayout.TOP:
      return layer >= num_layers - num_attention_layers
    else:
      return False

  def _init_mixing_sublayer(self, layer, model_arch,
                            mixing_key):
    """Initializes config-dependent mixing sublayer."""
    if model_arch == ModelArchitecture.BERT:
      mixing_sublayer = nn.SelfAttention(
          num_heads=self.config.num_heads,
          qkv_features=self.config.d_model,
          broadcast_dropout=False,
          kernel_init=default_kernel_init,
          bias_init=default_bias_init,
          dropout_rate=self.config.mixing_dropout_rate,
          use_bias=True,
          name=f"self_attention_{layer}")
    elif model_arch == ModelArchitecture.F_NET:
      mixing_sublayer = layers.FourierTransform(
          fourier_transform=self.fourier_transform,
          name=f"fourier_transform_{layer}")
    elif model_arch == ModelArchitecture.FF_ONLY:
      mixing_sublayer = layers.IdentityTransform(
          name=f"identity_transform_{layer}")
    elif model_arch == ModelArchitecture.LINEAR:
      mixing_sublayer = layers.LinearTransform(
          precision=lax.Precision.DEFAULT, name=f"linear_transform_{layer}")
    elif model_arch == ModelArchitecture.RANDOM:
      mixing_sublayer = layers.RandomTransform(
          max_seq_length=self.config.max_seq_length,
          d_model=self.config.d_model,
          key=mixing_key,
          precision=lax.Precision.DEFAULT,
          name=f"random_transform_{layer}")
    else:
      raise ValueError("Unexpected model architecture: %s" % model_arch.name)

    return mixing_sublayer


class PreTrainingModel(nn.Module):
  """Masked Language Modelling and Next-Sentence Prediction pre-training model.

  Attributes:
    config: Model specification.
    random_seed: Random number generator seed. Only used by
      ModelArchitecture.RANDOM architecture.
  """
  config: ml_collections.FrozenConfigDict
  random_seed: int = 0

  @nn.compact
  def __call__(self,
               input_ids,
               input_mask,
               type_ids,
               masked_lm_positions,
               masked_lm_labels,
               masked_lm_weights,
               next_sentence_labels,
               deterministic = False):
    """Applies pre-training model on inputs.

    Args:
      input_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] tokenized inputs.
      input_mask: <bool>[BATCH_SIZE, MAX_SEQ_LENGTH] mask separating actual
        inputs from padding. Only used by BERT.
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] Ids partitioning input into
        different types.
      masked_lm_positions: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] indices
        indicating which inputs are masked.
      masked_lm_labels: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] true labels
        for masked inputs.
      masked_lm_weights: <float>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] relative
        weighting for masked inputs.
      next_sentence_labels: <int>[BATCH_SIZE, 1] Labels for next sentence
        prediction task.
      deterministic: Whether or not to apply dropout to input.

    Returns:
      Loss and metrics for given inputs.
    """
    sequence_output, pooled_output = EncoderModel(
        self.config, random_seed=self.random_seed, name="encoder")(
            input_ids, input_mask, type_ids, deterministic=deterministic)

    masked_lm_output = layers.gather(sequence_output, masked_lm_positions)
    masked_lm_output = nn.Dense(
        self.config.d_emb,
        kernel_init=default_kernel_init,
        name="predictions_dense")(
            masked_lm_output)
    masked_lm_output = nn.gelu(masked_lm_output)
    masked_lm_output = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, name="predictions_layer_norm")(
            masked_lm_output)
    masked_lm_logits = layers.OutputProjection(
        kernel=self._get_embedding_table(), name="predictions_output")(
            masked_lm_output)

    next_sentence_logits = layers.OutputProjection(
        n_out=2, kernel_init=default_kernel_init, name="classification")(
            pooled_output)

    return _compute_pretraining_metrics(masked_lm_logits, next_sentence_logits,
                                        masked_lm_labels, masked_lm_weights,
                                        next_sentence_labels)

  def _get_embedding_table(self):
    """Returns kernel weights for word embeddings."""
    return self.variables["params"]["encoder"]["embedder"]["word"]["embedding"]


def _compute_pretraining_metrics(
    masked_lm_logits, next_sentence_logits,
    masked_lm_labels, masked_lm_weights,
    next_sentence_labels):
  """Computes the pre-training loss and its components.

  Args:
    masked_lm_logits: <float>[BATCH_SIZE * MAX_PREDICTIONS_PER_SEQ, EMB_DIM]
      predicted logits for masked LM task.
    next_sentence_logits: <float>[BATCH_SIZE, 2] predicted logits for NSP task.
    masked_lm_labels: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] true labels for
      masked inputs.
    masked_lm_weights: <float>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] weights for
      masked inputs.
    next_sentence_labels: <float>[BATCH_SIZE, 1] true labels for NSP task.

  Returns:
    Model loss and raw metrics.
  """

  def _compute_weighted_cross_entropy(
      logits,
      targets,
      weights = None):
    """Computes weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: <float>[NUM_EXAMPLES, NUM_CLASSES] predicted logits.
     targets: <int>[NUM_EXAMPLES] true labels.
     weights: <float>[NUM_EXAMPLES] relative weights for labels.

    Returns:
      Loss and normalizing factor for input batch.
    """
    onehot_targets = onehot(targets, logits.shape[-1])
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    normalizing_factor = onehot_targets.sum()
    if weights is not None:
      loss = loss * weights
      normalizing_factor = weights.sum()
    return loss.sum(), normalizing_factor

  masked_lm_correct = jnp.sum(
      (masked_lm_logits.argmax(-1) == masked_lm_labels.ravel()) *
      masked_lm_weights.ravel())
  masked_lm_loss, masked_lm_normalization = _compute_weighted_cross_entropy(
      masked_lm_logits,
      masked_lm_labels.ravel(),
      weights=masked_lm_weights.ravel())

  next_sentence_loss, num_next_sentence_labels = _compute_weighted_cross_entropy(
      next_sentence_logits, next_sentence_labels.ravel())
  next_sentence_correct = jnp.sum(
      next_sentence_logits.argmax(-1) == next_sentence_labels.ravel())

  return {
      "loss":
          masked_lm_loss / masked_lm_normalization +
          next_sentence_loss / num_next_sentence_labels,
      "masked_lm_loss":
          masked_lm_loss,
      "masked_lm_normalization":
          masked_lm_normalization,
      "masked_lm_correct":
          masked_lm_correct,
      "masked_lm_total":
          masked_lm_weights.sum(),
      "next_sentence_loss":
          next_sentence_loss,
      "num_next_sentence_labels":
          num_next_sentence_labels,
      "next_sentence_correct":
          next_sentence_correct,
  }


class SequenceClassificationModel(nn.Module):
  """Sequence classification model.

  Attributes:
    config: Model specification.
    n_classes: Number of output (prediction) classes.
  """
  config: ml_collections.FrozenConfigDict
  n_classes: int

  @nn.compact
  def __call__(
      self,
      input_ids,
      input_mask,
      type_ids,
      labels = None,
      deterministic = False
  ):
    """Applies model for sequence classification.

    Args:
      input_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] tokenized inputs.
      input_mask: <bool>[BATCH_SIZE, MAX_SEQ_LENGTH] mask separating actual
        inputs from padding. Only used by BERT.
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] Ids partitioning input into
        different types.
      labels: True labels associated with inputs. Generally only required for
        training. Shape depends on task type:
        * Classification: <int>[BATCH_SIZE]
        * Regression: <float>[BATCH_SIZE]
      deterministic: Whether or not to apply dropout to input.

    Returns:
      * If labels supplied (training mode): Model loss and metrics.
      * If no labels supplied (prediction / evaluation mode): Logits of shape
        <float>[BATCH_SIZE, n_classes].
    """
    _, pooled_output = EncoderModel(
        self.config, name="encoder")(
            input_ids, input_mask, type_ids, deterministic=deterministic)

    logits = layers.OutputProjection(
        n_out=self.n_classes,
        kernel_init=default_kernel_init,
        name="classification")(
            pooled_output)

    if labels is None:
      # Code path used during evaluation or prediction; metrics can be computed
      # from logits by the caller.
      return logits

    # Code path used during training.
    if self.config.dataset_name == "glue/stsb":  # Regression task
      loss = jnp.mean((logits[Ellipsis, 0] - labels)**2)
      return {"loss": loss, "num_labels": labels.size}
    else:  # Classification task
      logits = nn.log_softmax(logits)
      loss = -jnp.mean(
          jnp.sum(onehot(labels, logits.shape[-1]) * logits, axis=-1))
      correct_predictions = jnp.sum(logits.argmax(-1) == labels)
      return {
          "loss": loss,
          "correct_predictions": correct_predictions,
          "num_labels": labels.size
      }
