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

"""Encoder models."""

from typing import Any, Optional, Tuple, Union

import flax
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.training.common_utils import onehot
from jax import lax
import jax.numpy as jnp
import ml_collections

from sparse_mixers import layers
from sparse_mixers import routing
from sparse_mixers.configs import base as base_config

# Type Stubs
DispatchAlgorithm = base_config.DispatchAlgorithm
DiversityMetrics = layers.DiversityMetrics
FrozenVariableDict = flax_scope.FrozenVariableDict
LayerLayout = base_config.LayerLayout
MixingLayer = layers.MixingLayer
ModelArchitecture = base_config.ModelArchitecture
PRNGKey = Any


default_kernel_init = nn.initializers.normal(stddev=2e-2)
default_bias_init = nn.initializers.zeros

LAYER_NORM_EPSILON = 1e-12


@flax.struct.dataclass
class EncoderOutput:
  """Output from core encoder model.

  Attributes:
    sequence_output: Hidden states of shape <float>[BATCH_SIZE, MAX_SEQ_LENGTH,
      HIDDEN_DIM].
    pooled_output: Pooled output <float>[BATCH_SIZE, HIDDEN_DIM] scaled to (-1,
      1).
  """
  sequence_output: jnp.ndarray
  pooled_output: jnp.ndarray


@flax.struct.dataclass
class PretrainingStats:
  """Pre-training loss and raw prediction statistics.

  Attributes:
    masked_lm_loss: Unweighted masked LM loss for current batch.
    next_sentence_loss: Unweighted next sentenced prediction loss for current
      batch.
    masked_lm_correct: Number of correct masked LM predictions for current
      batch.
    masked_lm_normalization: Sample weighted normalization factor for masked LM
      labels for current batch.
    masked_lm_total: Number of masked LM labels for current batch.
    next_sentence_correct: Number of correct next sentence predictions for
      current batch.
    num_next_sentence_labels: Number of next sentence prediction labels for
      current batch.
    grad_l2_sum: L2-norm of gradient. Typically populated by the trainer.
    expert_metrics: Metrics for analyzing diversity among experts in mixture of
      experts architectures. Typically populated by the trainer.
  """
  masked_lm_loss: float
  next_sentence_loss: float

  masked_lm_correct: int
  masked_lm_normalization: float
  masked_lm_total: int

  next_sentence_correct: int
  num_next_sentence_labels: float

  # The following attributes are typically populated by the trainer as opposed
  # to the model.
  grad_l2_sum: Optional[float] = None
  expert_metrics: Optional[DiversityMetrics] = None


@flax.struct.dataclass
class ClassificationStats:
  """Classification loss and raw prediction statistics.

  Attributes:
    batch_loss: Loss for current batch.
    num_labels: Number of possible labels for current batch. correct_predictions
      will also be less than or equal to num_labels.
    correct_predictions: Number of correct predictions for current batch. This
      field is typically only populated for classification tasks.
    grad_l2_sum: L2-norm of gradient. Typically populated by the
      trainer.
    expert_metrics: Metrics for analyzing diversity among experts in mixture of
      experts architectures. Typically populated by the trainer.
  """
  batch_loss: float
  num_labels: int
  correct_predictions: Optional[int] = None

  # The following attributes are typically populated by the trainer as opposed
  # to the model.
  grad_l2_sum: Optional[float] = None
  expert_metrics: Optional[DiversityMetrics] = None


class EncoderModel(nn.Module):
  """Encoder model without any task-specific heads.

  Attributes:
    config: Model specifications.
  """
  config: ml_collections.FrozenConfigDict

  def setup(self):
    """Initializes encoder with config-dependent mixing layer."""
    encoder_blocks = []  # Attributes are immutable so use temporary list
    for layer in range(self.config.num_layers):
      if self._is_attention_layer(layer):
        attention_sublayer = layers.AttentionLayer(  # pytype: disable=wrong-arg-types  # jax-types
            num_heads=self.config.num_heads,
            d_model=self.config.d_model,
            dtype=self.config.dtype,
            kernel_init=default_kernel_init,
            bias_init=default_bias_init,
            dropout_rate=self.config.mixing_dropout_rate,
            pad_id=self.config.pad_id,
            name=f"attention_{layer}")
        mixing_sublayer = None
      else:
        attention_sublayer = None
        mixing_sublayer = self._init_mixing_sublayer(layer)

      feed_forward_sublayer = self._init_feed_forward_sublayer(layer)

      encoder_blocks.append(
          layers.EncoderBlock(
              mixing_sublayer=mixing_sublayer,
              attention_sublayer=attention_sublayer,
              feed_forward_sublayer=feed_forward_sublayer,
              name=f"encoder_{layer}"))
    self.encoder_blocks = encoder_blocks

    self.embedder = layers.EmbeddingLayer(config=self.config, name="embedder")

    self.pooler = nn.DenseGeneral(
        self.config.d_model,
        use_bias=True,
        dtype=self.config.dtype,
        kernel_init=default_kernel_init,
        name="pooler")

  def __call__(self,
               input_ids,
               type_ids,
               deterministic = False):
    """Applies model on the inputs.

    Args:
      input_ids: Tokenized inputs of shape <int>[BATCH_SIZE, MAX_SEQ_LENGTH].
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] ids partitioning input into
        different types.
      deterministic: Whether to apply dropout in each layer.

    Returns:
      Hidden states and pooled output.
    """
    hidden_states = self.embedder(
        input_ids, type_ids, deterministic=deterministic)

    for encoder_block in self.encoder_blocks:
      hidden_states = encoder_block(
          hidden_states, input_ids=input_ids, deterministic=deterministic)

    pooled_output = self.pooler(hidden_states[:, 0])
    pooled_output = jnp.tanh(pooled_output)

    return EncoderOutput(
        sequence_output=hidden_states, pooled_output=pooled_output)

  def _init_mixing_sublayer(self, layer):
    """Initializes config-dependent mixing sublayer."""
    if self.config.model_arch == ModelArchitecture.F_NET.name:
      mixing_sublayer = layers.FourierTransform(
          use_fft=self.config.use_fft,
          name=f"fourier_transform_{layer}")
    elif self.config.model_arch == ModelArchitecture.H_NET.name:
      mixing_sublayer = layers.HartleyTransform(
          use_fft=self.config.use_fft,
          name=f"hartley_transform_{layer}")
    elif self.config.model_arch == ModelArchitecture.LINEAR.name:
      mixing_sublayer = layers.LinearTransform(
          precision=lax.Precision.DEFAULT,
          name=f"linear_transform_{layer}")
    elif self.config.model_arch == ModelArchitecture.C_NET.name:
      mixing_sublayer = layers.CirculantTransform(
          use_fft=self.config.use_fft,
          name=f"circulant_transform_{layer}")
    elif self.config.model_arch == ModelArchitecture.T_NET.name:
      mixing_sublayer = layers.ToeplitzTransform(
          use_fft=self.config.use_fft,
          name=f"toeplitz_transform_{layer}")
    else:
      raise ValueError("Unexpected model architecture: %s" %
                       self.config.model_arch)

    return mixing_sublayer

  def _init_feed_forward_sublayer(
      self, layer):
    """Initializes config-dependent feed-forward sublayer."""
    if self._is_moe_layer(layer):
      expert = layers.FeedForwardLayer(
          d_ff=self.config.expert_d_ff,
          dropout_rate=self.config.expert_dropout_rate,
          dtype=layers.truncated_dtype(),
          name=f"expert_{layer}")
      ff_sublayer = layers.MoeLayer(
          num_experts=self.config.num_experts,
          router=self._init_routers(layer),
          max_group_size=self.config.max_group_size,
          train_capacity_factor=self.config.train_capacity_factor,
          eval_capacity_factor=self.config.eval_capacity_factor,
          expert=expert,
          min_expert_capacity=self.config.min_expert_capacity,
          dropout_rate=self.config.expert_dropout_rate,
          dtype=self.config.dtype,
          name=f"moe_{layer}")
    else:
      ff_sublayer = layers.FeedForwardLayer(
          d_ff=self.config.d_ff,
          dropout_rate=self.config.dropout_rate,
          dtype=self.config.dtype,
          name=f"feed_forward_{layer}")

    return ff_sublayer

  def _is_attention_layer(self, layer):
    """Returns true if the current layer should be an attention layer."""
    if self.config.model_arch == ModelArchitecture.BERT.name:
      return True
    return _is_replacement_layer(layer, self.config.num_layers,
                                 self.config.num_attention_layers,
                                 self.config.attention_layout)

  def _is_moe_layer(self, layer):
    """Returns true if the current layer should be an MoE layer."""
    return _is_replacement_layer(layer, self.config.num_layers,
                                 self.config.num_moe_layers,
                                 self.config.moe_layout)

  def _init_routers(self, layer):
    """Returns mask and scatter routers, only one of which will be non-empty."""
    if self.config.dispatch_algorithm == DispatchAlgorithm.MASK_TOKENS_CHOOSE:
      return routing.TokensChooseMaskedRouter(
          router_weights=routing.RouterWeights(name=f"router_weights_{layer}"),
          jitter_noise=self.config.jitter_noise,
          num_selected_experts=self.config.num_selected_experts,
          batch_prioritized_routing=self.config.batch_prioritized_routing,
          dtype=layers.truncated_dtype())
    elif (self.config.dispatch_algorithm ==
          DispatchAlgorithm.SCATTER_TOKENS_CHOOSE):
      return routing.TokensChooseScatterRouter(
          router_weights=routing.RouterWeights(name=f"router_weights_{layer}"),
          jitter_noise=self.config.jitter_noise,
          num_selected_experts=self.config.num_selected_experts,
          batch_prioritized_routing=self.config.batch_prioritized_routing,
          dtype=layers.truncated_dtype())
    elif (self.config.dispatch_algorithm ==
          DispatchAlgorithm.MASK_EXPERTS_CHOOSE):
      return routing.ExpertsChooseMaskedRouter(
          router_weights=routing.RouterWeights(name=f"router_weights_{layer}"),
          jitter_noise=self.config.jitter_noise,
          dtype=layers.truncated_dtype())
    else:
      raise ValueError(
          f"Unrecognized dispatch_algorithm: {self.config.dispatch_algorithm}")


def _is_replacement_layer(layer, num_layers, num_replace_layers,
                          layer_layout):
  """Returns true if, per layer_layout, current layer should be replaced."""
  if layer_layout == LayerLayout.BOTTOM:
    return layer < num_replace_layers
  elif layer_layout == LayerLayout.MIDDLE:
    return (num_layers - num_replace_layers <= 2 * layer <
            num_layers + num_replace_layers)
  elif layer_layout == LayerLayout.MIXED and num_replace_layers > 0:
    return layer % (num_layers // num_replace_layers) == 0
  elif layer_layout == LayerLayout.TOP:
    return layer >= num_layers - num_replace_layers
  else:
    return False


class PreTrainingModel(nn.Module):
  """Masked Language Modelling and Next-Sentence Prediction pre-training model.

  Attributes:
    config: Model specification.
  """
  config: ml_collections.FrozenConfigDict

  @nn.compact
  def __call__(self,
               input_ids,
               type_ids,
               masked_lm_positions,
               masked_lm_labels,
               masked_lm_weights,
               next_sentence_labels,
               deterministic = False):
    """Applies pre-training model on inputs.

    Args:
      input_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] tokenized inputs.
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] Ids partitioning input into
        different types.
      masked_lm_positions: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] indices
        indicating which inputs are masked.
      masked_lm_labels: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] true labels
        for masked inputs.
      masked_lm_weights: <float>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] relative
        weighting for masked inputs.
      next_sentence_labels: <int>[BATCH_SIZE] Labels for next sentence
        prediction task.
      deterministic: Whether to apply dropout to input.

    Returns:
      Loss and metrics for given inputs.
    """
    encoder_output = EncoderModel(
        self.config, name="encoder")(
            input_ids, type_ids, deterministic=deterministic)

    masked_lm_output = layers.gather(encoder_output.sequence_output,
                                     masked_lm_positions)
    masked_lm_output = nn.DenseGeneral(
        self.config.d_emb,
        use_bias=True,
        dtype=self.config.dtype,
        kernel_init=default_kernel_init,
        name="predictions_dense")(
            masked_lm_output)
    masked_lm_output = nn.gelu(masked_lm_output)
    masked_lm_output = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON,
        dtype=self.config.dtype,
        name="predictions_layer_norm")(
            masked_lm_output)
    masked_lm_logits = layers.OutputProjection(
        kernel=self._get_embedding_table(), name="predictions_output")(
            masked_lm_output)

    next_sentence_logits = layers.OutputProjection(  # pytype: disable=wrong-arg-types  # jax-types
        n_out=2, kernel_init=default_kernel_init, name="classification")(
            encoder_output.pooled_output)

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
    next_sentence_labels: <float>[BATCH_SIZE] true labels for NSP task.

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
    return loss.sum(), normalizing_factor  # pytype: disable=bad-return-type  # jax-ndarray

  masked_lm_correct = jnp.sum(
      (masked_lm_logits.argmax(-1) == masked_lm_labels.ravel()) *
      masked_lm_weights.ravel())
  masked_lm_loss, masked_lm_normalization = _compute_weighted_cross_entropy(
      masked_lm_logits,
      masked_lm_labels.ravel(),
      weights=masked_lm_weights.ravel())
  masked_lm_total = masked_lm_weights.sum()

  next_sentence_loss, num_next_sentence_labels = _compute_weighted_cross_entropy(
      next_sentence_logits, next_sentence_labels.ravel())
  next_sentence_correct = jnp.sum(
      next_sentence_logits.argmax(-1) == next_sentence_labels.ravel())

  return PretrainingStats(masked_lm_loss, next_sentence_loss, masked_lm_correct,  # pytype: disable=wrong-arg-types  # jax-ndarray
                          masked_lm_normalization, masked_lm_total,
                          next_sentence_correct, num_next_sentence_labels)


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
      type_ids,
      labels = None,
      deterministic = False):
    """Applies model for sequence classification.

    Args:
      input_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] tokenized inputs.
      type_ids: <int>[BATCH_SIZE, MAX_SEQ_LENGTH] Ids partitioning input into
        different types.
      labels: True labels associated with inputs. Generally only required for
        training. Shape depends on task type:
        * Classification: <int>[BATCH_SIZE],
        * Regression: <float>[BATCH_SIZE].
      deterministic: Whether to apply dropout to input.

    Returns:
      * If labels supplied (training mode): Model loss and metrics.
      * If no labels supplied (prediction / evaluation mode): Logits with shape
        <float>[BATCH_SIZE, n_classes].
    """
    encoder_output = EncoderModel(
        self.config, name="encoder")(
            input_ids, type_ids, deterministic=deterministic)

    # All other classification and regression tasks use the pooled output.
    output = encoder_output.pooled_output
    # TODO(jamesleethorp): For WiC, the original SuperGLUE paper
    #  (https://arxiv.org/abs/1905.00537) concatenates the "CLS" and "word"
    #  output representations. We only use the pooled output.

    logits = layers.OutputProjection(  # pytype: disable=wrong-arg-types  # jax-types
        n_out=self.n_classes,
        kernel_init=default_kernel_init,
        name="classification")(
            output)

    if labels is None:
      # Code path used during evaluation or prediction; metrics can be computed
      # from logits by the caller.
      return logits

    # Code path used during training.
    if (self.config.dataset_name == "glue/stsb" or  # Regression task
        self.config.dataset_name == "super_glue/copa" or  # "Regression" task
        self.config.dataset_name == "super_glue/record"):  # "Regression" task
      # Logits have shape: [BATCH_SIZE, 1].
      per_example_loss = jnp.sum((logits[Ellipsis, 0] - labels)**2, axis=-1)
      batch_loss = jnp.mean(per_example_loss)
      return ClassificationStats(batch_loss=batch_loss, num_labels=labels.size)  # pytype: disable=wrong-arg-types  # jnp-type

    else:  # Classification task
      # Logits have shape: [BATCH_SIZE, self.n_classes].
      logits = nn.log_softmax(logits, axis=-1)
      per_example_loss = -jnp.sum(
          onehot(labels, logits.shape[-1]) * logits, axis=-1)
      batch_loss = jnp.mean(per_example_loss)
      correct_predictions = jnp.sum(logits.argmax(-1) == labels)
      return ClassificationStats(  # pytype: disable=wrong-arg-types  # jnp-type
          batch_loss=batch_loss,
          num_labels=labels.size,
          correct_predictions=correct_predictions)
