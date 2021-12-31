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

"""Various loss functions to be used by ReadItTwice model."""

from typing import Optional, Text

import dataclasses
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.layers import tensor_utils
from readtwice.layers import tpu_utils


class BatchSpanCrossEntropyLoss(tf.keras.layers.Layer):
  """Cross entropy loss for multiple correct answers across the whole batch.

  Loss is a negative log of correct answer probabilities' sum.
  See https://arxiv.org/abs/1710.10723 for additional details.

  For the future, we can consider implementing different loss functions
  as described here - https://arxiv.org/abs/2005.01898. Note that
  the current implementation corresponds to their H1-Document loss.
  """

  def call(self,
           logits,
           annotation_begins,
           annotation_ends,
           annotation_labels,
           block_ids,
           num_replicas=None,
           eps=0):
    """Calls the layer.

    Args:
      logits: <float32>[batch_size, main_seq_len, 2] Logits per position.
      annotation_begins: <int32>[batch_size, main_seq_len] Positions of
        beginnings of answer spans.
      annotation_ends: <int32>[batch_size, main_seq_len] Positions of endings of
        answer spans.
      annotation_labels: <int32>[batch_size, main_seq_len] Positions of labels
        of answer spans. Label is 0 when the span is a placeholder one (included
        only for padding purposes) and should be ignored.
      block_ids: <int32>[batch_size] Block IDs of every sample in the batch.
      num_replicas: Number of replicas to gather summaries from. If None
        (default) then cross-replicas summaries are not used.
      eps: <float> Small constant for numerical stability.

    Returns:
        total_loss: <float>
    """
    seq_length = tf.shape(logits)[1]

    # (1) Aggregate block_ids across global batch. Compute cross block mask.
    all_block_ids = block_ids
    if num_replicas:
      all_block_ids = tpu_utils.cross_replica_concat(
          tensor=all_block_ids,
          num_replicas=num_replicas,
          name='block_ids_concat')

    # [batch_size, global_batch_size]
    cross_blocks_eq_mask = tf.cast(
        tf.equal(
            tf.expand_dims(block_ids, 1), tf.expand_dims(all_block_ids, 0)),
        tf.float32)

    # (2) Apply softmax over all positions in the (global) batch
    # across the blocks with the same `block_id`.

    # [batch_size, seq_len, 2]
    probs = cross_batch_softmax(logits, cross_blocks_eq_mask, num_replicas)

    # (3) Prepare one-hot labels based on annotation begins and ends

    # [batch_size, seq_len, 1]
    annotation_begins_one_hot = _one_hot_multi(
        annotation_begins,
        annotation_labels > 0,
        seq_length,
    )
    # [batch_size, seq_len, 1]
    annotation_ends_one_hot = _one_hot_multi(
        annotation_ends,
        annotation_labels > 0,
        seq_length,
    )
    # [batch_size, seq_len, 2]
    one_hot_labels = tf.concat(
        [annotation_begins_one_hot, annotation_ends_one_hot], 2)

    # (4) Compute the probability of the current begin / end positions across
    # the blocks with the same `block_id`.

    # [batch_size, 2]
    correct_probs = tf.reduce_sum(probs * one_hot_labels, axis=1)
    if num_replicas:
      # [global_batch_size, 2]
      correct_probs = tpu_utils.cross_replica_concat(
          tensor=correct_probs,
          num_replicas=num_replicas,
          name='correct_probs_concat')

    # [batch_size, 2]
    correct_probs = tf.matmul(cross_blocks_eq_mask, correct_probs)

    # (5) Compute log probability. We allow cases when there are no correct
    # labels not only for the current sample, but for the whole document
    # across the whole batch. In that case the probability of the correct label
    # would be 0 and the loss would be infinite. Therefore, we just do not
    # compute loss on these documents.

    # [batch_size, 1]
    num_annotations_per_sample = tf.reduce_sum(
        annotation_labels, 1, keepdims=True)
    if num_replicas:
      # [global_batch_size, 1]
      num_annotations_per_sample = tpu_utils.cross_replica_concat(
          tensor=num_annotations_per_sample,
          num_replicas=num_replicas,
          name='num_annotations_per_sample_concat')

    # [batch_size, 1]
    num_annotations_per_doc = tf.matmul(
        cross_blocks_eq_mask, tf.cast(num_annotations_per_sample, tf.float32))
    # [batch_size, 2]
    doc_with_annotations_mask = tf.stop_gradient(
        tf.cast(tf.tile(num_annotations_per_doc > 0, [1, 2]), tf.float32))
    doc_without_annotations_mask = tf.stop_gradient(1 -
                                                    doc_with_annotations_mask)
    log_correct_probs = tf.log(
        correct_probs + eps +
        doc_without_annotations_mask) * doc_with_annotations_mask

    # (6) Divide by the number of blocks per block_id
    # If there are K blocks with the same block_id, then on step (4) we'll
    # compute loss for this document K times. So we need to divide it back by K.

    # [batch_size, 2]
    log_correct_probs /= tf.reduce_sum(cross_blocks_eq_mask, 1, keepdims=True)

    # (7) Sum over blocks and begin/end predictions

    loss = -tf.reduce_sum(log_correct_probs)
    return loss


def cross_batch_softmax(logits, cross_blocks_eq_mask, num_replicas=None):
  """Computes softmax across the whole (global) batch.

  The computations are independent with respect to the 3rd, innermost dimension.
  In case of the span prediction, the size of this dimension is K=2, which
  corresponds to beginings and ends of annotations.

  Args:
    logits: <float32>[batch_size, seq_len, K] Tensor of logits.
    cross_blocks_eq_mask: <float32>[batch_size, global_batch_size] The mask
      which indicates which samples in the batch have the same block IDs.
    num_replicas: Optional[int]. If provided the function performs computations
      over the global (multi-devices) batch. Should be equal to the number of
      devices.

  Returns:
      probs: <float32>[batch_size, seq_len, K]
  """
  # (1) Apply max-trick to improve softmax numerical stability.

  # [batch_size, K]
  max_logits_per_sample = tf.math.reduce_max(logits, axis=1)
  if num_replicas:
    # [global_batch_size, K]
    max_logits_per_sample = tpu_utils.cross_replica_concat(
        tensor=max_logits_per_sample,
        num_replicas=num_replicas,
        name='max_logits_per_sample_concat')
  # [1, global_batch_size, K]
  max_logits_per_sample = tf.expand_dims(max_logits_per_sample, 0)

  # [batch_size, global_batch_size, 1]
  one_minus_one_mask = 2 * tf.expand_dims(cross_blocks_eq_mask, 2) - 1
  # [batch_size, global_batch_size, K]
  masked_max_logits_per_sample = tf.minimum(max_logits_per_sample,
                                            one_minus_one_mask * np.inf)
  # [batch_size, K]
  max_logits_per_sample = tf.reduce_max(masked_max_logits_per_sample, axis=1)

  # [batch_size, seq_len, K]
  logits -= tf.expand_dims(max_logits_per_sample, 1)

  # (2) Take exponent
  unnormalized_probs = tf.exp(logits)

  # (3) Compute softmax's denominator (normalization constant)

  # [batch_size, K]
  softmax_denominator_per_sample = tf.math.reduce_sum(
      unnormalized_probs, axis=1)
  if num_replicas:
    # [global_batch_size, K]
    softmax_denominator_per_sample = tpu_utils.cross_replica_concat(
        tensor=softmax_denominator_per_sample,
        num_replicas=num_replicas,
        name='softmax_denominator_per_sample_concat')

  # [batch_size, K]
  softmax_denominator_per_sample = tf.matmul(cross_blocks_eq_mask,
                                             softmax_denominator_per_sample)

  # (4) Compute probabilities

  # [batch_size, seq_len, K]
  probs = unnormalized_probs / tf.expand_dims(softmax_denominator_per_sample, 1)
  return probs


def _one_hot_multi(values, mask, depth):
  # [batch_size, depth, 1]
  return tf.expand_dims(
      tf.minimum(
          tf.reduce_sum(
              tf.one_hot(values, depth=depth, dtype=tf.float32) *
              tf.cast(tf.expand_dims(mask, 2), tf.float32), 1), 1), 2)


class BatchCoreferenceResolutionLoss(tf.keras.layers.Layer):
  """Loss for the task whether items come from the same entities."""

  def __init__(self,
               apply_linear_layer = False,
               padding_id = 0,
               name = 'batch_coreference_resolution_loss',
               **kwargs):
    """Constructor for BatchCoreferenceResolutionLoss.

    Args:
      apply_linear_layer: bool. Whether to apply linear layer before computing
        scores via dot product.
      padding_id: int. padding ID. Loss will be ignored for these samples.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: The config is invalid.
    """
    super(BatchCoreferenceResolutionLoss, self).__init__(name=name, **kwargs)
    self.apply_linear_layer = apply_linear_layer
    self.padding_id = padding_id

  def build(self, input_shape):
    """Keras build function.

    Args:
      input_shape: TensorShape of the input.
    """
    hidden_size = input_shape.as_list()[-1]
    if hidden_size is None:
      raise ValueError('`input_shape[-1]` must be statically known.')
    if self.apply_linear_layer:
      self.linear_fn = tf.keras.layers.Dense(hidden_size, use_bias=False)
    self.bias_term = self.add_weight(
        name='batch_coref_loss_bias',
        shape=(1),
        initializer='zeros',
        trainable=True)

    super(BatchCoreferenceResolutionLoss, self).build(input_shape)

  def call(self,
           item_states,
           item_ids,
           global_item_states,
           global_item_ids,
           labels_mask=None,
           labels_weight=None):
    """Calls the layer.

    Args:
      item_states: <float32>[batch_size, hidden_size]
      item_ids: <int32>[batch_size, hidden_size]
      global_item_states: <float32>[global_batch_size, hidden_size]
      global_item_ids: <int32>[global_batch_size, hidden_size]
      labels_mask: <int32>[batch_size, global_batch_size]
      labels_weight: <float32>[batch_size, global_batch_size]

    Returns:
        total_loss: <float>
    """
    # [batch_size, 1]
    item_ids_expanded = tf.expand_dims(item_ids, 1)
    # [1, global_batch_size]
    global_item_ids_expanded = tf.expand_dims(global_item_ids, 0)

    # Positive labels when IDs are the same
    # [batch_size, global_batch_size]
    labels = tf.equal(item_ids_expanded, global_item_ids_expanded)
    if labels_mask is not None:
      labels = tf.logical_and(labels, labels_mask)

    # In two cases the loss is ignored (label_weight is 0):
    # (1) Either of IDs is the padding ID
    # (2) Loss is computed when comparisng a sample to itself
    both_ids_are_not_padding = tf.logical_and(
        tf.not_equal(item_ids_expanded, self.padding_id),
        tf.not_equal(global_item_ids_expanded, self.padding_id))
    if labels_weight is None:
      labels_weight = tf.cast(both_ids_are_not_padding, tf.float32)
    else:
      labels_weight = labels_weight * tf.cast(both_ids_are_not_padding,
                                              tf.float32)
    # Hacky way to tell if samples are exactly the same --
    # their IDs are the same and their states are approximately the same.
    samples_are_the_same = tf.logical_and(
        tf.less(
            tf.norm(
                tf.expand_dims(item_states, 1) -
                tf.expand_dims(global_item_states, 0),
                axis=2), 1e-5), labels)
    # [batch_size, global_batch_size]
    labels_weight = (
        labels_weight * (1 - tf.cast(samples_are_the_same, tf.float32)))

    # [batch_size, global_batch_size]
    labels = tf.stop_gradient(tf.cast(labels, tf.float32))
    labels_weight = tf.stop_gradient(tf.cast(labels_weight, tf.float32))

    if self.apply_linear_layer:
      item_states = self.linear_fn(item_states)

    # [batch_size, global_batch_size]
    logits = tf.matmul(item_states, global_item_states, transpose_b=True)
    logits += self.bias_term

    # [batch_size, global_batch_size]
    loss_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss_per_sample *= labels_weight
    # Here we compute mean because otherwise the loss becomes too large
    loss_per_sample = tf.reduce_sum(loss_per_sample, 1)
    loss_per_sample /= (tf.reduce_sum(labels_weight, 1) + 1e-5)
    return tf.reduce_sum(loss_per_sample)


@dataclasses.dataclass(frozen=True)
class LanguageModelOutput:
  """Outputs of LanguageModelLoss.

    loss: <float32>[] the overall loss
    mlm_loss_per_sample: <float32>[batch_size]
    mlm_accuracy_per_sample: <float32>[batch_size]
    mlm_weight_per_sample: <float32>[batch_size]
  """
  loss: tf.Tensor
  mlm_predictions: tf.Tensor
  mlm_loss_per_sample: tf.Tensor
  mlm_accuracy_per_sample: tf.Tensor
  mlm_weight_per_sample: tf.Tensor
  mlm_loss_per_entity_sample: Optional[tf.Tensor] = None
  mlm_accuracy_per_entity_sample: Optional[tf.Tensor] = None
  mlm_weight_per_entity_sample: Optional[tf.Tensor] = None
  mlm_loss_per_non_entity_sample: Optional[tf.Tensor] = None
  mlm_accuracy_per_non_entity_sample: Optional[tf.Tensor] = None
  mlm_weight_per_non_entity_sample: Optional[tf.Tensor] = None


class LanguageModelLoss(tf.keras.layers.Layer):
  """Loss for the (masked) language model."""

  def __init__(self,
               output_weights,
               hidden_size,
               name = 'language_model_loss',
               activation = 'gelu',
               initializer_range = 0.02,
               **kwargs):
    """Constructor for LanguageModelLoss.

    Args:
      output_weights: Embeddings table
      hidden_size: Input size
      name: (Optional) name of the layer.
      activation: The non-linear activation function (function or string) in the
        1 layer MLP decoder. Default is "gelu".
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: Shape of the output_weights is not statically known.
    """
    super(LanguageModelLoss, self).__init__(name=name, **kwargs)
    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.output_weights = output_weights
    self.activation = activation
    self.initializer_range = initializer_range
    self.hidden_size = hidden_size

    self.vocab_size, self.embedding_size = tensor_utils.get_shape_list(
        self.output_weights, expected_rank=2, name='word embeddings table')
    if self.vocab_size is None:
      raise ValueError('`output_weights[0]` must be statically known.')

    self.linear_fn = tf.keras.layers.Dense(
        self.embedding_size,
        activation=tensor_utils.get_activation(self.activation),
        use_bias=True,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.initializer_range))
    self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001)

  def build(self, input_shape):
    """Keras build function.

    Args:
      input_shape: TensorShape of the input.
    """
    self.output_bias = self.add_weight(
        name='cls/predictions/output_bias',
        shape=(self.vocab_size),
        initializer='zeros',
        trainable=True)

  def call(self,
           input_tensor,
           label_ids,
           positions = None,
           label_weights = None,
           padding_token_id = None,
           mlm_is_entity_mask = None,
           mlm_is_not_entity_mask = None):
    """Get loss and log probs for the masked LM."""
    if padding_token_id is not None:
      pad_mask = tf.cast(tf.not_equal(label_ids, padding_token_id), tf.float32)
    if label_weights is not None:
      if padding_token_id is not None:
        label_weights *= pad_mask
    else:
      if padding_token_id is not None:
        label_weights = pad_mask
      else:
        label_weights = tf.ones_like(label_ids, tf.float32)

    if positions is not None:
      input_tensor = gather_indexes(input_tensor, positions)
    else:
      input_tensor = tf.reshape(input_tensor, [-1, self.hidden_size])
    input_tensor.set_shape([None, self.hidden_size])

    with tf.variable_scope('cls/predictions'):
      with tf.variable_scope('transform'):
        input_tensor = self.linear_fn(input_tensor)
        input_tensor = self.layer_norm(input_tensor)

      logits = tf.matmul(input_tensor, self.output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, self.output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      batch_size = tf.shape(label_ids)[0]
      mlm_labels_per_sample = tf.shape(label_ids)[1]
      label_ids_flattened = tf.reshape(label_ids, [-1])

      label_weights_flattened = tf.reshape(label_weights, [-1])

      one_hot_labels = tf.one_hot(
          label_ids_flattened, depth=self.vocab_size, dtype=tf.float32)

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
      mlm_predictions = tf.argmax(log_probs, axis=-1, output_type=tf.int32)
      loss = tf.reduce_sum(label_weights_flattened * per_example_loss) / (
          tf.reduce_sum(label_weights) + 1e-5)

      def weighted_sum_per_sample(values1, values2, weights):
        weights_per_sample = tf.reduce_sum(weights, 1)
        weights_denominator = weights_per_sample + 1e-5
        return (tf.reduce_sum(values1 * weights, 1) / weights_denominator,
                tf.reduce_sum(values2 * weights, 1) / weights_denominator,
                weights_per_sample)

      mlm_loss = tf.reshape(per_example_loss,
                            [batch_size, mlm_labels_per_sample])
      mlm_accuracy = tf.reshape(
          tf.cast(tf.equal(mlm_predictions, label_ids_flattened), tf.float32),
          [batch_size, mlm_labels_per_sample])
      (mlm_loss_per_sample, mlm_accuracy_per_sample,
       mlm_weight_per_sample) = weighted_sum_per_sample(mlm_loss, mlm_accuracy,
                                                        label_weights)

      if mlm_is_entity_mask is not None:
        (mlm_loss_per_entity_sample, mlm_accuracy_per_entity_sample,
         mlm_weight_per_entity_sample) = weighted_sum_per_sample(
             mlm_loss, mlm_accuracy, label_weights * mlm_is_entity_mask)
      else:
        mlm_loss_per_entity_sample = None
        mlm_accuracy_per_entity_sample = None
        mlm_weight_per_entity_sample = None

      if mlm_is_not_entity_mask is not None:
        (mlm_loss_per_non_entity_sample, mlm_accuracy_per_non_entity_sample,
         mlm_weight_per_non_entity_sample) = weighted_sum_per_sample(
             mlm_loss, mlm_accuracy, label_weights * mlm_is_not_entity_mask)
      else:
        mlm_loss_per_non_entity_sample = None
        mlm_accuracy_per_non_entity_sample = None
        mlm_weight_per_non_entity_sample = None

    return LanguageModelOutput(
        loss=loss,
        mlm_predictions=mlm_predictions,
        mlm_loss_per_sample=mlm_loss_per_sample,
        mlm_accuracy_per_sample=mlm_accuracy_per_sample,
        mlm_weight_per_sample=mlm_weight_per_sample,
        mlm_loss_per_entity_sample=mlm_loss_per_entity_sample,
        mlm_accuracy_per_entity_sample=mlm_accuracy_per_entity_sample,
        mlm_weight_per_entity_sample=mlm_weight_per_entity_sample,
        mlm_loss_per_non_entity_sample=mlm_loss_per_non_entity_sample,
        mlm_accuracy_per_non_entity_sample=mlm_accuracy_per_non_entity_sample,
        mlm_weight_per_non_entity_sample=mlm_weight_per_non_entity_sample)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = tensor_utils.get_shape_list(
      sequence_tensor, expected_rank=3, name='')
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor
