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

"""Defines loss layers and Felix models."""
from typing import Union

import numpy as np
from official.modeling import activations
from official.nlp.bert import configs
from official.nlp.modeling import losses
from official.nlp.modeling import models
from official.nlp.modeling import networks
import tensorflow as tf

from felix import felix_tagger


class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def _add_metrics(self, lm_output,
                   lm_labels,
                   lm_label_weights,
                   lm_example_loss):
    """Adds metrics.

    Args:
      lm_output: [batch_size, max_predictions_per_seq, vocab_size] tensor with
        language model logits.
      lm_labels: [batch_size, max_predictions_per_seq] tensor with gold outputs.
      lm_label_weights: [batch_size, max_predictions_per_seq] tensor with
        per-token weights.
      lm_example_loss: scalar loss.
    """
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    numerator = tf.reduce_sum(accuracy * lm_label_weights)
    denominator = tf.reduce_sum(lm_label_weights)
    masked_lm_accuracy = tf.math.divide_no_nan(numerator, denominator)

    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    masked_lm_predictions = tf.argmax(lm_output, axis=-1, output_type=tf.int32)
    per_token_accuracy = tf.equal(lm_labels, masked_lm_predictions)
    sent_level_accuracy = tf.cast(
        tf.reduce_all(
            tf.logical_or(per_token_accuracy, (lm_label_weights <= 0)), axis=1),
        tf.float32)
    self.add_metric(sent_level_accuracy, name='exact_match', aggregation='mean')

  def call(self, lm_output, lm_label_ids, lm_label_weights):
    """Implements call() for the layer.

    Args:
      lm_output: [batch_size, max_predictions_per_seq, vocab_size] tensor with
        language model logits.
      lm_label_ids: [batch_size, max_predictions_per_seq] tensor with gold
        outputs.
      lm_label_weights: [batch_size, max_predictions_per_seq] tensor with
        per-token weights.

    Returns:
      final_loss: scalar MLM loss.
    """
    lm_label_weights = tf.cast(lm_label_weights, tf.float32)
    lm_output = tf.cast(lm_output, tf.float32)

    mask_label_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=lm_label_ids,
        predictions=lm_output,
        weights=lm_label_weights,
        from_logits=True)

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      mask_label_loss)
    return mask_label_loss

  def get_config(self):
    return self._config


def get_insertion_model(bert_config,
                        seq_length,
                        max_predictions_per_seq,
                        is_training = True):
  """Returns a Felix MLM insertion model.

  Args:
      bert_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of masked tokens in sequence.
      is_training: Will the model be trained or is it inference time.

  Returns:
      Felix MLM insertion model as well as core BERT submodel from which to save
      weights after training.
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
  masked_lm_positions = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_positions',
      dtype=tf.int32)

  bert_encoder = networks.BertEncoder(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=activations.gelu,
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      sequence_length=seq_length,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))

  pretrainer_model = models.BertPretrainerV2(
      encoder_network=bert_encoder,
      mlm_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))

  felix_inputs = [
      input_word_ids,
      input_mask,
      input_type_ids,
      masked_lm_positions,
  ]
  outputs = pretrainer_model(felix_inputs)
  if is_training:
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32)
    output_loss = BertPretrainLossAndMetricLayer()(outputs['mlm_logits'],
                                                   masked_lm_ids,
                                                   masked_lm_weights)
    felix_inputs.append(masked_lm_ids)
    felix_inputs.append(masked_lm_weights)
    keras_model = tf.keras.Model(inputs=felix_inputs, outputs=output_loss)
  else:
    keras_model = tf.keras.Model(
        inputs=felix_inputs, outputs=outputs['mlm_logits'])

  return keras_model, bert_encoder


class FelixTagLoss(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for felix tagger."""

  def __init__(self, use_pointing = True, pointing_weight = 1.0):
    self._use_pointing = use_pointing
    self._pointing_weight = pointing_weight
    super(FelixTagLoss, self).__init__()

  def _add_metrics(self,
                   tag_logits,
                   tag_labels,
                   tag_loss,
                   input_mask,
                   labels_mask,
                   total_loss,
                   point_logits = None,
                   point_labels = None,
                   point_loss = None):
    """Adds metrics.

    Args:
      tag_logits: [batch_size, seq_length, vocab_size] tensor with tag logits.
      tag_labels: [batch_size, seq_length] tensor with gold outputs.
      tag_loss: [batch_size, seq_length] scalar loss for tagging model.
      input_mask: [batch_size, seq_length] tensor with mask (1s or 0s).
      labels_mask: [batch_size, seq_length] mask for labels, may be a binary
        mask or a weighted float mask.
      total_loss:   scalar loss for whole model.
      point_logits: [batch_size, seq_length, seq_length] optional tensor with
        point logits.
      point_labels: [batch_size, seq_length] optional tensor with gold outputs.
      point_loss: : optional scalar loss for pointing model.
    """
    self.add_metric(tag_loss, name='tag_loss', aggregation='mean')

    tag_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        tag_labels, tag_logits)
    numerator = tf.reduce_sum(tag_accuracy * tf.cast(labels_mask, tf.float32))
    denominator = tf.reduce_sum(tf.cast(labels_mask, tf.float32))
    tag_accuracy = tf.math.divide_no_nan(numerator, denominator)
    self.add_metric(tag_accuracy, name='tag_accuracy', aggregation='mean')

    tag_predictions = tf.argmax(tag_logits, axis=-1, output_type=tf.int32)
    tag_acc = tf.equal(tag_labels, tag_predictions)
    sent_level_tag_accuracy = tf.cast(
        tf.reduce_all(tf.logical_or(tag_acc, (labels_mask <= 0)), axis=1),
        tf.float32)
    self.add_metric(
        sent_level_tag_accuracy, name='tag_exact_match', aggregation='mean')

    if not self._use_pointing:
      self.add_metric(
          sent_level_tag_accuracy, name='exact_match', aggregation='mean')
      self.add_metric(total_loss, name='total_loss', aggregation='mean')
      return
    else:
      self.add_metric(point_loss, name='point_loss', aggregation='mean')

      point_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
          point_labels, point_logits)
      numerator = tf.reduce_sum(point_accuracy *
                                tf.cast(input_mask, tf.float32))
      denominator = tf.reduce_sum(tf.cast(input_mask, tf.float32))
      point_accuracy = tf.math.divide_no_nan(numerator, denominator)
      self.add_metric(point_accuracy, name='point_accuracy', aggregation='mean')

      point_predictions = tf.argmax(point_logits, axis=-1, output_type=tf.int32)
      point_acc = tf.equal(point_labels, point_predictions)
      sent_level_point_accuracy = tf.cast(
          tf.reduce_all(tf.logical_or(point_acc, (input_mask <= 0)), axis=1),
          tf.float32)
      self.add_metric(
          sent_level_point_accuracy,
          name='point_exact_match',
          aggregation='mean')

      sent_level_accuracy = tf.cast((tf.logical_and(
          tf.cast(sent_level_point_accuracy, tf.bool),
          tf.cast(sent_level_tag_accuracy, tf.bool))), tf.float32)

      self.add_metric(
          sent_level_accuracy, name='exact_match', aggregation='mean')

      self.add_metric(total_loss, name='total_loss', aggregation='mean')

  def call(self,
           tag_logits,
           tag_labels,
           input_mask,
           labels_mask,
           point_logits = None,
           point_labels = None):
    """Implements call() for the layer.

    Args:
      tag_logits: [batch_size, seq_length, vocab_size] tensor with tag logits.
      tag_labels: [batch_size, seq_length] tensor with gold outputs.
      input_mask: [batch_size, seq_length]tensor with mask (1s or 0s).
      labels_mask: [batch_size, seq_length] mask for labels, may be a binary
        mask or a weighted float mask.
      point_logits: [batch_size, seq_length, seq_length] optional tensor with
        point logits.
      point_labels: [batch_size, seq_length] optional tensor with gold outputs.

    Returns:
      Scalar loss of the model.
    """
    tag_logits = tf.cast(tag_logits, tf.float32)
    labels_mask = tf.cast(labels_mask, tf.float32) * tf.math.reduce_sum(
        tf.cast(input_mask, tf.float32), axis=-1, keepdims=True)
    tag_logits_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=tag_labels,
        predictions=tag_logits,
        weights=tf.cast(labels_mask, tf.float32),
        from_logits=True)
    if self._use_pointing:
      point_logits_loss = losses.weighted_sparse_categorical_crossentropy_loss(
          labels=point_labels,
          predictions=point_logits,
          weights=tf.cast(input_mask, tf.float32),
          from_logits=True)
      total_loss = tag_logits_loss + tf.cast(
          tf.constant(self._pointing_weight), tf.float32) * point_logits_loss
      self._add_metrics(tag_logits, tag_labels, tag_logits_loss, input_mask,
                        labels_mask, total_loss, point_logits, point_labels,
                        point_logits_loss)
    else:
      total_loss = tag_logits_loss
      self._add_metrics(tag_logits, tag_labels, tag_logits_loss, input_mask,
                        labels_mask, total_loss)

    return total_loss

  def get_config(self):
    return self._config


def get_tagging_model(bert_config,
                      seq_length,
                      use_pointing = True,
                      pointing_weight = 1.0,
                      is_training = True):
  """Returns model to be used for pre-training.

  Args:
      bert_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      use_pointing: If FELIX should use a pointer (reordering) model.
      pointing_weight: How much to weigh the pointing loss, in contrast to
        tagging loss. Note, if pointing is set to false this is ignored.
      is_training: Will the model be trained or is it inferance time.

  Returns:
      Felix model as well as core BERT submodel from which to save
      weights after pretraining.
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)

  bert_encoder = networks.BertEncoder(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=activations.gelu,
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      sequence_length=seq_length,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))

  felix_model = felix_tagger.FelixTagger(
      bert_encoder,
      seq_length=seq_length,
      use_pointing=use_pointing,
      bert_config=bert_config,
      is_training=is_training)
  felix_inputs = [input_word_ids, input_mask, input_type_ids]
  if is_training:
    edit_tags = tf.keras.layers.Input(
        shape=(seq_length,), name='edit_tags', dtype=tf.int32)

    felix_inputs.append(edit_tags)
    felix_outputs = felix_model(felix_inputs)
    labels_mask = tf.keras.layers.Input(
        shape=(seq_length,), name='labels_mask', dtype=tf.float32)
    felix_inputs.append(labels_mask)
    if use_pointing:
      pointers = tf.keras.layers.Input(
          shape=(seq_length,), name='pointers', dtype=tf.int32)
      tag_logits, pointing_logits = felix_outputs
      felix_inputs.append(pointers)
    else:
      tag_logits = felix_outputs[0]
      pointing_logits = None
      pointers = None
    loss_function = FelixTagLoss(use_pointing, pointing_weight)
    felix_tag_loss = loss_function(tag_logits, edit_tags, input_mask,
                                   labels_mask, pointing_logits, pointers)
    keras_model = tf.keras.Model(inputs=felix_inputs, outputs=felix_tag_loss)
  else:
    felix_inputs = [input_word_ids, input_mask, input_type_ids]
    felix_outputs = felix_model(felix_inputs)
    if use_pointing:
      tag_logits, pointing_logits = felix_outputs
      keras_model = tf.keras.Model(
          inputs=felix_inputs, outputs=[tag_logits, pointing_logits])
    else:
      tag_logits = felix_outputs[0]
      pointing_logits = None
      keras_model = tf.keras.Model(inputs=felix_inputs, outputs=tag_logits)
  return keras_model, bert_encoder
