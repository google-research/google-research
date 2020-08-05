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

"""CuBERT finetuning library for variable-misuse localization and repair.

This module implements the pointer model for joint classification, localization
and repair from the following paper on top of CuBERT:
Neural Program Repair by Jointly Learning to Localize and Repair [ICLR'19]
"""

from typing import Callable, Dict, Optional, Text, Tuple

from bert import modeling
from bert import optimization
import tensorflow as tf

from tensorflow import contrib


def create_original_varmisuse_model(
    bert_config,
    is_training,
    enable_sequence_masking,
    input_ids,
    input_mask,
    segment_ids,
    candidate_mask,
    target_mask,
    error_location_mask,
    use_one_hot_embeddings,
    multi_head_count = 2,
):
  """Creates a two-headed pointer model."""

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_sequence = model.get_sequence_output()

  final_sequence_shape = modeling.get_shape_list(final_sequence,
                                                 expected_rank=3)
  batch_size, sequence_length, hidden_size = final_sequence_shape

  cls_output = model.get_pooled_output()

  # Calculate pointer probabilities as the attention vector over program tokens.
  # Pointer network equations:
  # (1) M = tanh(Y * Wy_extend + h_extend * Wh_extend)
  # (2) multi_headed_alpha = softmax(M * w_extend)
  # Vector shapes:
  #  (1) M:     [batch_size, sequence_length, hidden_size]
  #  (2) Wy:    [hidden_size, hidden_size]
  #  (3) Wh:    [hidden_size, hidden_size]
  #  (4) h:     [batch_size, hidden_size]
  #  (5) Y:     [batch_size, sequence_length, hidden_size]
  #  (6) w:     [hidden_size, multi_head_count]
  #  (7) multi_headed_alpha: [batch_size, sequence_length, multi_head_count]
  #  (8) Wy_extend: Wy extended to [batch_size, hidden_size, hidden_size]
  #  (9) Wh_extend: Wh extended to [batch_size, hidden_size, hidden_size]
  # (10) h_extend: h extended to [batch_size, sequence_length, hidden_size]
  # (11) w_extend: w extended to [batch_size, hidden_size, multi_head_count]

  wy = tf.get_variable(
      "Wy",
      shape=[hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=contrib.layers.xavier_initializer())
  wh = tf.get_variable(
      "Wh",
      shape=[hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=contrib.layers.xavier_initializer())
  w = tf.get_variable(
      "w",
      shape=[hidden_size, multi_head_count],
      dtype=tf.float32,
      initializer=contrib.layers.xavier_initializer())

  # Dimensions: [batch_size, hidden_size, hidden_size]
  wy_extend = tf.tile(tf.expand_dims(wy, 0), [batch_size, 1, 1])
  # Dimensions: [batch_size, hidden_size, hidden_size]
  wh_extend = tf.tile(tf.expand_dims(wh, 0), [batch_size, 1, 1])
  # Dimensions: [batch_size, sequence_length, hidden_size]
  cls_output_extend = tf.tile(
      tf.expand_dims(cls_output, 1), [1, sequence_length, 1])

  candidate_mask_expanded = tf.expand_dims(candidate_mask, 2)
  if enable_sequence_masking:
    # Mask sequence using `candidate_mask`.
    candidates_mask_extend = tf.tile(candidate_mask_expanded,
                                     [1, 1, hidden_size])
    final_sequence_masked = tf.multiply(final_sequence,
                                        tf.to_float(candidates_mask_extend))
    m = tf.tanh(
        tf.matmul(final_sequence_masked, wy_extend) +
        tf.matmul(cls_output_extend, wh_extend))
  else:
    m = tf.tanh(
        tf.matmul(final_sequence, wy_extend) +
        tf.matmul(cls_output_extend, wh_extend))

  # Dimension: [batch_size, hidden_size, multi_head_count]
  w_extend = tf.tile(tf.expand_dims(w, 0), [batch_size, 1, 1])

  # Dimension: [batch_size, sequence_length, multi_head_count]
  logits = tf.matmul(m, w_extend)

  # Dimension: [batch_size, sequence_length, multi_head_count]
  candidates_mask_extend_to_heads = tf.tile(candidate_mask_expanded,
                                            [1, 1, multi_head_count])

  # Mask logits using `candidate_mask`.
  logits_masked = tf.multiply(
      logits, tf.to_float(candidates_mask_extend_to_heads))
  probabilities = tf.nn.softmax(logits_masked, axis=1)

  location_probabilities, repair_probabilities = tf.unstack(
      probabilities, axis=2)

  def compute_loss(labels, probabilities):
    return -tf.reduce_sum(
        tf.multiply(tf.to_float(labels),
                    tf.log(tf.clip_by_value(probabilities, 1e-10, 1.0))),
        axis=1)

  localization_loss = compute_loss(error_location_mask,
                                   location_probabilities)
  repair_loss = compute_loss(target_mask, repair_probabilities)

  per_example_loss = localization_loss + repair_loss

  loss = tf.reduce_mean(per_example_loss)

  return loss, per_example_loss, logits_masked, probabilities


def model_fn_builder(
    bert_config,
    enable_sequence_masking,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings
):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(
      features,
      labels,
      mode,
      params
  ):
    """The `model_fn` for TPUEstimator."""

    # The function signature is fixed as part of the estimator interface.
    # We pass task-specific labels as part of `features` and hence `labels` is
    # unused. `params` is for runtime parameters passed around by the estimator
    # framework and they are not used by us.
    # The unused parameters are deleted below.
    del labels, params

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    candidate_mask = features["candidate_mask"]
    error_location_mask = features["error_location_mask"]
    target_mask = features["target_mask"]

    sequence_length = tf.shape(input_ids)[1]

    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(input_ids)[0], dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, _, probabilities) = (
        create_original_varmisuse_model(
            bert_config=bert_config,
            is_training=is_training,
            enable_sequence_masking=enable_sequence_masking,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            candidate_mask=candidate_mask,
            target_mask=target_mask,
            error_location_mask=error_location_mask,
            use_one_hot_embeddings=use_one_hot_embeddings))

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names) = (
          modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint))
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
      return output_spec

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(
          per_example_loss,
          probabilities,
          error_location_mask,
          target_mask,
          is_real_example):
        """Metric function."""

        buggy_mask = tf.equal(error_location_mask[:, 0], 0)
        non_buggy_mask = tf.logical_not(buggy_mask)

        location_probabilities, repair_probabilities = tf.unstack(
            probabilities, axis=2)
        predicted_error_locations = tf.argmax(
            location_probabilities, axis=1, output_type=tf.int32)
        predicted_repair_locations = tf.argmax(
            repair_probabilities, axis=1, output_type=tf.int32)

        non_buggy_predictions = tf.equal(predicted_error_locations, 0)

        predicted_error_locations_one_hot = tf.one_hot(
            predicted_error_locations, sequence_length, dtype=tf.int32)
        predicted_repair_locations_one_hot = tf.one_hot(
            predicted_repair_locations, sequence_length, dtype=tf.int32)

        classification_accuracy = tf.metrics.accuracy(
            labels=non_buggy_mask,
            predictions=non_buggy_predictions,
            weights=is_real_example)

        true_positive_rate = tf.metrics.accuracy(
            labels=non_buggy_mask,
            predictions=non_buggy_predictions,
            weights=is_real_example * tf.cast(non_buggy_mask, tf.float32))

        correct_location_predictions = tf.reduce_sum(
            tf.multiply(
                predicted_error_locations_one_hot, error_location_mask), axis=1)
        # We can have more than one valid repair locations, so `target_mask`
        # can have multiple ones in it. The following calculation yields 1
        # if the predicted repair location is one of the valid repair locations.
        correct_repair_predictions = tf.reduce_sum(
            tf.multiply(
                predicted_repair_locations_one_hot, target_mask), axis=1)
        correct_localization_repair_predictions = (
            correct_location_predictions * correct_repair_predictions)

        localization_accuracy = tf.metrics.accuracy(
            labels=tf.cast(buggy_mask, tf.int32),
            predictions=correct_location_predictions,
            weights=is_real_example * tf.cast(buggy_mask, tf.float32))

        repair_accuracy = tf.metrics.accuracy(
            labels=tf.cast(buggy_mask, tf.int32),
            predictions=correct_repair_predictions,
            weights=is_real_example * tf.cast(buggy_mask, tf.float32))

        localization_repair_accuracy = tf.metrics.accuracy(
            labels=tf.cast(buggy_mask, tf.int32),
            predictions=correct_localization_repair_predictions,
            weights=is_real_example * tf.cast(buggy_mask, tf.float32))

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

        return {
            "eval_accuracy_classification": classification_accuracy,
            "eval_true_positive_rate": true_positive_rate,
            "eval_accuracy_localization": localization_accuracy,
            "eval_accuracy_repair": repair_accuracy,
            "eval_accuracy_localization_repair": localization_repair_accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, probabilities, error_location_mask,
                       target_mask, is_real_example])
      output_spec = contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
      return output_spec

    else:
      output_spec = contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
      return output_spec

  return model_fn
