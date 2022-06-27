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

"""Input and model functions for ETC HotpotQA model."""
from typing import Dict

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from etcmodel import tensor_utils
from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models import multihop_utils as qa_input_utils
from etcmodel.models import optimization
from etcmodel.models.hotpotqa import generate_tf_examples_lib


def get_inference_name_to_features(long_seq_length, global_seq_length):
  """Returns the feature spec of the model."""
  name_to_features = {
      "long_token_ids":
          tf.FixedLenFeature([long_seq_length], tf.int64),
      "long_sentence_ids":
          tf.FixedLenFeature([long_seq_length], tf.int64),
      "long_paragraph_ids":
          tf.FixedLenFeature([long_seq_length], tf.int64),
      "long_paragraph_breakpoints":
          tf.FixedLenFeature([long_seq_length], tf.int64),
      "long_token_type_ids":
          tf.FixedLenFeature([long_seq_length], tf.int64),
      "global_token_ids":
          tf.FixedLenFeature([global_seq_length], tf.int64),
      "global_paragraph_breakpoints":
          tf.FixedLenFeature([global_seq_length], tf.int64),
      "global_token_type_ids":
          tf.FixedLenFeature([global_seq_length], tf.int64),
  }
  return name_to_features


def input_fn_builder(input_filepattern: str, long_seq_length: int,
                     global_seq_length: int, is_training: bool,
                     answer_encoding_method: str, drop_remainder: bool):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = get_inference_name_to_features(long_seq_length,
                                                    global_seq_length)
  if is_training:
    name_to_features.update({
        "supporting_facts": tf.FixedLenFeature([global_seq_length], tf.int64),
        "answer_types": tf.FixedLenFeature([], tf.int64)
    })
    if answer_encoding_method == "span":
      name_to_features.update({
          "answer_begins": tf.FixedLenFeature([], tf.int64),
          "answer_ends": tf.FixedLenFeature([], tf.int64),
      })
    else:
      name_to_features.update({
          "answer_bio_ids": tf.FixedLenFeature([long_seq_length], tf.int64),
      })

  def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      if name != "unique_ids":
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        example[name] = t
    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    filenames = tf.gfile.Glob(input_filepattern)
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      np.random.seed(1234)
      np.random.shuffle(filenames)
      # Assigns a subset of examples to each worker, as recommended in
      # in https://arxiv.org/pdf/1706.02677.pdf.
      tpu_context = params["context"]
      host_filenames = np.array_split(
          filenames, tpu_context.num_hosts)[tpu_context.current_host]
      host_filenames = host_filenames.tolist()
      tf.logging.info("Num input files on host %d of %d: %d",
                      tpu_context.current_host, tpu_context.num_hosts,
                      len(host_filenames))
      d = tf.data.Dataset.from_tensor_slices(host_filenames).repeat()
      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset, sloppy=True, cycle_length=4))
      d = d.shuffle(buffer_size=batch_size * 4)
    else:
      d = tf.data.TFRecordDataset(filenames)
    d = d.map(
        map_func=decode_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn


def build_model(etc_model_config: modeling.EtcConfig,
                features: Dict[str, tf.Tensor], flat_sequence: bool,
                is_training: bool, answer_encoding_method: str, use_tpu: bool,
                use_wordpiece: bool):
  """Build the ETC HotpotQA model."""
  long_token_ids = features["long_token_ids"]
  long_sentence_ids = features["long_sentence_ids"]
  long_paragraph_ids = features["long_paragraph_ids"]
  long_paragraph_breakpoints = features["long_paragraph_breakpoints"]
  long_token_type_ids = features["long_token_type_ids"]
  global_token_ids = features["global_token_ids"]
  global_paragraph_breakpoints = features["global_paragraph_breakpoints"]
  global_token_type_ids = features["global_token_type_ids"]

  model = modeling.EtcModel(
      config=etc_model_config,
      is_training=is_training,
      use_one_hot_relative_embeddings=use_tpu)

  model_inputs = dict(
      token_ids=long_token_ids,
      global_token_ids=global_token_ids,
      segment_ids=long_token_type_ids,
      global_segment_ids=global_token_type_ids)

  cls_token_id = (
      generate_tf_examples_lib
      .SENTENCEPIECE_DEFAULT_GLOBAL_TOKEN_IDS["CLS_TOKEN_ID"])
  if use_wordpiece:
    cls_token_id = (
        generate_tf_examples_lib
        .WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS["CLS_TOKEN_ID"])

  model_inputs.update(
      qa_input_utils.make_global_local_transformer_side_inputs(
          long_paragraph_breakpoints=long_paragraph_breakpoints,
          long_paragraph_ids=long_paragraph_ids,
          long_sentence_ids=long_sentence_ids,
          global_paragraph_breakpoints=global_paragraph_breakpoints,
          local_radius=etc_model_config.local_radius,
          relative_pos_max_distance=etc_model_config.relative_pos_max_distance,
          use_hard_g2l_mask=etc_model_config.use_hard_g2l_mask,
          ignore_hard_g2l_mask=tf.cast(
              tf.equal(global_token_ids, cls_token_id),
              dtype=long_sentence_ids.dtype),
          flat_sequence=flat_sequence,
          use_hard_l2g_mask=etc_model_config.use_hard_l2g_mask).to_dict(
              exclude_none_values=True))

  long_output, global_output = model(**model_inputs)

  batch_size, long_seq_length, long_hidden_size = tensor_utils.get_shape_list(
      long_output, expected_rank=3)
  _, global_seq_length, global_hidden_size = tensor_utils.get_shape_list(
      global_output, expected_rank=3)

  long_output_matrix = tf.reshape(
      long_output, [batch_size * long_seq_length, long_hidden_size])
  global_output_matrix = tf.reshape(
      global_output, [batch_size * global_seq_length, global_hidden_size])

  # Get the logits for the supporting facts predictions.
  supporting_facts_output_weights = tf.get_variable(
      "supporting_facts_output_weights", [1, global_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  supporting_facts_output_bias = tf.get_variable(
      "supporting_facts_output_bias", [1], initializer=tf.zeros_initializer())
  supporting_facts_logits = tf.matmul(
      global_output_matrix, supporting_facts_output_weights, transpose_b=True)
  supporting_facts_logits = tf.nn.bias_add(supporting_facts_logits,
                                           supporting_facts_output_bias)
  supporting_facts_logits = tf.reshape(supporting_facts_logits,
                                       [batch_size, global_seq_length])

  # Get the logits for the answer type prediction.
  num_answer_types = 3  # SPAN, YES, NO
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, global_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())
  answer_type_logits = tf.matmul(
      global_output[:, 0, :], answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  extra_model_losses = model.losses

  if answer_encoding_method == "span":
    # Get the logits for the begin and end indices.
    answer_span_output_weights = tf.get_variable(
        "answer_span_output_weights", [2, long_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    answer_span_output_bias = tf.get_variable(
        "answer_span_output_bias", [2], initializer=tf.zeros_initializer())
    answer_span_logits = tf.matmul(
        long_output_matrix, answer_span_output_weights, transpose_b=True)
    answer_span_logits = tf.nn.bias_add(answer_span_logits,
                                        answer_span_output_bias)
    answer_span_logits = tf.reshape(answer_span_logits,
                                    [batch_size, long_seq_length, 2])
    answer_span_logits = tf.transpose(answer_span_logits, [2, 0, 1])
    answer_begin_logits, answer_end_logits = tf.unstack(
        answer_span_logits, axis=0)

    return (supporting_facts_logits, (answer_begin_logits, answer_end_logits),
            answer_type_logits, extra_model_losses)
  else:
    # Get the logits for the answer BIO encodings.
    answer_bio_output_weights = tf.get_variable(
        "answer_bio_output_weights", [3, long_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    answer_type_output_bias = tf.get_variable(
        "answer_bio_output_bias", [3], initializer=tf.zeros_initializer())
    answer_bio_logits = tf.matmul(
        long_output_matrix, answer_bio_output_weights, transpose_b=True)
    answer_bio_logits = tf.nn.bias_add(answer_bio_logits,
                                       answer_type_output_bias)
    answer_bio_logits = tf.reshape(answer_bio_logits,
                                   [batch_size, long_seq_length, 3])

    return (supporting_facts_logits, answer_bio_logits, answer_type_logits,
            extra_model_losses)


def model_fn_builder(etc_model_config: modeling.EtcConfig,
                     learning_rate: float,
                     num_train_steps: int,
                     num_warmup_steps: int,
                     flat_sequence: bool,
                     answer_encoding_method: str,
                     use_tpu: bool,
                     use_wordpiece: bool,
                     optimizer: str,
                     poly_power: float,
                     start_warmup_step: int,
                     learning_rate_schedule: str,
                     init_checkpoint=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del labels, params  # Unused by model_fn

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    if answer_encoding_method == "span":
      (supporting_facts_logits, answer_span_logits, answer_type_logits,
       extra_model_losses) = (
           build_model(etc_model_config, features, flat_sequence, is_training,
                       answer_encoding_method, use_tpu, use_wordpiece))
    else:
      (supporting_facts_logits, answer_bio_logits, answer_type_logits,
       extra_model_losses) = (
           build_model(etc_model_config, features, flat_sequence, is_training,
                       answer_encoding_method, use_tpu, use_wordpiece))

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = input_utils.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
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

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:
      # Computes the loss for supporting facts.
      def compute_supporting_facts_loss(logits, supporting_facts):
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(supporting_facts, logits.dtype), logits=logits)
        return tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))

      # Computes the loss for begin/end indices.
      def compute_answer_span_loss(logits, positions):
        long_seq_length = logits.shape.as_list()[-1]
        assert long_seq_length is not None
        one_hot_positions = tf.one_hot(
            positions, depth=long_seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      # Computes the loss for BIO encodings.
      def compute_answer_bio_loss(logits, bio_ids):
        one_hot_bio_ids = tf.one_hot(bio_ids, depth=3, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_bio_ids * log_probs, axis=[-2, -1]))
        return loss

      # Computes the loss for labels.
      def compute_answer_type_loss(logits, labels):
        one_hot_labels = tf.one_hot(labels, depth=3, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        return loss

      losses = [
          compute_supporting_facts_loss(supporting_facts_logits,
                                        features["supporting_facts"]),
          compute_answer_type_loss(answer_type_logits,
                                   features["answer_types"]),
      ]
      if answer_encoding_method == "span":
        losses.extend([
            compute_answer_span_loss(answer_span_logits[0],
                                     features["answer_begins"]),
            compute_answer_span_loss(answer_span_logits[1],
                                     features["answer_ends"]),
        ])
      else:
        losses.append(
            compute_answer_bio_loss(answer_bio_logits,
                                    features["answer_bio_ids"]))
      loss = sum(losses) / len(losses)
      if extra_model_losses:
        loss += tf.math.add_n(extra_model_losses)

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          optimizer, poly_power, start_warmup_step, learning_rate_schedule)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "long_token_ids": tf.identity(features["long_token_ids"]),
          "long_sentence_ids": tf.identity(features["long_sentence_ids"]),
          "long_token_type_ids": tf.identity(features["long_token_type_ids"]),
          "global_token_ids": tf.identity(features["global_token_ids"]),
          "supporting_facts_probs": tf.math.sigmoid(supporting_facts_logits),
          "answer_types": tf.math.top_k(answer_type_logits).indices,
      }
      if answer_encoding_method == "span":
        answer_begin_top_probs, answer_begin_top_indices = tf.math.top_k(
            tf.nn.softmax(answer_span_logits[0], axis=-1), k=50)
        answer_end_top_probs, answer_end_top_indices = tf.math.top_k(
            tf.nn.softmax(answer_span_logits[1], axis=-1), k=50)
        predictions.update({
            "answer_begin_top_probs": answer_begin_top_probs,
            "answer_begin_top_indices": answer_begin_top_indices,
            "answer_end_top_probs": answer_end_top_probs,
            "answer_end_top_indices": answer_end_top_indices,
        })
      else:
        answer_bio_probs, answer_bio_ids = tf.math.top_k(
            tf.nn.softmax(answer_bio_logits, axis=-1))
        predictions.update({
            "answer_bio_probs": tf.squeeze(answer_bio_probs, axis=-1),
            "answer_bio_ids": tf.squeeze(answer_bio_ids, axis=-1),
        })

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn
