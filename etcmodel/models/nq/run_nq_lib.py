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

"""ETC model for NQ."""

import collections
import functools
from typing import Mapping, Text

import attr
import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel import tensor_utils
from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models import optimization


def input_fn_builder(input_file, flags, etc_model_config, is_training,
                     drop_remainder, num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      "unique_ids":
          tf.FixedLenFeature([], tf.int64),
      "token_ids":
          tf.FixedLenFeature([flags.max_seq_length], tf.int64),
      "token_pos":
          tf.FixedLenFeature([flags.max_seq_length], tf.int64),
      "candidate_ids":
          tf.FixedLenFeature([flags.max_seq_length], tf.int64),
      "sentence_ids":
          tf.FixedLenFeature([flags.max_seq_length], tf.int64),
      "long_breakpoints":
          tf.FixedLenFeature([flags.max_seq_length], tf.int64),
      "global_token_ids":
          tf.FixedLenFeature([flags.max_global_seq_length], tf.int64),
      "global_breakpoints":
          tf.FixedLenFeature([flags.max_global_seq_length], tf.int64),
      "sa_start":
          tf.FixedLenFeature([], tf.int64),
      "sa_end":
          tf.FixedLenFeature([], tf.int64),
      "la_start":
          tf.FixedLenFeature([], tf.int64),
      "la_end":
          tf.FixedLenFeature([], tf.int64),
      "answer_type":
          tf.FixedLenFeature([], tf.int64)
  }

  def decode_record(record, name_to_features):
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
    d = tf.data.Dataset.list_files(input_file, shuffle=is_training)
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            functools.partial(tf.data.TFRecordDataset, compression_type="GZIP"),
            cycle_length=num_cpu_threads,
            sloppy=False))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(
        functools.partial(input_utils.add_side_input_features,
                          etc_model_config), tf.data.experimental.AUTOTUNE)
    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn


def build_model(etc_model_config, features, is_training, flags):
  """Build an ETC model."""
  token_ids = features["token_ids"]
  global_token_ids = features["global_token_ids"]

  model = modeling.EtcModel(
      config=etc_model_config,
      is_training=is_training,
      use_one_hot_relative_embeddings=flags.use_tpu)

  model_inputs = dict(token_ids=token_ids, global_token_ids=global_token_ids)
  for field in attr.fields(input_utils.GlobalLocalTransformerSideInputs):
    if field.name in features:
      model_inputs[field.name] = features[field.name]

  # Get the logits for the start and end predictions.
  l_final_hidden, _ = model(**model_inputs)

  l_final_hidden_shape = tensor_utils.get_shape_list(
      l_final_hidden, expected_rank=3)

  batch_size = l_final_hidden_shape[0]
  l_seq_length = l_final_hidden_shape[1]
  hidden_size = l_final_hidden_shape[2]

  num_answer_types = 5  # NULL, YES, NO, LONG, SHORT

  # We add a dense layer to the long output:
  l_output_weights = tf.get_variable(
      "cls/nq/long_output_weights", [4, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  l_output_bias = tf.get_variable(
      "cls/nq/long_output_bias", [4], initializer=tf.zeros_initializer())
  l_final_hidden_matrix = tf.reshape(l_final_hidden,
                                     [batch_size * l_seq_length,
                                      hidden_size])
  l_logits = tf.matmul(l_final_hidden_matrix, l_output_weights,
                       transpose_b=True)
  l_logits = tf.nn.bias_add(l_logits, l_output_bias)
  l_logits = tf.reshape(l_logits, [batch_size, l_seq_length, 4])

  if flags.mask_long_output:
    # Mask out invalid SA/LA start/end positions:
    # 1) find the SEP and CLS tokens:
    long_sep = tf.cast(tf.equal(token_ids, flags.sep_tok_id), tf.int32)
    long_not_sep = 1 - long_sep
    long_cls = tf.cast(tf.equal(token_ids, flags.cls_tok_id), tf.int32)

    # 2) accum sum the SEPs, and the only possible answers are those with sum
    #    equal to 1 (except SEPs) and the CLS position
    l_mask = tf.cast(tf.equal(tf.cumsum(long_sep, axis=-1), 1), tf.int32)
    l_mask = 1 - ((l_mask * long_not_sep) + long_cls)

    # 3) apply the mask to the logits
    l_mask = tf.expand_dims(tf.cast(l_mask, tf.float32) * -10E8, 2)
    l_logits = tf.math.add(l_logits, l_mask)

  # Get the logits for the answer type prediction.
  answer_type_output_layer = l_final_hidden[:, 0, :]
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types,
                                     answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  extra_model_losses = model.losses

  l_logits = tf.transpose(l_logits, [2, 0, 1])
  l_unstacked_logits = tf.unstack(l_logits, axis=0)
  return ([l_unstacked_logits[i] for i in range(4)], answer_type_logits,
          extra_model_losses)


def model_fn_builder(etc_model_config, num_train_steps, num_warmup_steps,
                     flags):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    position_logits, answer_type_logits, extra_model_losses = build_model(
        etc_model_config, features, is_training, flags)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if flags.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = input_utils.get_assignment_map_from_checkpoint(
          tvars, flags.init_checkpoint)
      if flags.use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(flags.init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(flags.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      else:
        init_string = ", *RANDOM_INIT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # Computes the loss for positions.
      def compute_loss(logits, positions, depth):
        one_hot_positions = tf.one_hot(
            positions, depth=depth, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      # Computes the loss for labels.
      def compute_label_loss(logits, labels):
        one_hot_labels = tf.one_hot(labels, depth=5, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        return loss

      labels = [
          features["sa_start"], features["sa_end"], features["la_start"],
          features["la_end"]]
      loss = (sum(
          compute_loss(position_logits[idx], label, flags.max_seq_length)
          for idx, label in enumerate(labels)) + compute_label_loss(
              answer_type_logits, features["answer_type"])) / 5.0

      if extra_model_losses:
        loss += tf.math.add_n(extra_model_losses)

      train_op = optimization.create_optimizer(
          loss, flags.learning_rate, num_train_steps, num_warmup_steps,
          flags.use_tpu, flags.optimizer, flags.poly_power,
          flags.start_warmup_step, flags.learning_rate_schedule)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": tf.identity(features["unique_ids"]),
          "token_ids": tf.identity(features["token_ids"]),
          "token_pos": tf.identity(features["token_pos"]),
          "candidate_ids": tf.identity(features["candidate_ids"]),
          "answer_type_logits": answer_type_logits,
      }
      output_names = ["sa_start", "sa_end", "la_start", "la_end"]

      for idx, output_name in enumerate(output_names):
        predictions[output_name] = tf.identity(features[output_name])
        if output_name == "la_global":
          # propagate the ground truth:
          predictions["la_start"] = tf.identity(features["la_start"])
          predictions["la_end"] = tf.identity(features["la_end"])
        values, indices = tf.compat.v1.math.top_k(position_logits[idx], k=50)
        predictions[output_name + "_pred"] = indices
        predictions[output_name + "_logit"] = values
        predictions[output_name + "_logit0"] = position_logits[idx][:, 0]
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def process_prediction(prediction: Mapping[Text, np.ndarray], writer
                       ) -> None:
  """Processes a single TF `Estimator.predict` prediction.

  Args:
    prediction: Prediction from `Estimator.predict` for a single example.
    writer: An open `tf.python_io.TFRecordWriter` to write to.
  """

  def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  features = collections.OrderedDict()

  # Scalar int64 features:
  features["unique_ids"] = create_int_feature([prediction["unique_ids"]])

  # Vector int64 features:
  # for name in ["token_ids", "token_pos", "candidate_ids"]:
  #   features[name] = create_int_feature(result[name])
  token_pos = prediction["token_pos"]

  # Span outputs:
  output_names = ["sa_start", "sa_end", "la_start", "la_end"]
  for name in output_names:
    # Scalar int64 output.
    features[name] = create_int_feature([prediction[name]])

    # ground truth:
    features[name + "_mapped"] = create_int_feature(
        [token_pos[prediction[name]]])

    # Vector int64 output.
    output_name = name + "_pred"
    features[output_name] = create_int_feature(prediction[output_name])
    features[output_name + "_mapped"] = create_int_feature(
        [token_pos[t] for t in prediction[output_name]])

    # Vector float output.
    output_name = name + "_logit"
    features[output_name] = create_float_feature(prediction[output_name])

    # Scalar float output.
    output_name = name + "_logit0"
    features[output_name] = create_float_feature([prediction[output_name]])

  features["answer_type_logits"] = create_float_feature(
      prediction["answer_type_logits"])

  writer.write(
      tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString())
