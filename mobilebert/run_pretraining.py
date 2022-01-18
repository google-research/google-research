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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os

import tensorflow.compat.v1 as tf

from mobilebert import distill_util
from mobilebert import modeling
from mobilebert import optimization
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import summary as contrib_summary


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("use_einsum", True, "Use tf.einsum to speed up.")

flags.DEFINE_bool("use_summary", False, "Use tf.summary to log training.")

flags.DEFINE_string(
    "bert_teacher_config_file", None,
    "The config json file corresponding to the teacher pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "first_input_file", None,
    "First round input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "first_max_seq_length", 128,
    "The first round maximum total input sequence length. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("first_train_batch_size", None,
                     "The first round total batch size for training.")

flags.DEFINE_integer("first_num_train_steps", 0,
                     "Number of the first training steps for second round.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"], "The optimizer.")

flags.DEFINE_bool("layer_wise_warmup", False,
                  "Whether to use layer-wise distillation warmup.")

flags.DEFINE_integer("num_distill_steps", 100000,
                     "Number of distillation steps.")

flags.DEFINE_float("distill_temperature", 1.0,
                   "The temperature factor of distill.")

flags.DEFINE_float("distill_ground_truth_ratio", 1.0,
                   "The ground truth factor of distill in 100%.")

flags.DEFINE_float("attention_distill_factor", 0.0,
                   "Whether to use attention distillation.")

flags.DEFINE_float("hidden_distill_factor", 0.0,
                   "Whether to use hidden distillation.")

flags.DEFINE_float("gamma_distill_factor", 0.0,
                   "Whether to use hidden statistics distillation.")

flags.DEFINE_float("beta_distill_factor", 0.0,
                   "Whether to use hidden statistics distillation.")

flags.DEFINE_float("weight_decay_rate", 0.01, "Weight decay rate.")


def _dicts_to_list(list_of_dicts, keys):
  """Transforms a list of dicts to a list of values.

  This is useful to create a list of Tensors to pass as an argument of
  `host_call_fn` and `metrics_fn`, because they take either a list as positional
  arguments or a dict as keyword arguments.

  Args:
    list_of_dicts: (list) a list of dicts. The keys of each dict must include
      all the elements in `keys`.
    keys: (list) a list of keys.

  Returns:
    (list) a list of values ordered by corresponding keys.
  """
  list_of_values = []
  for key in keys:
    list_of_values.extend([d[key] for d in list_of_dicts])
  return list_of_values


def _list_to_dicts(list_of_values, keys):
  """Restores a list of dicts from a list created by `_dicts_to_list`.

  `keys` must be the same as what was used in `_dicts_to_list` to create the
  list. This is used to restore the original dicts inside `host_call_fn` and
  `metrics_fn`.

  Transforms a list to a list of dicts.

  Args:
    list_of_values: (list) a list of values. The length must a multiple of the
      length of keys.
    keys: (list) a list of keys.

  Returns:
    (list) a list of dicts.
  """
  num_dicts = len(list_of_values) // len(keys)
  list_of_dicts = [collections.OrderedDict() for i in range(num_dicts)]
  for i, key in enumerate(keys):
    for j in range(num_dicts):
      list_of_dicts[j][key] = list_of_values[i * num_dicts + j]
  return list_of_dicts


def model_fn_builder(bert_config,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     bert_teacher_config=None,
                     init_from_teacher=False,
                     optimizer="lamb",
                     attention_distill_factor=0.0,
                     hidden_distill_factor=0.0,
                     gamma_distill_factor=0.0,
                     beta_distill_factor=0.0,
                     distill_temperature=1.0,
                     distill_ground_truth_ratio=1.0,
                     layer_wise_warmup=False,
                     use_einsum=True,
                     summary_dir=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if bert_teacher_config is None:
      model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings,
          use_einsum=use_einsum)

      label_ids = tf.reshape(masked_lm_ids, [-1])
      true_labels = tf.one_hot(
          label_ids, depth=bert_config.vocab_size,
          dtype=model.get_sequence_output().dtype)
      one_hot_labels = true_labels
    else:
      model = modeling.BertModel(
          config=bert_config,
          is_training=False,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings,
          use_einsum=use_einsum)

      with tf.variable_scope("teacher"):
        teacher_model = modeling.BertModel(
            config=bert_teacher_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum)

        label_ids = tf.reshape(masked_lm_ids, [-1])

        true_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size,
            dtype=model.get_sequence_output().dtype)

        teacher_logits = get_logits(
            bert_teacher_config,
            distill_temperature * teacher_model.get_sequence_output(),
            teacher_model.get_embedding_table(),
            masked_lm_positions)

        teacher_labels = tf.nn.softmax(teacher_logits, axis=-1)

        if distill_ground_truth_ratio == 1.0:
          one_hot_labels = true_labels
        else:
          one_hot_labels = (
              teacher_labels * (1 - distill_ground_truth_ratio)
              + true_labels * distill_ground_truth_ratio)

        teacher_attentions = teacher_model.get_all_attention_maps()
        student_attentions = model.get_all_attention_maps()

        teacher_hiddens = teacher_model.get_all_encoder_layers()
        student_hiddens = model.get_all_encoder_layers()

    (masked_lm_loss, _, masked_lm_example_loss,
     masked_lm_log_probs, _) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, tf.stop_gradient(one_hot_labels),
         true_labels, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    extra_loss1 = 0.0
    extra_loss2 = 0.0
    extra_loss3 = 0.0
    extra_loss4 = 0.0

    scalars_to_summarize = {}

    def get_layerwise_gate(layer_id):
      steps_per_phase = num_train_steps // bert_config.num_hidden_layers
      layer_wise_gate = distill_util.layer_wise_learning_rate(
          layer_id=layer_id, steps_per_phase=steps_per_phase, binary=True)
      return layer_wise_gate

    if layer_wise_warmup and hidden_distill_factor != 0.0:
      layer_id = 0
      for teacher_hidden, student_hidden in (
          zip(teacher_hiddens[1:], student_hiddens[1:])):
        with tf.variable_scope("hidden_distill_%d" % layer_id):
          mse_loss = tf.losses.mean_squared_error(
              tf.stop_gradient(
                  contrib_layers.layer_norm(
                      inputs=teacher_hidden,
                      begin_norm_axis=-1,
                      begin_params_axis=-1,
                      trainable=False)),
              contrib_layers.layer_norm(
                  inputs=student_hidden,
                  begin_norm_axis=-1,
                  begin_params_axis=-1,
                  trainable=False))
          layer_wise_gate = get_layerwise_gate(layer_id)
          extra_loss1 += layer_wise_gate * mse_loss
        layer_id += 1
      extra_loss1 = extra_loss1 * hidden_distill_factor / layer_id

    if layer_wise_warmup and (
        beta_distill_factor != 0 and gamma_distill_factor != 0.0):
      layer_id = 0
      for teacher_hidden, student_hidden in (
          zip(teacher_hiddens[1:], student_hiddens[1:])):
        with tf.variable_scope("hidden_distill_%d" % layer_id):
          teacher_mean = tf.reduce_mean(
              teacher_hiddens, axis=[-1], keepdims=True)
          student_mean = tf.reduce_mean(
              student_hidden, axis=[-1], keepdims=True)
          teacher_variance = tf.reduce_mean(
              tf.squared_difference(teacher_hiddens, teacher_mean),
              axis=[-1], keepdims=True)
          student_variance = tf.reduce_mean(
              tf.squared_difference(student_hidden, student_mean),
              axis=[-1], keepdims=True)
          beta_distill_loss = tf.reduce_mean(
              tf.squared_difference(
                  tf.stop_gradient(teacher_mean), student_mean))
          gamma_distill_loss = tf.reduce_mean(
              tf.abs(tf.stop_gradient(teacher_variance) - student_variance))
          layer_wise_gate = get_layerwise_gate(layer_id)
          extra_loss3 += layer_wise_gate * beta_distill_loss
          extra_loss4 += layer_wise_gate * gamma_distill_loss
        layer_id += 1
      extra_loss3 = extra_loss3 * beta_distill_factor / layer_id
      extra_loss4 = extra_loss4 * gamma_distill_factor / layer_id

    if layer_wise_warmup and attention_distill_factor != 0.0:
      layer_id = 0
      for teacher_attention, student_attention in (
          zip(teacher_attentions, student_attentions)):
        with tf.variable_scope("attention_distill_%d" % layer_id):
          teacher_attention_prob = tf.nn.softmax(
              teacher_attention, axis=-1)
          student_attention_log_prob = tf.nn.log_softmax(
              student_attention, axis=-1)
          kl_divergence = - (
              tf.stop_gradient(teacher_attention_prob)
              * student_attention_log_prob)
          kl_divergence = tf.reduce_mean(tf.reduce_sum(kl_divergence, axis=-1))
          layer_wise_gate = get_layerwise_gate(layer_id)
          extra_loss2 += layer_wise_gate * kl_divergence
        layer_id += 1
      extra_loss2 = extra_loss2 * attention_distill_factor / layer_id

    if layer_wise_warmup:
      total_loss = extra_loss1 + extra_loss2 + extra_loss3 + extra_loss4
    else:
      total_loss = masked_lm_loss + next_sentence_loss

    if summary_dir is not None:
      if layer_wise_warmup:
        scalars_to_summarize["feature_map_transfer_loss"] = extra_loss1
        scalars_to_summarize["attention_transfer_loss"] = extra_loss2
        scalars_to_summarize["mean_transfer_loss"] = extra_loss3
        scalars_to_summarize["variance_transfer_loss"] = extra_loss4
      else:
        scalars_to_summarize["masked_lm_loss"] = masked_lm_loss
        scalars_to_summarize["next_sentence_loss"] = next_sentence_loss

        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_accuracy = tf.cast(tf.math.equal(
            tf.reshape(masked_lm_ids, [-1]),
            tf.reshape(masked_lm_predictions, [-1])), tf.float32)
        numerator = tf.reduce_sum(
            tf.reshape(masked_lm_weights, [-1]) * masked_lm_accuracy)
        denominator = tf.reduce_sum(masked_lm_weights) + 1e-5
        masked_lm_accuracy = numerator / denominator
        scalars_to_summarize["masked_lm_accuracy"] = masked_lm_accuracy

        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_accuracy = tf.reduce_mean(
            tf.cast(tf.math.equal(
                tf.reshape(next_sentence_labels, [-1]),
                tf.reshape(next_sentence_predictions, [-1])), tf.float32))
        scalars_to_summarize["next_sentence_accuracy"] = next_sentence_accuracy

      scalars_to_summarize["global_step"] = tf.train.get_or_create_global_step()
      scalars_to_summarize["loss"] = total_loss

    host_call = None
    if summary_dir is not None:
      if use_tpu:
        for name in scalars_to_summarize:
          scalars_to_summarize[name] = tf.reshape(
              scalars_to_summarize[name], [1])

        def host_call_fn(*args):
          """Host call function to compute training summaries."""
          scalars = _list_to_dicts(args, scalars_to_summarize.keys())[0]
          for name in scalars:
            scalars[name] = scalars[name][0]

          with contrib_summary.create_file_writer(
              summary_dir, max_queue=1000).as_default():
            with contrib_summary.always_record_summaries():
              for name, value in scalars.items():
                if name not in ["global_step"]:
                  contrib_summary.scalar(
                      name, value, step=scalars["global_step"])

          return contrib_summary.all_summary_ops()

        host_call = (host_call_fn, _dicts_to_list([scalars_to_summarize],
                                                  scalars_to_summarize.keys()))
      else:
        for name in scalars_to_summarize:
          tf.summary.scalar(name, scalars_to_summarize[name])

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    teacher_initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
      if not init_from_teacher:
        # Initializes from the checkpoint for all variables.
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      elif bert_teacher_config is not None:
        # Initializes from the pre-trained checkpoint only for teacher model
        # and embeddings for distillation.
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint, init_embedding=True)
        (teacher_assignment_map, teacher_initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint, init_from_teacher=True)
        if use_tpu:

          def teacher_tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.train.init_from_checkpoint(init_checkpoint,
                                          teacher_assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = teacher_tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          tf.train.init_from_checkpoint(init_checkpoint, teacher_assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    total_size = 0
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      if var.name in teacher_initialized_variable_names:
        init_string = ", *INIT_FROM_TEACHER_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
      if not var.name.startswith("teacher"):
        total_size += functools.reduce(lambda x, y: x * y,
                                       var.get_shape().as_list())
    tf.logging.info("  total variable parameters: %d", total_size)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      if layer_wise_warmup:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps,
            num_warmup_steps, use_tpu, optimizer,
            end_lr_rate=1.0, use_layer_wise_warmup=True,
            total_warmup_phases=bert_config.num_hidden_layers)
      else:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps,
            num_warmup_steps, use_tpu, optimizer)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_logits(bert_config, input_tensor, output_weights, positions):
  """Get logits for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())

    if bert_config.hidden_size != bert_config.embedding_size:
      extra_output_weights = tf.get_variable(
          name="extra_output_weights",
          shape=[
              bert_config.vocab_size,
              bert_config.hidden_size - bert_config.embedding_size],
          initializer=modeling.create_initializer(
              bert_config.initializer_range))
      output_weights = tf.concat(
          [output_weights, extra_output_weights], axis=1)
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         one_hot_labels, true_labels, label_weights):
  """Get loss and log probs for the masked LM."""
  logits = get_logits(bert_config, input_tensor, output_weights, positions)

  with tf.variable_scope("cls/predictions"):
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_weights = tf.reshape(label_weights, [-1])

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    true_per_example_loss = -tf.reduce_sum(log_probs * true_labels, axis=[-1])
    true_numerator = tf.reduce_sum(label_weights * true_per_example_loss)
    true_denominator = tf.reduce_sum(label_weights) + 1e-5
    true_loss = true_numerator / true_denominator
  return (loss, true_loss, per_example_loss, log_probs, logits)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
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


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.interleave(
          tf.data.TFRecordDataset,
          num_parallel_calls=cycle_length,
          deterministic=(not is_training))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=num_cpu_threads).batch(
            batch_size, drop_remainder=True)
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  bert_teacher_config = None
  if FLAGS.bert_teacher_config_file is not None:
    bert_teacher_config = modeling.BertConfig.from_json_file(
        FLAGS.bert_teacher_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  summary_path = None
  warmup_path = None
  main_summary_path = None
  if FLAGS.use_summary:
    summary_path = os.path.join(FLAGS.output_dir, "my_summary")
    warmup_summary_path = os.path.join(summary_path, "warmup")
    main_summary_path = os.path.join(summary_path, "main")
    tf.gfile.MakeDirs(summary_path)
    tf.gfile.MakeDirs(warmup_summary_path)
    tf.gfile.MakeDirs(main_summary_path)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2

  if FLAGS.do_train and FLAGS.layer_wise_warmup:
    warmup_path = os.path.join(FLAGS.output_dir, "warmup")
    tf.gfile.MakeDirs(warmup_path)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=warmup_path,
        keep_checkpoint_max=0,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        bert_teacher_config=bert_teacher_config,
        init_checkpoint=FLAGS.init_checkpoint,
        init_from_teacher=True,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_distill_steps,
        num_warmup_steps=0,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        optimizer=FLAGS.optimizer,
        attention_distill_factor=FLAGS.attention_distill_factor,
        hidden_distill_factor=FLAGS.hidden_distill_factor,
        gamma_distill_factor=FLAGS.gamma_distill_factor,
        beta_distill_factor=FLAGS.beta_distill_factor,
        layer_wise_warmup=True,
        use_einsum=FLAGS.use_einsum,
        summary_dir=warmup_summary_path)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
      input_files.extend(tf.gfile.Glob(input_pattern))

    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_distill_steps)

  init_checkpoint = FLAGS.init_checkpoint
  if FLAGS.layer_wise_warmup:
    init_checkpoint = warmup_path

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=0,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host))

  if FLAGS.distill_ground_truth_ratio == 1.0:
    bert_teacher_config = None

  model_fn = model_fn_builder(
      bert_config=bert_config,
      bert_teacher_config=bert_teacher_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      optimizer=FLAGS.optimizer,
      distill_temperature=FLAGS.distill_temperature,
      distill_ground_truth_ratio=FLAGS.distill_ground_truth_ratio,
      use_einsum=FLAGS.use_einsum,
      summary_dir=main_summary_path)

  # First round training.
  if FLAGS.do_train and FLAGS.first_num_train_steps > 0:
    input_files = []
    for input_pattern in FLAGS.first_input_file.split(","):
      input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
      tf.logging.info("  %s" % input_file)

    if (FLAGS.first_train_batch_size is not None and
        FLAGS.first_train_batch_size != FLAGS.train_batch_size):
      estimator = tf.estimator.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.first_train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size)
    else:
      estimator = tf.estimator.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size)

    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.first_max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn,
                    max_steps=FLAGS.first_num_train_steps)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
