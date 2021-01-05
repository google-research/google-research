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

"""Library for OpenKP finetuning."""

import collections
import functools
import os
from typing import Dict, Mapping, Text

import attr
import tensorflow.compat.v1 as tf

from etcmodel import tensor_utils
from etcmodel.layers import wrappers
from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models import multihop_utils
from etcmodel.models import optimization

# Use symbols from TF2 as not all of them are backwards compatible.
tf_summary = tf.compat.v2.summary


def model_fn_builder(model_config,
                     model_dir,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     optimizer,
                     poly_power,
                     start_warmup_step,
                     learning_rate_schedule,
                     add_final_layer,
                     weight_decay_rate,
                     label_smoothing=0.0):
  """Constructs the model fn."""

  def record_summary_host_fn(metrics_dir, steps_per_summary, **kwargs):
    """A host_fn function for the host_call in TPUEstimatorSpec.

    Args:
      metrics_dir: Directory where tf summary events should be written.
      steps_per_summary: Number of steps for how often summaries should be
        recorded.
      **kwargs: Contains tensors for which summaries are to be recorded. It must
        contain a key of `global_step`.

    Returns:
      A summary op for each tensor to be recorded.
    """

    global_step = kwargs.pop("global_step")[0]
    with tf_summary.create_file_writer(metrics_dir).as_default():
      with tf_summary.record_if(
          lambda: tf.math.equal(global_step % steps_per_summary, 0)):
        for name, tensor in kwargs.items():
          tf_summary.scalar(name, tf.reduce_mean(tensor), step=global_step)
        return tf.summary.all_v2_summary_ops()

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    long_token_ids = features["long_token_ids"]
    long_token_type_ids = features["long_token_type_ids"]
    global_token_ids = features["global_token_ids"]
    global_token_type_ids = features["global_token_type_ids"]

    model_inputs = dict(
        token_ids=long_token_ids,
        global_token_ids=global_token_ids,
        global_segment_ids=global_token_type_ids,
        segment_ids=long_token_type_ids)

    for field in attr.fields(input_utils.GlobalLocalTransformerSideInputs):
      model_inputs[field.name] = features[field.name]

    labels = tf.cast(features["label_id"], dtype=tf.int32)

    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(labels), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = modeling.EtcModel(
        config=model_config,
        is_training=is_training,
        use_one_hot_embeddings=use_one_hot_embeddings,
        use_one_hot_relative_embeddings=use_tpu)

    _, global_output = model(**model_inputs)
    (total_loss, per_example_loss, logits) = (
        process_model_output(model_config, mode, global_output,
                             global_token_type_ids, labels, is_real_example,
                             add_final_layer, label_smoothing))
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
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          optimizer, poly_power, start_warmup_step, learning_rate_schedule,
          weight_decay_rate)

      metrics_dict = metric_fn(
          per_example_loss=per_example_loss,
          logits=logits,
          labels=labels,
          is_real_example=is_real_example,
          is_train=True)

      host_inputs = {
          "global_step":
              tf.expand_dims(tf.train.get_or_create_global_step(), 0),
      }

      host_inputs.update({
          metric_name: tf.expand_dims(metric_tensor, 0)
          for metric_name, metric_tensor in metrics_dict.items()
      })

      host_call = (functools.partial(
          record_summary_host_fn,
          metrics_dir=os.path.join(model_dir, "train_metrics"),
          steps_per_summary=50), host_inputs)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)

    elif mode == tf.estimator.ModeKeys.EVAL:
      metric_fn_tensors = dict(
          per_example_loss=per_example_loss,
          logits=logits,
          labels=labels,
          is_real_example=is_real_example)

      eval_metrics = (functools.partial(metric_fn,
                                        is_train=False), metric_fn_tensors)
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "logits": logits,
              # Wrap in `tf.identity` to avoid b/130501786.
              "label_ids": tf.identity(labels),
          },
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Unexpected mode {} encountered".format(mode))

    return output_spec

  return model_fn


def metric_fn(per_example_loss,
              logits,
              labels,
              is_real_example,
              is_train=False):
  """Computes the loss and accuracy of the model."""

  metrics_dict = collections.OrderedDict()

  def _add_metric(metric_name, metric_tensor):
    """Adds a given metric to the metric dict."""
    if is_train:
      metric_name = "train_metrics/" + metric_name
    else:
      # Convert to a streaming metric for eval.
      metric_tensor = tf.metrics.mean(metric_tensor)
      metric_name = "eval_metrics/" + metric_name
    metrics_dict[metric_name] = metric_tensor

  sum_weights = tf.reduce_sum(is_real_example) + 1e-5
  loss = tf.reduce_sum(per_example_loss * is_real_example) / sum_weights
  _add_metric(metric_name="loss", metric_tensor=loss)

  predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

  acc_numerator = tf.cast(
      tf.equal(predictions, labels), dtype=tf.float32) * is_real_example
  acc_numerator = tf.reduce_sum(acc_numerator)
  accuracy = acc_numerator / sum_weights

  if is_train:
    _add_metric(metric_name="accuracy", metric_tensor=accuracy)

  if not is_train:
    eval_accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions, weights=is_real_example)
    metrics_dict["eval_metrics/accuracy"] = eval_accuracy

  return metrics_dict


def process_model_output(model_config,
                         mode,
                         global_output_tensor,
                         global_token_type_ids_tensor,
                         labels,
                         is_real_example,
                         add_final_layer=True,
                         label_smoothing=0.0):
  """Process model output embeddings and computes loss, logits etc."""

  global_output_tensor_shape = tensor_utils.get_shape_list(
      global_output_tensor, expected_rank=3)
  batch_size = global_output_tensor_shape[0]
  global_seq_len = global_output_tensor_shape[1]
  hidden_size = global_output_tensor_shape[2]

  global_output_tensor = tf.reshape(global_output_tensor,
                                    [batch_size * global_seq_len, hidden_size])

  if add_final_layer:
    with tf.variable_scope("global_output_layer/transform"):
      is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
      final_layer = wrappers.ResidualBlock(
          inner_intermediate_size=model_config.intermediate_size,
          inner_activation=tensor_utils.get_activation(model_config.hidden_act),
          use_pre_activation_order=False,
          dropout_probability=model_config.hidden_dropout_prob)
      global_output_tensor = final_layer(
          global_output_tensor, training=is_training)

  output_weights = tf.get_variable(
      "output_weights", [1, model_config.hidden_size],
      initializer=tf.truncated_normal_initializer(
          stddev=model_config.initializer_range))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    logits = tf.matmul(global_output_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    tf.logging.info("*** logits initial are {} *** ".format(logits))
    logits = tf.reshape(logits, [batch_size, global_seq_len])
    tf.logging.info("*** logits after reshape are {} *** ".format(logits))

    # Consider only candidate global tokens in the global output.
    multiplier_mask = tf.cast(
        tf.equal(global_token_type_ids_tensor,
                 multihop_utils.CANDIDATE_GLOBAL_TOKEN_TYPE_ID),
        dtype=logits.dtype)

    adder_mask = -10000.0 * (1.0 - multiplier_mask)

    logits = (logits * multiplier_mask + adder_mask)

    tf.logging.info("*** global_token_type_ids_tensor is {} *** ".format(
        global_token_type_ids_tensor))
    tf.logging.info("*** adder_mask is {} *** ".format(adder_mask))
    tf.logging.info("*** multiplier_mask is {} *** ".format(multiplier_mask))
    tf.logging.info("*** logits computed are {} *** ".format(logits))

    # probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=global_seq_len, dtype=tf.float32)
    if label_smoothing > 0:
      num_classes = tf.reduce_sum(multiplier_mask, axis=-1)
      num_classes = tf.expand_dims(num_classes, -1)
      one_hot_labels = (1 - label_smoothing) * one_hot_labels
      one_hot_labels += (label_smoothing / num_classes)
      # Ensure smoothing of labels only for applicable global (candidate)
      # tokens.
      one_hot_labels *= multiplier_mask

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    numerator = tf.reduce_sum(per_example_loss * is_real_example)
    denominator = tf.reduce_sum(is_real_example) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, logits)


def get_model_schema(long_seq_len, global_seq_len):
  """Returns the feature spec of the model."""
  name_to_features = {
      "long_token_ids":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "long_token_type_ids":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "long_sentence_ids":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "long_paragraph_ids":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "long_paragraph_breakpoints":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "l2g_linked_ids":
          tf.FixedLenFeature([long_seq_len], tf.int64),
      "global_token_ids":
          tf.FixedLenFeature([global_seq_len], tf.int64),
      "global_token_type_ids":
          tf.FixedLenFeature([global_seq_len], tf.int64),
      "global_paragraph_breakpoints":
          tf.FixedLenFeature([global_seq_len], tf.int64),
      "label_id":
          tf.FixedLenFeature([], tf.float32, default_value=0),
      "is_real_example":
          tf.FixedLenFeature([], tf.int64, default_value=1),
  }
  return name_to_features


def _add_side_input_features(
    features: Mapping[Text, tf.Tensor],
    model_config: modeling.EtcConfig,
    candidate_ignore_hard_g2l: bool = True,
    query_ignore_hard_g2l: bool = True,
    enable_l2g_linking: bool = True,
) -> Dict[Text, tf.Tensor]:
  """Replaces raw input features with derived ETC side inputs."""

  features = dict(features)
  global_token_type_ids = features["global_token_type_ids"]

  if candidate_ignore_hard_g2l:
    # Have all the candidate global tokens attend to everything in the long
    # even when `use_hard_g2l_mask` is enabled.
    candidate_ignore_hard_g2l_mask = tf.cast(
        tf.equal(global_token_type_ids,
                 multihop_utils.CANDIDATE_GLOBAL_TOKEN_TYPE_ID),
        dtype=global_token_type_ids.dtype)
  else:
    candidate_ignore_hard_g2l_mask = tf.zeros_like(global_token_type_ids)

  if query_ignore_hard_g2l:
    query_ignore_hard_g2l_mask = tf.cast(
        tf.equal(global_token_type_ids,
                 multihop_utils.QUESTION_GLOBAL_TOKEN_TYPE_ID),
        dtype=global_token_type_ids.dtype)

  else:
    query_ignore_hard_g2l_mask = tf.zeros_like(global_token_type_ids)

  ignore_hard_g2l_mask = (
      query_ignore_hard_g2l_mask + candidate_ignore_hard_g2l_mask)

  if enable_l2g_linking:
    l2g_linked_ids = features["l2g_linked_ids"]
  else:
    l2g_linked_ids = None

  side_inputs = (
      multihop_utils.make_global_local_transformer_side_inputs(
          long_paragraph_breakpoints=features["long_paragraph_breakpoints"],
          long_paragraph_ids=features["long_paragraph_ids"],
          long_sentence_ids=features["long_sentence_ids"],
          global_paragraph_breakpoints=features["global_paragraph_breakpoints"],
          local_radius=model_config.local_radius,
          relative_pos_max_distance=model_config.relative_pos_max_distance,
          use_hard_g2l_mask=model_config.use_hard_g2l_mask,
          ignore_hard_g2l_mask=ignore_hard_g2l_mask,
          use_hard_l2g_mask=model_config.use_hard_l2g_mask,
          l2g_linked_ids=l2g_linked_ids))

  features.update(side_inputs.to_dict())
  return features


def input_fn_builder(input_file_pattern,
                     model_config,
                     long_seq_len,
                     global_seq_len,
                     is_training,
                     drop_remainder,
                     num_cpu_threads=4,
                     candidate_ignore_hard_g2l=True,
                     query_ignore_hard_g2l=True,
                     enable_l2g_linking=True):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

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

  def input_fn(params):
    """The actual input function."""
    name_to_features = get_model_schema(long_seq_len, global_seq_len)
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.Dataset.list_files(input_file_pattern, shuffle=is_training)

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_cpu_threads,
            sloppy=False))

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = d.shuffle(buffer_size=100).repeat()

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(
        functools.partial(
            _add_side_input_features,
            model_config=model_config,
            candidate_ignore_hard_g2l=candidate_ignore_hard_g2l,
            query_ignore_hard_g2l=query_ignore_hard_g2l,
            enable_l2g_linking=enable_l2g_linking),
        tf.data.experimental.AUTOTUNE)

    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn


def serving_input_receiver_fn(
    model_config: modeling.EtcConfig,
    long_seq_len: int,
    global_seq_len: int,
    candidate_ignore_hard_g2l: bool = True,
    query_ignore_hard_g2l: bool = True,
    enable_l2g_linking: bool = True
) -> tf.estimator.export.ServingInputReceiver:
  """Creates an input function to parse input features for inference."""

  # An input receiver that expects a vector of serialized `tf.Example`s.
  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name="serialized_tf_example")
  receiver_tensors = {"serialized_tf_example": serialized_tf_example}
  schema = get_model_schema(long_seq_len, global_seq_len)
  features = tf.parse_example(serialized_tf_example, schema)
  features = _add_side_input_features(
      features=features,
      model_config=model_config,
      candidate_ignore_hard_g2l=candidate_ignore_hard_g2l,
      query_ignore_hard_g2l=query_ignore_hard_g2l,
      enable_l2g_linking=enable_l2g_linking)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def run_export(estimator, model_dir, export_ckpts, model_config, long_seq_len,
               global_seq_len, candidate_ignore_hard_g2l=True,
               query_ignore_hard_g2l=True, enable_l2g_linking=True):
  """Exports `tf.SavedModel` for the specified checkpoints the in model_dir."""

  if export_ckpts is None:
    # Export all the checkpoints within the `model_dir`.
    ckpts = [
        f[0:f.rfind(".")]
        for f in os.listdir(model_dir)
        if f.startswith("model.ckpt-")
    ]
    ckpts = set(ckpts)
  else:
    ckpts = [ckpt.strip() for ckpt in export_ckpts.split(",")]

  for ckpt_name in ckpts:
    ckpt_path = os.path.join(model_dir, ckpt_name)
    export_path = estimator.export_saved_model(
        export_dir_base=os.path.join(model_dir, "saved_models", ckpt_name),
        serving_input_receiver_fn=functools.partial(
            serving_input_receiver_fn,
            model_config=model_config,
            long_seq_len=long_seq_len,
            global_seq_len=global_seq_len,
            candidate_ignore_hard_g2l=candidate_ignore_hard_g2l,
            query_ignore_hard_g2l=query_ignore_hard_g2l,
            enable_l2g_linking=enable_l2g_linking),
        checkpoint_path=ckpt_path)

    tf.logging.info("WikiHop ETC Model exported to %s for checkpoint %s ",
                    export_path, ckpt_path)
