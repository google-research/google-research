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

"""Run NarrativeQA fine-tuning for ReadItTwice BERT."""

import collections
import csv
import functools
import json
import os
import re
import sys
import time
from typing import List, Optional, Text

from absl import app
from absl import flags
from absl import logging
import dataclasses
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.data_utils import tokenization
from readtwice.models import checkpoint_utils
from readtwice.models import config
from readtwice.models import input_utils
from readtwice.models import losses
from readtwice.models import modeling
from readtwice.models import optimization
from readtwice.models.narrative_qa import evaluation

FLAGS = flags.FLAGS

## Model parameters

flags.DEFINE_string(
    "read_it_twice_bert_config_file", None,
    "The config json file corresponding to the pre-trained ReadItTwice BERT "
    "model. This specifies the model architecture.")

flags.DEFINE_bool(
    "enable_side_inputs", False,
    "If True, enables read-it-twice model. Otherwise, the model becomes "
    "equivalent to the standard Transformer model.")

flags.DEFINE_enum(
    "cross_block_attention_mode", "doc", ["block", "doc", "batch"],
    "The policy on how summaries between different "
    "blocks are allowed to interact with each other.")

## Input parameters

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("input_qaps", None, "qaps.csv input file.")

flags.DEFINE_string("eval_data_split", "valid",
                    "Data set split for evaluation: train, test or valid")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None, "Initial checkpoint "
    "(usually from a pre-trained ReadItTwice BERT model).")

flags.DEFINE_integer(
    "padding_token_id", 0,
    "The token id of the padding token according to the WordPiece vocabulary.")

flags.DEFINE_integer(
    "num_replicas_concat", 1, "Number of replicas to gather summaries from. "
    "If None (default) then cross-replicas summaries are not used.")

## Other parameters

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("summary_loss_weight", None,
                   "Loss for the summary prediction task.")

flags.DEFINE_integer("summary_num_layers", None,
                     "Number of layers for the summary prediction task.")

flags.DEFINE_integer(
    "summary_num_cross_attention_heads", None,
    "Number of attention heads for the summary prediction task.")

flags.DEFINE_bool(
    "summary_enable_default_side_input", False,
    "Add a default side input, which acts like a no-op attention, "
    "effective allowing attention weights to sum up to something less than 1.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_enum(
    "learning_rate_schedule", "poly_decay", ["poly_decay", "inverse_sqrt"],
    "The learning rate schedule to use. The default of "
    "`poly_decay` uses tf.train.polynomial_decay, while "
    "`inverse_sqrt` uses inverse sqrt of time after the warmup.")

flags.DEFINE_float("poly_power", 1.0,
                   "The power of poly decay if using `poly_decay` schedule.")

flags.DEFINE_float("num_train_epochs", 3.0, "Number of training epochs.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_string(
    "spm_model_path",
    "/namespace/webanswers/generative_answers/checkpoints/vocab_gpt.model",
    "Path to sentence piece model.")

flags.DEFINE_integer("decode_top_k", 40,
                     "Maximum number of logits to consider for begin/end.")

flags.DEFINE_integer("decode_max_size", 10,
                     "Maximum number of sentence pieces in an answer.")

flags.DEFINE_integer("save_checkpoints_steps", 15000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("steps_per_summary", 1000, "How often to write summaries.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("cross_attention_top_k", None,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "Whether to use one-hot multiplication instead of gather for embedding "
    "lookups.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from"
    " default tpu_worker.")

flags.DEFINE_integer("num_tpu_cores", None, "Number of tpu cores for the job.")

flags.DEFINE_integer("num_tpu_tasks", None, "Number of tpu tasks in total.")



def _get_global_step_for_checkpoint(checkpoint_path):
  """Returns the global step for the checkpoint path, or -1 if not found."""
  re_match = re.search(r"ckpt-(\d+)$", checkpoint_path)
  return -1 if re_match is None else int(re_match.group(1))


def _get_all_checkpoints(model_dir):
  checkpoint_pattern = re.compile("(model.ckpt-([0-9]+)).meta")
  checkpoints = []
  for path in tf.io.gfile.listdir(model_dir):
    checkpoint_match = re.match(checkpoint_pattern, path)
    if not checkpoint_match:
      continue
    checkpoint_path = os.path.join(model_dir, checkpoint_match.group(1))
    if (not tf.io.gfile.exists(checkpoint_path + ".meta") or
        not tf.io.gfile.exists(checkpoint_path + ".index")):
      raise ValueError("Invalid checkpoint path: {}".format(checkpoint_path))
    checkpoints.append((int(checkpoint_match.group(2)), checkpoint_path))
  checkpoints.sort()
  return [os.path.normpath(x[1]) for x in checkpoints]


def _is_checkpoint_in_the_model_dir():
  x1 = (
      os.path.normpath(FLAGS.init_checkpoint) in _get_all_checkpoints(
          FLAGS.output_dir))
  x2 = (
      os.path.normpath(os.path.split(
          FLAGS.init_checkpoint)[0]) == os.path.normpath(FLAGS.output_dir))
  assert x1 == x2
  return x1


def make_scalar_summary(tag, value):
  """Returns a TF Summary proto for a scalar summary value.

  Args:
    tag: The name of the summary.
    value: The scalar float value of the summary.

  Returns:
    A TF Summary proto.
  """
  return tf.summary.Summary(
      value=[tf.summary.Summary.Value(tag=tag, simple_value=value)])


def write_eval_results(global_step, eval_results, eval_name, summary_writer):
  logging.info("***** EVAL on step %d (%s) *****", global_step,
               eval_name.upper())
  for metric, value in eval_results.items():
    value_float = float(value)
    logging.info("  %s = %.3f", metric, value_float)
    summary_writer.add_summary(
        make_scalar_summary(
            tag="%s_metrics_%s/%s" % (FLAGS.eval_data_split, eval_name, metric),
            value=value_float),
        global_step=global_step)
  summary_writer.flush()


def _convert_prediction_logits_to_probs(nbest_predictions):
  """Compute probabilities for answers."""
  nbest_predictions_probs = collections.OrderedDict()
  for question, predictions in nbest_predictions.items():
    # pylint: disable=g-complex-comprehension
    all_logits = np.array(
        [logit for answer, logits in predictions.items() for logit in logits])
    if len(all_logits) == 0:  # pylint: disable=g-explicit-length-test
      # There are no valid predictions for this question
      continue
    max_logit = all_logits.max()
    norm_const = np.exp(all_logits - max_logit).sum()
    nbest_predictions_probs[question] = {
        answer:
        [float(np.exp(logit - max_logit) / norm_const) for logit in logits]
        for answer, logits in predictions.items()
    }
  return nbest_predictions_probs


def _get_best_predictions(nbest_predictions, reduction_fn):
  best_predictions = collections.OrderedDict()
  for question, predictions in nbest_predictions.items():
    scores_predictions = [(reduction_fn(scores), answer)
                          for answer, scores in predictions.items()]
    scores_predictions.sort()
    best_predictions[question] = scores_predictions[-1][1]
  return best_predictions


def record_summary_host_fn(metrics_dir, **kwargs):
  """A host_fn function for the host_call in TPUEstimatorSpec.

  Args:
    metrics_dir: Directory where tf summary events should be written. recorded.
    **kwargs: Contains tensors for which summaries are to be recorded. It must
      contain a key of `global_step`.

  Returns:
    A summary op for each tensor to be recorded.
  """
  # It's not documented, but when recording summaries via TPUEstimator,
  # you need to pass in the global_step value to your host_call function.

  # Describe the difference between sum and max. All tensors are supposed to
  # be of shape [num_cores]
  global_step = kwargs.pop("global_step")[0]
  with tf.compat.v2.summary.create_file_writer(metrics_dir).as_default():
    with tf.compat.v2.summary.record_if(True):
      for name, tensor in kwargs.items():
        tf.compat.v2.summary.scalar(
            name, tf.reduce_mean(tensor), step=global_step)
        tf.compat.v2.summary.scalar(
            name + "_sum", tf.reduce_sum(tensor), step=global_step)
      return tf.summary.all_v2_summary_ops()


def model_fn_builder(
    model_config, padding_token_id,
    enable_side_inputs, num_replicas_concat,
    cross_block_attention_mode, init_checkpoint,
    learning_rate, summary_loss_weight,
    summary_num_layers, summary_num_cross_attention_heads,
    summary_enable_default_side_input, num_train_steps,
    num_warmup_steps, use_tpu, use_one_hot_embeddings,
    optimizer, poly_power, start_warmup_step,
    learning_rate_schedule, nbest_logits_for_eval):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    logging.info("*** Model: Params ***")
    for name in sorted(params.keys()):
      logging.info("  %s = %s", name, params[name])
    logging.info("*** Model: Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s", name, features[name].shape)

    model = modeling.ReadItTwiceBertModel(
        config=model_config, use_one_hot_embeddings=use_one_hot_embeddings)

    span_prediction_layer = modeling.SpanPredictionHead(
        intermediate_size=model_config.intermediate_size,
        dropout_rate=model_config.hidden_dropout_prob)

    # [batch_size, main_seq_length]
    token_ids = features["token_ids"]
    main_seq_length = tf.shape(token_ids)[1]
    block_ids = features["block_ids"]
    block_pos = features["block_pos"]

    annotation_begins = features.get("entity_annotation_begins")
    annotation_ends = features.get("entity_annotation_ends")
    annotation_labels = features.get("entity_annotation_labels")

    # Do not attend padding tokens
    # [batch_size, main_seq_length, main_seq_length]
    att_mask = tf.tile(
        tf.expand_dims(tf.not_equal(token_ids, padding_token_id), 1),
        [1, main_seq_length, 1])
    att_mask = tf.cast(att_mask, dtype=tf.int32)

    main_output = model(
        token_ids=token_ids,
        training=(mode == tf.estimator.ModeKeys.TRAIN),
        block_ids=block_ids,
        block_pos=block_pos,
        att_mask=att_mask,
        annotation_begins=annotation_begins,
        annotation_ends=annotation_ends,
        annotation_labels=annotation_labels,
        enable_side_inputs=enable_side_inputs,
        num_replicas_concat=num_replicas_concat,
        cross_block_attention_mode=cross_block_attention_mode)

    span_logits = span_prediction_layer(
        hidden_states=main_output.final_hidden_states,
        token_ids=token_ids,
        padding_token_id=padding_token_id,
        ignore_prefix_length=features["prefix_length"],
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    is_summary_loss_enabled = (
        mode == tf.estimator.ModeKeys.TRAIN and
        summary_loss_weight is not None and summary_loss_weight > 0)
    if is_summary_loss_enabled:
      logging.info("Using summary prediction loss with weight %.3f",
                   summary_loss_weight)
      summary_token_ids = features["summary_token_ids"]
      summary_labels = tf.roll(summary_token_ids, shift=-1, axis=1)
      decoder = modeling.ReadItTwiceDecoderModel(
          config=model_config,
          num_layers_override=summary_num_layers,
          num_cross_attention_heads=summary_num_cross_attention_heads,
          enable_default_side_input=summary_enable_default_side_input,
          use_one_hot_embeddings=use_one_hot_embeddings)
      summary_token_logits = decoder(
          token_ids=summary_token_ids,
          side_input=main_output.global_summary.states,
          token2side_input_att_mask=modeling.get_cross_block_att(
              block_ids,
              block_pos,
              main_output.global_summary.block_ids,
              main_output.global_summary.block_pos,
              cross_block_attention_mode="doc"),
          training=True)
      language_model_loss_fn = losses.LanguageModelLoss(
          decoder.get_token_embedding_table(),
          hidden_size=model_config.hidden_size)
      language_model_loss = language_model_loss_fn(
          summary_token_logits,
          summary_labels,
          padding_token_id=padding_token_id).loss
    else:
      language_model_loss = None

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = checkpoint_utils.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                   init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      host_inputs = dict()

      span_prediction_loss = losses.BatchSpanCrossEntropyLoss()

      qa_loss = span_prediction_loss(
          logits=span_logits,
          annotation_begins=features["answer_annotation_begins"],
          annotation_ends=features["answer_annotation_ends"],
          annotation_labels=features["answer_annotation_labels"],
          block_ids=block_ids,
          num_replicas=num_replicas_concat,
          eps=1e-5)
      host_inputs["train_metrics/qa_loss"] = tf.expand_dims(qa_loss, 0)

      if language_model_loss is not None:
        total_loss = (1.0 / (1.0 + summary_loss_weight) * qa_loss +
                      summary_loss_weight /
                      (1.0 + summary_loss_weight) * language_model_loss)
        host_inputs["train_metrics/summary_lm_loss"] = tf.expand_dims(
            language_model_loss, 0)
      else:
        total_loss = qa_loss

      # Add regularization losses.
      if model.losses:
        total_loss += tf.math.add_n(model.losses)

      train_op = optimization.create_optimizer(
          total_loss,
          learning_rate,
          num_train_steps,
          num_warmup_steps,
          use_tpu,
          optimizer,
          poly_power,
          start_warmup_step,
          learning_rate_schedule,
          reduce_loss_sum=True)

      host_inputs.update({
          "global_step":
              tf.expand_dims(tf.train.get_or_create_global_step(), 0),
          "train_metrics/loss":
              tf.expand_dims(total_loss, 0),
      })

      host_call = (functools.partial(
          record_summary_host_fn,
          metrics_dir=os.path.join(FLAGS.output_dir,
                                   "train_metrics")), host_inputs)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      begin_logits_values, begin_logits_indices = tf.math.top_k(
          span_logits[:, :, 0],
          k=nbest_logits_for_eval,
      )
      end_logits_values, end_logits_indices = tf.math.top_k(
          span_logits[:, :, 1],
          k=nbest_logits_for_eval,
      )

      predictions = {
          "block_ids": tf.identity(block_ids),
          "begin_logits_values": begin_logits_values,
          "begin_logits_indices": begin_logits_indices,
          "end_logits_values": end_logits_values,
          "end_logits_indices": end_logits_indices,
          "token_ids": tf.identity(token_ids),
      }
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes is supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_files,
                     is_training,
                     num_cpu_threads = 4,
                     include_summary_feature=False):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  num_blocks_per_example, block_length = input_utils.get_block_params_from_input_file(
      input_files[0])
  max_num_annotations = input_utils.get_num_annotations_from_input_file(
      input_files[0])
  logging.info("***** Building Input pipeline *****")
  logging.info("  Number of blocks per example = %d", num_blocks_per_example)
  logging.info("  Block length = %d", block_length)
  logging.info("  Number of anntotations per block = %d", max_num_annotations)

  def input_fn(params):
    """The actual input function."""
    logging.info("*** Input: Params ***")
    for name in sorted(params.keys()):
      logging.info("  %s = %s", name, params[name])

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))

      # From https://www.tensorflow.org/guide/data#randomly_shuffling_input_data
      # Dataset.shuffle doesn't signal the end of an epoch until the shuffle
      # buffer is empty. So a shuffle placed before a repeat will show every
      # element of one epoch before moving to the next.
      d = d.shuffle(buffer_size=len(input_files))
      d = d.repeat()

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=1000)
    else:
      d = tf.data.TFRecordDataset(input_files)

    extra_int_features_shapes = dict()
    if include_summary_feature:
      extra_int_features_shapes["summary_token_ids"] = [
          num_blocks_per_example, block_length
      ]
    d = d.map(
        input_utils.get_span_prediction_example_decode_fn(
            num_blocks_per_example,
            block_length,
            max_num_answer_annotations=max_num_annotations,
            max_num_entity_annotations=max_num_annotations,
            extra_int_features_shapes=extra_int_features_shapes,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    return d

  return input_fn


def validate_flags():
  """Basic flag validation."""
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  if FLAGS.summary_loss_weight:
    if not FLAGS.enable_side_inputs:
      raise ValueError("`enable_side_inputs` should be specified "
                       "if summary loss is enabled")
    if not FLAGS.summary_num_layers or FLAGS.summary_num_layers <= 0:
      raise ValueError("`summary_num_layers` should be specified "
                       "if summary loss is enabled")
    if (FLAGS.summary_num_cross_attention_heads is None or
        FLAGS.summary_num_cross_attention_heads < 0):
      raise ValueError(
          "`summary_num_cross_attention_heads` should be specified "
          "if summary loss is enabled")


def main(_):
  logging.set_verbosity(logging.INFO)

  validate_flags()
  tf.io.gfile.makedirs(FLAGS.output_dir)


  for flag in FLAGS.flags_by_module_dict()[sys.argv[0]]:
    logging.info("  %s = %s", flag.name, flag.value)

  model_config = config.get_model_config(
      model_dir=FLAGS.output_dir,
      source_file=FLAGS.read_it_twice_bert_config_file,
      source_base64=None,
      write_from_source=False)

  if FLAGS.cross_attention_top_k is not None:
    model_config = dataclasses.replace(
        model_config, cross_attention_top_k=FLAGS.cross_attention_top_k)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  logging.info("*** Input Files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  num_blocks_per_example, block_length = input_utils.get_block_params_from_input_file(
      input_files[0])

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # PER_HOST_V1: iterator.get_next() is called 1 time with per_worker_batch_size
  # PER_HOST_V2: iterator.get_next() is called 8 times with per_core_batch_size
  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      # Keep all checkpoints
      keep_checkpoint_max=None,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          per_host_input_for_training=is_per_host,
          experimental_host_call_every_n_steps=FLAGS.steps_per_summary))

  # TODO(urikz): Is there a better way to compute the number of tasks?
  # the code below doesn't work because `tpu_cluster_resolver.cluster_spec()`
  # returns None. Therefore, I have to pass number of total tasks via CLI arg.
  # num_tpu_tasks = tpu_cluster_resolver.cluster_spec().num_tasks()
  batch_size = (FLAGS.num_tpu_tasks or 1) * num_blocks_per_example

  num_train_examples = input_utils.get_num_examples_in_tf_records(input_files)
  num_train_steps = int(num_train_examples * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  logging.info("***** Input configuration *****")
  logging.info("  Number of blocks per example = %d", num_blocks_per_example)
  logging.info("  Block length = %d", block_length)
  logging.info("  Number of TPU tasks = %d", FLAGS.num_tpu_tasks or 1)
  logging.info("  Batch size = %d", batch_size)
  logging.info("  Number of TPU cores = %d", FLAGS.num_tpu_cores or 0)
  logging.info("  Number training steps = %d", num_train_steps)
  logging.info("  Number warmup steps = %d", num_warmup_steps)

  model_fn = model_fn_builder(
      model_config=model_config,
      padding_token_id=FLAGS.padding_token_id,
      enable_side_inputs=FLAGS.enable_side_inputs,
      num_replicas_concat=FLAGS.num_tpu_cores,
      cross_block_attention_mode=FLAGS.cross_block_attention_mode,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      summary_loss_weight=FLAGS.summary_loss_weight,
      summary_num_layers=FLAGS.summary_num_layers,
      summary_num_cross_attention_heads=FLAGS.summary_num_cross_attention_heads,
      summary_enable_default_side_input=FLAGS.summary_enable_default_side_input,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      learning_rate_schedule=FLAGS.learning_rate_schedule,
      nbest_logits_for_eval=FLAGS.decode_top_k)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size)

  training_done_path = os.path.join(FLAGS.output_dir, "training_done")

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    train_input_fn = input_fn_builder(
        input_files=input_files,
        is_training=True,
        include_summary_feature=bool(FLAGS.summary_loss_weight))
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    # Write file to signal training is done.
    with tf.gfile.GFile(training_done_path, "w") as writer:
      writer.write("\n")

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    eval_input_fn = input_fn_builder(input_files=input_files, is_training=False)

    if FLAGS.input_qaps is None:
      raise ValueError("Must specify `input_qaps` for eval.")

    if FLAGS.eval_data_split is None:
      raise ValueError("Must specify `eval_data_split` for eval.")

    ground_truth = {}
    # We skip the first question ID as it corresponds on a padding document.
    question_ids = [None]
    with tf.io.gfile.GFile(FLAGS.input_qaps) as f:
      reader = csv.DictReader(f)
      for datum in reader:
        if datum["set"] != FLAGS.eval_data_split:
          continue
        question_id = datum["document_id"] + ": " + datum["question"]
        ground_truth[question_id] = [datum["answer1"], datum["answer2"]]
        question_ids.append(question_id)

    logging.info("Loaded %d questions for evaluation from %s for set %s",
                 len(ground_truth), FLAGS.input_qaps, FLAGS.eval_data_split)

    tokenizer = tokenization.FullTokenizer(FLAGS.spm_model_path)
    logging.info("Loaded SentencePiece model from %s", FLAGS.spm_model_path)

    # Writer for TensorBoard.
    summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.output_dir, "%s_metrics" % FLAGS.eval_data_split))

    if _is_checkpoint_in_the_model_dir():
      logging.info("FINAL EVALUATION: checkpoint = %s, data split = %s",
                   FLAGS.init_checkpoint, FLAGS.eval_data_split)
      checkpoints = [FLAGS.init_checkpoint]
    else:
      # for checkpoint_path in _get_all_checkpoints(FLAGS.output_dir):
      checkpoints = tf.train.checkpoints_iterator(
          FLAGS.output_dir, min_interval_secs=5 * 60, timeout=8 * 60 * 60)

    for checkpoint_path in checkpoints:
      start_time = time.time()
      global_step = _get_global_step_for_checkpoint(checkpoint_path)
      if global_step == 0:
        continue
      logging.info("Starting eval on step %d on checkpoint: %s", global_step,
                   checkpoint_path)

      nbest_predictions = collections.OrderedDict()

      try:
        for prediction_index, prediction in enumerate(
            estimator.predict(
                eval_input_fn,
                checkpoint_path=checkpoint_path,
                yield_single_examples=True)):
          block_id = prediction["block_ids"]
          if block_id == 0:
            # Padding document
            continue
          question_id = question_ids[block_id]
          if question_id not in nbest_predictions:
            nbest_predictions[question_id] = {}
          token_ids = prediction["token_ids"]
          for begin_index, begin_logit in zip(
              prediction["begin_logits_indices"],
              prediction["begin_logits_values"]):
            for end_index, end_logit in zip(prediction["end_logits_indices"],
                                            prediction["end_logits_values"]):
              if begin_index > end_index or end_index - begin_index + 1 > FLAGS.decode_max_size:
                continue
              tokens = list(
                  tokenizer.convert_ids_to_tokens([
                      int(token_id)
                      for token_id in token_ids[begin_index:end_index + 1]
                  ]))

              answer = "".join(tokens)
              answer = answer.replace(tokenization.SPIECE_UNDERLINE,
                                      " ").strip()
              if not answer:
                continue
              if answer not in nbest_predictions[question_id]:
                nbest_predictions[question_id][answer] = []
              nbest_predictions[question_id][answer].append(begin_logit +
                                                            end_logit)

        nbest_predictions_probs = _convert_prediction_logits_to_probs(
            nbest_predictions)
        best_predictions_max = _get_best_predictions(nbest_predictions_probs,
                                                     max)
        with tf.gfile.GFile(
            checkpoint_path +
            ".%s.best_predictions_max.json" % FLAGS.eval_data_split, "w") as f:
          json.dump(best_predictions_max, f, indent=2)

        best_predictions_max_results = evaluation.evaluate_narrative_qa(
            ground_truth, best_predictions_max)
        write_eval_results(global_step, best_predictions_max_results, "max",
                           summary_writer)

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info("Checkpoint %s no longer exists, skipping checkpoint",
                        checkpoint_path)
        continue

      if tf.io.gfile.exists(training_done_path):
        # Break if the checkpoint we just processed is the last one.
        last_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
        if last_checkpoint is None:
          continue
        last_global_step = _get_global_step_for_checkpoint(last_checkpoint)
        if global_step == last_global_step:
          break

      global_step = _get_global_step_for_checkpoint(checkpoint_path)
      logging.info("Finished eval on step %d in %d seconds", global_step,
                   time.time() - start_time)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  flags.mark_flags_as_required(["input_file", "output_dir"])
  app.run(main)
