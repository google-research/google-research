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

"""Run MLM and coref. resolution loss pre-training for ReadTwice."""
import functools
import os
import time
from typing import Text

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from readtwice.models import checkpoint_utils
from readtwice.models import config
from readtwice.models import input_utils
from readtwice.models import losses
from readtwice.models import metric_utils
from readtwice.models import modeling
from readtwice.models import optimization

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "source_model_config_file", None,
    "The source config file corresponding to the ETC ReadItTwiceBERT model. "
    "This specifies the model architecture. When first training the model, "
    "this file will be copied to a `read_it_twice_bert_config.json` file in the "
    "model directory, and future calls to this binary will use that file "
    "instead, ignoring this flag.")

flags.DEFINE_string(
    "source_model_config_base64", None,
    "A source config json Base64 string corresponding to the ETC ReadItTwiceBERT "
    "model. This has the same role as `source_model_config_file` and serves as "
    "an alternative. Only one should be specified, not both. When first "
    "training the model, this json config will be copied to a "
    "`read_it_twice_bert_config.json` file in the model directory, and future calls "
    "to this binary will use that file instead, ignoring this flag.")

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

## Parameters for the input pipeline -- `PretrainInputConfig`
flags.DEFINE_float(
    "mlm_fraction_to_mask", 0.15,
    "The fraction of tokens to mask for masked language model loss.")

flags.DEFINE_float(
    "mlm_entity_fraction_to_mask", None,
    "The fraction of entities to mask for masked language model loss.")

flags.DEFINE_string(
    "mention_mask_mode", "whole_mention",
    "Mentions masking strategy. Possible options: "
    "`whole_mention`, `whole_entity` and `whole_entity_batch`.")

flags.DEFINE_integer(
    "mlm_max_consecutive_masks", 5,
    "Maximum number of consecutive tokens to mask at a time. The actual number "
    "of consecutive masks will be uniformly sampled between 1 and this number "
    "(both inclusive).")

flags.DEFINE_bool(
    "mlm_use_whole_word", True,
    "Whether to mask whole words for the MLM task instead of just WordPieces. "
    "This requires the `is_continuation` feature to be present in the "
    "tensorflow Examples.")

flags.DEFINE_integer(
    "mask_token_id", 4,
    "The token id of the mask token according to the WordPiece vocabulary.")

flags.DEFINE_integer(
    "padding_token_id", 0,
    "The token id of the padding token according to the WordPiece vocabulary.")

# Parameters for the training objective
flags.DEFINE_bool(
    "enable_side_inputs", False,
    "If True, enables read-it-twice model. Otherwise, the model becomes equivalent to the standard Transformer model."
)

flags.DEFINE_integer(
    "num_replicas_concat", None,
    "Number of replicas to gather summaries from. If None (default) then cross-replicas summaries are not used."
)

flags.DEFINE_enum(
    "cross_block_attention_mode", "doc",
    ["block", "doc", "batch", "other_blocks"],
    "The policy on how summaries between different "
    "blocks are allowed to interact with each other.")

flags.DEFINE_string("extra_loss", None,
                    "Auxiliary loss to use. Options are sdp, spd_linear")

flags.DEFINE_integer("summary_num_layers", None,
                     "Number of layers for the summary prediction task.")

flags.DEFINE_integer(
    "summary_num_cross_attention_heads", None,
    "Number of attention heads for the summary prediction task.")

flags.DEFINE_bool(
    "summary_enable_default_side_input", False,
    "Add a default side input, which acts like a no-op attention, "
    "effective allowing attention weights to sum up to something less than 1.")

flags.DEFINE_string("metrics_name", None, "Name for logging metrics.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_enum(
    "learning_rate_schedule", "poly_decay", ["poly_decay", "inverse_sqrt"],
    "The learning rate schedule to use. The default of "
    "`poly_decay` uses tf.train.polynomial_decay, while "
    "`inverse_sqrt` uses inverse sqrt of time after the warmup.")

flags.DEFINE_float("poly_power", 1.0,
                   "The power of poly decay if using `poly_decay` schedule.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("num_eval_epochs", 2, "Number of eval epochs.")

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


def record_summary_host_fn(metrics_dir, metrics_name, **kwargs):
  """A host_fn function for the host_call in TPUEstimatorSpec.

  Args:
    metrics_dir: Directory where tf summary events should be written.
    metrics_name: Name for the metrics collection
    **kwargs: Contains tensors for which summaries are to be recorded. It must
      contain a key of `global_step`.

  Returns:
    A summary op for each tensor to be recorded.
  """
  # It's not documented, but when recording summaries via TPUEstimator,
  # you need to pass in the global_step value to your host_call function.
  global_step = kwargs.pop("global_step")[0]

  mlm_loss_per_sample = kwargs.pop("mlm_loss_per_sample")
  mlm_accuracy_per_sample = kwargs.pop("mlm_accuracy_per_sample")
  mlm_weight_per_sample = kwargs.pop("mlm_weight_per_sample")
  block_ids = kwargs.pop("block_ids")

  mlm_loss_per_entity_sample = kwargs.pop("mlm_loss_per_entity_sample", None)
  mlm_accuracy_per_entity_sample = kwargs.pop("mlm_accuracy_per_entity_sample",
                                              None)
  mlm_weight_per_entity_sample = kwargs.pop("mlm_weight_per_entity_sample",
                                            None)
  mlm_loss_per_non_entity_sample = kwargs.pop("mlm_loss_per_non_entity_sample",
                                              None)
  mlm_accuracy_per_non_entity_sample = kwargs.pop(
      "mlm_accuracy_per_non_entity_sample", None)
  mlm_weight_per_non_entity_sample = kwargs.pop(
      "mlm_weight_per_non_entity_sample", None)

  other_metrics = metric_utils.masked_lm_metrics(
      mlm_loss_per_sample,
      mlm_accuracy_per_sample,
      mlm_weight_per_sample,
      block_ids,
      mlm_loss_per_entity_sample=mlm_loss_per_entity_sample,
      mlm_accuracy_per_entity_sample=mlm_accuracy_per_entity_sample,
      mlm_weight_per_entity_sample=mlm_weight_per_entity_sample,
      mlm_loss_per_non_entity_sample=mlm_loss_per_non_entity_sample,
      mlm_accuracy_per_non_entity_sample=mlm_accuracy_per_non_entity_sample,
      mlm_weight_per_non_entity_sample=mlm_weight_per_non_entity_sample,
      is_train=True,
      metrics_name=metrics_name or "train_metrics")

  with tf.compat.v2.summary.create_file_writer(metrics_dir).as_default():
    with tf.compat.v2.summary.record_if(True):
      for name, tensor in kwargs.items():
        tf.compat.v2.summary.scalar(
            name, tf.reduce_mean(tensor), step=global_step)
      for name, tensor in other_metrics.items():
        tf.compat.v2.summary.scalar(
            name, tf.reduce_mean(tensor), step=global_step)
      return tf.summary.all_v2_summary_ops()


def model_fn_builder(model_config, padding_token_id, enable_side_inputs,
                     num_replicas_concat, cross_block_attention_mode,
                     extra_loss, summary_num_layers,
                     summary_num_cross_attention_heads,
                     summary_enable_default_side_input, init_checkpoint,
                     learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step, learning_rate_schedule, metrics_name):
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

    # [batch_size, main_seq_length]
    token_ids = features["token_ids"]
    batch_size = tf.shape(token_ids)[0]
    main_seq_length = tf.shape(token_ids)[1]
    block_ids = features["block_ids"]
    block_pos = features["block_pos"]

    annotation_begins = features.get("annotation_begins")
    annotation_ends = features.get("annotation_ends")
    annotation_labels = features.get("annotation_labels")

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

    mlm_loss_fn = losses.LanguageModelLoss(
        model.get_token_embedding_table(),
        hidden_size=model_config.hidden_size,
        name="mlm_loss")
    mlm_loss_output = mlm_loss_fn(
        input_tensor=main_output.final_hidden_states,
        label_ids=features["masked_lm_ids"],
        positions=features["masked_lm_positions"],
        label_weights=features["masked_lm_weights"],
        mlm_is_entity_mask=features.get("mlm_is_entity_mask"),
        mlm_is_not_entity_mask=features.get("mlm_is_not_entity_mask"),
        padding_token_id=padding_token_id)
    mlm_loss = mlm_loss_output.loss

    loss_to_log = dict(mlm_loss=tf.expand_dims(mlm_loss, 0))
    loss_weight_denominator = 1.0 + sum(extra_loss.values())
    total_loss = mlm_loss * (1.0 / loss_weight_denominator)
    for loss_name, loss_weight in extra_loss.items():
      logging.info("EXTRA LOSS: %s with weight %.2f", loss_name,
                   loss_weight / loss_weight_denominator)

      if model_config.summary_mode == "entity":
        # entity label "1" corresponds to unknown entity
        # there is no need to compute coreferense resolution loss
        # for these unknown entities.
        labels_weight = tf.cast(
            tf.logical_and(
                tf.not_equal(
                    tf.expand_dims(main_output.local_summary.labels, 1), 1),
                tf.not_equal(
                    tf.expand_dims(main_output.global_summary.labels, 0), 1)),
            tf.float32)
      else:
        labels_weight = None

      if loss_name == "sdp":
        loss_fn = losses.BatchCoreferenceResolutionLoss(
            apply_linear_layer=False)
        loss_value = loss_fn(
            main_output.local_summary.states,
            main_output.local_summary.labels,
            main_output.global_summary.states,
            main_output.global_summary.labels,
            labels_weight=labels_weight)
      elif loss_name == "sdp_linear":
        loss_fn = losses.BatchCoreferenceResolutionLoss(apply_linear_layer=True)
        loss_value = loss_fn(
            main_output.local_summary.states,
            main_output.local_summary.labels,
            main_output.global_summary.states,
            main_output.global_summary.labels,
            labels_weight=labels_weight)
      elif loss_name == "spp_linear":
        loss_fn = losses.BatchCoreferenceResolutionLoss(apply_linear_layer=True)
        # Positive examples are blocks which go one after another in the
        # original document.
        labels_mask = tf.less_equal(
            tf.abs(
                tf.expand_dims(main_output.local_summary.block_pos, 1) -
                tf.expand_dims(main_output.global_summary.block_pos, 0)), 1)
        loss_value = loss_fn(
            main_output.local_summary.states,
            main_output.local_summary.labels,
            main_output.global_summary.states,
            main_output.global_summary.labels,
            labels_mask=labels_mask,
            labels_weight=labels_weight)
      elif loss_name == "lm":
        token_labels = tf.roll(token_ids, shift=-1, axis=1)
        # [batch_size, global_batch_size]
        token2side_input_att_mask = modeling.get_cross_block_att(
            block_ids,
            block_pos,
            main_output.global_summary.block_ids,
            main_output.global_summary.block_pos,
            cross_block_attention_mode=cross_block_attention_mode,
            cast_to_int32=False)
        # We want to exclude the summary of the block itself
        # from decoder side input. As a proxy for this, we use block_ids AND
        # block_pos.
        samples_are_the_same = tf.logical_and(
            tf.equal(
                tf.expand_dims(block_ids, 1),
                tf.expand_dims(main_output.global_summary.block_ids, 0)),
            tf.equal(
                tf.expand_dims(block_pos, 1),
                tf.expand_dims(main_output.global_summary.block_pos, 0)))
        token2side_input_att_mask = tf.stop_gradient(
            tf.cast(
                tf.logical_and(token2side_input_att_mask,
                               tf.logical_not(samples_are_the_same)),
                dtype=tf.int32))

        decoder = modeling.ReadItTwiceDecoderModel(
            config=model_config,
            num_layers_override=summary_num_layers,
            num_cross_attention_heads=summary_num_cross_attention_heads,
            enable_default_side_input=summary_enable_default_side_input,
            use_one_hot_embeddings=use_one_hot_embeddings)
        summary_token_logits = decoder(
            token_ids=token_ids,
            side_input=main_output.global_summary.states,
            token2side_input_att_mask=token2side_input_att_mask,
            training=True)
        language_model_loss_fn = losses.LanguageModelLoss(
            decoder.get_token_embedding_table(),
            hidden_size=model_config.hidden_size)

        # We don't penalize the first and last 32 tokens, so the model does not
        # have incentive to memoize tokens at the border of blocks.
        labels_weights = tf.concat([
            tf.zeros([batch_size, 32], dtype=tf.bool),
            tf.ones([batch_size, main_seq_length - 32 * 2], dtype=tf.bool),
            tf.zeros([batch_size, 32], dtype=tf.bool)
        ],
                                   axis=1)
        labels_weights = tf.logical_and(
            labels_weights, tf.not_equal(token_labels, padding_token_id))
        labels_weights = tf.stop_gradient(
            tf.cast(labels_weights, dtype=tf.float32))

        loss_value = language_model_loss_fn(
            summary_token_logits, token_labels,
            label_weights=labels_weights).loss
      else:
        raise ValueError("Unknown extra loss: {}".format(loss_name))

      loss_to_log[loss_name] = tf.expand_dims(loss_value, 0)
      total_loss += loss_value * (loss_weight / loss_weight_denominator)

    if model.losses:
      total_loss += tf.math.add_n(model.losses)

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

    metric_fn_tensors = dict(
        mlm_loss_per_sample=mlm_loss_output.mlm_loss_per_sample,
        mlm_accuracy_per_sample=mlm_loss_output.mlm_accuracy_per_sample,
        mlm_weight_per_sample=mlm_loss_output.mlm_weight_per_sample,
        mlm_loss_per_entity_sample=mlm_loss_output.mlm_loss_per_entity_sample,
        mlm_accuracy_per_entity_sample=mlm_loss_output
        .mlm_accuracy_per_entity_sample,
        mlm_weight_per_entity_sample=mlm_loss_output
        .mlm_weight_per_entity_sample,
        mlm_loss_per_non_entity_sample=mlm_loss_output
        .mlm_loss_per_non_entity_sample,
        mlm_accuracy_per_non_entity_sample=mlm_loss_output
        .mlm_accuracy_per_non_entity_sample,
        mlm_weight_per_non_entity_sample=mlm_loss_output
        .mlm_weight_per_non_entity_sample,
        block_ids=block_ids)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          optimizer, poly_power, start_warmup_step, learning_rate_schedule)

      metric_fn_tensors.update({
          "global_step":
              tf.expand_dims(tf.train.get_or_create_global_step(), 0),
          "loss":
              tf.expand_dims(total_loss, 0),
      })
      metric_fn_tensors.update(loss_to_log)

      host_call = (functools.partial(
          record_summary_host_fn,
          metrics_dir=os.path.join(FLAGS.output_dir, "train_metrics"),
          metrics_name=metrics_name or "train_metrics"), metric_fn_tensors)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)

    elif mode == tf.estimator.ModeKeys.EVAL:

      eval_metrics = (functools.partial(
          metric_utils.masked_lm_metrics,
          is_train=False,
          metrics_name=metrics_name or "eval_metrics"), metric_fn_tensors)
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % mode)

    return output_spec

  return model_fn


def input_fn_builder(input_files,
                     input_config,  # pylint: disable=unused-argument
                     model_config,  # pylint: disable=unused-argument
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

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
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    raise ValueError("THIS SCRIPT IS NOT RUNNABLE AND FOR DEMONSTATION ONLY.")
    # d = d.map(
    #     pretrain_input_utils.get_pretrain_example_decode_fn(
    #         input_config, model_config),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # d = d.prefetch(tf.data.experimental.AUTOTUNE)
    # return d

  return input_fn


def validate_flags():
  """Basic flag validation."""
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")


def parse_extra_loss_flag(extra_loss_flag):
  """Return extra loss config."""
  extra_loss_dict = dict()
  if not extra_loss_flag:
    return extra_loss_dict
  extra_loss = extra_loss_flag.split(",")
  for loss_name_weight in extra_loss:
    loss_name, loss_weight = loss_name_weight.split(":")
    loss_weight = float(loss_weight)
    if loss_weight < 0 or loss_name in extra_loss_dict:
      raise ValueError("Invalid `extra_loss`: {}".format(extra_loss_flag))
    extra_loss_dict[loss_name] = loss_weight
  return extra_loss_dict


def main(_):
  logging.set_verbosity(logging.INFO)

  validate_flags()

  tf.io.gfile.makedirs(FLAGS.output_dir)

  model_config = config.get_model_config(
      model_dir=FLAGS.output_dir,
      source_file=FLAGS.source_model_config_file,
      source_base64=FLAGS.source_model_config_base64,
      write_from_source=FLAGS.do_train)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  num_blocks_per_example, block_length = input_utils.get_block_params_from_input_file(
      input_files[0])
  max_num_annotations = None
  if FLAGS.mlm_entity_fraction_to_mask is not None:
    max_num_annotations = input_utils.get_num_annotations_from_input_file(
        input_files[0])

  # TODO(urikz): Define `pretrain_input_utils`.
  input_config = pretrain_input_utils.PretrainInputConfig(  # pylint: disable=undefined-variable
      num_blocks_per_example=num_blocks_per_example,
      block_length=block_length,
      mlm_fraction_to_mask=FLAGS.mlm_fraction_to_mask,
      mlm_max_consecutive_masks=FLAGS.mlm_max_consecutive_masks,
      mlm_use_whole_word=FLAGS.mlm_use_whole_word,
      mask_token_id=FLAGS.mask_token_id,
      padding_token_id=FLAGS.padding_token_id,
      max_num_annotations=max_num_annotations,
      mlm_entity_fraction_to_mask=FLAGS.mlm_entity_fraction_to_mask,
      mention_mask_mode=FLAGS.mention_mask_mode)

  logging.info("*** Input Files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  tpu_cluster_resolver, num_tpu_tasks, num_tpu_cores = None, None, None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_system_metadata = tpu_cluster_resolver.get_tpu_system_metadata()
    num_tpu_tasks = tpu_cluster_resolver.get_tpu_system_metadata().num_hosts
    num_tpu_cores = tpu_system_metadata.num_cores

  # PER_HOST_V1: iterator.get_next() is called 1 time with per_worker_batch_size
  # PER_HOST_V2: iterator.get_next() is called 8 times with per_core_batch_size
  # pylint: enable=line-too-long
  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          per_host_input_for_training=is_per_host,
          experimental_host_call_every_n_steps=FLAGS.iterations_per_loop))

  batch_size = (num_tpu_tasks or 1) * num_blocks_per_example
  logging.info("***** Input configuration *****")
  logging.info("  Number of blocks per example = %d", num_blocks_per_example)
  logging.info("  Block length = %d", block_length)
  logging.info("  Number of TPU tasks = %d", num_tpu_tasks or 0)
  logging.info("  Batch size = %d", batch_size)
  logging.info("  Number of TPU cores = %d", num_tpu_cores or 0)
  logging.info("  Number of annotations per example = %d",
               input_config.max_num_annotations or 0)

  model_fn = model_fn_builder(
      model_config=model_config,
      padding_token_id=FLAGS.padding_token_id,
      enable_side_inputs=FLAGS.enable_side_inputs,
      num_replicas_concat=num_tpu_cores,
      cross_block_attention_mode=FLAGS.cross_block_attention_mode,
      extra_loss=parse_extra_loss_flag(FLAGS.extra_loss),
      summary_num_layers=FLAGS.summary_num_layers,
      summary_num_cross_attention_heads=FLAGS.summary_num_cross_attention_heads,
      summary_enable_default_side_input=FLAGS.summary_enable_default_side_input,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      learning_rate_schedule=FLAGS.learning_rate_schedule,
      metrics_name=FLAGS.metrics_name)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size)

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    train_input_fn = input_fn_builder(
        input_files=input_files,
        input_config=input_config,
        model_config=model_config,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    num_eval_examples = input_utils.get_num_examples_in_tf_records(input_files)
    eval_steps_per_epoch = num_eval_examples
    max_eval_steps = eval_steps_per_epoch * FLAGS.num_eval_epochs
    logging.info("  Number of eval examples = %d", num_eval_examples)
    logging.info("  Number of TPU tasks = %d", num_tpu_tasks or 1)
    logging.info("  Number of eval steps per epoch = %d", eval_steps_per_epoch)
    logging.info("  Eval steps = %d", max_eval_steps)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        input_config=input_config,
        model_config=model_config,
        is_training=False)

    # Run evaluation for each new checkpoint.
    for ckpt in tf.train.checkpoints_iterator(FLAGS.output_dir):
      logging.info("Starting eval on new checkpoint: %s", ckpt)
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn, steps=max_eval_steps, checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        logging.info("Eval results: %s. Elapsed seconds: %d", eval_results,
                     elapsed_time)

        # Terminate eval job when final checkpoint is reached.
        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= FLAGS.num_train_steps:
          logging.info("Evaluation finished after training step %d",
                       current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info("Checkpoint %s no longer exists, skipping checkpoint",
                     ckpt)


if __name__ == "__main__":
  flags.mark_flags_as_required(["input_file", "output_dir"])
  app.run(main)
