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

# Lint as: python3
"""Run training and evaluation for DrFact models."""

import collections
import functools
import json
import os
import re
import time

from absl import flags
from garcon.albert import tokenization as albert_tokenization
from bert import modeling
from bert import optimization
from bert import tokenization as bert_tokenization
from language.google.drfact import evaluate
from language.google.drfact import input_fns
from language.google.drfact import model_fns
from language.labs.drkit import search_utils
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import memory_stats as contrib_memory_stats

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("tokenizer_type", "bert_tokenization",
                    "The tokenizier type that the BERT model was trained on.")
flags.DEFINE_string("tokenizer_model_file", None,
                    "The tokenizier model that the BERT was trained with.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "output_prediction_file", "test_predictions.json",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None, "JSON for training.")

flags.DEFINE_string("predict_file", None, "JSON for predictions.")
flags.DEFINE_string("predict_prefix", "dev", "JSON for predictions.")

flags.DEFINE_string("test_file", None, "JSON for predictions.")

flags.DEFINE_string("data_type", "onehop",
                    "Whether queries are `onehop` or `twohop`.")

flags.DEFINE_string("model_type", "drkit",
                    "Whether to use `onehop` or `twohop` model.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("train_data_dir", None,
                    "Location of entity/mention/fact files for training data.")

flags.DEFINE_string("f2f_index_dir", None,
                    "Location of fact2fact files for training data.")

flags.DEFINE_string("test_data_dir", None,
                    "Location of entity/mention/fact files for test data.")

flags.DEFINE_string("model_ckpt_toload", "best_model",
                    "Name of the checkpoints.")

flags.DEFINE_string("test_model_ckpt", "best_model", "Name of the checkpoints.")

flags.DEFINE_string("embed_index_prefix", "bert_large", "Prefix of indexes.")

flags.DEFINE_integer("num_hops", 2, "Number of hops in rule template.")

flags.DEFINE_integer("max_entity_len", 4,
                     "Maximum number of tokens in an entity name.")

flags.DEFINE_integer(
    "num_mips_neighbors", 100,
    "Number of nearest neighbor mentions to retrieve for queries in each hop.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "projection_dim", None, "Number of dimensions to project embeddings to. "
    "Set to None to use full dimensions.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", False, "Whether to run eval on the test set.")

flags.DEFINE_float(
    "subject_mention_probability", 0.0,
    "Fraction of training instances for which we use subject "
    "mentions in the text as opposed to canonical names.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 300,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("supervision", "entity",
                    "Type of supervision -- `mention` or `entity`.")

flags.DEFINE_float("entity_score_threshold", 1e-2,
                   "Minimum score of an entity to retrieve sparse neighbors.")
flags.DEFINE_float("fact_score_threshold", 1e-2,
                   "Minimum score of a fact to retrieve sparse neighbors.")
flags.DEFINE_float("softmax_temperature", 2.,
                   "Temperature before computing softmax.")

flags.DEFINE_string(
    "sparse_reduce_fn", "max",
    "Function to aggregate sparse search results for a set of "
    "entities.")

flags.DEFINE_string("sparse_strategy", "dense_first",
                    "How to combine sparse and dense components.")

flags.DEFINE_boolean("intermediate_loss", False,
                     "Compute loss on intermediate layers.")

flags.DEFINE_boolean("light", False, "If true run in light mode.")
flags.DEFINE_boolean("is_excluding", False,
                     "If true exclude question and wrong choices' concepts.")

flags.DEFINE_string(
    "qry_layers_to_use", "-1",
    "Comma-separated list of layer representations to use as the fixed "
    "query representation.")

flags.DEFINE_string(
    "qry_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`qry_layers`.")

flags.DEFINE_string(
    "entity_score_aggregation_fn", "max",
    "Aggregation method for combining the mention logits to entities.")

flags.DEFINE_float("question_dropout", 0.2,
                   "Dropout probability for question BiLSTMs.")

flags.DEFINE_integer("question_num_layers", 2,
                     "Number of layers for question BiLSTMs.")

flags.DEFINE_integer("num_preds", 100, "Use -1 for all predictions.")

flags.DEFINE_boolean(
    "ensure_answer_sparse", False,
    "If true, ensures answer is among sparse retrieval results"
    "during training.")

flags.DEFINE_boolean(
    "ensure_answer_dense", False,
    "If true, ensures answer is among dense retrieval results "
    "during training.")

flags.DEFINE_boolean(
    "train_with_sparse", True,
    "If true, multiplies logits with sparse retrieval results "
    "during training.")

flags.DEFINE_boolean(
    "predict_with_sparse", True,
    "If true, multiplies logits with sparse retrieval results "
    "during inference.")

flags.DEFINE_boolean("fix_sparse_to_one", True,
                     "If true, sparse search matrix is fixed to {0,1}.")

flags.DEFINE_boolean("l2_normalize_db", False,
                     "If true, pre-trained embeddings are normalized to 1.")

flags.DEFINE_boolean("load_only_bert", False,
                     "To load only BERT variables from init_checkpoint.")

flags.DEFINE_boolean(
    "use_best_ckpt_for_predict", False,
    "If True, loads the best_model checkpoint in model_dir, "
    "instead of the latest one.")

flags.DEFINE_bool("profile_model", False, "Whether to run profiling.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("random_seed", 1, "Random seed for reproducibility.")

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

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool("debug", False,
                  "If true, only print the flags but not run anything.")


class QAConfig(object):
  """Hyperparameters for the QA model."""

  def __init__(self, qry_layers_to_use, qry_aggregation_fn, dropout,
               qry_num_layers, projection_dim, num_entities, max_entity_len,
               ensure_answer_sparse, ensure_answer_dense, train_with_sparse,
               predict_with_sparse, fix_sparse_to_one, supervision,
               l2_normalize_db, entity_score_aggregation_fn,
               entity_score_threshold, fact_score_threshold,
               softmax_temperature, sparse_reduce_fn, intermediate_loss,
               train_batch_size, predict_batch_size, light, sparse_strategy,
               load_only_bert):
    self.qry_layers_to_use = [int(vv) for vv in qry_layers_to_use.split(",")]
    self.qry_aggregation_fn = qry_aggregation_fn
    self.dropout = dropout
    self.qry_num_layers = qry_num_layers
    self.projection_dim = projection_dim
    self.num_entities = num_entities
    self.max_entity_len = max_entity_len
    self.load_only_bert = load_only_bert
    self.ensure_answer_sparse = ensure_answer_sparse
    self.ensure_answer_dense = ensure_answer_dense
    self.train_with_sparse = train_with_sparse
    self.predict_with_sparse = predict_with_sparse
    self.fix_sparse_to_one = fix_sparse_to_one
    self.supervision = supervision
    self.l2_normalize_db = l2_normalize_db
    self.entity_score_aggregation_fn = entity_score_aggregation_fn
    self.entity_score_threshold = entity_score_threshold
    self.fact_score_threshold = fact_score_threshold
    self.softmax_temperature = softmax_temperature
    self.sparse_reduce_fn = sparse_reduce_fn
    self.intermediate_loss = intermediate_loss
    self.train_batch_size = train_batch_size
    self.predict_batch_size = predict_batch_size
    self.light = light
    self.sparse_strategy = sparse_strategy


class MIPSConfig(object):
  """Hyperparameters for the MIPS model of mention index."""

  def __init__(self, ckpt_path, ckpt_var_name, num_mentions, emb_size,
               num_neighbors):
    self.ckpt_path = ckpt_path
    self.ckpt_var_name = ckpt_var_name
    self.num_mentions = num_mentions
    self.emb_size = emb_size
    self.num_neighbors = num_neighbors


class FactMIPSConfig(object):
  """Hyperparameters for the MIPS model of fact index."""

  def __init__(self, ckpt_path, ckpt_var_name, num_facts, emb_size,
               num_neighbors):
    self.ckpt_path = ckpt_path
    self.ckpt_var_name = ckpt_var_name
    self.num_facts = num_facts
    self.emb_size = emb_size
    self.num_neighbors = num_neighbors


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     exclude_bert):
  """Creates an optimizer training op, optionally excluding BERT vars."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.estimator.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  if exclude_bert:
    bert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bert")
    tvars = [vv for vv in tvars if vv not in bert_vars]

  tf.logging.info("Training the following variables:")
  for vv in tvars:
    tf.logging.info(vv.name)

  grads = tf.gradients(loss, tvars, colocate_gradients_with_ops=True)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


def get_assignment_map_from_checkpoint(tvars,
                                       init_checkpoint,
                                       load_only_bert=False):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    if load_only_bert and ("bert" not in name):
      continue
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def model_fn_builder(bert_config,
                     qa_config,
                     mips_config,
                     fact_mips_config,
                     init_checkpoint,
                     e2m_checkpoint,
                     m2e_checkpoint,
                     e2f_checkpoint,
                     f2e_checkpoint,
                     f2f_checkpoint,
                     entity_id_checkpoint,
                     entity_mask_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     create_model_fn,
                     summary_obj=None):
  """Returns `model_fn` closure for TPUEstimator."""
  tf.random.set_random_seed(FLAGS.random_seed)

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del labels, params  # Not used.
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    entity_ids = search_utils.load_database(
        "entity_ids", [qa_config.num_entities, qa_config.max_entity_len],
        entity_id_checkpoint,
        dtype=tf.int32)
    entity_mask = search_utils.load_database(
        "entity_mask", [qa_config.num_entities, qa_config.max_entity_len],
        entity_mask_checkpoint)

    if FLAGS.model_type == "drkit":
      # Initialize sparse tensor of ent2ment.
      with tf.device("/cpu:0"):
        tf_e2m_data, tf_e2m_indices, tf_e2m_rowsplits = (
            search_utils.load_ragged_matrix("ent2ment", e2m_checkpoint))
        with tf.name_scope("RaggedConstruction_e2m"):
          e2m_ragged_ind = tf.RaggedTensor.from_row_splits(
              values=tf_e2m_indices,
              row_splits=tf_e2m_rowsplits,
              validate=False)
          e2m_ragged_val = tf.RaggedTensor.from_row_splits(
              values=tf_e2m_data, row_splits=tf_e2m_rowsplits, validate=False)

      tf_m2e_map = search_utils.load_database(
          "coref", [mips_config.num_mentions], m2e_checkpoint, dtype=tf.int32)

      total_loss, predictions = create_model_fn(
          bert_config=bert_config,
          qa_config=qa_config,
          mips_config=mips_config,
          is_training=is_training,
          features=features,
          ent2ment_ind=e2m_ragged_ind,
          ent2ment_val=e2m_ragged_val,
          ment2ent_map=tf_m2e_map,
          entity_ids=entity_ids,
          entity_mask=entity_mask,
          use_one_hot_embeddings=use_one_hot_embeddings,
          summary_obj=summary_obj,
          num_preds=FLAGS.num_preds,
          is_excluding=FLAGS.is_excluding,
      )
    elif FLAGS.model_type == "drfact":
      # Initialize sparse tensor of ent2fact.
      with tf.device("/cpu:0"):  # Note: cpu or gpu?
        tf_e2f_data, tf_e2f_indices, tf_e2f_rowsplits = (
            search_utils.load_ragged_matrix("ent2fact", e2f_checkpoint))
        with tf.name_scope("RaggedConstruction_e2f"):
          e2f_ragged_ind = tf.RaggedTensor.from_row_splits(
              values=tf_e2f_indices,
              row_splits=tf_e2f_rowsplits,
              validate=False)
          e2f_ragged_val = tf.RaggedTensor.from_row_splits(
              values=tf_e2f_data, row_splits=tf_e2f_rowsplits, validate=False)
      # Initialize sparse tensor of fact2ent.
      with tf.device("/cpu:0"):
        tf_f2e_data, tf_f2e_indices, tf_f2e_rowsplits = (
            search_utils.load_ragged_matrix("fact2ent", f2e_checkpoint))
        with tf.name_scope("RaggedConstruction_f2e"):
          f2e_ragged_ind = tf.RaggedTensor.from_row_splits(
              values=tf_f2e_indices,
              row_splits=tf_f2e_rowsplits,
              validate=False)
          f2e_ragged_val = tf.RaggedTensor.from_row_splits(
              values=tf_f2e_data, row_splits=tf_f2e_rowsplits, validate=False)
      # Initialize sparse tensor of fact2fact.
      with tf.device("/cpu:0"):
        tf_f2f_data, tf_f2f_indices, tf_f2f_rowsplits = (
            search_utils.load_ragged_matrix("fact2fact", f2f_checkpoint))
        with tf.name_scope("RaggedConstruction_f2f"):
          f2f_ragged_ind = tf.RaggedTensor.from_row_splits(
              values=tf_f2f_indices,
              row_splits=tf_f2f_rowsplits,
              validate=False)
          f2f_ragged_val = tf.RaggedTensor.from_row_splits(
              values=tf_f2f_data, row_splits=tf_f2f_rowsplits, validate=False)

      total_loss, predictions = create_model_fn(
          bert_config=bert_config,
          qa_config=qa_config,
          fact_mips_config=fact_mips_config,
          is_training=is_training,
          features=features,
          ent2fact_ind=e2f_ragged_ind,
          ent2fact_val=e2f_ragged_val,
          fact2ent_ind=f2e_ragged_ind,
          fact2ent_val=f2e_ragged_val,
          fact2fact_ind=f2f_ragged_ind,
          fact2fact_val=f2f_ragged_val,
          entity_ids=entity_ids,
          entity_mask=entity_mask,
          use_one_hot_embeddings=use_one_hot_embeddings,
          summary_obj=summary_obj,
          num_preds=FLAGS.num_preds,
          is_excluding=FLAGS.is_excluding,
      )

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = get_assignment_map_from_checkpoint(
           tvars, init_checkpoint, load_only_bert=qa_config.load_only_bert)
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
      one_mb = tf.constant(1024 * 1024, dtype=tf.int64)
      devices = tf.config.experimental.list_logical_devices("GPU")
      memory_footprints = []
      for device in devices:
        memory_footprint = tf.print(
            device.name,
            contrib_memory_stats.MaxBytesInUse() / one_mb, " / ",
            contrib_memory_stats.BytesLimit() / one_mb)
        memory_footprints.append(memory_footprint)

      with tf.control_dependencies(memory_footprints):
        train_op = create_optimizer(total_loss, learning_rate, num_train_steps,
                                    num_warmup_steps, use_tpu, False)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def train(dataset, estimator, num_train_steps):
  """Run one training loop over given TFRecords file."""
  if FLAGS.profile_model:
    hooks = [
        tf.train.ProfilerHook(
            output_dir=estimator.model_dir, save_secs=100, show_memory=False)
    ]
    tf.logging.info("Saving profiling output to %s", estimator.model_dir)
  else:
    hooks = None
  estimator.train(
      input_fn=dataset.input_fn, max_steps=num_train_steps, hooks=hooks)


def single_eval(eval_dataset, estimator, ckpt_path, mention2text, entityid2name,
                supervision, output_prediction_file, eval_fn, paragraphs,
                mentions, **kwargs):
  """Run one evaluation using given checkpoint."""
  del mentions  # Not used.

  tf.logging.info("***** Running predictions using %s *****", ckpt_path)
  tf.logging.info("  Num eval examples = %d", len(eval_dataset.examples))
  tf.logging.info("  Eval Batch size = %d", FLAGS.predict_batch_size)

  # Collect ground truth answers.
  if supervision == "mention":
    name_map = mention2text
  else:
    name_map = entityid2name

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_results = []
  for batched_result in estimator.predict(
      eval_dataset.input_fn,
      yield_single_examples=False,
      checkpoint_path=ckpt_path):
    if not all_results:
      t_st = time.time()
    # print("batched_result", batched_result)  # Debug
    cur_bsz = len(batched_result["qas_ids"])
    for ii in range(cur_bsz):
      result = {}
      for r_key, values in batched_result.items():
        result[r_key] = values[ii]
      all_results.append(result)
      if len(all_results) % 100 == 0:
        tf.logging.info("Processing example: %d at single_eval",
                        len(all_results))
  total_time = time.time() - t_st

  # Compute metrics.
  metrics = eval_fn(
      eval_dataset,
      all_results,
      name_map,
      output_prediction_file,
      paragraphs,
      supervision=supervision,
      **kwargs)
  metrics["QPS"] = float(len(all_results)) / total_time

  return metrics


def _copy_model(in_path, out_path):
  """Copy model checkpoint for future use."""
  tf.logging.info("Copying checkpoint from %s to %s.", in_path, out_path)
  tf.gfile.Copy(
      in_path + ".data-00000-of-00001",
      out_path + ".data-00000-of-00001",
      overwrite=True)
  tf.gfile.Copy(in_path + ".index", out_path + ".index", overwrite=True)
  tf.gfile.Copy(in_path + ".meta", out_path + ".meta", overwrite=True)


def continuous_eval(eval_dataset, estimator, mention2text, entityid2name,
                    supervision, eval_fn, paragraphs, mentions, **kwargs):
  """Run continuous evaluation on given TFRecords file."""
  current_ckpt = 0
  best_acc = 0
  stop_evaluating = False
  if not tf.gfile.Exists(os.path.join(FLAGS.output_dir, "eval")):
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, "eval"))
  event_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_dir, "eval"))
  while not stop_evaluating:
    if FLAGS.use_best_ckpt_for_predict:
      ckpt_path = os.path.join(FLAGS.output_dir, FLAGS.model_ckpt_toload)
      if not tf.gfile.Exists(ckpt_path + ".meta"):
        tf.logging.info("No best_model checkpoint found in %s",
                        FLAGS.output_dir)
        tf.logging.info("Skipping evaluation.")
        break
      output_prediction_file = os.path.join(
          FLAGS.output_dir, "%s.predictions.json" % FLAGS.predict_prefix)
      stop_evaluating = True
    else:
      ckpt_path = tf.train.latest_checkpoint(FLAGS.output_dir)
      if ckpt_path == current_ckpt:
        tf.logging.info("No new checkpoint in %s", FLAGS.output_dir)
        tf.logging.info("Waiting for 10s")
        time.sleep(10)
        continue
      current_ckpt = ckpt_path
      model_name = None
      if ckpt_path is not None:
        model_name = os.path.basename(ckpt_path)
      output_prediction_file = os.path.join(FLAGS.output_dir,
                                            "predictions_%s.json" % model_name)

    metrics = single_eval(eval_dataset, estimator, ckpt_path, mention2text,
                          entityid2name, supervision, output_prediction_file,
                          eval_fn, paragraphs, mentions, **kwargs)
    tf.logging.info("Previous best accuracy:  %.4f", best_acc)
    tf.logging.info("Current accuracy:  %.4f", metrics["accuracy"])
    if ckpt_path is not None and not FLAGS.use_best_ckpt_for_predict:
      ckpt_number = int(ckpt_path.rsplit("-", 1)[1])
      if metrics["accuracy"] > best_acc:
        best_acc = metrics["accuracy"]
        if tf.gfile.Exists(ckpt_path + ".meta"):
          _copy_model(ckpt_path, os.path.join(FLAGS.output_dir, "best_model"))
    else:
      ckpt_number = 0
    for metric, value in metrics.items():
      tf.logging.info("%s: %.4f", metric, value)
      if not FLAGS.use_best_ckpt_for_predict:
        curr_summary = tf.Summary(value=[
            tf.Summary.Value(tag=metric, simple_value=value),
        ])
        event_writer.add_summary(curr_summary, global_step=ckpt_number)


def validate_flags_or_throw():
  """Validate the input FLAGS or throw an exception."""
  if (not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_test):
    raise ValueError("At least one of `do_train`, `do_predict` or "
                     "`do_test` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.debug:
    print(FLAGS)
    return

  # Decide data type.
  if FLAGS.data_type == "hotpotqa":
    dataset_class = input_fns.OpenCSRDataset
    eval_fn = evaluate.opencsr_eval_fn

  # Decide model type.
  if FLAGS.model_type == "drkit":
    create_model_fn = functools.partial(
        model_fns.create_drkit_model, num_hops=FLAGS.num_hops)
  elif FLAGS.model_type == "drfact":
    create_model_fn = functools.partial(
        model_fns.create_drfact_model, num_hops=FLAGS.num_hops)
  else:
    tf.logging.info("Wrong model_type...")
  # Load BERT.
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # Load mention and entity files.
  mention2text = json.load(
      tf.gfile.Open(os.path.join(FLAGS.train_data_dir, "mention2text.json")))
  tf.logging.info("Loading metadata about entities and mentions...")
  entity2id, entity2name = json.load(
      tf.gfile.Open(os.path.join(FLAGS.train_data_dir, "entities.json")))
  entityid2name = {str(i): entity2name[e] for e, i in entity2id.items()}
  all_paragraphs = json.load(
      tf.gfile.Open(os.path.join(FLAGS.train_data_dir, "subparas.json")))
  all_mentions = np.load(
      tf.gfile.Open(os.path.join(FLAGS.train_data_dir, "mentions.npy"), "rb"))

  qa_config = QAConfig(
      qry_layers_to_use=FLAGS.qry_layers_to_use,
      qry_aggregation_fn=FLAGS.qry_aggregation_fn,
      dropout=FLAGS.question_dropout,
      qry_num_layers=FLAGS.question_num_layers,
      projection_dim=FLAGS.projection_dim,
      load_only_bert=FLAGS.load_only_bert,
      num_entities=len(entity2id),
      max_entity_len=FLAGS.max_entity_len,
      ensure_answer_sparse=FLAGS.ensure_answer_sparse,
      ensure_answer_dense=FLAGS.ensure_answer_dense,
      train_with_sparse=FLAGS.train_with_sparse,
      predict_with_sparse=FLAGS.predict_with_sparse,
      fix_sparse_to_one=FLAGS.fix_sparse_to_one,
      supervision=FLAGS.supervision,
      l2_normalize_db=FLAGS.l2_normalize_db,
      entity_score_aggregation_fn=FLAGS.entity_score_aggregation_fn,
      entity_score_threshold=FLAGS.entity_score_threshold,
      fact_score_threshold=FLAGS.fact_score_threshold,
      softmax_temperature=FLAGS.softmax_temperature,
      sparse_reduce_fn=FLAGS.sparse_reduce_fn,
      intermediate_loss=FLAGS.intermediate_loss,
      light=FLAGS.light,
      sparse_strategy=FLAGS.sparse_strategy,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  mips_config = MIPSConfig(
      ckpt_path=os.path.join(FLAGS.train_data_dir,
                             "%s_mention_feats" % FLAGS.embed_index_prefix),
      ckpt_var_name="db_emb",
      num_mentions=len(mention2text),
      emb_size=FLAGS.projection_dim * 2,
      num_neighbors=FLAGS.num_mips_neighbors)
  fact_mips_config = FactMIPSConfig(
      ckpt_path=os.path.join(FLAGS.train_data_dir,
                             "%s_fact_feats" % FLAGS.embed_index_prefix),
      ckpt_var_name="fact_db_emb",
      num_facts=len(all_paragraphs),
      emb_size=FLAGS.projection_dim * 2,
      num_neighbors=FLAGS.num_mips_neighbors)
  validate_flags_or_throw()

  tf.gfile.MakeDirs(FLAGS.output_dir)

  # Save training flags.
  if FLAGS.do_train:
    json.dump(tf.app.flags.FLAGS.flag_values_dict(),
              tf.gfile.Open(os.path.join(FLAGS.output_dir, "flags.json"), "w"))

  # tokenizer = tokenization.FullTokenizer(
  #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  if FLAGS.tokenizer_type == "bert_tokenization":
    tokenizer = bert_tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)
  elif FLAGS.tokenizer_type == "albert_tokenization":
    tokenizer = albert_tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=False,
        spm_model_file=FLAGS.tokenizer_model_file)
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=50,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=tf.ConfigProto(log_device_placement=False))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.num_preds < 0:
    FLAGS.num_preds = len(entity2id)
  if FLAGS.do_train:
    train_dataset = dataset_class(
        in_file=FLAGS.train_file,
        tokenizer=tokenizer,
        subject_mention_probability=FLAGS.subject_mention_probability,
        max_qry_length=FLAGS.max_query_length,
        is_training=True,
        entity2id=entity2id,
        tfrecord_filename=os.path.join(FLAGS.output_dir, "train.tf_record"))
    num_train_steps = int(train_dataset.num_examples / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  if FLAGS.do_predict:
    eval_dataset = dataset_class(
        in_file=FLAGS.predict_file,
        tokenizer=tokenizer,
        subject_mention_probability=0.0,
        max_qry_length=FLAGS.max_query_length,
        is_training=False,
        entity2id=entity2id,
        tfrecord_filename=os.path.join(
            FLAGS.output_dir, "eval.%s.tf_record" % FLAGS.predict_prefix))
    qa_config.predict_batch_size = FLAGS.predict_batch_size
  summary_obj = None
  # summary_obj = summary.TPUSummary(FLAGS.output_dir,
  #                                  FLAGS.save_checkpoints_steps)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      qa_config=qa_config,
      mips_config=mips_config,
      fact_mips_config=fact_mips_config,
      init_checkpoint=FLAGS.init_checkpoint,
      e2m_checkpoint=os.path.join(FLAGS.train_data_dir, "ent2ment.npz"),
      m2e_checkpoint=os.path.join(FLAGS.train_data_dir, "coref.npz"),
      e2f_checkpoint=os.path.join(FLAGS.train_data_dir, "ent2fact.npz"),
      f2e_checkpoint=os.path.join(FLAGS.train_data_dir, "fact_coref.npz"),
      f2f_checkpoint=os.path.join(FLAGS.f2f_index_dir, "fact2fact.npz"),
      entity_id_checkpoint=os.path.join(FLAGS.train_data_dir, "entity_ids"),
      entity_mask_checkpoint=os.path.join(FLAGS.train_data_dir, "entity_mask"),
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      create_model_fn=create_model_fn,
      summary_obj=summary_obj)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  if FLAGS.do_train or FLAGS.do_predict:
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", train_dataset.num_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train(train_dataset, estimator, num_train_steps)

  if FLAGS.do_predict:
    continuous_eval(
        eval_dataset,
        estimator,
        mention2text,
        entityid2name,
        qa_config.supervision,
        eval_fn,
        paragraphs=all_paragraphs,
        mentions=all_mentions)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
