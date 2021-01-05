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

"""seq2act estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import optimize
import tensorflow.compat.v1 as tf
from seq2act.models import input as input_utils
from seq2act.models import seq2act_model
from seq2act.utils import decode_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_files", None, "the file names for training")
flags.DEFINE_string("eval_files", None, "the file names for eval")
flags.DEFINE_integer("worker_gpu", 1, "worker_gpu")
flags.DEFINE_integer("worker_replicas", 1, "num_workers")
flags.DEFINE_integer("train_steps", 100000, "train_steps")
flags.DEFINE_string("hparams", "", "the hyper parameters")
flags.DEFINE_string("domain_hparams", "",
                    "the domain-specific hyper parameters")
flags.DEFINE_string("domain_train_files", None, "the file names for training")
flags.DEFINE_string("domain_eval_files", None, "the file names for eval")
flags.DEFINE_string("metric_types",
                    "final_accuracy,ref_accuracy,basic_accuracy",
                    "metric types")
flags.DEFINE_boolean("post_processing", True,
                     "post processing the predictions")


def _ref_accuracy(features, pred_dict, nonpadding, name, metrics,
                  decode_refs=None,
                  measure_beginning_eos=False,
                  debug=False):
  """Computes the accuracy of reference prediction.

  Args:
    features: the feature dict.
    pred_dict: the dictionary to hold the prediction results.
    nonpadding: a 2D boolean tensor for masking out paddings.
    name: the name of the feature to be predicted.
    metrics: the eval metrics.
    decode_refs: decoded references.
    measure_beginning_eos: whether to measure the beginning and the end.
    debug: whether to output mismatches.
  """
  if decode_refs is not None:
    gt_seq_lengths = decode_utils.verb_refs_to_lengths(features["task"],
                                                       features["verb_refs"])
    pr_seq_lengths = decode_utils.verb_refs_to_lengths(decode_refs["task"],
                                                       decode_refs["verb_refs"])
    full_acc, partial_acc = decode_utils.sequence_accuracy(
        features[name], decode_refs[name], gt_seq_lengths, pr_seq_lengths,
        debug=debug, name=name)
    metrics[name + "_full_accuracy"] = tf.metrics.mean(full_acc)
    metrics[name + "_partial_accuracy"] = tf.metrics.mean(partial_acc)
  if measure_beginning_eos:
    nonpadding = tf.reshape(nonpadding, [-1])
    refs = tf.reshape(features[name], [-1, 2])
    predict_refs = tf.reshape(pred_dict[name], [-1, 2])
    metrics[name + "_start"] = tf.metrics.accuracy(
        labels=tf.boolean_mask(refs[:, 0], nonpadding),
        predictions=tf.boolean_mask(predict_refs[:, 0], nonpadding),
        name=name + "_start_accuracy")
    metrics[name + "_end"] = tf.metrics.accuracy(
        labels=tf.boolean_mask(refs[:, 1], nonpadding),
        predictions=tf.boolean_mask(predict_refs[:, 1], nonpadding),
        name=name + "_end_accuracy")


def _eval(metrics, pred_dict, loss_dict, features, areas, compute_seq_accuracy,
          hparams, metric_types, decode_length=20):
  """Internal eval function."""
  # Assume data sources are not mixed within each batch
  if compute_seq_accuracy:
    decode_features = {}
    for key in features:
      if not key.endswith("_refs"):
        decode_features[key] = features[key]
    decode_utils.decode_n_step(seq2act_model.compute_logits,
                               decode_features, areas,
                               hparams, n=decode_length, beam_size=1)
    decode_features["input_refs"] = decode_utils.unify_input_ref(
        decode_features["verbs"], decode_features["input_refs"])
    acc_metrics = decode_utils.compute_seq_metrics(
        features, decode_features)
    metrics["seq_full_acc"] = tf.metrics.mean(acc_metrics["complete_refs_acc"])
    metrics["seq_partial_acc"] = tf.metrics.mean(
        acc_metrics["partial_refs_acc"])
    if "final_accuracy" in metric_types:
      metrics["complet_act_accuracy"] = tf.metrics.mean(
          acc_metrics["complete_acts_acc"])
      metrics["partial_seq_acc"] = tf.metrics.mean(
          acc_metrics["partial_acts_acc"])
      print0 = tf.print("*** lang", features["raw_task"], summarize=100)
      with tf.control_dependencies([print0]):
        loss_dict["total_loss"] = tf.identity(loss_dict["total_loss"])
  else:
    decode_features = None
  if "ref_accuracy" in metric_types:
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(features["verb_refs"]), 3),
        tf.assert_equal(tf.shape(features["verb_refs"])[-1], 2)]):
      _ref_accuracy(features, pred_dict,
                    tf.less(features["verb_refs"][:, :, 0],
                            features["verb_refs"][:, :, 1]),
                    "verb_refs", metrics, decode_features,
                    measure_beginning_eos=True)
    _ref_accuracy(features, pred_dict,
                  tf.less(features["obj_refs"][:, :, 0],
                          features["obj_refs"][:, :, 1]),
                  "obj_refs", metrics, decode_features,
                  measure_beginning_eos=True)
    _ref_accuracy(features, pred_dict,
                  tf.less(features["input_refs"][:, :, 0],
                          features["input_refs"][:, :, 1]),
                  "input_refs", metrics, decode_features,
                  measure_beginning_eos=True)
  if "basic_accuracy" in metric_types:
    target_verbs = tf.reshape(features["verbs"], [-1])
    verb_nonpadding = tf.greater(target_verbs, 1)
    target_verbs = tf.boolean_mask(target_verbs, verb_nonpadding)
    predict_verbs = tf.boolean_mask(tf.reshape(pred_dict["verbs"], [-1]),
                                    verb_nonpadding)
    metrics["verb"] = tf.metrics.accuracy(
        labels=target_verbs,
        predictions=predict_verbs,
        name="verb_accuracy")
    input_mask = tf.reshape(
        tf.less(features["verb_refs"][:, :, 0],
                features["verb_refs"][:, :, 1]), [-1])
    metrics["input"] = tf.metrics.accuracy(
        labels=tf.boolean_mask(
            tf.reshape(tf.to_int32(
                tf.less(features["input_refs"][:, :, 0],
                        features["input_refs"][:, :, 1])), [-1]), input_mask),
        predictions=tf.boolean_mask(
            tf.reshape(pred_dict["input"], [-1]), input_mask),
        name="input_accuracy")
    metrics["object"] = tf.metrics.accuracy(
        labels=tf.boolean_mask(tf.reshape(features["objects"], [-1]),
                               verb_nonpadding),
        predictions=tf.boolean_mask(tf.reshape(pred_dict["objects"], [-1]),
                                    verb_nonpadding),
        name="object_accuracy")
    metrics["eval_object_loss"] = tf.metrics.mean(
        tf.reduce_mean(
            tf.boolean_mask(tf.reshape(loss_dict["object_losses"], [-1]),
                            verb_nonpadding)))
    metrics["eval_verb_loss"] = tf.metrics.mean(
        tf.reduce_mean(
            tf.boolean_mask(tf.reshape(loss_dict["verbs_losses"], [-1]),
                            verb_nonpadding)))


def decode_sequence(features, areas, hparams, decode_length,
                    post_processing=True):
  """Decodes the entire sequence in an auto-regressive way."""
  decode_utils.decode_n_step(seq2act_model.compute_logits,
                             features, areas,
                             hparams, n=decode_length, beam_size=1)
  if post_processing:
    features["input_refs"] = decode_utils.unify_input_ref(
        features["verbs"], features["input_refs"])
    pred_lengths = decode_utils.verb_refs_to_lengths(features["task"],
                                                     features["verb_refs"],
                                                     include_eos=False)
  predicted_actions = tf.concat([
      features["verb_refs"],
      features["obj_refs"],
      features["input_refs"],
      tf.to_int32(tf.expand_dims(features["verbs"], 2)),
      tf.to_int32(tf.expand_dims(features["objects"], 2))], axis=-1)
  if post_processing:
    predicted_actions = tf.where(
        tf.tile(tf.expand_dims(
            tf.sequence_mask(pred_lengths,
                             maxlen=tf.shape(predicted_actions)[1]),
            2), [1, 1, tf.shape(predicted_actions)[-1]]), predicted_actions,
        tf.zeros_like(predicted_actions))
  return predicted_actions


def create_model_fn(hparams, compute_additional_loss_fn=None,
                    compute_additional_metric_fn=None,
                    compute_seq_accuracy=False,
                    decode_length=20):
  """Creates the model function.

  Args:
    hparams: the hyper parameters.
    compute_additional_loss_fn: the optional callback for calculating
        additional loss.
    compute_additional_metric_fn: the optional callback for computing
        additional metrics.
    compute_seq_accuracy: whether to compute seq accuracy.
    decode_length: the maximum decoding length.
  Returns:
    the model function for estimator.
  """
  def model_fn(features, labels, mode):
    """The model function for creating an Estimtator."""
    del labels
    input_count = tf.reduce_sum(
        tf.to_int32(tf.greater(features["input_refs"][:, :, 1],
                               features["input_refs"][:, :, 0])))
    tf.summary.scalar("input_count", input_count)
    loss_dict, pred_dict, areas = seq2act_model.core_graph(
        features, hparams, mode, compute_additional_loss_fn)
    if mode == tf.estimator.ModeKeys.PREDICT:
      pred_dict["sequences"] = decode_sequence(
          features, areas, hparams, decode_length,
          post_processing=FLAGS.post_processing)
      return tf.estimator.EstimatorSpec(mode, predictions=pred_dict)
    elif mode == tf.estimator.ModeKeys.EVAL:
      metrics = {}
      _eval(metrics, pred_dict, loss_dict, features,
            areas, compute_seq_accuracy,
            hparams,
            metric_types=FLAGS.metric_types.split(","),
            decode_length=decode_length)
      if compute_additional_metric_fn:
        compute_additional_metric_fn(metrics, pred_dict, features)
      return tf.estimator.EstimatorSpec(
          mode, loss=loss_dict["total_loss"], eval_metric_ops=metrics)
    else:
      assert mode == tf.estimator.ModeKeys.TRAIN
      loss = loss_dict["total_loss"]
      for loss_name in loss_dict:
        if loss_name == "total_loss":
          continue
        if loss_name.endswith("losses"):
          continue
        tf.summary.scalar(loss_name, loss_dict[loss_name])
      step_num = tf.to_float(tf.train.get_global_step())
      schedule_string = hparams.learning_rate_schedule
      names = schedule_string.split("*")
      names = [name.strip() for name in names if name.strip()]
      ret = tf.constant(1.0)
      for name in names:
        ret *= learning_rate.learning_rate_factor(name, step_num, hparams)
      train_op = optimize.optimize(loss, ret, hparams)
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  return model_fn


def create_input_fn(files,
                    batch_size,
                    repeat,
                    required_agreement,
                    data_source,
                    max_range,
                    max_dom_pos,
                    max_pixel_pos,
                    mean_synthetic_length,
                    stddev_synthetic_length,
                    load_extra=False,
                    load_screen=True,
                    buffer_size=8 * 1024,
                    shuffle_size=8 * 1024,
                    load_dom_dist=False):
  """Creats the input function."""
  def input_fn():
    return input_utils.input_fn(data_files=files,
                                batch_size=batch_size,
                                repeat=repeat,
                                required_agreement=required_agreement,
                                data_source=data_source,
                                max_range=max_range,
                                max_dom_pos=max_dom_pos,
                                max_pixel_pos=max_pixel_pos,
                                mean_synthetic_length=mean_synthetic_length,
                                stddev_synthetic_length=stddev_synthetic_length,
                                load_extra=load_extra,
                                buffer_size=buffer_size,
                                load_screen=load_screen,
                                shuffle_size=shuffle_size,
                                load_dom_dist=load_dom_dist)
  return input_fn


def create_hybrid_input_fn(data_files_list,
                           data_source_list,
                           batch_size_list,
                           max_range,
                           max_dom_pos,
                           max_pixel_pos,
                           mean_synthetic_length,
                           stddev_synthetic_length,
                           batch_size,
                           boost_input=False,
                           load_screen=True,
                           buffer_size=1024 * 8,
                           shuffle_size=1024,
                           load_dom_dist=False):
  """Creats the input function."""
  def input_fn():
    return input_utils.hybrid_input_fn(
        data_files_list,
        data_source_list,
        batch_size_list,
        max_range=max_range,
        max_dom_pos=max_dom_pos,
        max_pixel_pos=max_pixel_pos,
        mean_synthetic_length=mean_synthetic_length,
        stddev_synthetic_length=stddev_synthetic_length,
        hybrid_batch_size=batch_size,
        boost_input=boost_input,
        load_screen=load_screen,
        buffer_size=buffer_size,
        shuffle_size=shuffle_size,
        load_dom_dist=load_dom_dist)
  return input_fn


def create_hparams():
  """Creates hyper parameters."""
  hparams = getattr(transformer, "transformer_base")()
  hparams.add_hparam("reference_warmup_steps", 0)
  hparams.add_hparam("max_span", 20)
  hparams.add_hparam("task_vocab_size", 59429)
  hparams.add_hparam("load_screen", True)

  hparams.set_hparam("hidden_size", 16)
  hparams.set_hparam("num_hidden_layers", 2)
  hparams.add_hparam("freeze_reference_model", False)
  hparams.add_hparam("mean_synthetic_length", 1.0)
  hparams.add_hparam("stddev_synthetic_length", .0)
  hparams.add_hparam("instruction_encoder", "transformer")
  hparams.add_hparam("instruction_decoder", "transformer")
  hparams.add_hparam("clip_norm", 0.)
  hparams.add_hparam("span_rep", "area")
  hparams.add_hparam("dom_dist_variance", 1.0)

  hparams.add_hparam("attention_mechanism", "luong")  # "bahdanau"
  hparams.add_hparam("output_attention", True)
  hparams.add_hparam("attention_layer_size", 128)

  # GAN-related hyper params
  hparams.add_hparam("dis_loss_ratio", 0.01)
  hparams.add_hparam("gen_loss_ratio", 0.01)
  hparams.add_hparam("gan_update", "center")
  hparams.add_hparam("num_joint_layers", 2)
  hparams.add_hparam("use_additional_loss", False)

  hparams.add_hparam("compute_verb_obj_separately", True)
  hparams.add_hparam("synthetic_screen_noise", 0.)
  hparams.add_hparam("screen_encoder", "mlp")
  hparams.add_hparam("screen_encoder_layers", 2)
  hparams.add_hparam("action_vocab_size", 6)
  hparams.add_hparam("max_pixel_pos", 100)
  hparams.add_hparam("max_dom_pos", 500)
  hparams.add_hparam("span_aggregation", "sum")
  hparams.add_hparam("obj_text_aggregation", "sum")
  hparams.add_hparam("screen_embedding_feature", "text_pos_type")
  hparams.add_hparam("alignment", "dot_product_attention")
  hparams.parse(FLAGS.hparams)

  hparams.set_hparam("use_target_space_embedding", False)
  hparams.set_hparam("filter_size", hparams.hidden_size * 4)
  hparams.set_hparam("attention_layer_size", hparams.hidden_size)
  hparams.set_hparam("dropout", hparams.layer_prepostprocess_dropout)
  return hparams


def save_hyperparams(hparams, output_dir):
  """Save the model hyperparameters."""
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)
  if not tf.gfile.Exists(os.path.join(output_dir, "hparams.json")):
    with tf.gfile.GFile(
        os.path.join(output_dir, "hparams.json"), mode="w") as f:
      f.write(hparams.to_json())


def load_hparams(checkpoint_path):
  """Prepares the hyper-parameters."""
  hparams = create_hparams()
  with tf.gfile.Open(os.path.join(checkpoint_path, "hparams.json"),
                     "r") as hparams_file:
    hparams_string = " ".join(hparams_file.readlines())
    hparams.parse_json(hparams_string)
  tf.logging.info("hparams: %s" % hparams)
  return hparams


def _pad_to_max(x, y, constant_values=0):
  """Pad x and y to their maximum shape."""
  shape_x = common_layers.shape_list(x)
  shape_y = common_layers.shape_list(y)
  assert len(shape_x) == len(shape_y)
  pad_x = [[0, 0]]
  pad_y = [[0, 0]]
  for dim in range(len(shape_x) - 1):
    add_y = shape_x[dim + 1] - shape_y[dim + 1]
    add_x = -add_y
    pad_x.append([0, tf.maximum(add_x, 0)])
    pad_y.append([0, tf.maximum(add_y, 0)])
  x = tf.pad(x, pad_x, constant_values=constant_values)
  y = tf.pad(y, pad_y, constant_values=constant_values)
  return x, y
