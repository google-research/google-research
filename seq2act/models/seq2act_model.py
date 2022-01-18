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

"""Seq2act model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from seq2act.models import seq2act_grounding
from seq2act.models import seq2act_reference


def compute_logits(features, hparams, mode,
                   use_cache=None, cache=None):
  """Computes the logits."""
  if mode != tf.estimator.ModeKeys.TRAIN:
    for key in hparams.values():
      if key.endswith("dropout"):
        setattr(hparams, key, 0.0)
    setattr(hparams, "synthetic_screen_noise", 0.0)
  tf.logging.info(hparams)
  references = seq2act_reference.compute_logits(
      features, hparams,
      train=(mode == tf.estimator.ModeKeys.TRAIN))
  if use_cache is not None and cache is not None:
    for key in cache:
      references[key] = tf.where(
          tf.equal(use_cache, 1), tf.concat([
              cache[key], cache[key][:, -1:, :]], axis=1), references[key])
  action_logits, obj_logits, consumed_logits = seq2act_grounding.compute_logits(
      features, references, hparams)
  return action_logits, obj_logits, consumed_logits, references


def compute_loss(loss_dict, features, action_logits, obj_logits,
                 consumed_logits, references, hparams):
  """Computes the loss."""
  total_loss = seq2act_reference.compute_losses(
      loss_dict, features, references, hparams)
  grounding_loss = seq2act_grounding.compute_losses(
      loss_dict, features, action_logits, obj_logits, consumed_logits)
  global_step = tf.train.get_global_step()
  if global_step:
    total_loss += tf.cond(
        tf.greater(global_step, hparams.reference_warmup_steps),
        lambda: grounding_loss,
        lambda: tf.constant(0.))
  else:
    total_loss += grounding_loss
  loss_dict["total_loss"] = total_loss


def predict(prediction_dict, action_logits, obj_logits, consumed_logits,
            references):
  """Compute predictions."""
  seq2act_reference.compute_predictions(prediction_dict, references)
  seq2act_grounding.compute_predictions(prediction_dict,
                                        action_logits, obj_logits,
                                        consumed_logits)


def core_graph(features, hparams, mode,
               compute_additional_loss=None):
  """The core TF graph for the estimator."""
  action_logits, obj_logits, consumed_logits, references = (
      compute_logits(features, hparams, mode))
  prediction_dict = {}
  loss_dict = {}
  if mode != tf.estimator.ModeKeys.PREDICT:
    compute_loss(loss_dict, features,
                 action_logits, obj_logits, consumed_logits, references,
                 hparams)
    if compute_additional_loss:
      compute_additional_loss(hparams, features, references["decoder_output"],
                              loss_dict, prediction_dict, mode)
  if mode != tf.estimator.ModeKeys.TRAIN:
    if mode == tf.estimator.ModeKeys.PREDICT:
      prediction_dict["task"] = features["task"]
      prediction_dict["raw_task"] = features["raw_task"]
      prediction_dict["data_source"] = features["data_source"]
    predict(prediction_dict, action_logits, obj_logits, consumed_logits,
            references)
  return loss_dict, prediction_dict, references["areas"]
