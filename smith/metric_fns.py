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

"""Metrics used in dual encoder SMITH model."""

import tensorflow.compat.v1 as tf


def metric_fn_pretrain(masked_lm_example_loss_1, masked_lm_weights_1,
                       masked_sent_per_example_loss_1, masked_sent_weight_1,
                       masked_lm_example_loss_2, masked_lm_weights_2,
                       masked_sent_per_example_loss_2, masked_sent_weight_2,
                       predicted_class, labels, is_real_example):
  """Computes the metrics of the model during pre-training.

  Note that the inputs of this metric_fn should be a list of tensors according
  to the documentation of tpu_estimator
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py#L315

  Args:
    masked_lm_example_loss_1: float Tensor with shape [batch *
      max_predictions_per_seq]. The per example loss for masked token LM
      prediction from sequence 1.
    masked_lm_weights_1: float Tensor with shape [batch,
      max_predictions_per_seq]. The weights of masked tokens from sequence 1.
    masked_sent_per_example_loss_1: float Tensor with shape [batch *
      max_masked_sent_per_doc]. The per example los for masked sentence LM
      prediction from sequence 1.
    masked_sent_weight_1: float Tensor with shape [batch,
      max_masked_sent_per_doc]. The weights of masked sentences from sequence 1.
    masked_lm_example_loss_2: float Tensor with shape [batch *
      max_predictions_per_seq]. The per example loss for masked token LM
      prediction from sequence 2.
    masked_lm_weights_2: float Tensor with shape [batch,
      max_predictions_per_seq]. The weights of masked tokens from sequence 2.
    masked_sent_per_example_loss_2: float Tensor with shape [batch *
      max_masked_sent_per_doc]. The per example los for masked sentence LM
      prediction from sequence 2.
    masked_sent_weight_2: float Tensor with shape [batch,
      max_masked_sent_per_doc]. The weights of masked sentences from sequence 2.
    predicted_class: int Tensor with shape [batch]. The predicted class for each
      example in the batch.
    labels: float Tensor with shape [batch]. The ground truth label for each
      example in the batch.
    is_real_example: float Tensor with shape [batch]. The Tensor to indicate
      whether an example is a real example or a padded fake example. It will be
      used as the weights in the metrics computation.

  Returns:
    The metrics dict to be used in the evaluation metrics.
  """
  masked_lm_example_loss_1 = tf.reshape(masked_lm_example_loss_1, [-1])
  masked_lm_weights_1 = tf.reshape(masked_lm_weights_1, [-1])
  masked_lm_mean_loss_1 = tf.metrics.mean(
      values=masked_lm_example_loss_1, weights=masked_lm_weights_1)
  masked_lm_example_loss_2 = tf.reshape(masked_lm_example_loss_2, [-1])
  masked_lm_weights_2 = tf.reshape(masked_lm_weights_2, [-1])
  masked_lm_mean_loss_2 = tf.metrics.mean(
      values=masked_lm_example_loss_2, weights=masked_lm_weights_2)
  metrics_dict = {
      "masked_lm_loss_1": masked_lm_mean_loss_1,
      "masked_lm_loss_2": masked_lm_mean_loss_2,
  }
  labels = tf.reshape(labels, [-1])
  predicted_class = tf.reshape(predicted_class, [-1])
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=predicted_class, weights=is_real_example)
  metrics_dict["accuracy"] = accuracy
  masked_sent_per_example_loss_1 = tf.reshape(masked_sent_per_example_loss_1,
                                              [-1])
  masked_sent_weight_1 = tf.reshape(masked_sent_weight_1, [-1])
  masked_sent_lm_mean_loss_1 = tf.metrics.mean(
      values=masked_sent_per_example_loss_1, weights=masked_sent_weight_1)
  metrics_dict["masked_sent_lm_loss_1"] = masked_sent_lm_mean_loss_1
  masked_sent_per_example_loss_2 = tf.reshape(masked_sent_per_example_loss_2,
                                              [-1])
  masked_sent_weight_2 = tf.reshape(masked_sent_weight_2, [-1])
  masked_sent_lm_mean_loss_2 = tf.metrics.mean(
      values=masked_sent_per_example_loss_2, weights=masked_sent_weight_2)
  metrics_dict["masked_sent_lm_loss_2"] = masked_sent_lm_mean_loss_2
  return metrics_dict


def metric_fn_finetune(predicted_class, labels, siamese_example_loss,
                       is_real_example):
  """Computes the metrics of the model during fine-tuning.

  Note that the inputs of this metric_fn should be a list of tensors according
  to the documentation of tpu_estimator
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py#L315

  Args:
    predicted_class: int Tensor with shape [batch]. The predicted class for each
      example in the batch.
    labels: float Tensor with shape [batch]. The ground truth label for each
      example in the batch.
    siamese_example_loss: float Tensor with shape [batch]. The per example text
      pair matching loss.
    is_real_example: float Tensor with shape [batch]. The Tensor to indicate
      whether an example is a real example or a padded fake example. It will be
      used as the weights in the metrics computation.

  Returns:
    The metrics dict to be used in the evaluation metrics.
  """
  labels = tf.reshape(labels, [-1])
  siamese_loss = tf.metrics.mean(
      values=siamese_example_loss, weights=is_real_example)
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=predicted_class, weights=is_real_example)
  precision = tf.metrics.precision(
      labels=labels, predictions=predicted_class, weights=is_real_example)
  recall = tf.metrics.recall(
      labels=labels, predictions=predicted_class, weights=is_real_example)
  metrics_dict = {
      "accuracy": accuracy,
      "siamese_loss": siamese_loss,
      "precision": precision,
      "recall": recall
  }
  return metrics_dict
