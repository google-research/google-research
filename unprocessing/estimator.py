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

"""Unprocessing model function and train and eval specs for Estimator.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from unprocessing import process
from tensorflow.contrib import layers as contrib_layers


def psnr(labels, predictions):
  """Computes average peak signal-to-noise ratio of `predictions`.

  Here PSNR is defined with respect to the maximum value of 1. All image tensors
  must be within the range [0, 1].

  Args:
    labels: Tensor of shape [B, H, W, N].
    predictions: Tensor of shape [B, H, W, N].

  Returns:
    Tuple of (psnr, update_op) as returned by tf.metrics.
  """
  predictions.shape.assert_is_compatible_with(labels.shape)
  with tf.control_dependencies([tf.assert_greater_equal(labels, 0.0),
                                tf.assert_less_equal(labels, 1.0)]):
    psnrs = tf.image.psnr(labels, predictions, max_val=1.0)
    psnrs = tf.boolean_mask(psnrs, tf.logical_not(tf.is_inf(psnrs)))
    return tf.metrics.mean(psnrs, name='psnr')


def create_model_fn(inference_fn, hparams):
  """Creates a model function for Estimator.

  Args:
    inference_fn: Model inference function with specification:
      Args -
        noisy_img - Tensor of shape [B, H, W, 4].
        variance - Tensor of shape [B, H, W, 4].
      Returns -
        Tensor of shape [B, H, W, 4].
    hparams: Hyperparameters for model as a tf.contrib.training.HParams object.

  Returns:
    `_model_fn`.
  """
  def _model_fn(features, labels, mode, params):
    """Constructs the model function.

    Args:
      features: Dictionary of input features.
      labels: Tensor of labels if mode is `TRAIN` or `EVAL`, otherwise `None`.
      mode: ModeKey object (`TRAIN` or `EVAL`).
      params: Parameter dictionary passed from the Estimator object.

    Returns:
      An EstimatorSpec object that encapsulates the model and its serving
        configurations.
    """
    del params  # Unused.

    def process_images(images):
      """Closure for processing images with fixed metadata."""
      return process.process(images, features['red_gain'],
                             features['blue_gain'], features['cam2rgb'])

    denoised_img = inference_fn(features['noisy_img'], features['variance'])

    noisy_img = process_images(features['noisy_img'])
    denoised_img = process_images(denoised_img)
    truth_img = process_images(labels)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      loss = tf.losses.absolute_difference(truth_img, denoised_img)
    else:
      loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
      train_op = contrib_layers.optimize_loss(
          loss=loss,
          global_step=tf.train.get_global_step(),
          learning_rate=None,
          optimizer=optimizer,
          name='')  # Prevents scope prefix.
    else:
      train_op = None

    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {'PSNR': psnr(truth_img, denoised_img)}

      def summary(images, name):
        """As a hack, saves image summaries by adding to `eval_metric_ops`."""
        images = tf.saturate_cast(images * 255 + 0.5, tf.uint8)
        eval_metric_ops[name] = (tf.summary.image(name, images, max_outputs=2),
                                 tf.no_op())

      summary(noisy_img, 'Noisy')
      summary(denoised_img, 'Denoised')
      summary(truth_img, 'Truth')

      diffs = (denoised_img - truth_img + 1.0) / 2.0
      summary(diffs, 'Diffs')

    else:
      eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

  return _model_fn


def create_train_and_eval_specs(train_dataset_fn,
                                eval_dataset_fn,
                                eval_steps=250):
  """Creates a TrainSpec and EvalSpec.

  Args:
    train_dataset_fn: Function returning a Dataset of training data.
    eval_dataset_fn: Function returning a Dataset of evaluation data.
    eval_steps: Number of steps for evaluating model.

  Returns:
    Tuple of (TrainSpec, EvalSpec).
  """
  train_spec = tf.estimator.TrainSpec(input_fn=train_dataset_fn, max_steps=None)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_dataset_fn, steps=eval_steps, name='')

  return train_spec, eval_spec
