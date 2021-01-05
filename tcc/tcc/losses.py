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

"""Loss functions imposing the cycle-consistency constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


def classification_loss(logits, labels, label_smoothing):
  """Loss function based on classifying the correct indices.

  In the paper, this is called Cycle-back Classification.

  Args:
    logits: Tensor, Pre-softmax scores used for classification loss. These are
      similarity scores after cycling back to the starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    label_smoothing: Float, label smoothing factor which can be used to
      determine how hard the alignment should be.
  Returns:
    loss: Tensor, A scalar classification loss calculated using standard softmax
      cross-entropy loss.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
  labels = tf.stop_gradient(labels)
  return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
      y_true=labels, y_pred=logits, from_logits=True,
      label_smoothing=label_smoothing))


def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type,
                    normalize_indices, variance_lambda, huber_delta):
  """Loss function based on regressing to the correct indices.

  In the paper, this is called Cycle-back Regression. There are 3 variants
  of this loss:
  i) regression_mse: MSE of the predicted indices and ground truth indices.
  ii) regression_mse_var: MSE of the predicted indices that takes into account
  the variance of the similarities. This is important when the rate at which
  sequences go through different phases changes a lot. The variance scaling
  allows dynamic weighting of the MSE loss based on the similarities.
  iii) regression_huber: Huber loss between the predicted indices and ground
  truth indices.


  Args:
    logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional temporal information to the alignment loss.
    loss_type: String, This specifies the kind of regression loss function.
      Currently supported loss functions: regression_mse, regression_mse_var,
      regression_huber.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities don't arise as sequence
      indices can be large numbers.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.

  Returns:
     loss: Tensor, A scalar loss calculated using a variant of regression.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
  labels = tf.stop_gradient(labels)
  steps = tf.stop_gradient(steps)

  if normalize_indices:
    float_seq_lens = tf.cast(seq_lens, tf.float32)
    tile_seq_lens = tf.tile(
        tf.expand_dims(float_seq_lens, axis=1), [1, num_steps])
    steps = tf.cast(steps, tf.float32) / tile_seq_lens
  else:
    steps = tf.cast(steps, tf.float32)

  beta = tf.nn.softmax(logits)
  true_time = tf.reduce_sum(steps * labels, axis=1)
  pred_time = tf.reduce_sum(steps * beta, axis=1)

  if loss_type in ['regression_mse', 'regression_mse_var']:
    if 'var' in loss_type:
      # Variance aware regression.
      pred_time_tiled = tf.tile(tf.expand_dims(pred_time, axis=1),
                                [1, num_steps])

      pred_time_variance = tf.reduce_sum(
          tf.square(steps - pred_time_tiled) * beta, axis=1)

      # Using log of variance as it is numerically stabler.
      pred_time_log_var = tf.math.log(pred_time_variance)
      squared_error = tf.square(true_time - pred_time)
      return tf.reduce_mean(tf.math.exp(-pred_time_log_var) * squared_error
                            + variance_lambda * pred_time_log_var)

    else:
      return tf.reduce_mean(
          tf.keras.losses.mean_squared_error(y_true=true_time,
                                             y_pred=pred_time))
  elif loss_type == 'regression_huber':
    return tf.reduce_mean(tf.keras.losses.huber_loss(
        y_true=true_time, y_pred=pred_time,
        delta=huber_delta))
  else:
    raise ValueError('Unsupported regression loss %s. Supported losses are: '
                     'regression_mse, regresstion_mse_var and regression_huber.'
                     % loss_type)
