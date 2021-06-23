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

"""Construct model and evaluation metrics for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import protein_dataset
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.layers.python.layers import optimizers as optimizers_lib


REPRESENTATION_KEY = 'representation'
LOGITS_KEY = 'logits'


def _set_padding_to_sentinel(padded_representations, sequence_lengths,
                             sentinel):
  """Set padding on batch of padded representations to a sentinel value.

  Useful for preparing a batch of sequence representations for max or average
  pooling.

  Args:
    padded_representations: float32 tensor, shape (batch, longest_sequence, d),
      where d is some arbitrary embedding dimension. E.g. the output of
      tf.data.padded_batch.
    sequence_lengths: tensor, shape (batch,). Each entry corresponds to the
      original length of the sequence (before padding) of that sequence within
      the batch.
    sentinel: float32 tensor, shape: broadcastable to padded_representations.

  Returns:
    tensor of same shape as padded_representations, where all entries
      in the sequence dimension that came from padding (i.e. are beyond index
      sequence_length[i]) are set to sentinel.
  """
  sequence_dimension = 1
  embedding_dimension = 2

  with tf.variable_scope('set_padding_to_sentinel', reuse=False):
    longest_sequence_length = tf.shape(
        padded_representations)[sequence_dimension]
    embedding_size = tf.shape(padded_representations)[embedding_dimension]

    seq_mask = tf.sequence_mask(sequence_lengths, longest_sequence_length)
    seq_mask = tf.expand_dims(seq_mask, [embedding_dimension])
    is_not_padding = tf.tile(seq_mask, [1, 1, embedding_size])

    full_sentinel = tf.zeros_like(padded_representations)
    full_sentinel = full_sentinel + tf.convert_to_tensor(sentinel)

    per_location_representations = tf.where(
        is_not_padding, padded_representations, full_sentinel)

    return per_location_representations


def _make_per_sequence_features(per_location_representations, raw_features,
                                hparams):
  """Aggregate representations across the sequence dimension."""
  del hparams

  sequence_lengths = raw_features[protein_dataset.SEQUENCE_LENGTH_KEY]
  per_location_representations = _set_padding_to_sentinel(
      per_location_representations, sequence_lengths, tf.constant(0.))
  pooled_representation = tf.reduce_max(per_location_representations, axis=1)

  pooled_representation = tf.identity(
      pooled_representation, name='pooled_representation')

  return pooled_representation


def _convert_representation_to_prediction_ops(representation, raw_features,
                                              num_output_classes, hparams):
  """Map per-location features to problem-specific prediction ops.

  Args:
    representation: [batch_size, sequence_length, feature_dim] Tensor.
    raw_features: dictionary containing the raw input Tensors; this is the
      sequence, keyed by sequence_key.
    num_output_classes: number of different labels.
    hparams: tf.contrib.HParams object.

  Returns:
    predictions: dictionary containing Tensors that Estimator
      will return as predictions.
    predictions_for_loss: Tensor that make_loss() consumes.
  """
  del hparams

  per_sequence_features = _make_per_sequence_features(
      per_location_representations=representation,
      raw_features=raw_features,
      hparams=hparams)
  logits = tf.layers.dense(
      per_sequence_features, num_output_classes, name=LOGITS_KEY)

  predictions = {
      protein_dataset.LABEL_KEY:
          tf.identity(tf.sigmoid(logits), name='predictions')
  }

  predictions_for_loss = logits
  return predictions, predictions_for_loss


def _make_representation(features, hparams, mode):
  """Produces [batch_size, sequence_length, embedding_dim] features.

  Args:
    features: dict from str to Tensor, containing sequence and sequence length.
    hparams: tf.contrib.training.HParams()
    mode: tf.estimator.ModeKeys instance.

  Returns:
    Tensor of shape [batch_size, sequence_length, embedding_dim].
  """
  sequence_features = features[protein_dataset.SEQUENCE_KEY]
  sequence_lengths = features[protein_dataset.SEQUENCE_LENGTH_KEY]

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  sequence_features = _conv_layer(
      sequence_features=sequence_features,
      sequence_lengths=sequence_lengths,
      num_units=hparams.filters,
      dilation_rate=1,
      kernel_size=hparams.kernel_size,
  )

  for layer_index in range(hparams.num_layers):
    sequence_features = _residual_block(
        sequence_features=sequence_features,
        sequence_lengths=sequence_lengths,
        hparams=hparams,
        layer_index=layer_index,
        activation_fn=tf.nn.relu,
        is_training=is_training)

  return sequence_features


def _make_prediction_ops(features, hparams, mode, num_output_classes):
  """Returns (predictions, predictions_for_loss)."""
  del hparams, mode
  logits = tf.layers.dense(
      features, num_output_classes, name='logits')

  confidences = tf.nn.softmax(logits)
  confidence_of_max_prediction = tf.reduce_max(confidences, axis=-1)
  predicted_index = tf.argmax(confidences, axis=-1)

  predictions = {
      'label': predicted_index,
      'logits': logits,
      'confidences': confidences,
      'confidence_of_max_prediction': confidence_of_max_prediction
  }

  predictions_for_loss = logits
  return predictions, predictions_for_loss


def _batch_norm(features, is_training):
  return tf.layers.batch_normalization(features, training=is_training)


def _conv_layer(sequence_features, sequence_lengths, num_units, dilation_rate,
                kernel_size):
  """Return a convolution of the input features that respects sequence len."""
  padding_zeroed = _set_padding_to_sentinel(sequence_features, sequence_lengths,
                                            tf.constant(0.))
  conved = tf.layers.conv1d(
      padding_zeroed,
      filters=num_units,
      kernel_size=[kernel_size],
      dilation_rate=dilation_rate,
      padding='same')

  # Re-zero padding, because shorter sequences will have their padding
  # affected by half the width of the convolution kernel size.
  re_zeroed = _set_padding_to_sentinel(conved, sequence_lengths,
                                       tf.constant(0.))
  return re_zeroed


def _residual_block(sequence_features, sequence_lengths, hparams, layer_index,
                    activation_fn, is_training):
  """Construct a single block for a residual network."""

  with tf.variable_scope('residual_block_{}'.format(layer_index), reuse=False):
    shifted_layer_index = layer_index - hparams.first_dilated_layer + 1
    dilation_rate = max(1, hparams.dilation_rate**shifted_layer_index)

    num_bottleneck_units = math.floor(
        hparams.resnet_bottleneck_factor * hparams.filters)

    features = _batch_norm(sequence_features, is_training)
    features = activation_fn(features)
    features = _conv_layer(
        sequence_features=features,
        sequence_lengths=sequence_lengths,
        num_units=num_bottleneck_units,
        dilation_rate=dilation_rate,
        kernel_size=hparams.kernel_size,
    )
    features = _batch_norm(features, is_training=is_training)
    features = activation_fn(features)

    # The second convolution is purely local linear transformation across
    # feature channels, as is done in
    # tensorflow_models/slim/nets/resnet_v2.bottleneck
    residual = _conv_layer(
        features,
        sequence_lengths,
        num_units=hparams.filters,
        dilation_rate=1,
        kernel_size=1)

    with_skip_connection = sequence_features + residual
    return with_skip_connection


def _make_loss(predictions_for_loss, labels, num_output_classes):
  """Make scalar loss."""
  del num_output_classes

  logits = predictions_for_loss
  labels_op = labels[protein_dataset.LABEL_KEY]

  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_op, logits=logits))


def _make_train_op(loss, hparams):
  """Create train op."""

  def learning_rate_decay_fn(learning_rate, global_step):
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               hparams.lr_decay_steps,
                                               hparams.lr_decay_rate)
    learning_rate = learning_rate * tf.minimum(
        tf.cast(global_step / hparams.lr_warmup_steps, tf.float32),
        tf.constant(1.))
    return learning_rate

  return contrib_layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      clip_gradients=optimizers_lib.adaptive_clipping_fn(
          decay=hparams.gradient_clipping_decay,
          report_summary=True,
      ),
      learning_rate=hparams.learning_rate,
      learning_rate_decay_fn=learning_rate_decay_fn,
      optimizer='Adam')


def make_model_fn(label_vocab, hparams):
  """Returns a model function for estimator given prediction base class.

  Args:
    label_vocab: list of string.
    hparams: tf.contrib.HParams object.

  Returns:
    A function that returns a tf.estimator.EstimatorSpec
  """
  del hparams

  def _model_fn(features, labels, params, mode=None):
    """Returns tf.estimator.EstimatorSpec."""

    num_output_classes = len(label_vocab)
    predictions, predictions_for_loss = _make_prediction_ops(
        features=features,
        hparams=params,
        mode=mode,
        num_output_classes=num_output_classes)

    evaluation_hooks = []
    if mode == tf.estimator.ModeKeys.TRAIN:
      loss = _make_loss(
          predictions_for_loss=predictions_for_loss,
          labels=labels,
          num_output_classes=num_output_classes)
      train_op = _make_train_op(loss=loss, hparams=params)
      eval_ops = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
      loss = None
      train_op = None
      eval_ops = None
    else:  # Eval mode.
      loss = _make_loss(
          predictions_for_loss=predictions_for_loss,
          labels=labels,
          num_output_classes=num_output_classes)

      train_op = None
      eval_ops = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_ops,
        evaluation_hooks=evaluation_hooks,
    )

  return _model_fn
