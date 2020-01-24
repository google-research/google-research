# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Observation based RNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from explaining_risk_increase import input_fn
from tensorflow.contrib import estimator as contrib_estimator
from tensorflow.contrib import lookup as contrib_lookup
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import training as contrib_training
from tensorflow.contrib.learn.python.learn.estimators import rnn_common


TOLERANCE = 0.2


class PredictionKeys(object):
  """Enum for prediction keys."""
  LOGITS = 'logits'
  PROBABILITIES = 'probs'
  CLASSES = 'classes'


def _most_recent_obs_value(obs_values, indicator, delta_time,
                           attribution_max_delta_time):
  """Returns the most recent lab result for each test within a time frame.

  The eligible lab values fall into a time window until time of prediction -
  attribution_max_delta_time. Among those we select their most recent value
  or zero if there are none.

  Args:
    obs_values: A dense representation of the observation_values at the position
      of their obs_code_ids. A padded Tensor of shape [batch_size,
      max_sequence_length, vocab_size] of type float32 where obs_values[b, t,
      id] = observation_values[b, t, 0] and id = observation_code_ids[b, t, 0]
      and obs_values[b, t, x] = 0 for all other x != id. If t is greater than
      the sequence_length of batch entry b then the result is 0 as well.
    indicator: A one-hot encoding of whether a value in obs_values comes from
      observation_values or is just filled in to be 0. A Tensor of shape
      [batch_size, max_sequence_length, vocab_size] and type float32.
    delta_time: A Tensor of shape [batch_size, max_sequence_length] describing
      the time to prediction.
    attribution_max_delta_time: Time threshold so that we return the most recent
      lab values among those that are at least attribution_max_delta_time
      seconds old at time of prediction.

  Returns:
    A Tensor of shape [batch_size, 1, vocab_size] of the most recent lab results
    for all lab tests that are at least attribution_max_delta_time old at time
    of prediction.
  """
  batch_size = tf.shape(indicator)[0]
  seq_len = tf.shape(indicator)[1]
  num_obs = indicator.shape[2]
  # Prepend a dummy so that for lab tests for which we have no eligible lab
  # values we will select 0.
  obs_values = tf.concat(
      [tf.zeros([batch_size, 1, num_obs]), obs_values], axis=1)
  indicator = tf.concat([tf.ones([batch_size, 1, num_obs]), indicator], axis=1)
  delta_time = tf.to_int32(delta_time)
  delta_time = tf.concat(
      [
          tf.zeros([batch_size, 1, 1], dtype=tf.int32) +
          attribution_max_delta_time, delta_time
      ],
      axis=1)
  # First we figure out what the eligible lab values are that are at least
  # attribution_max_delta_time old.
  indicator = tf.to_int32(indicator)
  indicator *= tf.to_int32(delta_time >= attribution_max_delta_time)
  range_val = tf.expand_dims(tf.range(seq_len + 1), axis=0)
  range_val = tf.tile(range_val, multiples=[tf.shape(indicator)[0], 1])
  # [[[0], [1], ..., [max_sequence_length]],
  #  [[0], [1], ..., [max_sequence_length]],
  #  ...]
  range_val = tf.expand_dims(range_val, axis=2)
  # [batch_size, max_sequence_length, vocab_size] with 1 non-zero number per
  # time-step equal to that time-step.
  seq_indicator = indicator * range_val
  # [batch_size, vocab_size] with the time-step of the last lab value.
  last_val_indicator = tf.reduce_max(seq_indicator, axis=1, keepdims=True)
  last_val_indicator = tf.tile(
      last_val_indicator, multiples=[1, tf.shape(indicator)[1], 1])

  # eq indicates which lab values are the most recent ones.
  eq = tf.logical_and(
      tf.equal(last_val_indicator, seq_indicator), indicator > 0)
  most_recent_obs_value_indicator = tf.where(eq)
  # Collect the lab values associated with those indices.
  res = tf.gather_nd(obs_values, most_recent_obs_value_indicator)
  # Reorder the values by batch and then by lab test.
  res_sorted = tf.sparse_reorder(
      tf.sparse_transpose(
          tf.SparseTensor(
              indices=most_recent_obs_value_indicator,
              values=res,
              dense_shape=tf.to_int64(
                  tf.stack([batch_size, seq_len + 1, num_obs]))),
          perm=[0, 2, 1])).values

  return tf.reshape(res_sorted, [batch_size, 1, num_obs])


def _predictions_for_gradients(predictions, seq_mask, delta_time,
                               attribution_max_delta_time, averaged):
  """Aggregates eligible predictions over time.

  Predictions are eligible if their are within the sequence_length (as indicated
  by seq_mask) and their associated delta_time is at most
  attribution_max_delta_time.
  Eligible predictions are either averaged across those eligble times (if
  averaged=True) or summed otherwise.

  Args:
    predictions: A Tensor of shape [batch_size, max_seq_len, 1]
      with the predictions in the sequence.
    seq_mask: A Tensor of shape [batch_size, max_sequence_length, 1] indicating
      which timesteps are padded.
    delta_time: A Tensor of shape [batch_size, max_sequence_length] describing
      the time to prediction.
    attribution_max_delta_time: Attribution is limited to values that are no
      older than that many seconds at time of prediction.
    averaged: Whether predictions are simply summed up across the time-steps
      or averaged over on the sequence length.
  Returns:
    A Tensor of shape [batch, 1, 1] of the eligible predictions
    aggregated across time.
  """
  mask = seq_mask * tf.to_float(delta_time < attribution_max_delta_time)
  predictions *= mask
  if averaged:
    predictions /= tf.reduce_sum(mask, axis=1, keepdims=True)
  return tf.reduce_sum(predictions, axis=1, keepdims=True)


def compute_gradient_attribution(predictions, obs_values, indicator):
  """Constructs the attribution of what inputs result in a higher prediction.

  Attribution here refers to the timesteps in which the predictions (derived
  from the logits) increased. We are only interested in increases in the
  previous 12h.

  Args:
    predictions: A Tensor of shape [batch_size, 1, 1] with the
      predictions in the sequence.
    obs_values: A dense representation of the observation_values with
      obs_values[b, t, :] has at most one non-zero value at the position
      of the corresponding lab test from obs_code_ids with the value of the lab
      result. A padded Tensor of shape [batch_size, max_sequence_length,
      vocab_size] of type float32 of possibly normalized observation values.
    indicator: A one-hot encoding of whether a value in obs_values comes from
      observation_values or is just filled in to be 0. A Tensor of
      shape [batch_size, max_sequence_length, vocab_size] and type float32.
  Returns:
    A Tensor of shape [batch, max_sequence_length, 1] of the gradient of the
    prediction as a function of the lab result at that batch-entry time.
  """
  attr = tf.gradients(tf.squeeze(predictions, axis=1,
                                 name='squeeze_pred_for_gradients'),
                      [obs_values])[0]
  # Zero-out gradients for other lab-tests and then sum up across lab tests
  # for which at most one gradient will be non-zero.
  attr *= indicator
  attr = tf.reduce_sum(attr, axis=2, keepdims=True)
  return attr


def compute_path_integrated_gradient_attribution(
    obs_values,
    indicator,
    diff_delta_time,
    delta_time,
    sequence_length,
    seq_mask,
    hparams,
    construct_logits_fn=None):
  """Constructs the attribution of what inputs result in a higher prediction.

  Attribution here refers to the integrated gradients as defined here
  https://arxiv.org/pdf/1703.01365.pdf and approximated for the j-th variable
  via

  (x-x') * 1/num_steps * sum_{i=1}^{num_steps} of the derivative of
  F(x'+(x-x')*i/num_steps) w.r.t. its j-th input.

  where we take x' the most recent value before attribution_max_delta_time and
  x to be the subsequent observation values from the same lab test.
  x'+(x-x')*i/num_steps is the linear interpolation between x' and x.

  Args:
    obs_values: A dense representation of the observation_values with
      obs_values[b, t, :] has at most one non-zero value at the position
      of the corresponding lab test from obs_code_ids with the value of the lab
      result. A padded Tensor of shape [batch_size, max_sequence_length,
      vocab_size] of type float32 of possibly normalized observation values.
    indicator: A one-hot encoding of whether a value in obs_values comes from
      observation_values or is just filled in to be 0. A Tensor of
      shape [batch_size, max_sequence_length, vocab_size] and type float32.
    diff_delta_time: Difference between two consecutive time steps.
    delta_time: A Tensor of shape [batch_size, max_sequence_length] describing
      the time to prediction.
    sequence_length: Sequence length (before padding), Tensor of shape
      [batch_size].
    seq_mask: A Tensor of shape [batch_size, max_sequence_length, 1]
      indicating which timesteps are padded.
    hparams: Hyper parameters.
    construct_logits_fn: A method with constructing the logits given input as
      construct_logits. If None using construct_logits.
  Returns:
    A Tensor of shape [batch, max_sequence_length, 1] of the gradient of the
    prediction as a function of the lab result at that batch-entry time.
  """
  last_obs_values_0 = _most_recent_obs_value(obs_values, indicator, delta_time,
                                             hparams.attribution_max_delta_time)
  gradients = []
  # We need to limit the diff over the base to timesteps after base.
  last_obs_values = last_obs_values_0 * (
      tf.to_float(indicator) *
      tf.to_float(delta_time < hparams.attribution_max_delta_time))
  obs_values_with_last_replaced = obs_values * tf.to_float(
      delta_time >= hparams.attribution_max_delta_time) + last_obs_values
  diff_over_base = obs_values - obs_values_with_last_replaced

  for i in range(hparams.path_integrated_gradients_num_steps):
    alpha = 1.0 * i / (hparams.path_integrated_gradients_num_steps - 1)
    step_obs_values = obs_values_with_last_replaced + diff_over_base * alpha
    if not construct_logits_fn:
      construct_logits_fn = construct_logits
    logits, _ = construct_logits_fn(
        diff_delta_time,
        step_obs_values,
        indicator,
        sequence_length,
        seq_mask,
        hparams,
        reuse=True)
    if hparams.use_rnn_attention:
      last_logits = logits
    else:
      last_logits = rnn_common.select_last_activations(
          logits, tf.to_int32(sequence_length))
    # Ideally, we'd like to get the gradients of the change in
    # value over the previous one to attribute it to both and not just a single
    # value.
    gradient = compute_gradient_attribution(last_logits, step_obs_values,
                                            indicator)
    gradients.append(
        tf.reduce_sum(diff_over_base, axis=2, keepdims=True) * gradient)
  return tf.add_n(gradients) / tf.to_float(
      hparams.path_integrated_gradients_num_steps)


def compute_attention(seq_output, last_output, hidden_layer_dim, seq_mask,
                      sequence_length):
  """Constructs attention of the last_output as query and the sequence output.

  The attention is the dot-product of the last_output (the final RNN output),
  with the seq_output (the RNN's output at each step). Here the final RNN output
  is considered as the "query" or "context" vector. The final attention output
  is a weighted sum of the RNN's outputs at all steps. Details:

    alpha_i = seq_output_i * last_output
    beta is then obtained by normalizing alpha:
    beta_i = exp(alpha_i) / sum_j exp(alpha_j)
    The new attention vector is then the beta-weighted sum over the seq_output:
    attention_vector = sum_i beta_i * seq_output_i

  If hidden_dim > 0 then before computing alpha the seq_output and the
  last_output are sent through two separate hidden layers.
  seq_output = hidden_layer(seq_output)
  last_output = hidden_layer(last_output)

  Args:
    seq_output: The raw rnn output of shape [batch_size, max_sequence_length,
      rnn_size].
    last_output: The last output of the rnn of shape [batch_size, rnn_size].
    hidden_layer_dim: If 0 no hidden layer is applied before multiplying the
      last_logits with the seq_logits.
    seq_mask: A Tensor of shape [batch_size, max_sequence_length, 1] indicating
      which timesteps are padded.
    sequence_length: Sequence length (before padding), Tensor of shape
      [batch_size].

  Returns:
    Attention output with shape [batch_size, rnn_size].
    The attention beta tensor.
  """
  # Compute the weights.
  if hidden_layer_dim > 0:
    last_output = tf.layers.dense(
        last_output, hidden_layer_dim, activation=tf.nn.relu6)
    seq_output = tf.layers.dense(
        seq_output, hidden_layer_dim, activation=tf.nn.relu6)
  last_output = tf.expand_dims(last_output, 1)  # [batch_size, 1, rnn_size]
  tmp = tf.multiply(seq_output, last_output)  # dim 1: broadcast
  alpha_tensor = tf.reduce_sum(tmp, 2)  # [b, max_seq_len]
  alpha_tensor *= tf.squeeze(seq_mask, axis=2)
  beta_tensor = tf.nn.softmax(alpha_tensor)  # using default dim -1
  beta_tensor = tf.expand_dims(beta_tensor, -1)  # [b, max_seq_len, 1]

  # Compute weighted sum of the original rnn_outputs over all steps
  tmp = seq_output * beta_tensor  # last dim: use "broadcast"
  rnn_outputs_weighted_sum = tf.reduce_sum(tmp, 1)  # [b, rnn_size]
  last_beta = rnn_common.select_last_activations(
      beta_tensor, tf.to_int32(sequence_length))
  tf.summary.histogram('last_beta_attention', last_beta)

  return rnn_outputs_weighted_sum, beta_tensor


def compute_prediction_diff_attribution(logits):
  """Constructs the attribution of what inputs result in a higher prediction.

  Attribution here refers to the timesteps in which the predictions (derived
  from the logits) increased.

  Args:
    logits: The logits of the model_fn.
  Returns:
    A Tensor of shape [batch_size, max_sequence_length, 1] with an attribution
    value at time t of prediction at time t minus prediction at time t-1.
  """
  predictions = tf.sigmoid(logits)
  shape = tf.shape(logits)
  zeros = tf.zeros(shape=[shape[0], 1, shape[2]], dtype=tf.float32)
  # Our basic notion of attribution at timestep i is how much the predicted
  # risk increased at that time compared to the previous prediction.
  return predictions - tf.concat(
      [zeros, predictions[:, :-1, :]], axis=1, name='attribution')


def convert_attribution(attribution, sequence_feature_map, seq_mask, delta_time,
                        attribution_threshold, attribution_max_delta_time,
                        prefix=''):
  """Constructs the attribution of what inputs result in a higher prediction.

  Attribution here refers to the timesteps in which the predictions (derived
  from the logits) increased. We are only interested in increases in the
  previous attribution_max_delta_time.

  Args:
    attribution: A Tensor of shape [batch, max_sequence_length, 1] computed
      using some attribution method.
    sequence_feature_map: A dictionary from name to (Sparse)Tensor.
    seq_mask: A Tensor of shape [batch_size, max_sequence_length, 1] indicating
      which timesteps are padded.
    delta_time: A Tensor of shape [batch_size, max_sequence_length] describing
      the time to prediction.
    attribution_threshold: Attribution values below this threshold will be
      dropped.
    attribution_max_delta_time: Attribution is limited to values that are no
      older than that many seconds at time of prediction.
    prefix: A string to prepend to the feature names for the attribution_dict.
  Returns:
    A dictionary from feature names to SparseTensors of
    dense_shape [batch_size, max_sequence_length, 1].
  """
  # We do not want attribution in the padding.
  attribution *= seq_mask

  # We focus on attribution in the past 12h.
  # [batch_size, max_sequence_length, 1]
  attribution *= tf.to_float(delta_time < attribution_max_delta_time)

  # We get rid of low attribution.
  attribution_indices = tf.where(attribution > attribution_threshold)
  attribution_values = tf.gather_nd(attribution, attribution_indices)

  # Now, attribution.indices indicate in the input timesteps which we should
  # attend to.
  attribution_dict = {}
  for feature, sp_feature in sequence_feature_map.items():
    # Limitation: This is not going to work for sequence feature in which
    # the third (last/token) dimension is > 1. In that case only the first
    # token would be highlighted.
    attribution_dict[prefix + feature] = tf.sparse.expand_dims(
        tf.SparseTensor(
            indices=attribution_indices,
            values=attribution_values,
            dense_shape=tf.to_int64(tf.shape(sp_feature))), axis=1)
  return attribution_dict


def normalize_each_feature(observation_values, obs_code, vocab_size, mode,
                           momentum):
  """Combines SparseTensors of observation codes and values into a Tensor.

  Args:
    observation_values: A SparseTensor of type float with the observation
      values of dense shape [batch_size, max_sequence_length, 1].
      There may be no time gaps in between codes.
    obs_code: A Tensor of shape [?, 3] of type int32 with the ids that go along
      with the observation_values. We will do the normalization separately for
      each lab test.
    vocab_size: The range of the values in obs_code is from 0 to vocab_size.
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    momentum: Mean and variance will be updated as
      momentum*old_value + (1-momentum) * new_value.
  Returns:
    observation_values as in the input only with normalized values.
  """
  with tf.variable_scope('batch_normalization'):
    new_indices = []
    new_values = []

    for i in range(vocab_size):
      with tf.variable_scope('bn' + str(i)):
        positions_of_feature_i = tf.where(tf.equal(obs_code, i))
        values_of_feature_i = tf.gather_nd(observation_values.values,
                                           positions_of_feature_i)
        if mode == tf.estimator.ModeKeys.TRAIN:
          tf.summary.scalar('avg_observation_values/' + str(i),
                            tf.reduce_mean(values_of_feature_i))
          tf.summary.histogram('observation_values/' + str(i),
                               values_of_feature_i)
        batchnorm_layer = tf.layers.BatchNormalization(
            axis=1,
            momentum=momentum,
            epsilon=0.01,
            trainable=True)
        normalized_values = tf.squeeze(
            batchnorm_layer.apply(
                tf.expand_dims(values_of_feature_i, axis=1),
                training=(mode == tf.estimator.ModeKeys.TRAIN)
            ),
            axis=1,
            name='squeeze_normalized_values')
        if mode == tf.estimator.ModeKeys.TRAIN:
          tf.summary.scalar('batchnorm_layer/moving_mean/' + str(i),
                            tf.squeeze(batchnorm_layer.moving_mean))
          tf.summary.scalar('batchnorm_layer/moving_variance/' + str(i),
                            tf.squeeze(batchnorm_layer.moving_variance))
          tf.summary.scalar('avg_normalized_values/' + str(i),
                            tf.reduce_mean(normalized_values))
          tf.summary.histogram('normalized_observation_values/' + str(i),
                               normalized_values)
        indices_i = tf.gather_nd(observation_values.indices,
                                 positions_of_feature_i)
        new_indices += [indices_i]
        normalized_values = tf.where(tf.is_nan(normalized_values),
                                     tf.zeros_like(normalized_values),
                                     normalized_values)
        new_values += [normalized_values]

    normalized_sp_tensor = tf.SparseTensor(
        indices=tf.concat(new_indices, axis=0),
        values=tf.concat(new_values, axis=0),
        dense_shape=observation_values.dense_shape)
    normalized_sp_tensor = tf.sparse_reorder(normalized_sp_tensor)
    return normalized_sp_tensor


def combine_observation_code_and_values(observation_code_ids,
                                        observation_values, vocab_size, mode,
                                        normalize, momentum, min_value,
                                        max_value):
  """Combines SparseTensors of observation codes and values into a Tensor.

  Args:
    observation_code_ids: A SparseTensor of type int32 with the ids of the
      observation codes of dense shape [batch_size, max_sequence_length, 1].
      There may be no time gaps in between codes.
    observation_values: A SparseTensor of type float with the observation
      values of dense shape [batch_size, max_sequence_length, 1].
      There may be no time gaps in between codes.
    vocab_size: The range of the values in obs_code_ids is from 0 to vocab_size.
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    normalize: Whether to normalize each lab test.
    momentum: For the batch normalization mean and variance will be updated as
      momentum*old_value + (1-momentum) * new_value.
    min_value: Observation values smaller than this will be capped to min_value.
    max_value: Observation values larger than this will be capped to max_value.

  Returns:
    - obs_values: A dense representation of the observation_values at the
                  position of their obs_code_ids. A padded Tensor of shape
                  [batch_size, max_sequence_length, vocab_size] of type float32
                  where obs_values[b, t, id] = observation_values[b, t, 0] and
                  id = observation_code_ids[b, t, 0] and obs_values[b, t, x] = 0
                  for all other x != id. If t is greater than the
                  sequence_length of batch entry b then the result is 0 as well.
    - indicator: A one-hot encoding of whether a value in obs_values comes from
                 observation_values or is just filled in to be 0. A Tensor of
                 shape [batch_size, max_sequence_length, vocab_size] and type
                 float32.
  """
  obs_code = observation_code_ids.values
  if normalize:
    with tf.variable_scope('values'):
      observation_values = normalize_each_feature(
          observation_values, obs_code, vocab_size, mode, momentum)
  observation_values_rank2 = tf.SparseTensor(
      values=observation_values.values,
      indices=observation_values.indices[:, 0:2],
      dense_shape=observation_values.dense_shape[0:2])
  obs_indices = tf.concat(
      [observation_values_rank2.indices,
       tf.expand_dims(obs_code, axis=1)],
      axis=1, name='obs_indices')
  obs_shape = tf.concat(
      [observation_values_rank2.dense_shape, [vocab_size]], axis=0,
      name='obs_shape')

  obs_values = tf.sparse_to_dense(obs_indices, obs_shape,
                                  observation_values_rank2.values)
  obs_values.set_shape([None, None, vocab_size])
  indicator = tf.sparse_to_dense(obs_indices, obs_shape,
                                 tf.ones_like(observation_values_rank2.values))
  indicator.set_shape([None, None, vocab_size])
  # clip
  obs_values = tf.minimum(obs_values, max_value)
  obs_values = tf.maximum(obs_values, min_value)
  return obs_values, indicator


def construct_input(sequence_feature_map, categorical_values,
                    categorical_seq_feature, feature_value, mode, normalize,
                    momentum, min_value, max_value, input_keep_prob):
  """Returns a function to build the model.

  Args:
    sequence_feature_map: A dictionary of (Sparse)Tensors of dense shape
      [batch_size, max_sequence_length, None] keyed by the feature name.
    categorical_values: Potential values of the categorical_seq_feature.
    categorical_seq_feature: Name of feature of observation code.
    feature_value: Name of feature of observation value.
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    normalize: Whether to normalize each lab test.
    momentum: For the batch normalization mean and variance will be updated as
      momentum*old_value + (1-momentum) * new_value.
    min_value: Observation values smaller than this will be capped to min_value.
    max_value: Observation values larger than this will be capped to max_value.
    input_keep_prob: Keep probability for input observation values.

  Returns:
    - diff_delta_time: Tensor of shape [batch_size, max_seq_length, 1]
      with the
    - obs_values: A dense representation of the observation_values with
                  obs_values[b, t, :] has at most one non-zero value at the
                  position of the corresponding lab test from obs_code_ids with
                  the value of the lab result. A padded Tensor of shape
                  [batch_size, max_sequence_length, vocab_size] of type float32
                  of possibly normalized observation values.
    - indicator: A one-hot encoding of whether a value in obs_values comes from
                 observation_values or is just filled in to be 0. A Tensor of
                 shape [batch_size, max_sequence_length, vocab_size] and type
                 float32.
  """
  with tf.variable_scope('input'):
    sequence_feature_map = {
        k: tf.sparse_reorder(s) if isinstance(s, tf.SparseTensor) else s
        for k, s in sequence_feature_map.items()
    }
    # Filter out invalid values.
    # For invalid observation values we do this through a sparse retain.
    # This makes sure that the invalid values will not be considered in the
    # normalization.
    observation_values = sequence_feature_map[feature_value]
    observation_code_sparse = sequence_feature_map[categorical_seq_feature]
    # Future work: Create a flag for the missing value indicator.
    valid_values = tf.abs(observation_values.values - 9999999.0) > TOLERANCE
    # apply input dropout
    if input_keep_prob < 1.0:
      random_tensor = input_keep_prob
      random_tensor += tf.random_uniform(tf.shape(observation_values.values))
      # 0. if [input_keep_prob, 1.0) and 1. if [1.0, 1.0 + input_keep_prob)
      dropout_mask = tf.floor(random_tensor)
      if mode == tf.estimator.ModeKeys.TRAIN:
        valid_values = tf.to_float(valid_values) * dropout_mask
        valid_values = valid_values > 0.5
    sequence_feature_map[feature_value] = tf.sparse_retain(
        observation_values, valid_values)
    sequence_feature_map[categorical_seq_feature] = tf.sparse_retain(
        observation_code_sparse, valid_values)

    # 1. Construct the sequence of observation values to feed into the RNN
    #    and their indicator.
    # We assign each observation code an id from 0 to vocab_size-1. At each
    # timestep we will lookup the id for the observation code and take the value
    # of the lab test and a construct a vector with all zeros but the id-th
    # position is set to the lab test value.
    obs_code = sequence_feature_map[categorical_seq_feature]
    obs_code_dense_ids = contrib_lookup.index_table_from_tensor(
        tuple(categorical_values), num_oov_buckets=0,
        name='vocab_lookup').lookup(obs_code.values)
    obs_code_sparse = tf.SparseTensor(
        values=obs_code_dense_ids,
        indices=obs_code.indices,
        dense_shape=obs_code.dense_shape)
    obs_code_sparse = tf.sparse_reorder(obs_code_sparse)
    observation_values = sequence_feature_map[feature_value]
    observation_values = tf.sparse_reorder(observation_values)
    vocab_size = len(categorical_values)
    obs_values, indicator = combine_observation_code_and_values(
        obs_code_sparse, observation_values, vocab_size, mode, normalize,
        momentum, min_value, max_value)

    # 2. We compute the diff_delta_time as additional sequence feature.
    # Note, the LSTM is very sensitive to how you encode time.
    delta_time = sequence_feature_map['deltaTime']
    diff_delta_time = tf.concat(
        [delta_time[:, :1, :], delta_time[:, :-1, :]], axis=1) - delta_time
    diff_delta_time = tf.to_float(diff_delta_time) / (60.0 * 60.0)

  return (diff_delta_time, obs_values, indicator)


def construct_rnn_logits(diff_delta_time,
                         obs_values,
                         indicator,
                         sequence_length,
                         rnn_size,
                         variational_recurrent_keep_prob,
                         variational_input_keep_prob,
                         variational_output_keep_prob,
                         reuse=False):
  """Computes logits combining inputs and applying an RNN.

  Args:
   diff_delta_time: Difference between two consecutive time steps.
   obs_values: A dense representation of the observation_values with
      obs_values[b, t, :] has at most one non-zero value at the position
      of the corresponding lab test from obs_code_ids with the value of the lab
      result. A padded Tensor of shape [batch_size, max_sequence_length,
      vocab_size] of type float32 of possibly normalized observation values.
    indicator: A one-hot encoding of whether a value in obs_values comes from
      observation_values or is just filled in to be 0. A Tensor of
      shape [batch_size, max_sequence_length, vocab_size] and type float32.
    sequence_length: Sequence length (before padding), Tensor of shape
      [batch_size].
    rnn_size: Size of the LSTM hidden state and output.
    variational_recurrent_keep_prob: 1 - droput for the hidden LSTM state.
    variational_input_keep_prob: 1 - dropout for the input to the LSTM.
    variational_output_keep_prob: 1 - dropout for the output of the LSTM.
    reuse: Whether to reuse existing variables or setup new ones.

  Returns:
    logits a Tensor of shape [batch_size, max_sequence_length, 1].
  """
  with tf.variable_scope('logits/rnn', reuse=reuse) as sc:
    rnn_inputs = [diff_delta_time, indicator, obs_values]
    sequence_data = tf.concat(rnn_inputs, axis=2, name='rnn_input')

    # Run a recurrent neural network across the time dimension.
    cell = contrib_rnn.LSTMCell(rnn_size, state_is_tuple=True)
    if (variational_recurrent_keep_prob < 1 or variational_input_keep_prob < 1
        or variational_output_keep_prob < 1):
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=variational_input_keep_prob,
          output_keep_prob=variational_output_keep_prob,
          state_keep_prob=variational_recurrent_keep_prob,
          variational_recurrent=True, input_size=tf.shape(sequence_data)[2],
          seed=12345678)

    output, _ = tf.nn.dynamic_rnn(
        cell,
        sequence_data,
        sequence_length=sequence_length,
        dtype=tf.float32,
        swap_memory=True,
        scope='rnn')

    # 3. Make a time-series of logits via a linear-mapping of the rnn-output
    #    to logits_dimension = 1.
    return tf.layers.dense(
        output, 1, name=sc, reuse=reuse, activation=None), output


def construct_logits(diff_delta_time, obs_values, indicator, sequence_length,
                     seq_mask, hparams, reuse):
  """Constructs logits through an RNN.

  Args:
    diff_delta_time: Difference between two consecutive time steps.
    obs_values: A dense representation of the observation_values with
      obs_values[b, t, :] has at most one non-zero value at the position
      of the corresponding lab test from obs_code_ids with the value of the lab
      result. A padded Tensor of shape [batch_size, max_sequence_length,
      vocab_size] of type float32 of possibly normalized observation values.
    indicator: A one-hot encoding of whether a value in obs_values comes from
      observation_values or is just filled in to be 0. A Tensor of
      shape [batch_size, max_sequence_length, vocab_size] and type float32.
    sequence_length: Sequence length (before padding), Tensor of shape
      [batch_size].
    seq_mask: A Tensor of shape [batch_size, max_sequence_length, 1] indicating
      which timesteps are padded.
    hparams: Hyper parameters.
    reuse: Boolean indicator of whether to re-use the variables.

  Returns:
    - Logits: A Tensor of shape [batch, {max_sequence_length,1}, 1].
    - Weights: Defaults to None. Only populated to a Tensor of shape
               [batch, max_sequence_length, 1] if
               hparams.use_rnn_attention is True.
  """

  logits, raw_output = construct_rnn_logits(
      diff_delta_time, obs_values, indicator, sequence_length, hparams.rnn_size,
      hparams.variational_recurrent_keep_prob,
      hparams.variational_input_keep_prob, hparams.variational_output_keep_prob,
      reuse)
  if hparams.use_rnn_attention:
    with tf.variable_scope('logits/rnn/attention', reuse=reuse) as sc:
      last_logits = rnn_common.select_last_activations(
          raw_output, tf.to_int32(sequence_length))
      weighted_final_output, weight = compute_attention(
          raw_output, last_logits, hparams.attention_hidden_layer_dim,
          seq_mask, sequence_length)
      return tf.layers.dense(
          weighted_final_output, 1, name=sc, reuse=reuse,
          activation=None), weight
  else:
    return logits, None


class ObservationSequenceModel(object):
  """Model that runs an RNN over the time series of observation values.

  Consider a single lab (e.g. heart rate) and its value (e.g. 60) at a time.
  The input to the RNN at that timestep will have a value of 60 at the unique
  position for heart rate. The positions of all other lab tests will be 0.

  Additional input to the RNN include an indicator (to be able to distinguish a
  true lab measurement of 0 from the padded ones) and a notion of time
  (in particular how many hours have passed since the previous time-step).

  Caution: This model can only be run on condensed SequenceExample with an
  observation present each time step.
  """

  def create_model_hparams(self):
    """Returns default hparams for observation sequence model."""
    categorical_values_str = 'loinc:2823-3,loinc:2160-0,loinc:804-5,loinc:3094-0,loinc:786-4,loinc:2075-0,loinc:2951-2,loinc:34728-6,mimic3:observation_code:834,mimic3:observation_code:678,loinc:2345-7,mimic3:observation_code:3603,mimic3:observation_code:223761,loinc:3173-2,loinc:5895-7,loinc:5902-2,loinc:2601-3,loinc:2000-8,loinc:2777-1,mimic3:observation_code:3655,loinc:32693-4,mimic3:observation_code:679,mimic3:observation_code:676,loinc:2339-0,loinc:1994-3,mimic3:observation_code:224690,loinc:1975-2,loinc:1742-6,loinc:1920-8,loinc:6768-6,mimic3:observation_code:3312,mimic3:observation_code:8502,mimic3:observation_code:3313,loinc:1751-7,loinc:6598-7,mimic3:observation_code:225309,mimic3:observation_code:225310,mimic3:observation_code:40069,loinc:3016-3,loinc:1968-7,loinc:4548-4,loinc:2093-3,loinc:2085-9,loinc:2090-9,mimic3:observation_code:6701,mimic3:observation_code:8555,mimic3:observation_code:6702,loinc:10839-9,mimic3:observation_code:3318,mimic3:observation_code:3319'
    return contrib_training.HParams(
        context_features=['sequenceLength'],
        batch_size=128,
        learning_rate=0.002,
        sequence_features=[
            'deltaTime',
            'Observation.code',
            'Observation.valueQuantity.value',
            'Observation.valueQuantity.unit',
            'Observation.code.harmonized:valueset-observation-name',
        ],
        feature_value='Observation.valueQuantity.value',
        categorical_values=categorical_values_str.split(','),
        categorical_seq_feature='Observation.code',
        label_key='label.in_hospital_death',
        input_keep_prob=1.0,
        attribution_threshold=0.0001,
        attribution_max_delta_time=12 * 60 * 60,
        rnn_size=64,
        variational_recurrent_keep_prob=0.99,
        variational_input_keep_prob=0.97,
        variational_output_keep_prob=0.98,
        sequence_prediction=False,
        normalize=True,
        momentum=0.75,
        min_value=-1000.0,
        max_value=1000.0,
        # If sequence_prediction is True then the loss will also include the
        # sum of the changes in predictions across the sequence as a way to
        # learn models with less volatile predictions.
        volatility_loss_factor=0.0,
        include_sequence_prediction=True,
        include_gradients_attribution=True,
        include_gradients_sum_time_attribution=False,
        include_gradients_avg_time_attribution=False,
        include_path_integrated_gradients_attribution=True,
        use_rnn_attention=False,
        attention_hidden_layer_dim=0,
        include_diff_sequence_prediction_attribution=True,
        # If include_path_integrated_gradients_attribution determines the number
        # of steps between the old and the current observation value.
        path_integrated_gradients_num_steps=10,
    )

  def create_model_fn(self, hparams):
    """Returns a function to build the model.

    Args:
      hparams: The hyperparameters.

    Returns:
      A function to build the model's graph. This function is called by
      the Estimator object to construct the graph.
    """

    def model_fn(features, labels, mode):
      """Creates the prediction, loss, and train ops.

      Args:
        features: A dictionary of tensors keyed by the feature name.
        labels: A dictionary of label tensors keyed by the label key.
        mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

      Returns:
        EstimatorSpec with the mode, prediction, loss, train_op and
        output_alternatives a dictionary specifying the output for a
        servo request during serving.
      """
      # 1. Construct input to RNN
      sequence_feature_map = {
          k: features[input_fn.SEQUENCE_KEY_PREFIX + k]
          for k in hparams.sequence_features
      }
      sequence_length = tf.squeeze(
          features[input_fn.CONTEXT_KEY_PREFIX + 'sequenceLength'],
          axis=1,
          name='sq_seq_len')
      tf.summary.scalar('sequence_length', tf.reduce_mean(sequence_length))
      diff_delta_time, obs_values, indicator = construct_input(
          sequence_feature_map, hparams.categorical_values,
          hparams.categorical_seq_feature, hparams.feature_value, mode,
          hparams.normalize, hparams.momentum, hparams.min_value,
          hparams.max_value, hparams.input_keep_prob)

      seq_mask = tf.expand_dims(
          tf.sequence_mask(sequence_length, dtype=tf.float32), axis=2)
      logits, weights = construct_logits(
          diff_delta_time,
          obs_values,
          indicator,
          sequence_length,
          seq_mask,
          hparams,
          reuse=False)

      all_attribution_dict = {}
      if mode == tf.estimator.ModeKeys.TRAIN:
        if hparams.sequence_prediction:
          assert not hparams.use_rnn_attention
          # If we train a sequence_prediction we repeat the labels over time.
          label_tensor = labels[hparams.label_key]
          labels[hparams.label_key] = tf.tile(
              tf.expand_dims(label_tensor, 2),
              multiples=[1, tf.shape(logits)[1], 1])
          if hparams.volatility_loss_factor > 0.0:
            volatility = tf.reduce_sum(
                tf.square(seq_mask *
                          compute_prediction_diff_attribution(logits)))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 volatility * hparams.volatility_loss_factor)
        elif not hparams.use_rnn_attention:
          logits = rnn_common.select_last_activations(
              logits, tf.to_int32(sequence_length))
      else:
        if hparams.sequence_prediction:
          last_logits = rnn_common.select_last_activations(
              logits, tf.to_int32(sequence_length))
        else:
          last_logits = logits
        if mode == tf.estimator.ModeKeys.PREDICT:
          delta_time = sequence_feature_map['deltaTime']
          all_attributions = {}
          if hparams.include_gradients_attribution:
            all_attributions['gradient_last'] = compute_gradient_attribution(
                last_logits, obs_values, indicator)
          if hparams.include_gradients_sum_time_attribution:
            assert not hparams.use_rnn_attention
            all_attributions['gradient_sum'] = compute_gradient_attribution(
                _predictions_for_gradients(
                    logits, seq_mask, delta_time,
                    hparams.attribution_max_delta_time, averaged=False),
                obs_values, indicator)
          if hparams.include_gradients_avg_time_attribution:
            assert not hparams.use_rnn_attention
            all_attributions['gradient_avg'] = compute_gradient_attribution(
                _predictions_for_gradients(
                    logits, seq_mask, delta_time,
                    hparams.attribution_max_delta_time, averaged=True),
                obs_values, indicator)
          if hparams.include_path_integrated_gradients_attribution:
            all_attributions['integrated_gradient'] = (
                compute_path_integrated_gradient_attribution(
                    obs_values, indicator, diff_delta_time, delta_time,
                    sequence_length, seq_mask, hparams))
          if hparams.use_rnn_attention:
            all_attributions['rnn_attention'] = weights
          if hparams.include_diff_sequence_prediction_attribution:
            all_attributions['diff_sequence'] = (
                compute_prediction_diff_attribution(logits))

          all_attribution_dict = {}
          for attribution_name, attribution in all_attributions.items():
            attribution_dict = convert_attribution(
                attribution,
                sequence_feature_map,
                seq_mask,
                delta_time,
                hparams.attribution_threshold,
                hparams.attribution_max_delta_time,
                prefix=attribution_name + '-')
            all_attribution_dict.update(attribution_dict)
          if hparams.include_sequence_prediction:
            # Add the predictions at each time step to the attention dictionary.
            attribution_indices = tf.where(seq_mask > 0.5)
            all_attribution_dict['predictions'] = tf.sparse.expand_dims(
                tf.SparseTensor(
                    indices=attribution_indices,
                    values=tf.gather_nd(
                        tf.sigmoid(logits), attribution_indices),
                    dense_shape=tf.to_int64(tf.shape(delta_time))),
                axis=1)
        # At test/inference time we only make a single prediction even if we did
        # sequence_prediction during training.
        logits = last_logits
        seq_mask = None

      probabilities = tf.sigmoid(logits)
      classes = probabilities > 0.5
      predictions = {
          PredictionKeys.LOGITS: logits,
          PredictionKeys.PROBABILITIES: probabilities,
          PredictionKeys.CLASSES: classes
      }
      # Calculate the loss for TRAIN and EVAL, but not PREDICT.
      if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
      else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels[hparams.label_key],
            logits=predictions[PredictionKeys.LOGITS])
        if hparams.sequence_prediction:
          loss *= seq_mask
        loss = tf.reduce_mean(loss)
        regularization_losses = tf.losses.get_regularization_losses()
        if regularization_losses:
          tf.summary.scalar('loss/prior_regularization', loss)
          regularization_loss = tf.add_n(regularization_losses)
          tf.summary.scalar('loss/regularization_loss', regularization_loss)
          loss += regularization_loss
        tf.summary.scalar('loss', loss)

      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparams.learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-8)
        optimizer = contrib_estimator.clip_gradients_by_norm(optimizer, 6.0)
        train_op = contrib_training.create_train_op(
            total_loss=loss, optimizer=optimizer, summarize_gradients=False)
      if mode != tf.estimator.ModeKeys.TRAIN:
        for k, v in all_attribution_dict.items():
          if not isinstance(v, tf.SparseTensor):
            raise ValueError('Expect attributions to be in SparseTensor, '
                             'getting %s for feature %s' %
                             (v.__class__.__name__, k))
          predictions['attention_attribution,%s,indices' % k] = v.indices
          predictions['attention_attribution,%s,values' % k] = v.values
          predictions['attention_attribution,%s,shape' % k] = v.dense_shape

      eval_metric_ops = {}
      if mode == tf.estimator.ModeKeys.EVAL:
        auc = tf.metrics.auc
        prob_k = PredictionKeys.PROBABILITIES
        class_k = PredictionKeys.CLASSES
        m = 'careful_interpolation'
        metric_fn_dict = {
            'auc-roc':
                lambda l, p: auc(l, p[prob_k], curve='ROC', summation_method=m),
            'auc-pr':
                lambda l, p: auc(l, p[prob_k], curve='PR', summation_method=m),
            'accuracy':
                lambda l, p: tf.metrics.accuracy(l, p[class_k]),
        }
        for (k, f) in metric_fn_dict.items():
          eval_metric_ops[k] = f(label_tensor, predictions)
      # Define the output for serving.
      export_outputs = {}
      if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'mortality': tf.estimator.export.PredictOutput(predictions)
        }

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs)

    return model_fn
