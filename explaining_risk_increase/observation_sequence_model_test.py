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

# Lint as: python2, python3

"""Tests for observation sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from six.moves import range
import tensorflow.compat.v1 as tf
from explaining_risk_increase import input_fn
from explaining_risk_increase import observation_sequence_model as osm
from tensorflow.contrib import training as contrib_training


class ObservationSequenceTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ObservationSequenceTest, self).setUp()
    self.observation_values = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 1, 0], [1, 2, 0]],
        values=[100.0, 2.3, 0.5, 0.0, 4.0],
        dense_shape=[2, 3, 1])

  def testGradientAttribution(self):
    factors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    observation_values = tf.constant(
        [[[0, 100.0, 0, 0, 0], [0, 2.3, 0, 0, 0], [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0.5, 0], [0.0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0]]])
    indicator = tf.constant(
        [[[0, 1.0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
         [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
    last_logits = tf.reduce_sum(
        tf.reduce_sum(
            observation_values * tf.expand_dims(factors, axis=2),
            axis=2,
            keepdims=True),
        axis=1,
        keepdims=True)
    gradients = osm.compute_gradient_attribution(
        last_logits, obs_values=observation_values, indicator=indicator)
    with self.test_session() as sess:
      acutal_gradients = sess.run(tf.squeeze(gradients))
      self.assertAllClose(factors, acutal_gradients, atol=0.01)

  def testAttention(self):
    seq_output = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                  [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]
    seq_mask = [[[1.0], [0.0], [0.0]], [[1.0], [1.0], [1.0]]]
    last_output = [[1.0, 2.0], [.5, .6]]

    results = osm.compute_attention(
        seq_output, last_output, hidden_layer_dim=0, seq_mask=seq_mask,
        sequence_length=[1, 3])
    expected_alpha = np.array(
        [[1.0 * 1 + 2 * 2, 0, 0],
         [0.1 * .5 + .2 * .6, .3 * .5 + .4 * .6, .5 * .5 + .6 * .6]])
    exp = np.exp(expected_alpha - np.max(expected_alpha, axis=1, keepdims=True))
    expected_beta = exp / np.sum(exp, axis=1, keepdims=True)
    expected_beta = np.expand_dims(expected_beta, 2)
    expected_attn = np.sum(np.array(seq_output) * expected_beta, axis=1)

    with self.test_session() as sess:
      actual_attention, acutal_beta = sess.run(results)
      self.assertAllClose(expected_beta, acutal_beta, atol=0.01)
      self.assertAllClose(expected_attn, actual_attention, atol=0.01)

  def testIntegratedGradientAttribution(self):
    # Due to complexity of the indicator we cannot easily extend this test to
    # > 1 lab test.
    obs_values = tf.constant([[[10000.0], [15000.0], [2.0]],
                              [[0.0], [100.0], [2000.0]]])

    # We compare these values to a linear interpolation between the second to
    # the last and the last value of the test.
    obs_values_base = tf.constant(
        [[[10000.0], [15000.0], [15000.0]], [[0.0], [100.0], [100.0]]])
    # For this test we need to select all attributions in order for consistency
    # to hold.
    indicator = tf.ones(shape=[2, 3, 1], dtype=tf.float32)
    delta_time = tf.constant(
        [[[1000], [999], [2]], [[1001], [500], [20]]], dtype=tf.float32)
    # Selected so that the attribution is only over the third time step in both
    # batch entries.
    attribution_max_delta_time = 100
    num_classes = 1

    diff_delta_time = tf.constant(
        [[[1000], [1], [997]], [[1001], [501], [480]]], dtype=tf.float32)
    # This is also important to not loose any time steps in the attribution.
    sequence_length = tf.constant([3, 3])

    # TODO(milah): Not clear why this test doesn't work for the RNN.
    def construct_logits_fn(
        unused_diff_delta_time, obs_values, unused_indicator,
        unused_sequence_length, unused_seq_mask, unused_hparams,
        reuse):
      result = tf.layers.dense(
          obs_values, num_classes, name='test1', reuse=reuse,
          activation=None) * (
              tf.expand_dims(obs_values[:, 0, :], axis=1) + 0.5)
      return result, None

    # First setup the weights of the RNN.
    logits, _ = construct_logits_fn(diff_delta_time, obs_values, indicator,
                                    sequence_length, None, None, False)
    # To verify the correctness of the attribution we compute the prediction at
    # the obs_values_base.
    base_logits, _ = construct_logits_fn(diff_delta_time, obs_values_base,
                                         indicator, sequence_length, None, None,
                                         True)

    # Set high for increased precision of the approximation.
    num_steps = 100
    hparams = contrib_training.HParams(
        sequence_prediction=True,
        use_rnn_attention=False,
        path_integrated_gradients_num_steps=num_steps,
        attribution_max_delta_time=attribution_max_delta_time)
    gradients = osm.compute_path_integrated_gradient_attribution(
        obs_values, indicator, diff_delta_time, delta_time, sequence_length,
        None, hparams, construct_logits_fn)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      actual_logits = sess.run(logits)
      actual_base_logits = sess.run(base_logits)
      actual_gradients = sess.run(gradients)
      self.assertAllClose(
          actual_logits - actual_base_logits, actual_gradients, atol=0.001)

  def testLastObservations(self):
    obs_values = tf.constant(
        [[[0, 100.0, 0, 0, 0], [0, 2.3, 0, 0, 0], [-1.0, 0, 0, 0, 0]],
         [[0, 0, 0, 0.5, 0], [0.0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0]]])
    indicator = tf.constant(
        [[[0, 1.0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
         [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])

    delta_time = tf.constant([
        [[1000], [1001], [2]],  # the last event is too new.
        [[10], [20], [22]]
    ])
    attribution_max_delta_time = 10

    expected_result = [[[0, 2.3, 0, 0, 0]], [[0, 4.0, 0, 0.5, 0]]]

    last_vals = osm._most_recent_obs_value(
        obs_values, indicator, delta_time, attribution_max_delta_time)
    with self.test_session() as sess:
      actual_last_vals = sess.run(last_vals)
      self.assertAllClose(expected_result, actual_last_vals, atol=0.01)

  def testGradientPredictions(self):
    logits = [[[0.0], [100.0], [200.0]], [[1000.0], [100.0], [5.0]]]

    delta_time = tf.constant([
        [[1000000], [1000001], [20]],  # first two events are too old.
        [[10], [20], [20]]
    ])

    seq_mask = tf.constant([
        [[1.0], [1.0], [1.0]],
        [[1.0], [1.0], [0.0]]  # last event is padded
    ])

    predictions = osm._predictions_for_gradients(
        logits, seq_mask, delta_time, attribution_max_delta_time=100,
        averaged=False)

    avg_predictions = osm._predictions_for_gradients(
        logits, seq_mask, delta_time, attribution_max_delta_time=100,
        averaged=True)
    expected_predictions = [[[200.0]], [[1100.0]]]
    avg_expected_predictions = [[[200.0]], [[550.0]]]

    with self.test_session() as sess:
      actual_pred, = sess.run([predictions])
      self.assertAllClose(expected_predictions, actual_pred)

      avg_actual_pred, = sess.run([avg_predictions])
      self.assertAllClose(avg_expected_predictions, avg_actual_pred)

  def testAttribution(self):
    """Low-level test for the correctness of compute_attribution."""
    logits = [[[0.0], [100.0], [200.0]], [[-1000.0], [100.0], [5.0]]]

    delta_time = tf.constant([
        [[1000000], [1000001], [20]],  # first two events are too old.
        [[10], [9], [8]]
    ])

    sequence_feature_map = {
        'obs_vals': self.observation_values,
        'deltaTime': delta_time
    }

    seq_mask = tf.constant([
        [[1.0], [1.0], [0.0]],  # last event is padded
        [[1.0], [1.0], [1.0]]
    ])

    attribution_threshold = 0.01

    expected_ixs = [1, 0, 1, 0]

    def _sigmoid(x):
      return math.exp(x) / (1 + math.exp(x))

    expected_values = [
        _sigmoid(logits[expected_ixs[0]][expected_ixs[2]][expected_ixs[3]]) -
        _sigmoid(logits[expected_ixs[0]][expected_ixs[2] - 1][expected_ixs[3]])
    ]

    attribution = osm.compute_prediction_diff_attribution(logits)
    attribution_dict = osm.convert_attribution(
        attribution, sequence_feature_map, seq_mask, delta_time,
        attribution_threshold, 12 * 60 * 60)
    self.assertEqual(
        set(sequence_feature_map.keys()), set(attribution_dict.keys()))
    with self.test_session() as sess:
      actual_attr_val, actual_attr_time = sess.run(
          [attribution_dict['obs_vals'], attribution_dict['deltaTime']])
      self.assertAllClose([expected_ixs], actual_attr_val.indices)
      self.assertAllClose(expected_values, actual_attr_val.values)
      self.assertAllClose([expected_ixs], actual_attr_time.indices)
      self.assertAllClose(expected_values, actual_attr_time.values)

  def testCombine(self):
    """Low-level test for the results of combine_observation_code_and_values."""

    observation_code_ids = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]],
        values=tf.constant([1, 1, 3, 0, 1], dtype=tf.int64),
        dense_shape=[2, 3, 1])

    vocab_size = 5

    expected_result = [[[0, 100.0, 0, 0, 0], [0, 2.3, 0, 0, 0],
                        [0, 0, 0, 0, 0]], [[0, 0, 0, 0.5, 0], [0.0, 0, 0, 0, 0],
                                           [0, 4.0, 0, 0, 0]]]
    expected_indicator = [[[0, 1.0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                          [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]]

    acutal_result, acutal_indicator = osm.combine_observation_code_and_values(
        observation_code_ids=observation_code_ids,
        observation_values=self.observation_values,
        vocab_size=vocab_size,
        mode=tf.estimator.ModeKeys.TRAIN,
        normalize=False,
        momentum=0.9,
        min_value=-10000000,
        max_value=10000000)
    with self.test_session() as sess:
      a_result, a_indicator = sess.run([acutal_result, acutal_indicator])
      self.assertAllClose(expected_result, a_result, atol=0.01)
      self.assertAllClose(expected_indicator, a_indicator, atol=0.01)

  def testRnnInput(self):
    observation_values = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0],
                 [1, 0, 0], [1, 1, 0], [1, 2, 0]],
        values=[100.0, 2.3, 9999999.0, 0.5, 0.0, 4.0],
        dense_shape=[2, 3, 1])
    observation_code_ids = tf.SparseTensor(
        indices=observation_values.indices,
        values=['loinc:2', 'loinc:1', 'loinc:2',
                'loinc:1', 'MISSING', 'loinc:1'],
        dense_shape=observation_values.dense_shape)
    delta_time, obs_values, indicator = osm.construct_input(
        {
            'Observation.code':
                observation_code_ids,
            'Observation.valueQuantity.value':
                observation_values,
            'deltaTime':
                tf.constant([[[2 * 60 * 60], [3 * 60 * 60], [0]],
                             [[1 * 60 * 60], [3 * 60 * 60], [6 * 60 * 60]]],
                            dtype=tf.int64)
        }, ['loinc:1', 'loinc:2', 'MISSING'],
        'Observation.code',
        'Observation.valueQuantity.value',
        mode=tf.estimator.ModeKeys.TRAIN,
        normalize=False,
        momentum=0.9,
        min_value=-10000000,
        max_value=10000000,
        input_keep_prob=1.0)

    result = tf.concat([delta_time, indicator, obs_values], axis=2)

    expected_result = [
        [[0, 0, 1, 0, 0, 100, 0],
         [-1, 1, 0, 0, 2.3, 0, 0],
         # value 9999999.0 was filtered.
         [3, 0, 0, 0, 0, 0, 0]
        ],
        [[0, 1, 0, 0, 0.5, 0, 0],
         [-2, 0, 0, 1, 0, 0, 0],
         [-3, 1, 0, 0, 4.0, 0, 0]]
    ]

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      actual_result = sess.run(result)
      print(actual_result)
      self.assertAllClose(expected_result, actual_result, atol=0.01)

  def testEmptyRnnInput(self):
    observation_values = tf.SparseTensor(
        indices=tf.reshape(tf.constant([], dtype=tf.int64), shape=[0, 3]),
        values=tf.constant([], dtype=tf.float32),
        dense_shape=[2, 0, 1])
    observation_code_ids = tf.SparseTensor(
        indices=observation_values.indices,
        values=tf.constant([], dtype=tf.string),
        dense_shape=observation_values.dense_shape)
    delta_time, obs_values, indicator = osm.construct_input(
        {
            'Observation.code':
                observation_code_ids,
            'Observation.valueQuantity.value':
                observation_values,
            'deltaTime':
                tf.reshape(tf.constant([[], []], dtype=tf.int64), [2, 0, 1])
        }, ['loinc:1', 'loinc:2', 'MISSING'],
        'Observation.code',
        'Observation.valueQuantity.value',
        mode=tf.estimator.ModeKeys.TRAIN,
        normalize=False,
        momentum=0.9,
        min_value=-10000000,
        max_value=10000000,
        input_keep_prob=1.0)

    result = tf.concat([delta_time, indicator, obs_values], axis=2)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      actual_result, = sess.run([tf.shape(result)])
      self.assertAllClose([2, 0, 7], actual_result)

  @parameterized.parameters(
      (True, True, False, False, False, False, False, 0, 0.0),
      (False, False, True, False, False, False, False, 0, 0.0),
      (True, False, False, True, False, False, False, 0, 0.0),
      (False, False, False, False, True, False, False, 0, 0.0),
      (False, False, False, False, False, True, False, 0, 0.0),
      (True, True, True, True, True, True, False, 0, 0.0),
      (True, True, True, True, True, True, False, 0, 1.0),
      (False, False, False, False, False, False, True, 0, 0.0),
      (False, True, False, False, True, False, True, 5, 0.0),
  )
  def testBasicModelFn(self, sequence_prediction, include_gradients,
                       include_gradients_sum_time, include_gradients_avg_time,
                       include_path_integrated_gradients,
                       include_diff_sequence_prediction, use_rnn_attention,
                       attention_hidden_layer_dim, volatility_loss_factor):
    """This high-level tests ensures there are no errors during training.

    It also checks that the loss is decreasing.

    Args:
      sequence_prediction: Whether to consider the recent predictions in the
        loss or only the most last prediction.
      include_gradients: Whether to generate attribution with the
        gradients of the last predictions.
      include_gradients_sum_time: Whether to generate attribution
        with the gradients of the sum of the predictions over time.
      include_gradients_avg_time: Whether to generate attribution
        with the gradients of the average of the predictions over time.
      include_path_integrated_gradients: Whether to generate
        attribution with the integrated gradients of last predictions compared
        to their most recent values before attribution_max_delta_time.
      include_diff_sequence_prediction: Whether to
        generate attribution from the difference of consecutive predictions.
      use_rnn_attention: Whether to use attention for the RNN.
      attention_hidden_layer_dim: If use_rnn_attention what the dimensionality
        of a hidden layer should be (or 0 if none) of last output and
        intermediates before multiplying to obtain a weight.
      volatility_loss_factor: Include the sum of the changes in predictions
        across the sequence in the loss multiplied by this factor.
    """
    num_steps = 2
    hparams = contrib_training.HParams(
        batch_size=2,
        learning_rate=0.008,
        sequence_features=[
            'deltaTime', 'Observation.code', 'Observation.valueQuantity.value'
        ],
        categorical_values=['loinc:1', 'loinc:2', 'MISSING'],
        categorical_seq_feature='Observation.code',
        context_features=['sequenceLength'],
        feature_value='Observation.valueQuantity.value',
        label_key='label.in_hospital_death',
        attribution_threshold=-1.0,
        rnn_size=6,
        variational_recurrent_keep_prob=1.1,
        variational_input_keep_prob=1.1,
        variational_output_keep_prob=1.1,
        sequence_prediction=sequence_prediction,
        time_decayed=False,
        normalize=True,
        momentum=0.9,
        min_value=-1000.0,
        max_value=1000.0,
        volatility_loss_factor=volatility_loss_factor,
        attribution_max_delta_time=100000,
        input_keep_prob=1.0,
        include_sequence_prediction=sequence_prediction,
        include_gradients_attribution=include_gradients,
        include_gradients_sum_time_attribution=include_gradients_sum_time,
        include_gradients_avg_time_attribution=include_gradients_avg_time,
        include_path_integrated_gradients_attribution=(
            include_path_integrated_gradients),
        include_diff_sequence_prediction_attribution=(
            include_diff_sequence_prediction),
        use_rnn_attention=use_rnn_attention,
        attention_hidden_layer_dim=attention_hidden_layer_dim,
        path_integrated_gradients_num_steps=10,
    )
    observation_values = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0],
                 [1, 0, 0], [1, 1, 0], [1, 2, 0]],
        values=[100.0, 2.3, 9999999.0, 0.5, 0.0, 4.0],
        dense_shape=[2, 3, 1])
    model = osm.ObservationSequenceModel()
    model_fn = model.create_model_fn(hparams)
    features = {
        input_fn.CONTEXT_KEY_PREFIX + 'sequenceLength':
            tf.constant([[2], [3]], dtype=tf.int64),
        input_fn.SEQUENCE_KEY_PREFIX + 'Observation.code':
            tf.SparseTensor(
                indices=observation_values.indices,
                values=[
                    'loinc:2', 'loinc:1', 'loinc:2', 'loinc:1', 'MISSING',
                    'loinc:1'
                ],
                dense_shape=observation_values.dense_shape),
        input_fn.SEQUENCE_KEY_PREFIX + 'Observation.valueQuantity.value':
            observation_values,
        input_fn.SEQUENCE_KEY_PREFIX + 'deltaTime':
            tf.constant([[[1], [2], [0]], [[1], [3], [4]]], dtype=tf.int64)
    }
    label_key = 'label.in_hospital_death'
    labels = {label_key: tf.constant([[1.0], [0.0]], dtype=tf.float32)}
    with tf.variable_scope('test'):
      model_fn_ops_train = model_fn(features, labels,
                                    tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope('test', reuse=True):
      features[input_fn.CONTEXT_KEY_PREFIX + 'label.in_hospital_death'
              ] = tf.SparseTensor(indices=[[0, 0]], values=['expired'],
                                  dense_shape=[2, 1])
      model_fn_ops_eval = model_fn(
          features, labels=None, mode=tf.estimator.ModeKeys.PREDICT)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # Test train.
      for i in range(num_steps):
        loss, _ = sess.run(
            [model_fn_ops_train.loss, model_fn_ops_train.train_op])
        if i == 0:
          initial_loss = loss
      self.assertLess(loss, initial_loss)
      # Test infer.
      sess.run(model_fn_ops_eval.predictions)

if __name__ == '__main__':
  absltest.main()
