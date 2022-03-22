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
"""Tests for input function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from absl.testing import absltest

import numpy as np

import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from tensorflow.compat.v1 import estimator as tf_estimator
from explaining_risk_increase import input_fn
from explaining_risk_increase import observation_sequence_model as osm
from tensorflow.contrib import training as contrib_training

TESTDATA_DIR = 'explaining_risk_increase/test_data/'


class InputProcessingTest(tf.test.TestCase):

  def test_intersect_indices_dense(self):
    delta_time = tf.constant([[1, 2, 3, 4, 5], [10, 20, 30, 40, 0],
                              [100, 200, 300, 0, 0]])
    obs_harm_code = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [1, 1, 2], [2, 1, 0]],
        values=['pulse', 'pulse', 'blood_pressure', 'temperature'],
        dense_shape=[3, 2, 3])
    expected_delta_time = [[[1], [2]], [[20], [0]], [[200], [0]]]
    new_delta_time = input_fn._intersect_indices(delta_time, obs_harm_code)
    with self.test_session() as sess:
      acutal_delta_time = sess.run(new_delta_time)
      self.assertAllClose(expected_delta_time, acutal_delta_time)

  def test_intersect_indices_sparse(self):
    obs_code = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0],
                 [1, 3, 0], [2, 0, 0], [2, 2, 0]],
        values=[
            'loinc:1', 'loinc:1', 'loinc:2', 'loinc:4', 'loinc:4', 'loinc:2',
            'loinc:1', 'loinc:4'
        ],
        dense_shape=[3, 4, 1])
    obs_harm_code = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0]],
        values=[
            'pulse', 'pulse', 'blood_pressure', 'temperature', 'temperature'
        ],
        dense_shape=[3, 2, 3])
    indices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [2, 1, 0]]
    values = [b'loinc:1', b'loinc:1', b'loinc:4', b'loinc:4']
    dense_shape = [3, 2, 1]
    new_obs_code = input_fn._intersect_indices(obs_code, obs_harm_code)
    with self.test_session() as sess:
      acutal_obs_code = sess.run(new_obs_code)
      self.assertAllEqual(values, acutal_obs_code.values)
      self.assertAllEqual(indices, acutal_obs_code.indices)
      self.assertAllEqual(dense_shape, acutal_obs_code.dense_shape)


class TestInputFn(tf.test.TestCase):

  def setUp(self):
    super(TestInputFn, self).setUp()
    directory = os.path.join(absltest.get_default_test_srcdir(), TESTDATA_DIR)
    seqex_list = [self.read_seqex_ascii(filename, directory)
                  for filename in ['example1.ascii', 'example2.ascii']]
    self.input_data_dir = self.create_input_tfrecord(
        seqex_list, tempfile.mkdtemp(), 'input')
    self.log_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(TestInputFn, self).tearDown()
    if self.log_dir:
      shutil.rmtree(self.log_dir)
    if self.input_data_dir:
      os.remove(self.input_data_dir)

  def read_seqex_ascii(self, filename, testdata_dir):
    """Read a tf.SequenceExample in ascii format from disk."""
    seqex_pb = tf.train.SequenceExample()
    example_file = os.path.join(testdata_dir, filename)
    with open(example_file, 'r') as f:
      text_format.Parse(f.read(), seqex_pb)
      return seqex_pb

  def create_input_tfrecord(self, seqex_list, tmp_dir, filename):
    """Create a temporary TFRecord file on disk with the tf.SequenceExamples.

    Args:
      seqex_list: A list of tf.SequenceExamples.
      tmp_dir: Path to the test tmp directory.
      filename: Temporary filename for TFRecord output.
    Returns:
      The path to an TFRecord table containing the provided examples.
    """
    path = os.path.join(tmp_dir, filename)
    with tf.python_io.TFRecordWriter(path) as writer:
      for seqex in seqex_list:
        writer.write(seqex.SerializeToString())
    return path

  def test_input_fn(self):
    feature_map, label = input_fn.get_input_fn(
        tf_estimator.ModeKeys.TRAIN,
        [self.input_data_dir],
        'label.in_hospital_death.class',
        sequence_features=[
            'Observation.code', 'Observation.value.quantity.value',
            'Observation.value.quantity.unit',
            'Observation.code.harmonized:valueset-observation-name'
        ],
        dense_sequence_feature='Observation.value.quantity.value',
        required_sequence_feature='Observation.code.harmonized:valueset-'
        'observation-name',
        batch_size=2,
        shuffle=False)()
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      feature_map.update(label)
      results = sess.run(feature_map)
      expected_dense_shape = [2, 3, 1]
      expected_indices = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0]]
      expected_code = [b'loinc:4', b'loinc:6', b'loinc:6', b'loinc:1']
      expected_harm_code = [b'pulse', b'temperature', b'temperature', b'pulse']
      expected_value = [2.0, 20.0, -2.0, 1.0]

      code = results['s-Observation.code']
      self.assertAllEqual(expected_code, code.values)
      self.assertAllEqual(expected_indices, code.indices)
      self.assertAllEqual(expected_dense_shape, code.dense_shape)

      harm_code = results[
          's-Observation.code.harmonized:valueset-observation-name']
      self.assertAllEqual(expected_harm_code, harm_code.values)
      self.assertAllEqual(expected_indices, harm_code.indices)
      self.assertAllEqual(expected_dense_shape, harm_code.dense_shape)

      value = results['s-Observation.value.quantity.value']
      self.assertAllClose(np.array(expected_value), value.values)
      self.assertAllEqual(expected_indices, value.indices)
      self.assertAllEqual(expected_dense_shape, value.dense_shape)

      unit = results['s-Observation.value.quantity.unit']
      self.assertAllEqual([b'F'], unit.values)
      self.assertAllEqual([[0, 2, 0]], unit.indices)
      self.assertAllEqual(expected_dense_shape, unit.dense_shape)

      self.assertAllClose(results['label.in_hospital_death.class'],
                          np.array([[1.], [0.]]))

      self.assertAllEqual([[3], [1]], results['c-sequenceLength'])

      self.assertAllEqual([[[12], [11], [10]], [[12], [0], [0]]],
                          results['s-deltaTime'])

  def test_model_integration(self):
    features, labels = input_fn.get_input_fn(
        tf_estimator.ModeKeys.TRAIN,
        [self.input_data_dir],
        'label.in_hospital_death.class',
        sequence_features=[
            'Observation.code', 'Observation.value.quantity.value',
            'Observation.value.quantity.unit',
            'Observation.code.harmonized:valueset-observation-name'
        ],
        dense_sequence_feature='Observation.value.quantity.value',
        required_sequence_feature='Observation.code.harmonized:valueset-observation-name',
        batch_size=2,
        shuffle=False)()
    num_steps = 2
    hparams = contrib_training.HParams(
        batch_size=2,
        learning_rate=0.008,
        sequence_features=[
            'deltaTime', 'Observation.code', 'Observation.value.quantity.value'
        ],
        categorical_values=['loinc:4', 'loinc:6', 'loinc:1'],
        categorical_seq_feature='Observation.code',
        context_features=['sequenceLength'],
        feature_value='Observation.value.quantity.value',
        label_key='label.in_hospital_death.class',
        attribution_threshold=-1.0,
        rnn_size=6,
        variational_recurrent_keep_prob=1.1,
        variational_input_keep_prob=1.1,
        variational_output_keep_prob=1.1,
        sequence_prediction=False,
        time_decayed=False,
        normalize=True,
        momentum=0.9,
        min_value=-1000.0,
        max_value=1000.0,
        volatility_loss_factor=0.0,
        attribution_max_delta_time=100000,
        input_keep_prob=1.0,
        include_sequence_prediction=False,
        include_gradients_attribution=True,
        include_gradients_sum_time_attribution=False,
        include_gradients_avg_time_attribution=False,
        include_path_integrated_gradients_attribution=True,
        include_diff_sequence_prediction_attribution=False,
        use_rnn_attention=True,
        attention_hidden_layer_dim=5,
        path_integrated_gradients_num_steps=10,
    )
    model = osm.ObservationSequenceModel()
    model_fn = model.create_model_fn(hparams)
    with tf.variable_scope('test'):
      model_fn_ops_train = model_fn(features, labels,
                                    tf_estimator.ModeKeys.TRAIN)
    with tf.variable_scope('test', reuse=True):
      model_fn_ops_eval = model_fn(
          features, labels=None, mode=tf_estimator.ModeKeys.PREDICT)

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
