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

# Lint as: python3
"""Tests for baseline_model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest
import tensorflow.compat.v1 as tf

from group_agnostic_fairness import baseline_model
from group_agnostic_fairness.data_utils.uci_adult_input import UCIAdultInput
from group_agnostic_fairness.fairness_metrics import RobustFairnessMetrics


class BaselineModelTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(BaselineModelTest, self).setUp()
    self.model_dir = tempfile.mkdtemp()
    self.hidden_units = [16, 4]
    self.batch_size = 8
    self.train_steps = 20
    self.test_steps = 5
    self.dataset_base_dir = os.path.join(os.path.dirname(__file__), 'data/toy_data')  # pylint: disable=line-too-long
    self.train_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/train.csv')]  # pylint: disable=line-too-long
    self.test_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/test.csv')]  # pylint: disable=line-too-long
    self.load_dataset = UCIAdultInput(
        dataset_base_dir=self.dataset_base_dir,
        train_file=self.train_file,
        test_file=self.test_file)
    self.label_column_name = 'income'
    self.protected_groups = ['sex', 'race']
    self.subgroups = [0, 1, 2, 3]
    self.fairness_metrics = RobustFairnessMetrics(
        label_column_name=self.label_column_name,
        protected_groups=self.protected_groups,
        subgroups=self.subgroups)

  def test_get_feature_columns_with_demographics(self):
    feature_columns, _, _, target_variable_column = self.load_dataset.get_feature_columns(include_sensitive_columns=True)  # pylint: disable=line-too-long
    self.assertLen(feature_columns, 14)
    self.assertEqual(target_variable_column, self.label_column_name)

  def test_get_feature_columns_without_demographics(self):
    feature_columns, _, _, target_variable_column = self.load_dataset.get_feature_columns(include_sensitive_columns=False)  # pylint: disable=line-too-long
    self.assertLen(feature_columns, 12)
    self.assertEqual(target_variable_column, self.label_column_name)

  def test_get_input_fn(self):
    input_fn = self.load_dataset.get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN, batch_size=self.batch_size)
    features, targets = input_fn()
    self.assertIn('sex', targets)
    self.assertIn('race', targets)
    self.assertIn('subgroup', targets)
    self.assertIn(self.label_column_name, targets)
    self.assertLen(features, 15)

  def _get_train_test_input_fn(self):
    train_input_fn = self.load_dataset.get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN, batch_size=self.batch_size)
    test_input_fn = self.load_dataset.get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL, batch_size=self.batch_size)
    return train_input_fn, test_input_fn

  def test_eval_results_baseline_model(self):
    config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=2)
    feature_columns, _, _, label_column_name = self.load_dataset.get_feature_columns(include_sensitive_columns=True)  # pylint: disable=line-too-long
    estimator = baseline_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        hidden_units=self.hidden_units,
        batch_size=self.batch_size)
    self.assertIsInstance(estimator, tf.estimator.Estimator)
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)
    self.assertNotEmpty(eval_results)
    # # Checks if all tp,tn,fp,fn keys are present in eval_results dictionary
    self.assertIn('auc', eval_results)
    self.assertIn('fp', eval_results)
    self.assertIn('fn', eval_results)
    self.assertIn('tp', eval_results)
    self.assertIn('tn', eval_results)

  def test_add_fairness_metrics_baseline_model(self):
    config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=2)
    feature_columns, _, _, label_column_name = self.load_dataset.get_feature_columns(include_sensitive_columns=True)  # pylint: disable=line-too-long
    estimator = baseline_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        hidden_units=self.hidden_units,
        batch_size=self.batch_size)
    self.assertIsInstance(estimator, tf.estimator.Estimator)

    # Adds additional fairness metrics to estimator
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn()
    estimator = tf.estimator.add_metrics(estimator, eval_metrics_fn)

    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)
    self.assertNotEmpty(eval_results)
    # # Checks if auc metric is computed for all subgroups
    for subgroup in self.subgroups:
      self.assertIn('auc subgroup {}'.format(subgroup), eval_results)

  def test_global_steps_baseline_model(self):
    config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=2)
    feature_columns, _, _, label_column_name = self.load_dataset.get_feature_columns(include_sensitive_columns=True)  # pylint: disable=line-too-long
    estimator = baseline_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        hidden_units=self.hidden_units,
        batch_size=self.batch_size)
    self.assertIsInstance(estimator, tf.estimator.Estimator)
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)
    # Checks if global step has reached specified number of train_steps
    self.assertIn('global_step', eval_results)
    self.assertEqual(eval_results['global_step'], self.train_steps)

  def test_create_baseline_estimator_with_demographics(self):
    config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=2)
    feature_columns, _, _, label_column_name = self.load_dataset.get_feature_columns(include_sensitive_columns=True)  # pylint: disable=line-too-long
    estimator = baseline_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        hidden_units=self.hidden_units,
        batch_size=self.batch_size,
        learning_rate=0.01,
        optimizer='Adagrad',
        activation=tf.nn.relu)
    self.assertIsInstance(estimator, tf.estimator.Estimator)

  def test_create_baseline_estimator_without_demographics(self):
    config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=2)
    feature_columns, _, _, label_column_name = self.load_dataset.get_feature_columns(include_sensitive_columns=False)  # pylint: disable=line-too-long
    estimator = baseline_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        hidden_units=self.hidden_units,
        batch_size=self.batch_size,
        learning_rate=0.01,
        optimizer='Adagrad',
        activation=tf.nn.relu)
    self.assertIsInstance(estimator, tf.estimator.Estimator)


if __name__ == '__main__':
  tf.test.main()
