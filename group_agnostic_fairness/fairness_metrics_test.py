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

# Lint as: python3
"""Tests for fairness_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from group_agnostic_fairness import adversarial_reweighting_model
from group_agnostic_fairness.data_utils.uci_adult_input import UCIAdultInput
from group_agnostic_fairness.fairness_metrics import RobustFairnessMetrics


class FairnessMetricsTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(FairnessMetricsTest, self).setUp()
    self.num_thresholds = 5
    self.label_column_name = 'income'
    self.protected_groups = ['sex', 'race']
    self.subgroups = [0, 1, 2, 3]
    self.model_dir = tempfile.mkdtemp()
    self.print_dir = tempfile.mkdtemp()
    self.primary_hidden_units = [16, 4]
    self.batch_size = 8
    self.train_steps = 10
    self.test_steps = 5
    self.pretrain_steps = 5
    self.dataset_base_dir = os.path.join(os.path.dirname(__file__), 'data/toy_data')  # pylint: disable=line-too-long
    self.train_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/train.csv')]  # pylint: disable=line-too-long
    self.test_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/test.csv')]  # pylint: disable=line-too-long
    self.load_dataset = UCIAdultInput(
        dataset_base_dir=self.dataset_base_dir,
        train_file=self.train_file,
        test_file=self.test_file)
    self.fairness_metrics = RobustFairnessMetrics(
        label_column_name=self.label_column_name,
        protected_groups=self.protected_groups,
        subgroups=self.subgroups)
    self.eval_metric_keys = [
        'accuracy', 'recall', 'precision', 'tp', 'tn', 'fp', 'fn', 'fpr', 'fnr'
    ]

  def _get_train_test_input_fn(self):
    train_input_fn = self.load_dataset.get_input_fn(
        mode=tf_estimator.ModeKeys.TRAIN, batch_size=self.batch_size)
    test_input_fn = self.load_dataset.get_input_fn(
        mode=tf_estimator.ModeKeys.EVAL, batch_size=self.batch_size)
    return train_input_fn, test_input_fn

  def _get_estimator(self):
    config = tf_estimator.RunConfig(model_dir=self.model_dir,
                                    save_checkpoints_steps=1)
    feature_columns, _, _, label_column_name = (
        self.load_dataset.get_feature_columns(include_sensitive_columns=True))
    estimator = adversarial_reweighting_model.get_estimator(
        feature_columns=feature_columns,
        label_column_name=label_column_name,
        config=config,
        model_dir=self.model_dir,
        primary_hidden_units=self.primary_hidden_units,
        batch_size=self.batch_size,
        pretrain_steps=self.pretrain_steps,
        primary_learning_rate=0.01,
        adversary_learning_rate=0.01,
        optimizer='Adagrad',
        activation=tf.nn.relu,
        adversary_loss_type='ce_loss',
        adversary_include_label=True,
        upweight_positive_instance_only=False)
    return estimator

  def test_create_and_add_fairness_metrics(self):
    # Instantiates a robust estimator
    estimator = self._get_estimator()
    self.assertIsInstance(estimator, tf_estimator.Estimator)

    # Adds additional fairness metrics to estimator
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn(
        num_thresholds=self.num_thresholds)
    estimator = tf_estimator.add_metrics(estimator, eval_metrics_fn)

    # Trains and evaluated robust model
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)

    # Checks if eval_results are computed
    self.assertNotEmpty(eval_results)

    for key in self.eval_metric_keys:
      self.assertIn(key, eval_results)

  def test_create_and_add_fairness_metrics_with_print_dir(self):
    # Instantiates a robust estimator
    estimator = self._get_estimator()
    self.assertIsInstance(estimator, tf_estimator.Estimator)

    # Adds additional fairness metrics to estimator
    self.fairness_metrics_with_print = RobustFairnessMetrics(
        label_column_name=self.label_column_name,
        protected_groups=self.protected_groups,
        subgroups=self.subgroups,
        print_dir=self.print_dir)
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn(
        num_thresholds=self.num_thresholds)
    estimator = tf_estimator.add_metrics(estimator, eval_metrics_fn)

    # Trains and evaluated robust model
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)

    # Checks if eval_results are computed
    self.assertNotEmpty(eval_results)
    for key in self.eval_metric_keys:
      self.assertIn(key, eval_results)

  def test_subgroup_metrics(self):

    # Instantiates a robust estimator
    estimator = self._get_estimator()
    self.assertIsInstance(estimator, tf_estimator.Estimator)

    # Adds additional fairness metrics to estimator
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn(
        num_thresholds=self.num_thresholds)
    estimator = tf_estimator.add_metrics(estimator, eval_metrics_fn)

    # Trains and evaluated robust model
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)

    # Checks if eval_results are computed
    self.assertNotEmpty(eval_results)

    # # Checks if auc metric is computed for all subgroups
    for subgroup in self.subgroups:
      self.assertIn('auc subgroup {}'.format(subgroup), eval_results)
      self.assertIn('fpr subgroup {}'.format(subgroup), eval_results)
      self.assertIn('fnr subgroup {}'.format(subgroup), eval_results)

  def test_protected_group_metrics(self):

    # Instantiates a robust estimator
    estimator = self._get_estimator()
    self.assertIsInstance(estimator, tf_estimator.Estimator)

    # Adds additional fairness metrics to estimator
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn(
        num_thresholds=self.num_thresholds)
    estimator = tf_estimator.add_metrics(estimator, eval_metrics_fn)

    # Trains and evaluated robust model
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)

    # Checks if eval_results are computed
    self.assertNotEmpty(eval_results)

    # # Checks if auc metric is computed for all protected_groups
    for group in self.protected_groups:
      self.assertIn('auc {} group 0'.format(group), eval_results)
      self.assertIn('auc {} group 1'.format(group), eval_results)

  def test_threshold_metrics(self):

    # Instantiates a robust estimator
    estimator = self._get_estimator()
    self.assertIsInstance(estimator, tf_estimator.Estimator)

    # Adds additional fairness metrics to estimator
    eval_metrics_fn = self.fairness_metrics.create_fairness_metrics_fn(
        num_thresholds=self.num_thresholds)
    estimator = tf_estimator.add_metrics(estimator, eval_metrics_fn)

    # Trains and evaluated robust model
    train_input_fn, test_input_fn = self._get_train_test_input_fn()
    estimator.train(input_fn=train_input_fn, steps=self.train_steps)
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=self.test_steps)

    # # Checks if tp,tn,fp,fn metrics are computed at thresholds
    self.assertIn('fp_th', eval_results)
    self.assertIn('fn_th', eval_results)
    self.assertIn('tp_th', eval_results)
    self.assertIn('tn_th', eval_results)

    # # Checks if the len of tp_th matches self.num_thresholds
    self.assertLen(eval_results['tp_th'], self.num_thresholds)

    # # Checks if threshold metrics are computed for protected_groups
    self.assertIn('fp_th subgroup {}'.format(self.subgroups[0]), eval_results)
    self.assertIn('fp_th {} group 0'.format(self.protected_groups[0]),
                  eval_results)
    self.assertIn('fp_th {} group 1'.format(self.protected_groups[0]),
                  eval_results)

if __name__ == '__main__':
  tf.test.main()
