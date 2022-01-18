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
# pylint: disable=line-too-long
"""Tests for creating and running all models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import absltest
import tensorflow.compat.v1 as tf

from group_agnostic_fairness import main_trainer

FLAGS = flags.FLAGS


def run_experiment(model_name, dataset):
  """Sets FLAGS and runs experiment."""
  FLAGS.model_name = model_name
  FLAGS.dataset_base_dir = os.path.join(os.path.dirname(__file__), 'data/toy_data')
  FLAGS.train_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/train.csv')]
  FLAGS.test_file = [os.path.join(os.path.dirname(__file__), 'data/toy_data/test.csv')]
  FLAGS.dataset = dataset
  FLAGS.primary_hidden_units = [4, 2]
  FLAGS.adversary_hidden_units = [2]
  FLAGS.batch_size = 2
  FLAGS.base_dir = tempfile.mkdtemp()
  FLAGS.total_train_steps = 8
  FLAGS.test_steps = 4
  main_trainer.run_model()


class RunModelTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(RunModelTest, self).setUp()
    self._model_name = 'baseline'
    self._dataset = 'uci_adult'

  # Test cases for whole training on various datasets
  def test_run_model_on_uci_adult_dataset(self):
    """Tests the whole model training can run end-to-end on uci_adult dataset."""
    self._dataset = 'uci_adult'
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  # Test cases for whole training on various settings of adversarial reweighting
  # model.
  def test_run_adversarial_reweighting_model_with_demographics(self):
    """Tests if adversarial_reweighting_model training can run end-to-end."""
    self._model_name = 'adversarial_reweighting'
    FLAGS.include_sensitive_columns = True
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  def test_run_adversarial_reweighting_model_without_demographics(self):
    """Tests if adversarial_reweighting_model training can run end-to-end."""
    self._model_name = 'adversarial_reweighting'
    FLAGS.include_sensitive_columns = False
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  def test_run_adversarial_reweighting_model_with_label(self):
    """Tests if adversarial_reweighting_model training can run end-to-end."""
    self._model_name = 'adversarial_reweighting'
    FLAGS.adversary_include_label = True
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  # Test case for whole training of other baseline robust-learning models
  def test_simple_baseline_model(self):
    """Tests if inverse_propensity_weighting model training can run end-to-end."""
    self._model_name = 'baseline'
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  def test_run_inverse_propensity_weighting_without_label_model(self):
    """Tests if inverse_propensity_weighting model training can run end-to-end."""
    self._model_name = 'inverse_propensity_weighting'
    FLAGS.reweighting_type = 'IPS_without_label'
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  def test_run_inverse_propensity_weighting_with_label_model(self):
    """Tests if inverse_propensity_weighting model training can run end-to-end."""
    self._model_name = 'inverse_propensity_weighting'
    FLAGS.reweighting_type = 'IPS_with_label'
    run_experiment(model_name=self._model_name, dataset=self._dataset)

  # Test cases that break various models due to incompatable flag settings
  def test_run_ips_model_on_not_implemented_weighing_scheme(self):
    """Tests if inverse_propensity_weighting model training can run end-to-end."""
    with self.assertRaises(ValueError):
      self._model_name = 'inverse_propensity_weighting'
      FLAGS.reweighting_type = 'not_implemented'
      run_experiment(model_name=self._model_name, dataset=self._dataset)

  def test_run_not_implemented_model(self):
    """Shoud raise ValueError as <dummy_name> model is not implemented."""
    with self.assertRaises(ValueError):
      run_experiment(model_name='dummy_name', dataset=self._dataset)


if __name__ == '__main__':
  tf.test.main()
