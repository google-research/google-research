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

"""Reinforcement Learning based Locally Interpretabel Modeling (RL-LIM) tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
from sklearn import linear_model
import tensorflow.compat.v1 as tf

from rllim import rllim


class RllimTest(tf.test.TestCase):
  """RL-LIM test class."""

  def setUp(self):
    """Sets parameters and datasets."""

    super(RllimTest, self).setUp()
    self.temp_dir = tempfile.mkdtemp()

    # Network parameters
    self.parameters = dict()
    self.parameters['hidden_dim'] = 5
    self.parameters['iterations'] = 10
    self.parameters['num_layers'] = 3
    self.parameters['batch_size'] = 10
    self.parameters['batch_size_inner'] = 2
    self.parameters['lambda'] = 1.0

    # Train / Valid / Test set
    self.x_train = np.random.rand(100, 10)
    self.y_train_hat = np.random.rand(100, 1)
    self.x_probe = np.random.rand(40, 10)
    self.y_probe_hat = np.random.rand(40, 1)
    self.x_test = np.random.rand(200, 10)

    # Others
    self.checkpoint_file_name = self.temp_dir+'/model1.ckpt'
    self.interp_model = linear_model.Ridge(alpha=1)
    self.baseline_model = linear_model.Ridge(alpha=1)
    self.baseline_model.fit(self.x_train, self.y_train_hat)

  def tearDown(self):
    super(RllimTest, self).tearDown()
    shutil.rmtree(self.temp_dir)

  def testRllimLocalExplanation(self):
    """Tests local explanation of RL-LIM."""

    tf.reset_default_graph()

    rllim_class = rllim.Rllim(
        x_train=self.x_train, y_train=self.y_train_hat,
        x_probe=self.x_probe, y_probe=self.y_probe_hat,
        parameters=self.parameters,
        interp_model=self.interp_model,
        baseline_model=self.baseline_model,
        checkpoint_file_name=self.checkpoint_file_name)

    rllim_class.rllim_train()

    _, test_coef = \
        rllim_class.rllim_interpreter(
            x_train=self.x_train, y_train=self.y_train_hat,
            x_test=self.x_test, interp_model=self.interp_model)

    self.assertAllEqual([200, 11], test_coef.shape)

  def testRllimPrediction(self):
    """Tests local predictions of RL-LIM."""

    tf.reset_default_graph()

    rllim_class = rllim.Rllim(
        x_train=self.x_train, y_train=self.y_train_hat,
        x_probe=self.x_probe, y_probe=self.y_probe_hat,
        parameters=self.parameters,
        interp_model=self.interp_model,
        baseline_model=self.baseline_model,
        checkpoint_file_name=self.checkpoint_file_name)

    rllim_class.rllim_train()

    test_y_fit, _ = \
        rllim_class.rllim_interpreter(
            x_train=self.x_train, y_train=self.y_train_hat,
            x_test=self.x_test, interp_model=self.interp_model)

    self.assertAllEqual([200,], test_y_fit.shape)

if __name__ == '__main__':
  tf.test.main()
