# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Data Valuation using Reinforcement Learning (DVRL) tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
from sklearn import linear_model
import tensorflow as tf

from dvrl import dvrl


class DvrlTest(tf.test.TestCase):
  """DVRL test class."""

  def setUp(self):
    """Sets parameters and datasets."""

    super(DvrlTest, self).setUp()
    self.temp_dir = tempfile.mkdtemp()

    # Network parameters
    self.parameters = dict()
    self.parameters['hidden_dim'] = 10
    self.parameters['comb_dim'] = 10
    self.parameters['iterations'] = 100
    self.parameters['activation'] = tf.nn.relu
    self.parameters['layer_number'] = 3
    self.parameters['batch_size'] = 100
    self.parameters['learning_rate'] = 0.01

    # Train / Valid / Test set
    self.x_train = np.random.rand(1000, 10)
    self.y_train = np.random.randint(0, 2, 1000)
    self.x_valid = np.random.rand(400, 10)
    self.y_valid = np.random.randint(0, 2, 400)
    self.x_test = np.random.rand(2000, 10)
    self.y_teste = np.random.randint(0, 2, 2000)

    # Others
    self.problem = 'classification'
    self.checkpoint_file_name = self.temp_dir+'/model.ckpt'
    self.flags = {'sgd': False, 'pretrain': False}
    self.pred_model = linear_model.LogisticRegression(solver='lbfgs')

  def testDvrlDataValuation(self):
    """Tests data valuation of DVRL."""

    tf.reset_default_graph()

    dvrl_class = dvrl.Dvrl(
        x_train=self.x_train, y_train=self.y_train,
        x_valid=self.x_valid, y_valid=self.y_valid,
        problem=self.problem, pred_model=self.pred_model,
        parameters=self.parameters,
        checkpoint_file_name=self.checkpoint_file_name,
        flags=self.flags)

    dvrl_class.train_dvrl('auc')

    dve_out = dvrl_class.data_valuator(
        x_train=self.x_train, y_train=self.y_train)

    self.assertAllEqual([1000,], dve_out.shape)

  def testDvrlPrediction(self):
    """Tests predictions of DVRL."""

    tf.reset_default_graph()

    dvrl_class = dvrl.Dvrl(
        x_train=self.x_train, y_train=self.y_train,
        x_valid=self.x_valid, y_valid=self.y_valid,
        problem=self.problem, pred_model=self.pred_model,
        parameters=self.parameters,
        checkpoint_file_name=self.checkpoint_file_name,
        flags=self.flags)

    dvrl_class.train_dvrl('auc')

    y_test_hat = dvrl_class.dvrl_predictor(x_test=self.x_test)

    self.assertAllEqual([2000, 2], y_test_hat.shape)

if __name__ == '__main__':
  tf.test.main()
