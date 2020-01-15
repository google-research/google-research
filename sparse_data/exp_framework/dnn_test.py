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

"""Tests for dnn.py in the exp_framework module."""

from absl.testing import absltest
import numpy as np
from scipy import sparse
from sparse_data.data import sim
from sparse_data.exp_framework import dnn


class TestModel(absltest.TestCase):

  def setUp(self):
    super(TestModel, self).setUp()
    self.submodule = None

  def test_pipeline(self):
    if self.submodule is None:
      return

    d = sim.LinearSimulation(
        num_sample=1000, problem='classification', num_feature=5)
    d.reset()
    x_train, y_train, x_test, y_test = d.get()

    _, metrics = self.submodule.pipeline(
        x_train, y_train, x_test, y_test, problem='classification')

    self.assertGreater(metrics['test_acc'], 0.5)
    self.assertGreater(metrics['train_acc'], 0.5)
    self.assertGreater(metrics['micro_auc'], 0.5)
    self.assertGreater(metrics['macro_auc'], 0.5)

    d = sim.LinearSimulation(
        num_sample=1000, problem='regression', num_feature=5)
    d.reset()
    x_train, y_train, x_test, y_test = d.get()

    _, metrics = self.submodule.pipeline(
        x_train, y_train, x_test, y_test, problem='regression')

    self.assertLess(metrics['test_mse'], 1e2)
    self.assertLess(metrics['train_mse'], 1e2)


class TestDNN(TestModel):

  def setUp(self):
    super(TestDNN, self).setUp()
    self.submodule = dnn


class TestFunctionalKerasClassifier(absltest.TestCase):
  """Tests the scikit-learn wrapper for a functional Keras model."""

  def test_predict(self):
    """Tests the predict() function that this class adds to its parent."""
    d = sim.LinearSimulation(num_sample=500, problem='classification')
    d.reset()
    x_train, y_train, x_test, _ = d.get()

    num_class = len(set(y_train))
    num_feature = x_train.shape[1]
    is_sparse = sparse.issparse(x_train)

    clf = dnn.FunctionalKerasClassifier(
        build_fn=dnn.keras_build_fn,
        num_feature=num_feature,
        num_output=num_class,
        is_sparse=is_sparse,
        verbose=False)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    # check shape
    self.assertEqual(y_pred.shape, (np.size(x_test, 0),))
    # check predicted values (should be integer labels)
    self.assertTrue(np.all(np.isclose(y_pred, y_pred.astype(int), 0.0001)))
    self.assertTrue(np.array_equal(y_pred, np.argmax(y_proba, axis=1)))


if __name__ == '__main__':
  absltest.main()
