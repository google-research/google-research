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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest
import tensorflow as tf
from extrapolation.classifier import classifier
from extrapolation.utils import dataset_utils


FLAGS = flags.FLAGS


def test_clf(test, clf, itr, n_classes):
  """Helper function for testing classifiers.

  Ensures that classifiers output appropriately shapes tensors and that
  weight regularization does something reasonable.

  Args:
    test (TestCase): a test case instance.
    clf (Classifier): a classifier.
    itr (Iterator): a data iterator.
    n_classes (int): number of classes in this problem.
  """
  batch_x, batch_y = itr.next()
  l_unreg, _, preds, _ = clf.get_loss(batch_x, batch_y, return_preds=True)
  test.assertEqual(preds.shape, [batch_x.shape[0], n_classes])
  test.assertTrue(tf.reduce_all(preds <= 1.))
  test.assertTrue(tf.reduce_all(preds >= 0.))
  l_reg = clf.get_loss_dampened(batch_x, batch_y, lam=1.)
  test.assertTrue(l_reg.shape[0], batch_x.shape[0])
  test.assertTrue(tf.reduce_all(l_reg >= l_unreg))


class RunClassifierMnistTest(absltest.TestCase):

  def test_CNN(self):

    x_shape = (64, 28, 28, 1)
    y_shape = (64, 1)
    conv_dims = [20, 10]
    conv_sizes = [5, 5]
    dense_sizes = [100]
    n_classes = 10
    clf = classifier.CNN(conv_dims, conv_sizes, dense_sizes, n_classes)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    test_clf(self, clf, itr, n_classes)

  def test_MLP(self):

    x_shape = (64, 28 * 28 * 1)
    y_shape = (64, 1)
    layer_dims = [20, 10]
    n_classes = 10
    clf = classifier.MLP(layer_dims, n_classes)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    test_clf(self, clf, itr, n_classes)

  def test_CNN_onehot(self):

    x_shape = (64, 28, 28, 1)
    y_shape = (64, 10)
    conv_dims = [20, 10]
    conv_sizes = [5, 5]
    dense_sizes = [100]
    n_classes = 10
    clf = classifier.CNN(conv_dims, conv_sizes, dense_sizes,
                         n_classes, onehot=True)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    test_clf(self, clf, itr, n_classes)

  def test_MLP_onehot(self):

    x_shape = (64, 28 * 28 * 1)
    y_shape = (64, 10)
    layer_dims = [20, 10]
    n_classes = 10
    clf = classifier.MLP(layer_dims, n_classes, onehot=True)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    test_clf(self, clf, itr, n_classes)


if __name__ == '__main__':
  absltest.main()
