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

"""Tests for Transform library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import numpy as np
import pandas as pd
import pandas.util.testing as pandas_testing
from six.moves import range
import tensorflow.compat.v1 as tf

from correct_batch_effects_wdn import metadata
from correct_batch_effects_wdn import transform

_ACTIVITY = "ACTIVE"
_PLATE = "plate1"
_SITE = 0
_TIMEPOINT = "0"
_SEQUENCE = "AGCT"
_CELL_DENSITY = "0"
_PASSAGE = "0"
_CELL_LINE_ID = ""


class TransformTest(tf.test.TestCase):

  def setUp(self):
    super(TransformTest, self).setUp()
    wells_384, rows_384, cols_384 = [], [], []
    for row in string.ascii_uppercase[:16]:
      for col in range(24):
        wells_384.append("%s%02d" % (row, col))
        rows_384.append("%s" % row)
        cols_384.append("%02d" % col)

    n_per_batch = 100
    n_each_control = 3 * n_per_batch
    n_other = 3 * n_per_batch
    np.random.seed(123)
    self.columns = [0, 1]

    neg_control_batches = []
    for i in range(0, n_each_control, n_per_batch):
      batch = "week%d" % (i % n_per_batch)
      control_tuples = []
      for j in range(n_per_batch):
        control_tuples.append(
            ("NEGATIVE_CONTROL", "DMSO", "DMSO", 1.0, _ACTIVITY, batch, _PLATE,
             wells_384[j], rows_384[j], cols_384[j], _SITE, _TIMEPOINT,
             _SEQUENCE, _CELL_DENSITY, _PASSAGE, _CELL_LINE_ID))
      neg_control_batches.append(
          pd.DataFrame(
              np.random.multivariate_normal(
                  mean=np.array([2.0 + i, 4.0 + i]),
                  cov=np.array([[3.0 + i, 1.0 + i], [1.0 + i, 2.0 + i]]),
                  size=n_per_batch),
              columns=self.columns,
              index=pd.MultiIndex.from_tuples(
                  control_tuples, names=metadata.METADATA_ORDER)))
    self.neg_controls = pd.concat(neg_control_batches)

    pos_control_batches = []
    for i in range(0, n_each_control, n_per_batch):
      batch = "week%d" % (i % n_per_batch)
      control_tuples = []
      for j in range(n_per_batch):
        control_tuples.append(
            ("POSITIVE_CONTROL", "Taxol", "Taxol", 1.0, _ACTIVITY, batch,
             _PLATE, wells_384[j], rows_384[j], cols_384[j], _SITE, _TIMEPOINT,
             _SEQUENCE, _CELL_DENSITY, _PASSAGE, _CELL_LINE_ID))
      pos_control_batches.append(
          pd.DataFrame(
              np.random.multivariate_normal(
                  mean=np.array([5.0 + i, 7.0 + i]),
                  cov=np.array([[6.0 + i, 4.0 + i], [4.0 + i, 5.0 + i]]),
                  size=n_per_batch),
              columns=self.columns,
              index=pd.MultiIndex.from_tuples(
                  control_tuples, names=metadata.METADATA_ORDER)))
    self.pos_controls = pd.concat(pos_control_batches)
    self.controls = pd.concat([self.neg_controls, self.pos_controls])

    experimental_batches = []
    for i in range(0, n_other, n_per_batch):
      batch = "week%d" % (i % n_per_batch)
      experimental_tuples = []
      for j in range(n_per_batch):
        experimental_tuples.append(
            ("EXPERIMENTAL", "other", "2", 1.0, _ACTIVITY, batch, _PLATE,
             wells_384[j], rows_384[j], cols_384[j], _SITE, _TIMEPOINT,
             _SEQUENCE, _CELL_DENSITY, _PASSAGE, _CELL_LINE_ID))
      experimental_batches.append(
          pd.DataFrame(
              np.random.multivariate_normal(
                  mean=np.array([1.0 + i, 2.0 + i]),
                  cov=np.array([[3.0 + i, 1.0 + i], [1.0 + i, 2.0 + i]]),
                  size=n_per_batch),
              columns=self.columns,
              index=pd.MultiIndex.from_tuples(
                  experimental_tuples, names=metadata.METADATA_ORDER)))
    self.experimental = pd.concat(experimental_batches)
    self.data = pd.concat([self.controls, self.experimental])

  def testGetNegativeControls(self):
    pandas_testing.assert_frame_equal(self.neg_controls,
                                      transform.get_negative_controls(
                                          self.data))

  def testEigSymmetric(self):
    q_expected = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
                           [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
    # q should be orthonormal - make sure it really is
    pandas_testing.assert_almost_equal(
        q_expected.T.dot(q_expected), np.identity(2))
    lambda_expected = np.diag([3.0, 2.0])
    a = q_expected.dot(lambda_expected).dot(q_expected.T)
    lambda_computed, q_computed = transform.eig_symmetric(a)
    pandas_testing.assert_almost_equal(
        np.diag(lambda_expected), lambda_computed)
    # make sure q_computed is orthonormal
    pandas_testing.assert_almost_equal(
        np.identity(2), q_expected.T.dot(q_expected))
    for i in range(q_expected.shape[0]):
      ev_expected = q_expected[:, i]
      ev_computed = q_computed[:, i]
      # In this example, the eigenvalues are discrete, so the eigenvectors are
      # unique up to sign.  Since the sign will depend on the particulars of
      # the algorithm used to generate the eigenvectors, just make sure that
      # the dot product with the expected eigenvectors is +/- 1
      pandas_testing.assert_almost_equal(1.0,
                                         np.abs(ev_expected.dot(ev_computed)))

  def testFactorAnalysisRun(self):
    transform.factor_analysis(self.data, 0.1, -1)

  def testGetBootstrapSampleRun(self):
    bootstrap_data = transform.get_bootstrap_sample(self.data)
    self.assertTupleEqual(self.data.shape, bootstrap_data.shape)

  def testTransformDf(self):
    df_small = pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]]))
    rotate_mat_np = np.array([[3.0, 4.0], [5.0, 6.0]])
    shift_vec_np = np.array([[-1.0], [-2.0]])
    expected = pd.DataFrame(np.array([[10.0, 15.0], [24.0, 37.0]]))
    df_trans = transform.transform_df(
        df_small, rotate_mat_np, shift_vec_np)
    pandas_testing.assert_frame_equal(df_trans, expected)

  def testSumOfSquare(self):
    a = tf.constant(np.array([1.0, 2.0]))
    expected = 5.0
    with self.session() as sess:
      a_sum_of_square = sess.run(transform.sum_of_square(a))
    self.assertEqual(a_sum_of_square, expected)

  def testDropUnevaluatedComp(self):
    pandas_testing.assert_frame_equal(
        pd.concat([self.pos_controls, self.experimental]),
        transform.drop_unevaluated_comp(self.data))


if __name__ == "__main__":
  tf.test.main()
