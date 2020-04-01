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
"""Tests for Evaluate library."""

import string

import numpy as np
import numpy.testing as numpy_testing
import pandas as pd
from six.moves import range
import tensorflow.compat.v1 as tf

from correct_batch_effects_wdn import evaluate
from correct_batch_effects_wdn import metadata


class EvaluateTest(tf.test.TestCase):

  def setUp(self):
    super(EvaluateTest, self).setUp()
    wells_384, rows_384, cols_384 = [], [], []
    for row in string.ascii_uppercase[:16]:
      for col in range(24):
        wells_384.append("%s%02d" % (row, col))
        rows_384.append("%s" % row)
        cols_384.append("%02d" % col)

    n_control = 100
    n_other = 200
    np.random.seed(123)
    self.columns = ["e1", "e2"]
    self.controls = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.array([2.0, 4.0]),
            cov=np.array([[3.0, 1.0], [1.0, 2.0]]),
            size=n_control),
        columns=self.columns)
    self.controls[metadata.TREATMENT_GROUP] = "NEGATIVE_CONTROL"
    self.controls[metadata.MOA] = "DMSO_MOA"
    self.controls[metadata.COMPOUND] = "DMSO"
    self.controls[metadata.CONCENTRATION] = 1.0
    self.controls[metadata.ACTIVITY] = "ACTIVE"
    self.controls[metadata.BATCH] = "week0"
    self.controls[metadata.PLATE] = "plate1"
    self.controls[metadata.WELL] = ["A1", "B2"] * int(n_control / 2)
    self.controls[metadata.ROW] = ["A", "B"] * int(n_control / 2)
    self.controls[metadata.COLUMN] = ["1", "2"] * int(n_control / 2)
    self.controls[metadata.SITE] = 0
    self.controls[metadata.UNIQUE_ID] = np.repeat(range(int(n_control / 2)), 2)
    self.controls[metadata.TIMEPOINT] = "0"
    self.controls[metadata.SEQUENCE] = "AGCT"
    self.controls[metadata.CELL_DENSITY] = "0"
    self.controls[metadata.PASSAGE] = "0"
    self.controls[metadata.CELL_LINE_ID] = ""
    self.controls.set_index(
        list(metadata.METADATA_ORDER) + [metadata.UNIQUE_ID], inplace=True)
    self.others = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.array([1.0, 2.0]),
            cov=np.array([[2.0, 3.0], [3.0, 4.0]]),
            size=n_other),
        columns=self.columns)
    self.others[metadata.TREATMENT_GROUP] = "EXPERIMENTAL"
    self.others[metadata.MOA] = "other"
    self.others[metadata.COMPOUND] = "2"
    self.others[metadata.CONCENTRATION] = 1.0
    self.others[metadata.ACTIVITY] = "ACTIVE"
    self.others[metadata.BATCH] = "week0"
    self.others[metadata.PLATE] = "plate1"
    self.others[metadata.WELL] = wells_384[:n_other]
    self.others[metadata.ROW] = rows_384[:n_other]
    self.others[metadata.COLUMN] = cols_384[:n_other]
    self.others[metadata.SITE] = 0
    self.others[metadata.UNIQUE_ID] = 0
    self.others[metadata.TIMEPOINT] = "0"
    self.others[metadata.SEQUENCE] = "AGCT"
    self.others[metadata.CELL_DENSITY] = "0"
    self.others[metadata.PASSAGE] = "0"
    self.others[metadata.CELL_LINE_ID] = ""
    self.others.set_index(
        list(metadata.METADATA_ORDER) + [metadata.UNIQUE_ID], inplace=True)
    self.data = pd.concat([self.controls, self.others])

    dists = pd.DataFrame(
        np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0],
                  [7.0, 8.0, 9.0, 10.0], [9.0, 10.0, 11.0, 12.0]]))
    dists[metadata.MOA] = ["a", "b", "a", "a"]
    dists[metadata.COMPOUND] = ["c", "d", "c", "b"]
    dists[metadata.CONCENTRATION] = [1.0, 1.5, 2.2, 3.1]
    dists[metadata.BATCH] = ["0", "0", "1", "1"]
    dists.set_index(
        [
            metadata.MOA, metadata.COMPOUND, metadata.CONCENTRATION,
            metadata.BATCH
        ],
        inplace=True)
    dists.columns = dists.index
    self.dists = dists

  def testIndexToDict(self):
    idx, row = next(self.dists.iterrows())
    self.assertEqual({
        metadata.MOA: "a",
        metadata.COMPOUND: "c",
        metadata.CONCENTRATION: 1.0,
        metadata.BATCH: "0"
    }, evaluate._index_to_dict(idx, row))

  def testNotSameCompoundFilter(self):
    expected = np.array([False, True, False, True])
    np.testing.assert_array_equal(expected,
                                  evaluate.not_same_compound_filter(
                                      self.dists.index[0], self.dists.iloc[0]))
    expected = np.array([True, False, True, True])
    np.testing.assert_array_equal(expected,
                                  evaluate.not_same_compound_filter(
                                      self.dists.index[1], self.dists.iloc[1]))

  def testNotSameCompoundOrBatchFilter(self):
    expected = np.array([False, False, False, True])
    np.testing.assert_array_equal(expected,
                                  evaluate.not_same_compound_or_batch_filter(
                                      self.dists.index[0], self.dists.iloc[0]))
    expected = np.array([False, False, True, True])
    np.testing.assert_array_equal(expected,
                                  evaluate.not_same_compound_or_batch_filter(
                                      self.dists.index[1], self.dists.iloc[1]))

  def testOneNearestNeighbor(self):
    result = evaluate.one_nearest_neighbor(self.dists,
                                           evaluate.not_same_compound_filter)
    expected = ([({
        "batch": "1",
        "compound": "b",
        "concentration": 3.1,
        "moa": "a"
    }, {
        "batch": "0",
        "compound": "c",
        "concentration": 1.0,
        "moa": "a"
    })], [({
        "batch": "0",
        "compound": "c",
        "concentration": 1.0,
        "moa": "a"
    }, {
        "batch": "0",
        "compound": "d",
        "concentration": 1.5,
        "moa": "b"
    }), ({
        "batch": "1",
        "compound": "c",
        "concentration": 2.2,
        "moa": "a"
    }, {
        "batch": "0",
        "compound": "d",
        "concentration": 1.5,
        "moa": "b"
    })])
    self.assertEqual(expected, result)

  def testKNearestNeighbors(self):
    print(self.dists)
    result = evaluate.k_nearest_neighbors(self.dists, 2,
                                          evaluate.not_same_compound_filter)
    expected = (
        [
            # nearest neighbor matches
            ({
                "batch": "1",
                "compound": "b",
                "concentration": 3.1,
                "moa": "a"
            }, {
                "batch": "0",
                "compound": "c",
                "concentration": 1.0,
                "moa": "a"
            }),
            # 2nd nearest neighbor matches
        ],
        [
            # nearest neighbor mismatches
            ({
                "batch": "0",
                "compound": "c",
                "concentration": 1.0,
                "moa": "a"
            }, {
                "batch": "0",
                "compound": "d",
                "concentration": 1.5,
                "moa": "b"
            }),
            ({
                "batch": "1",
                "compound": "c",
                "concentration": 2.2,
                "moa": "a"
            }, {
                "batch": "0",
                "compound": "d",
                "concentration": 1.5,
                "moa": "b"
            }),
            # 2nd nearest neighbor mismatches
            ({
                "batch": "1",
                "compound": "b",
                "concentration": 3.1,
                "moa": "a"
            }, {
                "batch": "0",
                "compound": "d",
                "concentration": 1.5,
                "moa": "b"
            }),
        ])
    self.assertEqual(expected, result)

  def testGetConfusionMatrix(self):
    (correct, mismatch) = evaluate.one_nearest_neighbor(
        self.dists,
        evaluate.not_same_compound_filter)
    confusion_matrix = evaluate.get_confusion_matrix(
        correct, mismatch, metadata.MOA, ["a", "b"])
    expected = np.array([[1, 2],
                         [0, 0]])
    numpy_testing.assert_array_equal(expected, confusion_matrix)


if __name__ == "__main__":
  tf.test.main()
