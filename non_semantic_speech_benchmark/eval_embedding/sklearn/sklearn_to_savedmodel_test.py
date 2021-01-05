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

"""Tests for non_semantic_speech_benchmark.eval_embedding.sklearn.sklearn_to_savedmodel."""

from absl.testing import absltest
import numpy as np
from sklearn import linear_model
import tensorflow as tf

from non_semantic_speech_benchmark.eval_embedding.sklearn import sklearn_to_savedmodel


class SklearnToSavedmodelTest(absltest.TestCase):

  def test_sklearn_logistic_regression_to_keras(self):
    for i in range(10):
      m = linear_model.LogisticRegression()
      n_samples, n_features = 10, 2048
      m.fit(np.random.rand(n_samples, n_features), [1] * 5 + [0] * 5)
      k = sklearn_to_savedmodel.sklearn_logistic_regression_to_keras(m)

      for j in range(20):
        rng = np.random.RandomState(i * j)
        data = rng.lognormal(size=n_features).reshape([1, -1])

        sklearn_prob = m.predict_proba(data)
        keras_prob = k(data)

        np.testing.assert_almost_equal(sklearn_prob, keras_prob.numpy(), 6)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
