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

"""Tests for non_semantic_speech_benchmark.eval_embedding.sklearn.models."""

import random as rn
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from non_semantic_speech_benchmark.eval_embedding.sklearn import models


def _get_some_data(num, dims=128):
  inputs = np.random.rand(num, dims) * 10000 - 5000
  sum_np = np.sum(inputs, axis=1)
  targets = np.where(sum_np < -3000, 0, np.where(sum_np < 1000, 1, 2))
  return inputs, targets


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      ({'model_name': k} for k in models.get_sklearn_models().keys())
  )
  def test_sklearn_models_sanity(self, model_name):
    # Set random seed according to:
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development.
    np.random.seed(42)
    rn.seed(42)

    model = models.get_sklearn_models()[model_name]()

    # Actually train.
    inputs, targets = _get_some_data(9000)
    model.fit(inputs, targets)

    # Check that performance is near perfect.
    inputs, targets = _get_some_data(512)
    acc = model.score(inputs, targets)
    expected = 0.5 if 'forest' in model_name.lower() else 0.9
    self.assertGreater(acc, expected)
    logging.info('%s final acc: %f', model, acc)


if __name__ == '__main__':
  absltest.main()
