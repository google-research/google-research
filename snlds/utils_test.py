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

# Lint as: python3
"""Tests for snlds.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from snlds import utils


class UtilsTest(tf.test.TestCase):

  def sefUp(self):
    super(UtilsTest, self).setUp()

  def test_build_dense_network(self):
    returned_nets = utils.build_dense_network(
        [3, 8, 3], ["relu", "relu", None])
    self.assertEqual(len(returned_nets.layers), 3)

  def test_normalize_logprob(self):
    input_prob = np.random.uniform(low=1e-6, high=1., size=[2, 3, 6])
    log_normalizer = np.log(np.sum(input_prob, axis=-1, keepdims=True))
    input_logprob = np.log(input_prob)

    target_logprob = input_logprob - log_normalizer
    self.assertAllClose(
        self.evaluate(utils.normalize_logprob(input_logprob)[0]),
        target_logprob)

    input_tensor = np.log([0.1, 0.3, 0.5])
    target_logprob = np.log([1./3., 1./3., 1./3.])
    temperature = 1e5
    self.assertAllClose(
        self.evaluate(utils.normalize_logprob(
            input_tensor, temperature=temperature)[0]),
        target_logprob,
        rtol=1e-4,
        atol=1e-4,
        )

  def test_get_posterior_crossentropy(self):
    input_logprob = np.log(np.random.uniform(low=1e-6, high=1., size=[2, 3, 6]))
    prior_prob = np.random.uniform(size=[6])

    result_entropy = utils.get_posterior_crossentropy(input_logprob, prior_prob)
    numpy_result = np.sum(input_logprob*prior_prob[None, None, :],
                          axis=(1, 2))

    self.assertAllClose(
        self.evaluate(result_entropy), numpy_result)


if __name__ == "__main__":
  tf.test.main()
