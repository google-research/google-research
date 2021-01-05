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

# Lint as: python2, python3
"""Tests for uq_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special
import tensorflow.compat.v2 as tf

from uq_benchmark_2019 import uq_utils


class UqUtilsTest(tf.test.TestCase):

  def test_np_inverse_softmax(self):
    batch_size, nclasses = [4, 3]
    logits_orig = np.random.rand(batch_size, nclasses)
    probs_orig = scipy.special.softmax(logits_orig, axis=-1)
    logits_new = uq_utils.np_inverse_softmax(probs_orig)
    probs_new = scipy.special.softmax(logits_new, axis=-1)
    self.assertAllClose(probs_orig, probs_new)

  def test_np_soften_probabilities(self):
    shape = [12, 5]
    logits = np.random.uniform(0, 1, size=shape)
    probs = scipy.special.softmax(logits, axis=-1)
    probs[0] = 0
    probs[0, 0] = 1
    soft_probs = uq_utils.np_soften_probabilities(probs, epsilon=1e-8)
    self.assertAllClose(probs[1:], soft_probs[1:])
    self.assertAllLess(soft_probs, 1)
    self.assertAllGreater(soft_probs, 0)
    self.assertAllClose(np.ones(shape[0]), soft_probs.sum(1), atol=1e-10)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
