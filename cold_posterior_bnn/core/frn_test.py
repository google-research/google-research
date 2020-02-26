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
"""Tests for google_research.google_research.cold_posterior_bnn.core.frn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from cold_posterior_bnn.core import frn


class FrnTest(tf.test.TestCase):

  def test_frn_serialization(self):
    layer0 = frn.FRN(reg_epsilon=1.0e-4)
    config0 = layer0.get_config()
    layer1 = layer0.__class__(**config0)
    config1 = layer1.get_config()
    self.assertEqual(config0, config1,
                     msg='Serialization does not capture all state.')

  def test_tlu_serialization(self):
    layer0 = frn.TLU(tau_regularizer=tf.keras.regularizers.l2(l=0.01))
    config0 = layer0.get_config()
    layer1 = layer0.__class__(**config0)
    config1 = layer1.get_config()
    self.assertEqual(config0, config1,
                     msg='Serialization does not capture all state.')


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()

