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

"""Tests for constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.utils import constants


class ConstantsTest(tf.test.TestCase):

  def test_dummy_lhs_symbol(self):
    cfg = nltk.CFG.fromstring([constants.DUMMY_PRODUCTION_RULE])
    self.assertEqual(cfg.productions()[0].lhs().symbol(),
                     constants.DUMMY_LHS_SYMBOL)

  def test_dummy_rhs_symbol(self):
    cfg = nltk.CFG.fromstring([constants.DUMMY_PRODUCTION_RULE])
    rhs = cfg.productions()[0].rhs()
    self.assertLen(rhs, 1)
    self.assertEqual(rhs[0].symbol(), constants.DUMMY_RHS_SYMBOL)


if __name__ == '__main__':
  tf.test.main()
