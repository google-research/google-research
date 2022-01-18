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

"""Tests for basic_specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tunas import basic_specs


class BasicSpecsTest(tf.test.TestCase):

  def test_filter_multiplier_ordering(self):
    # Verify that instances of FilterMultiplier are compatible with Python's
    # native "less than" and "greater than" operations. This works because
    # Python namedtuples like FilterMultiplier inherit from `tuple`, and tuples
    # are ordered lexicographically. For example, when determining whether
    # (x1, x2, x3) < (y1, y2, y3), we first compare x1 and y1. If x1 < y1, the
    # inequality is true. If they're equal, we next compare x2 and y2.
    # And so on.
    multiplier = basic_specs.FilterMultiplier
    self.assertLess(multiplier(3.1), multiplier(4.1))
    self.assertGreater(multiplier(20), multiplier(10))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
