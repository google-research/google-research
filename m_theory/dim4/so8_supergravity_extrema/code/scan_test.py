# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Basic tests for the solution scanner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dim4.so8_supergravity_extrema.code import extrema


class SO8SupergravityTest(tf.test.TestCase):

  def testScan(self):
    solutions = extrema.scan_for_solutions(1, 0.1, 10, None)
    # Assert that we did indeed find at least one solution in this
    # near-origin search.
    self.assertTrue(bool(solutions))


if __name__ == "__main__":
  tf.test.main()
