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

"""Basic tests for the solution scanner."""

import tensorflow as tf

from dim4.so8_supergravity_extrema.code import extrema


class SO8SupergravityTest(tf.test.TestCase):

  def testScan(self):
    pot, stat, _ = next(extrema.scan_for_solutions())
    # Check that we did indeed find an AdS solution.
    self.assertLess(pot, 0)
    self.assertLess(stat, 1e-3)


if __name__ == "__main__":
  tf.test.main()
