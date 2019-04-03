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

"""Tests for fit_partition_spline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from robust_loss import distribution
from robust_loss import fit_partition_spline


class FitPartitionSplineTest(tf.test.TestCase):

  def testNumericalPartitionIsAccurate(self):
    """Test _numerical_base_partition_function against some golden data."""
    for (numer, denom) in [(0, 1), (1, 8), (1, 2), (1, 1), (2, 1), (8, 1)]:
      alpha = tf.cast(numer, tf.float64) / tf.cast(denom, tf.float64)
      z_true = distribution.analytical_base_partition_function(numer, denom)
      with self.session():
        z = fit_partition_spline.numerical_base_partition_function(alpha).eval()
      self.assertAllClose(z, z_true, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
  tf.test.main()
