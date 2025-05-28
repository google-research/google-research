# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for bottlenecks."""

import tensorflow as tf

from vct.src import bottlenecks


class CompressionTest(tf.test.TestCase):

  def test_compress(self):

    def _make_bottleneck(compression):
      return bottlenecks.ConditionalLocScaleShiftBottleneck(
          num_means=5,
          num_scales=64,
          coding_rank=1,
          compression=compression,
      )

    model = _make_bottleneck(compression=False)
    data = tf.cast(tf.linspace(-10, 10, 100), tf.float32)
    mean = tf.ones((100,), tf.float32) * 0.4
    scale = tf.ones((100,), tf.float32) / 100.

    data_out_est, bits_est = model(data, mean, scale, training=False)
    error = tf.abs(data_out_est - data)
    self.assertAllLessEqual(error, 0.5)

    # Re-create model with compression=True to test range coding.
    model = _make_bottleneck(compression=True)
    _, bytestring = model.compress(data, mean, scale)
    data_out_real = model.decompress(bytestring, mean, scale)
    num_bytes = len(bytestring.numpy())

    self.assertAllClose(data_out_est, data_out_real)
    self.assertLessEqual(num_bytes * 8., bits_est)


if __name__ == "__main__":
  tf.test.main()
