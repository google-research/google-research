# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import numpy as np
import tensorflow as tf

import sparse_deferred.np as sdnp


_DATA = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype='int32')
_INDICES = np.array([0, 1, 0], dtype='int32')


def engine():
  return sdnp._NumpyEngine()


class NpEngineTest(tf.test.TestCase):

  def test_np_engine_constructs_ok(self):
    """Verify that `_NumpyEngine` implements all abstract methods."""
    _ = engine()

  def test_unsorted_segment_sum(self):
    self.assertAllEqual(
        engine().unsorted_segment_sum(_DATA, _INDICES, num_segments=3),
        tf.math.unsorted_segment_sum(_DATA, _INDICES, num_segments=3))

  def test_unsorted_segment_max(self):
    self.assertAllEqual(
        engine().unsorted_segment_max(_DATA, _INDICES, num_segments=3),
        tf.math.unsorted_segment_max(_DATA, _INDICES, num_segments=3))
    float_data = np.array(_DATA, dtype='float32')
    self.assertAllEqual(
        engine().unsorted_segment_max(float_data, _INDICES, num_segments=3),
        tf.math.unsorted_segment_max(float_data, _INDICES, num_segments=3))


if __name__ == '__main__':
  tf.test.main()
