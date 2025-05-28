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

import jax.numpy as jnp
import tensorflow as tf

import sparse_deferred.jax as sdjnp


class JaxEngineTest(tf.test.TestCase):

  def test_jax_engine_constructs_ok(self):
    """Tests verifies that JaxEngine implements all abstract methods."""
    sdjnp._JaxEngine()

  def test_unsorted_segment_sum(self):
    engine = sdjnp._JaxEngine()
    segment_sum = engine.unsorted_segment_sum(
        jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype='int32'),
        jnp.array([0, 1, 0], dtype='int32'),
        num_segments=3)
    expected_segment_sum = jnp.array(
        [[5, 5, 5, 5],
         [5, 6, 7, 8],
         [0, 0, 0, 0]], dtype='int32')
    if not jnp.allclose(segment_sum, expected_segment_sum):
      raise AssertionError()

if __name__ == '__main__':
  tf.test.main()
