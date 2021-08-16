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

"""Tests for f_net.fourier."""

import functools

from absl.testing import absltest
from jax import lax
from jax import numpy as jnp
import numpy as np
from scipy import linalg

from f_net import fourier


class FourierTest(absltest.TestCase):

  def test_two_dim_matmul(self):
    max_seq_length = 3
    hidden_dim = 8

    # We test the 2D matmul function on the DFT calculation (primary use-case).
    dft_mat_seq = jnp.asarray(linalg.dft(max_seq_length))
    dft_mat_hidden = jnp.asarray(linalg.dft(hidden_dim))
    two_dim_matmul = functools.partial(
        fourier.two_dim_matmul,
        matrix_dim_one=dft_mat_seq,
        matrix_dim_two=dft_mat_hidden)

    inputs = jnp.array([[1, 0, 0, 11, 9, 2, 0.4, 2], [1, 1, 0, 1, 0, 2, 8, 1],
                        [1, 4, 0, 5, 5, 0, -3, 1]],
                       dtype=jnp.float32)

    expected_output = np.fft.fftn(inputs)

    for precision, delta in zip(
        [lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST],
        [1e-4, 1e-5, 1e-5]):
      actual_output = two_dim_matmul(inputs, precision=precision)
      # Compare results row by row.
      for i in range(max_seq_length):
        self.assertSequenceAlmostEqual(
            actual_output[i], expected_output[i], delta=delta)

  def test_loop_fftn(self):
    inputs = jnp.array(
        [[1, 1, 0, 1, 0, -2, 8, -1], [0, -11, 11, 12, 0, -22, 348, -1],
         [1, 15, 4, 4, 0, 0, -8, 1]],
        dtype=jnp.float32)
    actual_output = fourier.fftn(inputs)
    expected_output = np.fft.fftn(inputs)

    for i in range(3):
      # Compare results row by row.
      self.assertSequenceAlmostEqual(
          actual_output[i], expected_output[i], delta=1e-4)


if __name__ == "__main__":
  absltest.main()
