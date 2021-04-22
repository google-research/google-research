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

"""Tests for aqt.fp_cast."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as onp
from aqt.jax import fp_cast


config.update('jax_numpy_rank_promotion', 'raise')

fp8_143_max_value = float.fromhex('0x1.ep4')
fp8_143_min_value = float.fromhex('0x1p-11')


def test_data(dtype):
  return (
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_special',
          dtype=dtype,
          argument_result_values=[
              [float('nan'), float('nan')],
              [float('inf'), float('inf')],
              [float('-inf'), float('-inf')],
          ],
      ),
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_zeros',
          dtype=dtype,
          argument_result_values=[
              [-0., -0.],
              [0., 0.],
          ],
      ),
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_saturation',
          dtype=dtype,
          argument_result_values=[
              [fp8_143_max_value, fp8_143_max_value],
              [float.fromhex('0x1.fp4'), fp8_143_max_value],
              [float.fromhex('0x1.cp4'),
               float.fromhex('0x1.cp4')],
              [-fp8_143_max_value, -fp8_143_max_value],
              [float.fromhex('-0x1.fp4'), -fp8_143_max_value],
              [float.fromhex('-0x1.cp4'),
               float.fromhex('-0x1.cp4')],
          ],
      ),
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_flush_to_zero',
          dtype=dtype,
          argument_result_values=[
              [fp8_143_min_value, fp8_143_min_value],
              [float.fromhex('0x1p-12'), 0.],
              [float.fromhex('0x1p-13'), 0.],
              [-fp8_143_min_value, -fp8_143_min_value],
              [float.fromhex('-0x1p-12'), -0.],
              [float.fromhex('-0x1p-13'), -0.],
          ],
      ),
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_round_near_even',
          dtype=dtype,
          argument_result_values=[
              [-1. + 1 / 32, -1.0],
              [-1. - 1 / 16, -1.0],
              [-2. + 1 / 16, -2.0],
              [-2. - 1 / 8, -2.0],
              [1. - 1 / 32, 1.0],
              [1. + 1 / 16, 1.0],
              [2. - 1 / 16, 2.0],
              [2. + 1 / 8, 2.0],
              [27.5, 28.],
              [28.5, 28.],
              [-27.5, -28.],
              [-28.5, -28.],
          ],
      ),
      dict(
          testcase_name=f'{dtype.dtype.name}_fp8_143_multi_dimensional',
          dtype=dtype,
          argument_result_values=[
              [
                  [
                      [
                          [7.09375, float('inf'), 8.125, 14.1875],
                          [48., 11.875, 8.5625, 9.3125],
                          [4.15625, 0.433594, 15.4375, 2.32812],
                      ],
                      [
                          [5.4375, 1.00781, 13.25, 1.57812],
                          [15.3125, 2.23438, 6.71875, 7.9375],
                          [3.0625, 6.6875, 4.625,
                           float('nan')],
                      ],
                  ],
                  [
                      [
                          [7, float('inf'), 8, 14],
                          [30., 12, 9, 9],
                          [4, 0.4375, 15, 2.25],
                      ],
                      [
                          [5.5, 1, 13, 1.625],
                          [15, 2.25, 6.5, 8],
                          [3, 6.5, 4.5, float('nan')],
                      ],
                  ],
              ],
          ],
      ),
  )


def gradient_test_data(sig_bits):
  fp_format = dict(exp_min=-11, exp_max=4, sig_bits=sig_bits)
  bounds = fp_cast.get_bounds(**fp_format)

  # Approximate float32 eps at bounds.flush_to_zero_bound
  eps_flush_to_zero = 1e-7
  # Approximate float32 eps at bounds.saturation_bound
  eps_saturation = 1e-5

  # Construct a list of pairs where the first element is the primal value
  # and the second element is the expected gradient of the floating-point
  # quantization function evaluated at that value.
  testcases = [

      # Test values above the saturation bound. Gradient should be zero since
      # these values are clipped to the saturation threshold during
      # quantization.
      (2.11e10, 0),
      (2.11e10, 0),
      (bounds.saturation_bound + eps_saturation, 0),
      (-bounds.saturation_bound - eps_saturation, 0),

      # Test values within the range of the fp format. The gradient should be 1
      # since values in this range use the straight-through estimator.
      (1.0, 1),
      (-1.0, 1),
      (2.11, 1),
      (-2.11, 1),
      (bounds.flush_to_zero_bound + eps_flush_to_zero, 1),
      (-bounds.flush_to_zero_bound - eps_flush_to_zero, 1),
      (bounds.saturation_bound - eps_saturation, 1),
      (-bounds.saturation_bound + eps_saturation, 1)
  ]
  return [{
      'primal': testcase[0], 'expected_grad': testcase[1], **fp_format
  } for testcase in testcases]


class FpCastTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *test_data(jnp.bfloat16),
      *test_data(jnp.float32))
  def test_downcast_sat_ftz(self, dtype, argument_result_values):
    argument_result = jnp.array(
        argument_result_values,
        dtype=dtype,
    )
    y = fp_cast.downcast_sat_ftz(
        argument_result[:, 0],
        exp_min=-11,
        exp_max=4,
        sig_bits=3,
    )
    onp.testing.assert_equal(
        onp.array(argument_result[:, 1], dtype=onp.float32),
        onp.array(y, dtype=onp.float32),
    )

  def test_invalid_argument_type(self):
    x_s8 = jnp.array(1, dtype=jnp.int8)
    with self.assertRaises(ValueError):
      fp_cast.downcast_sat_ftz(x_s8, exp_min=-11, exp_max=4, sig_bits=3)

  def test_return_type(self):
    x_bf16 = jnp.array(1.0, dtype=jnp.bfloat16)
    y_bf16 = fp_cast.downcast_sat_ftz(
        x_bf16, exp_min=-11, exp_max=4, sig_bits=3)
    self.assertEqual(x_bf16.dtype, y_bf16.dtype)

    xf32 = jnp.array(1.0, dtype=jnp.bfloat16)
    yf32 = fp_cast.downcast_sat_ftz(
        xf32, exp_min=-11, exp_max=4, sig_bits=3)
    self.assertEqual(xf32.dtype, yf32.dtype)

  def test_sig_bits_zero(self):
    x = jnp.array(2.11111)
    y = fp_cast.downcast_sat_ftz(x, exp_min=-11, exp_max=4, sig_bits=0)
    self.assertEqual(y.item(), 2.0)

  @parameterized.parameters(*gradient_test_data(sig_bits=3),
                            *gradient_test_data(sig_bits=0))
  def test_grad(self, primal, expected_grad, sig_bits, exp_min, exp_max):
    # We use a 'sum' here so that each element of the gradient of the sum
    # corresponds to the partial derivative of output of `downcast_sat_ftz`
    # with respect to the corresponding input.
    def downcast_and_sum(x):
      return jnp.sum(
          fp_cast.downcast_sat_ftz(
              x, sig_bits=sig_bits, exp_min=exp_min, exp_max=exp_max))

    x_grad = jax.grad(downcast_and_sum)(jnp.array(primal))
    onp.testing.assert_equal(onp.array(x_grad), expected_grad)


if __name__ == '__main__':
  absltest.main()
