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

"""Tests for fast_grad_norm."""


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dp_sgd_clipping import fast_cnn_grad_norm


class FastGradNormTest(parameterized.TestCase):
  """Correcntness of results and error handling tests."""

  @parameterized.product(
      (
          dict(
              testcase_name="small_positive_input_matrix_and_partial_gradient",
              input_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=2,
              stride=1,
              expected_result=3564,
          ),
          dict(
              testcase_name="zero_input_matrix_and_positive_partial_gradient",
              input_matrix=np.array([[0, 0, 0], [0, 0, 0]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=2,
              stride=1,
              expected_result=0,
          ),
          dict(
              testcase_name="input_dimension_size_2_output_dimension_size_1",
              input_matrix=np.array([[1, 1], [2, 2]]),
              partial_gradient=np.array([[1]]),
              kernel_size=2,
              stride=1,
              expected_result=10,
          ),
          dict(
              testcase_name="input_dimension_size_2_output_dimension_size_2",
              input_matrix=np.array([[1, 1], [2, 2]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=1,
              stride=1,
              expected_result=290,
          ),
          dict(
              testcase_name="input_dimension_size_4_output_dimension_size_3",
              input_matrix=np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
              partial_gradient=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
              kernel_size=2,
              stride=1,
              expected_result=3528,
          ),
          dict(
              testcase_name="stride_2",
              input_matrix=np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
              partial_gradient=np.array([[1, 1], [2, 2], [3, 3]]),
              kernel_size=2,
              stride=2,
              expected_result=1568,
          ),
          dict(
              testcase_name="stride_3",
              input_matrix=np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
              partial_gradient=np.array([[1], [2], [3]]),
              kernel_size=2,
              stride=3,
              expected_result=392,
          ),
          dict(
              testcase_name="kernel_size_same_as_input_dimension_4",
              input_matrix=np.array(
                  [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
              ),
              partial_gradient=np.array([[1]]),
              kernel_size=4,
              stride=1,
              expected_result=120,
          ),
          dict(
              testcase_name="kernel_size_same_as_input_dimension_5",
              input_matrix=np.array([
                  [1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2],
                  [3, 3, 3, 3, 3],
                  [4, 4, 4, 4, 4],
              ]),
              partial_gradient=np.array([[1]]),
              kernel_size=5,
              stride=1,
              expected_result=150,
          ),
          dict(
              testcase_name="kernel_size_same_as_input_dimension_6",
              input_matrix=np.array([
                  [1, 1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2, 2],
                  [3, 3, 3, 3, 3, 3],
                  [4, 4, 4, 4, 4, 4],
              ]),
              partial_gradient=np.array([[1], [2]]),
              kernel_size=6,
              stride=1,
              expected_result=900,
          ),
      ),
      (
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_fast_grad_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_ghost_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_norm_fft"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "naive_fast_grad_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "naive_ghost_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "grad_norm_computation_selector"
              )
          ),
      ),
  )
  def test_norm_computation_functions_return_values_are_correct(
      self,
      testcase_name,
      input_matrix,
      partial_gradient,
      kernel_size,
      stride,
      expected_result,
      norm_computation_function,
  ):
    """Checks the correctness of the all the norm computation functions."""

    result = norm_computation_function(
        input_matrix, partial_gradient, kernel_size, stride
    )
    self.assertAlmostEqual(result, expected_result)

  @parameterized.product(
      (
          dict(
              testcase_name="negative_kernel_size",
              input_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=-1,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="negative_stride",
              input_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=2,
              stride=-1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="empty_input_matrix",
              input_matrix=np.array([[]]),
              partial_gradient=np.array([[1, 2], [3, 4]]),
              kernel_size=2,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="empty_partial_gradient",
              input_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
              partial_gradient=np.array([[]]),
              kernel_size=2,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name=(
                  "columns_partial_gradient_do_not_match_output_dimension"
              ),
              input_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
              partial_gradient=np.array([[1]]),
              kernel_size=2,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="input_matrix_is_not_2d_matrix",
              input_matrix=np.array([1, 2, 3]),
              partial_gradient=np.array([[1, 2]]),
              kernel_size=2,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="partial_gradient_is_not_2d_matrix",
              input_matrix=np.array([[1, 2, 3]]),
              partial_gradient=np.array([1, 2]),
              kernel_size=2,
              stride=1,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="output_size_mismatch_stride_2",
              input_matrix=np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
              partial_gradient=np.array([[1, 2], [1, 2]]),
              kernel_size=2,
              stride=2,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="output_size_mismatch_stride_3",
              input_matrix=np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
              partial_gradient=np.array([[1, 2], [1, 2]]),
              kernel_size=2,
              stride=3,
              expected_result=ValueError,
          ),
          dict(
              testcase_name="output_size_mismatch_stride_4",
              input_matrix=np.array(
                  [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
              ),
              partial_gradient=np.array([[1, 2], [1, 2]]),
              kernel_size=2,
              stride=4,
              expected_result=ValueError,
          ),
      ),
      (
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_fast_grad_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_ghost_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "in_place_norm_fft"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "naive_fast_grad_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "naive_ghost_norm"
              )
          ),
          dict(
              norm_computation_function=getattr(
                  fast_cnn_grad_norm, "grad_norm_computation_selector"
              )
          ),
      ),
  )
  def test_norm_computation_functions_raise_value_error_when_args_are_invalid(
      self,
      testcase_name,
      input_matrix,
      partial_gradient,
      kernel_size,
      stride,
      expected_result,
      norm_computation_function,
  ):
    """Checks validity of the arguments of the norm computation functions."""

    with self.assertRaises(expected_result):
      norm_computation_function(
          input_matrix,
          partial_gradient,
          kernel_size,
          stride,
      )


if __name__ == "__main__":
  absltest.main()
