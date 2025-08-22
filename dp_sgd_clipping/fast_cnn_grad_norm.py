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

"""This is a library for fast and memory efficient gradient norm computation.

The functions operate on a single Convolutional Neural Network (CNN) layer and
take as input one sample of the batch and the corresponding partial gradient.
The sample is assumed to be a 2D matrix whose rows correspond o the 1D vectors
of the different input channels.The partial gradient is assumed to be a 2D
matrix whose rows correspond to the 1D vectors of the different output channels.
The other two args are the kernel size and stride of the layer.
"""

import math

import numpy as np


def _check_value_and_shape_of_arguments(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Checks the arguments of the functions in this library."""
  if input_matrix.ndim != 2:
    raise ValueError("input_matrix must be a 2D matrix")
  if partial_gradient.ndim != 2:
    raise ValueError("partial_gradient must be a 2D matrix")
  if input_matrix.shape[1] == 0:
    raise ValueError("input_matrix must be non-empty")
  if partial_gradient.shape[1] == 0:
    raise ValueError("partial_gradient must be non-empty")
  if kernel_size <= 0:
    raise ValueError("kernel_size must be a positive integer")
  if stride <= 0:
    raise ValueError("stride must be a positive integer")
  if (
      # This is the formula for the output dimension of a CNN layer.
      math.floor((input_matrix.shape[1] - kernel_size) / stride + 1)
      != partial_gradient.shape[1]
  ):
    raise ValueError(
        "Number of columns of partial_gradient must be equal to the"
        " output dimension of the layer"
    )


def in_place_fast_grad_norm(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Computes the gradient norm squared of a single sample in a batch.

  This function avoids explicitly instantiating the intermediate matrix that is
  used in the gradient norm computation. This is useful when the batch size is
  large and the gradient norm is computed for each sample in the batch.

  More formally, it implements the following logic: let x be the input matrix, g
  be the partial gradient, U(x[i]) be the matrix whose rows correspond to
  the different kernel windows of the i-th input channel, n_in be the number of
  input channels, n_out be the number of output channels, and res be the
  l_2 gradient norm squared. Then,
  res = sum_{i in n_in} sum_{j in n_out} ||U(x[i])^T g[j]||^2.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.

  Returns:
    l_2 norm squared of the gradient as a float.
  """

  # This function is checking that the values and shapes of args are valid.
  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  res = 0

  for input_vector in input_matrix:
    # We use the sliding window view to avoid explicitly instantiating the
    # intermediate matrix.
    u_input_vector = np.lib.stride_tricks.sliding_window_view(
        input_vector, window_shape=(kernel_size,)
    )[::stride]
    u_input_vector_transpose = u_input_vector.T
    for output in partial_gradient:
      res += np.sum(
          np.square(np.tensordot(u_input_vector_transpose, output, (1, 0)))
      )

  return res


def _select_method(
    prioritize_memory,
    fft_beats_memory_efficient_implementations,
    ghost_beats_memory_efficient_implementations,
    fft_beats_naive_implementations,
    ghost_beats_naive_implementations,
):
  """Helper function that selects the appropriate norm computation function.

  Args:
    prioritize_memory: flag that indicates whether to use the memory efficient
      implementation of the gradient norm computation or increase memory usage
      to The function that computes the gradient norm. gain better computational
      efficiency.
    fft_beats_memory_efficient_implementations: flag that indicates whether the
      FFT based norm computation function is faster than the memory efficient
      implementations.
    ghost_beats_memory_efficient_implementations: flag that indicates whether
      the ghost norm computation function is faster than the memory efficient
      implementations.
    fft_beats_naive_implementations: flag that indicates whether the FFT based
      norm computation function is faster than the naive implementations.
    ghost_beats_naive_implementations: flag that indicates whether the ghost
      norm computation function is faster than the naive implementations.

  Returns:
    The appropriate norm computation function.
  """
  if prioritize_memory:
    if fft_beats_memory_efficient_implementations:
      return in_place_norm_fft
    elif ghost_beats_memory_efficient_implementations:
      return in_place_ghost_norm
    else:
      return in_place_fast_grad_norm
  else:
    if fft_beats_naive_implementations:
      return in_place_norm_fft
    elif ghost_beats_naive_implementations:
      return naive_ghost_norm
    else:
      return naive_fast_grad_norm


def in_place_ghost_norm(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Computes the gradient norm squared of a single sample in a batch.

  This function uses the ghost norm trick to compute the gradient norm squared.
  It avoids explicitly instantiating the intermediate matrices that are
  used in the gradient norm computations. This is useful when the batch size is
  large and the gradient norm is computed for each sample in the batch.

  More formally, it implements the following logic: let x be the input matrix, g
  be the partial gradient, U(x[i]) be the matrix whose rows correspond to
  the different kernel windows of the i-th input channel, n_in be the number of
  input channels, n_out be the number of output channels, and res be the
  l_2 gradient norm squared. Then,
  res = <sum_{i in n_in} U(x[i]) U(x[i])^T, sum_{j in n_out} g[j] (g[j])^T>,
  where <,> is the Frobenius inner product.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.

  Returns:
    l_2 norm squared of the gradient as a float.
  """

  # checking shapes and values of the arguments
  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  output_dimension = partial_gradient.shape[1]
  number_input_channels = input_matrix.shape[0]
  number_output_channels = partial_gradient.shape[0]
  res = 0

  for j_1 in range(output_dimension):
    for j_2 in range(output_dimension):
      # This expression computes the following:
      # sum_{i in n_in} (U(x[i]) U(x[i])^T)[j_1][j_2])
      # Recall that the j_1 row of U(x[i]) corresponds to the kernel
      # window of the i-th input channel at the j_1-th position.
      temp_1 = 0
      for i in range(number_input_channels):
        temp_1 += np.dot(
            input_matrix[i][j_1 * stride : j_1 * stride + kernel_size],
            input_matrix[i][j_2 * stride : j_2 * stride + kernel_size],
        )
      # This expression computes the following:
      # sum_{k in n_out} g[k](g[k])^T[j_1][j_2]
      temp_2 = 0
      for k in range(number_output_channels):
        temp_2 += np.dot(partial_gradient[k][j_1], partial_gradient[k][j_2])
      res += temp_1 * temp_2

  return res


def in_place_norm_fft(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Computes the gradient norm squared of a single sample in a batch.

  This function uses the Fast Fourier Transform to compute the gradient
  norm squared, by efficiently transforming the gradient norm computation to a
  multiplication of a circulant matrix with a vector. It avoids explicitly
  instantiating the intermediate matrices and vectors that are used in the
  computations. This is useful when the batch size is large and
  the gradient norm is computed for each sample in the batch.

  More formally, it implements the following logic: let x be the input matrix, g
  be the partial gradient, U(x[i]) be the matrix whose rows correspond to
  the different kernel windows of the i-th input channel, n_in be the number of
  input channels, n_out be the number of output channels, and res be the l_2
  gradient norm squared. Then,
  res = sum_{i in n_in} sum_{j in n_out} ||R P U'(x[i])^T g'[j]||^2, where R is
  an operator that returns some specific entries of the vector it is applied to,
  P is an appropriate permutation matrix, U'(x[i]) is some circulantmatrix that
  is defined based on U(x[i]), and g'[j] is the vector that is obtained by
  padding g[j] appropriately.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.

  Returns:
    l_2 norm squared of the gradient as a float.
  """

  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  input_dimension = input_matrix.shape[1]
  output_dimension = partial_gradient.shape[1]
  number_input_channels = input_matrix.shape[0]
  number_output_channels = partial_gradient.shape[0]
  res = 0

  # We only allocate memory for the entries of the padded partial gradient once
  padded_partial_gradient = np.zeros(
      input_dimension, dtype=partial_gradient.dtype
  )

  for j in range(number_output_channels):
    # We start by populating the non-zero entries of the padded partial gradient
    upper_bound = (output_dimension - 1) * stride + 1
    padded_partial_gradient[:upper_bound:stride] = partial_gradient[j]

    partial_derivative_fft = np.fft.fft(padded_partial_gradient)
    for i in range(number_input_channels):

      # FFT of the first column of the circulant matrix that is defined based
      # on U(x[i])
      vector_in_fft = np.fft.fft(np.flip(input_matrix[i]))

      temp = np.flip(
          np.fft.ifft(np.multiply(vector_in_fft, partial_derivative_fft))
      )

      res += np.sum(np.real(temp[0:kernel_size] ** 2))

  return res


def _unfold(
    input_matrix,
    kernel_size,
    stride,
    output_dimension,
):
  """Unfolds the input matrix.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.
    output_dimension: output dimension of the layer.

  Returns:
    A 2D matrix whose rows correspond to the different kernel windows of
    all the input channels.
  """

  input_channels = input_matrix.shape[0]

  # Create slices of the input_matrix and stack them
  # Create an array of indices to slice
  indices = (
      np.arange(kernel_size)[None, :]
      + np.arange(output_dimension)[:, None] * stride
  )

  # Extract slices for each channel
  slices = np.stack(
      [input_matrix[j, indices] for j in range(input_channels)], axis=-1
  )
  res = slices.reshape(output_dimension, -1)
  return res


def naive_fast_grad_norm(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Computes the gradient norm squared of a single sample in a batch.

  This function is a memory inefficient implementation of the gradient norm
  computation. It can be faster than the in-place implementations of this
  function, but it requires more memory.

  More formally, it implements the following logic: let x be the input matrix, g
  be the partial gradient, U be the matrix whose i-th row consists of
  consecutive blocks of the i-th kernel windows of all input channels, n_in be
  the number of input channels, n_out be the number of output channels, and res
  be the l_2 gradient norm squared. Then,
  res = sum_{i in n_in} sum_{j in n_out} ||U(x[i])^T g[j]||^2.

  Unlike the in-place implementations, this function explicitly instantiates the
  intermediate matrix that is used in the gradient norm computation.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.

  Returns:
    l_2 norm squared of the gradient as a float.
  """

  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  unfolded_input_matrix = _unfold(
      input_matrix, kernel_size, stride, partial_gradient.shape[1]
  )
  grad = unfolded_input_matrix.T @ partial_gradient.T
  norm_grad = (
      np.einsum("ij,ij->", grad, grad)
  )

  return norm_grad


def naive_ghost_norm(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
):
  """Computes the gradient norm squared of a single sample in a batch.

  This function is a memory inefficient implementation of the gradient norm
  computation. It can be faster than the in-place implementations of this
  function, but it requires more memory.

  More formally, it implements the following logic: let x be the input matrix, g
  be the partial gradient, U be the matrix whose i-th row consists of
  consecutive blocks of the i-th kernel windows of all input channels, n_in be
  the number of input channels, n_out be the number of output channels, and res
  be the l_2 gradient norm squared. Then, res = <U U^T, g g^T>, where <,> is the
  Frobenius inner product.

  Unlike the in-place implementations, this function explicitly instantiates the
  intermediate matrices that are used in the gradient norm computation.

  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.

  Returns:
    l_2 norm squared of the gradient as a float.
  """
  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  # computation of UU^T
  unfolded_input_matrix = _unfold(
      input_matrix, kernel_size, stride, partial_gradient.shape[1]
  )
  v = unfolded_input_matrix @ unfolded_input_matrix.T

  # computation of gg^T
  partial_matrix = partial_gradient.T @ partial_gradient

  norm_grad = np.tensordot(v, partial_matrix, axes=[[0, 1], [0, 1]])
  return norm_grad


def grad_norm_computation_selector(
    input_matrix,
    partial_gradient,
    kernel_size,
    stride,
    memory_efficient_implementation_required=True,
):
  """Selects the norm computation function to use based on the parameters.

  The three gradient norm computation functions provided in this library have
  incomparable guarantees and outperform one another in different regimes of the
  parameters of the underlying CNN. This is a meta-function which, given
  a specific set of parameters, calls the appropriate norm computation function
  which performs better in the given regime. The choice of the appropriate
  function is guided both by the theoretical analysis of the running time of the
  functions provided in this module as well as experimental evaluation of their
  behavior in representative settings.


  Args:
    input_matrix: 2D matrix whose rows are 1D vectors of the input to the layer.
    partial_gradient: 2D matrix whose rows are 1D vectors of the partial
      gradient across the output channels.
    kernel_size: kernel size of the layer.
    stride: stride of the layer.
    memory_efficient: flag that indicates whether to use the memory efficient
    implementation of the gradient norm computation or increase memory usage to
    gain better computational efficiency.

  Returns:
    l_2 norm squared of the gradient as a float.
  """

  _check_value_and_shape_of_arguments(
      input_matrix, partial_gradient, kernel_size, stride
  )

  # dimension, number of channels computation
  input_channels = input_matrix.shape[0]
  input_dimension = input_matrix.shape[1]
  output_channels = partial_gradient.shape[0]
  output_dimension = partial_gradient.shape[1]

  # heuristic choice of the constants hidden in the big-oh notation of the
  # asymptotic running times
  constant_fft = 20
  constant_in_place_ghost = 6
  constant_in_place_fast_grad = 4
  constant_naive_ghost = 3
  constant_naive_fast_grad = 2

  # asymptotic running time of various norm computation functions
  fft_asymptotic_running_time = (
      input_channels
      * output_channels
      * input_dimension
      * math.log(input_dimension)
  )
  ghost_asymptotic_running_time = output_dimension**2 * (
      input_channels * kernel_size + output_channels
  )
  fast_grad_asymptotic_running_time = (
      input_channels * output_channels * output_dimension * kernel_size
  )

  # heuristic computation of the actual running time taking into account the
  # heuristic choice of constants
  fft_running_time = constant_fft * fft_asymptotic_running_time
  in_place_ghost_running_time = (
      constant_in_place_ghost * ghost_asymptotic_running_time
  )
  in_place_fast_grad_running_time = (
      constant_in_place_fast_grad * fast_grad_asymptotic_running_time
  )
  naive_ghost_running_time = (
      constant_naive_ghost * ghost_asymptotic_running_time
  )
  naive_fast_grad_running_time = (
      constant_naive_fast_grad * fast_grad_asymptotic_running_time
  )

  # predicates that indicate which norm computation function should be used
  fft_beats_memory_efficient_implementations = (
      fft_running_time <= in_place_ghost_running_time
      and fft_running_time <= in_place_fast_grad_running_time
  )
  ghost_beats_memory_efficient_implementations = (
      in_place_ghost_running_time <= fft_running_time
      and in_place_ghost_running_time <= in_place_fast_grad_running_time
  )
  fft_beats_naive_implementations = (
      fft_running_time <= naive_ghost_running_time
      and fft_running_time <= naive_fast_grad_running_time
  )
  ghost_beats_naive_implementations = (
      naive_ghost_running_time <= naive_fast_grad_running_time
      and naive_ghost_running_time <= fft_running_time
  )

  method = _select_method(
      memory_efficient_implementation_required,
      fft_beats_memory_efficient_implementations,
      ghost_beats_memory_efficient_implementations,
      fft_beats_naive_implementations,
      ghost_beats_naive_implementations,
  )

  return method(
      input_matrix, partial_gradient, kernel_size, stride
  )
