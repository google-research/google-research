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

"""Various helper functions related to `tf.keras.layers.EinsumDense`."""

import enum
import itertools
import re

import numpy as np
import tensorflow as tf

EquationType = enum.Enum(
    "EquationType",
    ["UNKNOWN", "NO_ELLIPSES", "LEFT_ELLIPSES", "RIGHT_ELLIPSES"],
)


def _is_batch_of_vectors(t):
  """Checks if an input is a batch of 1D vectors."""
  num_nontrivial_indices = 0
  for s in t.shape[1:]:
    if num_nontrivial_indices > 1:
      return False
    if s > 1:
      num_nontrivial_indices += 1
  return num_nontrivial_indices <= 1


def _parse_einsum_equation(
    equation,
):
  """Returns a case number and I/O substrings of an einsum equation."""
  eqn_type = EquationType.UNKNOWN
  match1 = re.fullmatch(r"([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", equation)
  if match1 is not None:
    eqn_type = EquationType.NO_ELLIPSES
  match2 = re.fullmatch(
      r"\.\.\.([a-zA-Z]+),([a-zA-Z]+)->\.\.\.([a-zA-Z]+)", equation
  )
  if match2 is not None:
    eqn_type = EquationType.LEFT_ELLIPSES
  match3 = re.fullmatch(
      r"([a-zA-Z]+)\.\.\.,([a-zA-Z]+)->([a-zA-Z]+)\.\.\.", equation
  )
  if match3 is not None:
    eqn_type = EquationType.RIGHT_ELLIPSES
  matched = [g for g in [match1, match2, match3] if g is not None]
  if len(matched) != 1:
    raise ValueError(
        "Invalid Einsum eqution string "
        + equation
        + " ."
        "Must be one of the forms {ab,bc->ac}, {...ab,bc->...ac}, "
        "{ab...,bc->ac...}"
    )
  return eqn_type, matched[0].groups()


def _reshape_einsum_inputs(
    input_tensor,
    equation,
):
  """Converts input tensor to a batched matrix according to an einsum equation.

  Args:
    input_tensor: A `tf.Tensor` corresponding to the first input of the einsum
      equation.
    equation: The einsum equation `string`.

  Returns:
    A rank-3 `tf.Tensor` representing a batch of rank-2 matrices with the same
    number of rows and columns. The output dimensions, in order, are:
    ```
      (num_batches, num_rows, num_columns)
    ```
    When `input_tensor` is a rank-2 `tf.Tensor`, the number of output rows is 1
    and the number of output columns is the second dimension of the input. The
    product of the non-trivial dimensions of the output should be equal to
    the product of the dimensions of `input_tensor`.
  """
  # Find the components `ab`, `bc`, and `ac` given that `equation` can only be
  # one of the following mutually exclusive forms:
  #
  #   (C1) ab,bc->ac,
  #   (C2) ...ab,bc->...ac
  #   (C3) ab...,bc->ac...
  #
  # NOTE: `a`, `b`, and `c` are (possibly) also substrings.

  # Compute the first index of the `b` part of the `ab` component.
  input_shape = input_tensor.shape
  input_len = len(input_shape)
  eqn_type, (ab_str, bc_str, ac_str) = _parse_einsum_equation(equation)
  if eqn_type == EquationType.LEFT_ELLIPSES:
    # In case (C2), the `a` part of this component can be empty, so we have no
    # choice but to compare the `c` part of `ac` with the `bc` component.
    c_len = 0
    for s1, s2 in itertools.zip_longest(reversed(bc_str), reversed(ac_str)):
      if s1 == s2:
        c_len += 1
      else:
        break
    b_len = len(bc_str) - c_len
    b_idx = input_len - b_len
  else:
    # For the other cases, we simply compare `ab` with `ac` to get the length
    # of the `a` component, i.e., the first index of `b`.
    b_idx = 0
    for s1, s2 in itertools.zip_longest(ab_str, ac_str):
      if s1 == s2:
        b_idx += 1
      else:
        break
  # Prepare `input_tensor` for reshaping and get the pivot index of the prepped
  # tensor. Note that case (C3) requires a transpose to ensure that matrix
  # multiplication is performed by the caller.
  if eqn_type == EquationType.RIGHT_ELLIPSES:
    ellipses_idx = len(ab_str)
    # Convert `ab...` to `a...b`.
    new_ordering = (
        list(range(0, b_idx))
        + list(range(ellipses_idx, input_len))
        + list(range(b_idx, ellipses_idx))
    )
    input_tensor = tf.transpose(input_tensor, perm=new_ordering)
    ellipses_len = input_len - ellipses_idx
    pivot_idx = b_idx + ellipses_len
  else:
    pivot_idx = b_idx
  # The output tensor is a batched set of matrices, split at the pivot index
  # of the previously prepped tensor.
  base_tensor_shape = input_tensor.shape
  batch_size = base_tensor_shape[0]
  num_rows = int(np.prod(base_tensor_shape[1:pivot_idx]))
  num_columns = int(np.prod(base_tensor_shape[pivot_idx:]))
  return tf.reshape(input_tensor, shape=[batch_size, num_rows, num_columns])


def _reshape_einsum_outputs(
    output_tensor,
    equation,
):
  """Converts output tensor to a batched matrix according to an einsum equation.

  The logic is almost the same as in `_reshape_einsum_inputs()` except
  in the case where the equation is left-elided by ellipses. For this case,
  we need to pass in a reversed kernel shape.

  Args:
    output_tensor: A `tf.Tensor` corresponding to the output of the einsum
      equation.
    equation: The einsum equation `string`.

  Returns:
    A rank-3 `tf.Tensor` whose first dimension is the batch dimension. The
    product of the non-trivial dimensions of the output should be equal to
    the product of the non-trivial dimensions of `output_tensor`.
  """
  match = re.fullmatch(r"([a-zA-Z|.]+),([a-zA-Z|.]+)->([a-zA-Z|.]+)", equation)
  if match is not None:
    s1, s2, s3 = match.groups()
  else:
    raise ValueError(
        "Invalid Einsum eqution string "
        + equation
        + " ."
        "Must be one of the forms {ab,bc->ac}, {...ab,bc->...ac}, "
        "{ab...,bc->ac...}"
    )
  reversed_equation = s3 + "," + s2[::-1] + "->" + s1
  return _reshape_einsum_inputs(output_tensor, reversed_equation)


def _get_einsum_bias_adjoint_reduction_axes(
    equation,
    bias_axes,
    input_rank,
):
  """Computes axes related to the adjoint of the einsum bias broadcast op.

  To describe the output of this computation, first recall that for each
  example the `EinsumDense` layer performs the following transformation:
  ```
  F(W, bias | X) = Einsum(W, X) + Q(bias)
  ```
  where `W` is a tensor of trainable variables, `bias` is a tensor of rank
  `len(bias_axes)`, `X` is a batch of inputs, and `Q` is a linear broadcast
  operator that roughly corresponds to `Q(bias) ~= tf.broadcast_to(bias, S)` for
  `S := tf.shape(Einsum(W, X))`. It is straightforward to show that the
  adjoint of `Q` is given by `Q'(Y) := tf.reduce_sum(Y, axes=R)` where `R`
  contains the broadcasting indices. This function returns `R` as an
  unordered list of `int`s.

  Args:
    equation: The einsum equation `string`.
    bias_axes: A substring of the output part of `equation` specifying which
      axes a bias `tf.Tensor` is added to.
    input_rank: The rank of the tensor that the adjoint operator is being
      applied to.

  Returns:
    A list of `int` containing axes in the `input` corresponding to
    `input_rank`. Each `int` is at most `input_rank-1` and excludes zero.
  """
  reduction_axes = []
  bias_char_set = set(bias_axes)
  eqn_type, (_, _, ac_str) = _parse_einsum_equation(equation)
  # If `equation` of the form `...ab,bc->...ac`, i.e., case (C2), we do a
  # right to left traversal; the other cases do a left to right traversal.
  input_indices = range(input_rank)
  traversal_zip = (
      itertools.zip_longest(reversed(input_indices), reversed(ac_str))
      if eqn_type == EquationType.LEFT_ELLIPSES
      else itertools.zip_longest(input_indices, ac_str)
  )
  # Traverse the output part of `equation` and add an index to the output if
  # the corresponding `char` in the `ac` part is NOT in `bias_axes` and the
  # index is not zero (batch dimension). Add all indices except index zero in
  # the `...` part of the output substring (if present).
  for idx, output_chr in traversal_zip:
    if idx != 0:
      if output_chr is not None and bias_char_set is not None:
        if output_chr not in bias_char_set:
          reduction_axes.append(idx)
        else:
          bias_char_set.remove(output_chr)
      else:
        reduction_axes.append(idx)
  return reduction_axes


def compute_fast_einsum_squared_gradient_norm(
    equation,
    input_tensor,
    grad_tensor,
    bias_axes,
):
  """Computes the batch gradient norms of an Einsum gradient decompostion.

  This logic generalizes the one for `tf.keras.layers.Dense`. For reference,
  we describe part of the mathematical analysis below. It can be safely skipped
  upon first reading of this docstring.

  -----------------------------------------------------------------------------
  BEGIN ANALYSIS
  -----------------------------------------------------------------------------
  Recall that the einsum dense computation for a single example is of the form
  ```
  output = tf.einsum(equation, input, kernel) + bias,
  ```
  where `bias` is broadcasted and summed with the output of the `tf.einsum()`
  call, and equation has one of the following forms:

    (C1) ab,bc->ac,
    (C2) ...ab,bc->...ac
    (C3) ab...,bc->ac...

  Mathematically, the above computation is equivalent to:
  ```
  output = tf.matmul(X, W) + Q(bias)
  ```
  where `X` (resp. `W`) is a 2D tensor reshaped from `input` (resp. `kernel`)
  and `Q` is a linear operator that transforms `bias` to comport with the
  tensor output by the `tf.matmul()` call. When generalizing to a batch of
  examples, `X` is a 3D tensor whose first dimension is the batch dimension.

  Following the same trick as for `tf.keras.layers.Dense` layers, suppose that
  we have:
  ```
  loss = f(base_vars)
  G = tape.gradient(loss, base_vars)
  ```
  Then, using the chain rule and denoting `A'` to be the adjoint of a matrix
  `A`, it is straightforward to show that the gradient of `loss` with respect
  to `W` is given by the block matrix `K := [X' G; Q' G]`. Hence, the square
  norm of `K`, i.e., what is returned by `sqr_norm_fn` is given by
  ```
  sqr_norm = <X X', G G'> + || Q' G ||_F^2
  ```
  where `||.||_F` is the Frobenius norm and `<.,.>` is the Euclidean inner
  product for matrices.
  -----------------------------------------------------------------------------
  END ANALYSIS
  -----------------------------------------------------------------------------

  Args:
    equation: A `string` representing the einsum equation.
    input_tensor: A `tf.Tensor` reprenting the einsum input.
    grad_tensor: A `tf.Tensor` that is the gradient of the scalar loss with
      respect to the pre-activation tensor.
    bias_axes: A `string` that specifies the einsum biases in `equation`.

  Returns:
    A 1D `tf.Tensor` whose i-th entry is the squared gradient corresponding
    to the i-th example in `input_tensor`.
  """
  # NOTE: When the input/gradient tensors are 1D, it is MUCH faster to do
  # a `tf.square()` + `tf.reduce_sum()` than a single `tf.matmul()`.

  # Compute the matrix `X X'` and `G G'` for each example.
  x = _reshape_einsum_inputs(input_tensor, equation)
  g = _reshape_einsum_outputs(grad_tensor, equation)
  if _is_batch_of_vectors(input_tensor) and _is_batch_of_vectors(grad_tensor):
    x_matrix = tf.reshape(x, [x.shape[0], -1])
    batch_xxt = tf.reduce_sum(tf.square(x_matrix), axis=1)
    g_matrix = tf.reshape(g, [g.shape[0], -1])
    batch_ggt = tf.reduce_sum(tf.square(g_matrix), axis=1)
  else:
    batch_xxt = tf.matmul(x, x, transpose_b=True)
    batch_ggt = tf.matmul(g, g, transpose_b=True)
  # Compute the inner product and adjust for bias (if it exists).
  reduction_axes = tf.range(1, len(batch_ggt.shape))
  sqr_norms = tf.reduce_sum(batch_xxt * batch_ggt, axis=reduction_axes)
  if bias_axes is not None:
    # The adjoint operator `Q` on `G` is a reduce sum on the axes in `G` that
    # are not broadcasted from `bias`.
    adjoint_reduction_axes = _get_einsum_bias_adjoint_reduction_axes(
        equation,
        bias_axes,
        len(grad_tensor.shape),
    )
    qg = tf.reduce_sum(grad_tensor, axis=adjoint_reduction_axes)
    qg_reduction_axes = tf.range(1, len(qg.shape))
    bias_sqr_norms = tf.reduce_sum(tf.square(qg), axis=qg_reduction_axes)
    sqr_norms += bias_sqr_norms

  return sqr_norms
