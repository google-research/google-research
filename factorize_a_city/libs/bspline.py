# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""This is a copy of tensorflow_graphics/math/interpolation/bspline.py.

This was included because tensorflow_graphics is built for tensorflow>=2.2
but the source code relies on dependencies from tensorflow<2.0.
"""

import enum
import tensorflow as tf


class Degree(enum.IntEnum):
  """Defines valid degrees for B-spline interpolation."""
  CONSTANT = 0
  LINEAR = 1
  QUADRATIC = 2
  CUBIC = 3
  QUARTIC = 4


def _constant(position):
  """B-Spline basis function of degree 0 for positions in the range [0, 1]."""
  # A piecewise constant spline is discontinuous at the knots.
  return tf.expand_dims(tf.clip_by_value(1.0 + position, 1.0, 1.0), axis=-1)


def _linear(position):
  """B-Spline basis functions of degree 1 for positions in the range [0, 1]."""
  # Piecewise linear splines are C0 smooth.
  return tf.stack((1.0 - position, position), axis=-1)


def _quadratic(position):
  """B-Spline basis functions of degree 2 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  pos_sq = tf.pow(position, 2.0)

  # Piecewise quadratic splines are C1 smooth.
  return tf.stack((tf.pow(1.0 - position, 2.0) / 2.0, -pos_sq + position + 0.5,
                   pos_sq / 2.0),
                  axis=-1)


def _cubic(position):
  """B-Spline basis functions of degree 3 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  neg_pos = 1.0 - position
  pos_sq = tf.pow(position, 2.0)
  pos_cb = tf.pow(position, 3.0)

  # Piecewise cubic splines are C2 smooth.
  return tf.stack(
      (tf.pow(neg_pos, 3.0) / 6.0, (3.0 * pos_cb - 6.0 * pos_sq + 4.0) / 6.0,
       (-3.0 * pos_cb + 3.0 * pos_sq + 3.0 * position + 1.0) / 6.0,
       pos_cb / 6.0),
      axis=-1)


def _quartic(position):
  """B-Spline basis functions of degree 4 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  neg_pos = 1.0 - position
  pos_sq = tf.pow(position, 2.0)
  pos_cb = tf.pow(position, 3.0)
  pos_qt = tf.pow(position, 4.0)

  # Piecewise quartic splines are C3 smooth.
  return tf.stack(
      (tf.pow(neg_pos, 4.0) / 24.0,
       (-4.0 * tf.pow(neg_pos, 4.0) + 4.0 * tf.pow(neg_pos, 3.0) +
        6.0 * tf.pow(neg_pos, 2.0) + 4.0 * neg_pos + 1.0) / 24.0,
       (pos_qt - 2.0 * pos_cb - pos_sq + 2.0 * position) / 4.0 + 11.0 / 24.0,
       (-4.0 * pos_qt + 4.0 * pos_cb + 6.0 * pos_sq + 4.0 * position + 1.0) /
       24.0, pos_qt / 24.0),
      axis=-1)


def knot_weights(positions,
                 num_knots,
                 degree,
                 cyclical,
                 sparse_mode=False,
                 name=None):
  """Function that converts cardinal B-spline positions to knot weights.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    positions: A tensor with shape `[A1, .. An]`. Positions must be between `[0,
      C - D)` for non-cyclical and `[0, C)` for cyclical splines, where `C` is
      the number of knots and `D` is the spline degree.
    num_knots: A strictly positive `int` describing the number of knots in the
      spline.
    degree: An `int` describing the degree of the spline, which must be smaller
      than `num_knots`.
    cyclical: A `bool` describing whether the spline is cyclical.
    sparse_mode: A `bool` describing whether to return a result only for the
      knots with nonzero weights. If set to True, the function returns the
      weights of only the `degree` + 1 knots that are non-zero, as well as the
      indices of the knots.
    name: A name for this op. Defaults to "bspline_knot_weights".

  Returns:
    A tensor with dense weights for each control point, with the shape
    `[A1, ... An, C]` if `sparse_mode` is False.
    Otherwise, returns a tensor of shape `[A1, ... An, D + 1]` that contains the
    non-zero weights, and a tensor with the indices of the knots, with the type
    tf.int32.

  Raises:
    ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
    InvalidArgumentError: If positions are not in the right range.
  """
  with tf.compat.v1.name_scope(name, "bspline_knot_weights", [positions]):
    positions = tf.convert_to_tensor(value=positions)

    if degree > 4 or degree < 0:
      raise ValueError("Degree should be between 0 and 4.")
    if degree > num_knots - 1:
      raise ValueError("Degree cannot be >= number of knots.")

    all_basis_functions = {
        # Maps valid degrees to functions.
        Degree.CONSTANT: _constant,
        Degree.LINEAR: _linear,
        Degree.QUADRATIC: _quadratic,
        Degree.CUBIC: _cubic,
        Degree.QUARTIC: _quartic
    }
    basis_functions = all_basis_functions[degree]

    if not cyclical and num_knots - degree == 1:
      # In this case all weights are non-zero and we can just return them.
      if not sparse_mode:
        return basis_functions(positions)
      else:
        shift = tf.zeros_like(positions, dtype=tf.int32)
        return basis_functions(positions), shift

    shape_batch = tf.shape(input=positions)
    positions = tf.reshape(positions, shape=(-1,))

    # Calculate the nonzero weights from the decimal parts of positions.
    shift = tf.floor(positions)
    sparse_weights = basis_functions(positions - shift)
    shift = tf.cast(shift, tf.int32)

    if sparse_mode:
      # Returns just the weights and the shift amounts, so that tf.gather_nd on
      # the knots can be used to sparsely activate knots if needed.
      shape_weights = tf.concat(
          (shape_batch, tf.constant((degree + 1,), dtype=tf.int32)), axis=0)
      sparse_weights = tf.reshape(sparse_weights, shape=shape_weights)
      shift = tf.reshape(shift, shape=shape_batch)
      return sparse_weights, shift

    num_positions = tf.size(input=positions)
    ind_row, ind_col = tf.meshgrid(
        tf.range(num_positions, dtype=tf.int32),
        tf.range(degree + 1, dtype=tf.int32),
        indexing="ij")

    tiled_shifts = tf.reshape(
        tf.tile(tf.expand_dims(shift, axis=-1), multiples=(1, degree + 1)),
        shape=(-1,))
    ind_col = tf.reshape(ind_col, shape=(-1,)) + tiled_shifts
    if cyclical:
      ind_col = tf.math.mod(ind_col, num_knots)
    indices = tf.stack((tf.reshape(ind_row, shape=(-1,)), ind_col), axis=-1)
    shape_indices = tf.concat((tf.reshape(
        num_positions, shape=(1,)), tf.constant(
            (degree + 1, 2), dtype=tf.int32)),
                              axis=0)
    indices = tf.reshape(indices, shape=shape_indices)
    shape_scatter = tf.concat((tf.reshape(
        num_positions, shape=(1,)), tf.constant((num_knots,), dtype=tf.int32)),
                              axis=0)
    weights = tf.scatter_nd(indices, sparse_weights, shape_scatter)
    shape_weights = tf.concat(
        (shape_batch, tf.constant((num_knots,), dtype=tf.int32)), axis=0)
    return tf.reshape(weights, shape=shape_weights)


def interpolate_with_weights(knots, weights, name=None):
  """Interpolates knots using knot weights.

  Note:
    In the following, A1 to An, and B1 to Bk are optional batch dimensions.

  Args:
    knots: A tensor with shape `[B1, ..., Bk, C]` containing knot values, where
      `C` is the number of knots.
    weights: A tensor with shape `[A1, ..., An, C]` containing dense weights for
      the knots, where `C` is the number of knots.
    name: A name for this op. Defaults to "bspline_interpolate_with_weights".

  Returns:
    A tensor with shape `[A1, ..., An, B1, ..., Bk]`, which is the result of
    spline interpolation.

  Raises:
    ValueError: If the last dimension of knots and weights is not equal.
  """
  with tf.compat.v1.name_scope(name, "bspline_interpolate_with_weights",
                               [knots, weights]):
    knots = tf.convert_to_tensor(value=knots)
    weights = tf.convert_to_tensor(value=weights)

  return tf.tensordot(weights, knots, (-1, -1))


def interpolate(knots, positions, degree, cyclical, name=None):
  """Applies B-spline interpolation to input control points (knots).

  Note:
    In the following, A1 to An, and B1 to Bk are optional batch dimensions.

  Args:
    knots: A tensor with shape `[B1, ..., Bk, C]` containing knot values, where
      `C` is the number of knots.
    positions: Tensor with shape `[A1, .. An]`. Positions must be between `[0, C
      - D)` for non-cyclical and `[0, C)` for cyclical splines, where `C` is the
      number of knots and `D` is the spline degree.
    degree: An `int` between 0 and 4, or an enumerated constant from the Degree
      class, which is the degree of the splines.
    cyclical: A `bool`, whether the splines are cyclical.
    name: A name for this op. Defaults to "bspline_interpolate".

  Returns:
    A tensor of shape `[A1, ... An, B1, ..., Bk]`, which is the result of spline
    interpolation.
  """
  with tf.compat.v1.name_scope(name, "bspline_interpolate", [knots, positions]):
    knots = tf.convert_to_tensor(value=knots)
    positions = tf.convert_to_tensor(value=positions)

    num_knots = knots.get_shape().as_list()[-1]
    weights = knot_weights(positions, num_knots, degree, cyclical, False, name)
    return interpolate_with_weights(knots, weights)
