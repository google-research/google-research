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

"""Util functions for manipulating camera geometry.

* Where possible, utility functions operate on the last one or two
  dimensions of their inputs, and will function irrespective of how many
  preceding dimensions are present. Where it makes sense, functions support
  broadcasting on the part of the shape preceding the fixed dimensions.
  This is to allow preceding dimensions to freely be used for batching or
  other purposes.

* Camera poses are representated as 3x4 matrices (consisting of a 3x3 rotation
  matrix and a 3-coordinate translation vector):
    [[ r r r tx ]
     [ r r r ty ]
     [ r r r tz ]]
  The matrix maps a position in world-space into a position relative to the
  camera position. (Conventionally, the camera position has the Z axis pointing
  into the screen and the Y axis pointing down.) Functions to manipulate such
  matrices have names beginning "mat34_".

* Camera intrinsics are represented as a tensor of last dimension 4. The four
  elements are fx, fy (focal length) and cx, cy (principal point). Intrinsics
  are independent of image-size, they are expressed as if the image runs from
  (0,0) to (1,1). So typically cx == cy == 0.5, and for a 90-degree field of
  view, fx == 0.5.

* Points (whether 2D or 3D) are represented using the last axis of a tensor.
  A set of N 3D points would have shape [N, 3].

* We use texture coordinates to represent points in an image. They go from (0,0)
  in the top-left corner of an image to (1,1) in the bottom right. It is
  convenient to work with these coordinates rather than counts of pixels,
  because they are resolution-independent.
"""
import tensorflow as tf


def broadcast_to_match(a, b, ignore_axes=0):
  """Returns (a', b') which are the inputs broadcast up to have the same shape.

  Suppose you want to apply an operation to tensors a and b but it doesn't
  support broadcasting. As an example maybe we have tensors of these shapes:
    a    [5, 1, 3, 4]
    b [2, 1, 8, 4, 2]
  Considering the last two dimensions as matrices, we may want to multiply
  a by b to get a tensor [2, 5, 8, 3, 2] of (2x3) matrices. However, tf.matmul
  doesn't support this because the outer dimensions don't match. Calling
  tf.matmul(a, b) directly will fail.

  However, the dimensions do match under broadcasting, so we can do the
  multiplication like this:
    a, b = broadcast_to_match(a, b, ignore_axes=2)
    c = tf.matmul(a, b)
  The ignore_axes parameter tells us to ignore the last two dimensions of a
  and b and just make the rest match.

  Args:
    a: Any shape
    b: Any shape
    ignore_axes: If present, broadcasting will not apply to the final this many
      axes. For example, if you are planning to call tf.matmul(a, b) on the
      result, then set ignore_axes=2 because tf.matmul operates on the last two
      axes, only the rest need to match. To ignore a different number of axes
      for inputs a and b, pass a pair of number to ignore_axes.

  Returns:
    a', b': Identical to the two inputs except tiled so that the shapes
        match. See https://www.tensorflow.org/performance/xla/broadcasting.
        If the shapes already match, no tensorflow graph operations are added,
        so this is cheap.
  """
  a = tf.convert_to_tensor(a)
  b = tf.convert_to_tensor(b)
  a_shape = a.shape.as_list()
  b_shape = b.shape.as_list()
  # Extract the part of the shape that is required to match.
  if isinstance(ignore_axes, tuple) or isinstance(ignore_axes, list):
    ignore_a = ignore_axes[0]
    ignore_b = ignore_axes[1]
  else:
    ignore_a = ignore_axes
    ignore_b = ignore_axes
  if ignore_a:
    a_shape = a_shape[:-ignore_a]
  if ignore_b:
    b_shape = b_shape[:-ignore_b]
  if a_shape == b_shape:
    return (a, b)
  # Addition supports broadcasting, so add a tensor of zeroes.
  za = tf.zeros(a_shape + [1] * ignore_b, dtype=b.dtype)
  zb = tf.zeros(b_shape + [1] * ignore_a, dtype=a.dtype)
  a += zb
  b += za

  a_new_shape = a.shape.as_list()
  b_new_shape = b.shape.as_list()
  if ignore_a:
    a_new_shape = a_new_shape[:-ignore_a]
  if ignore_b:
    b_new_shape = b_new_shape[:-ignore_b]
  assert a_new_shape == b_new_shape
  return (a, b)


def mat34_to_mat44(matrix):
  """Converts 3x4 matrices to 4x4 matrices by adding filler.

  Considering the last two dimensions of the input tensor, where m
  indicates a matrix coefficient and t a matrix coefficient for translation,
  this function does the following:
       [[m, m, m, t],           [[m, m, m, t],
        [m, m, m, t],    ===>    [m, m, m, t],
        [m, m, m, t]]            [m, m, m, t],
                                 [0, 0, 0, 1]]

  Args:
    matrix: [..., 3, 4] matrix

  Returns:
    A [..., 4, 4] tensor with an extra row [0, 0, 0, 1] added to each matrix.
    Dimensions other than that last two are the same as for the input.
  """
  shape = matrix.shape.as_list()

  extra_dims = shape[:-2]
  filler = tf.constant([0.0, 0.0, 0.0, 1.0],
                       shape=len(extra_dims) * [1] + [1, 4])
  filler = tf.tile(filler, extra_dims + [1, 1])
  return tf.concat([matrix, filler], axis=-2)


def mat34_transform(m, v):
  """Transform a set of 3d points by a 3x4 pose matrix.

  Args:
    m: [..., 3, 4] matrix
    v: [..., N, 3] set of N 3d points.

  Returns:
    The transformed points mv. The transform is computed as if we added an
    extra coefficient with value 1.0 to each point, performed a matrix
    multiplication, and removed the extra coefficient again. The parts of the
    shape indicated by "..." must match, either directly or via broadcasting.

  Raises:
    ValueError: if inputs are the wrong shape.
  """
  (m, v) = broadcast_to_match(m, v, ignore_axes=2)
  rotation = m[Ellipsis, :3]

  translation = m[Ellipsis, 3]
  translation = translation[Ellipsis, tf.newaxis, :]  # Now shape is [..., 1, 3].
  # Points are stored as (N * 3) rather than (3 * N), so multiply in reverse
  # rather than transposing them.
  return tf.matmul(v, rotation, transpose_b=True) + translation


def mat34_product(a, b):
  """Returns the product of a and b, 3x4 matrices.

  Args:
    a: [..., 3, 4] matrix
    b: [..., 3, 4] matrix

  Returns:
    The product ab. The product is computed as if we added an extra row
    [0, 0, 0, 1] to each matrix, multiplied them, and then removed the extra
    row. The shapes of a and b must match, either directly or via
    broadcasting.

  Raises:
    ValueError: if a or b are not 3x4 matrices.
  """

  (a, b) = broadcast_to_match(a, b, ignore_axes=2)
  # Split translation part off from the rest
  a33, a_translate = tf.split(a, [3, 1], axis=-1)
  b33, b_translate = tf.split(b, [3, 1], axis=-1)
  # Compute parts of the product
  ab33 = tf.matmul(a33, b33)
  ab_translate = a_translate + tf.matmul(a33, b_translate)
  # Assemble
  return tf.concat([ab33, ab_translate], axis=-1)


def mat34_pose_inverse(matrix):
  """Invert a 3x4 matrix.

  Args:
    matrix: [..., 3, 4] matrix where [..., 3, 3] is a rotation matrix

  Returns:
    The inverse matrix, of the same shape as the input. It is computed as
    if we added an extra row with values [0, 0, 0, 1], inverted the
    matrix, and removed the row again.
  """
  rest, translation = tf.split(matrix, [3, 1], axis=-1)
  inverse = tf.linalg.matrix_transpose(rest)
  inverse_translation = -tf.matmul(inverse, translation)
  return tf.concat([inverse, inverse_translation], axis=-1)


def homogenize(coords):
  """Convert (x, y) to (x, y, 1), or (x, y, z) to (x, y, z, 1)."""
  ones = tf.ones_like(coords[Ellipsis, :1])
  return tf.concat([coords, ones], axis=-1)


def texture_to_camera_coordinates(coords, intrinsics):
  """Convert texture coordinates to x,y,1 coordinates relative to camera.

  Args:
    coords: [..., 2] texture coordinates
    intrinsics: [..., 4] (resolution-independent) camera intrinsics. Last
      dimension (fx, fy, cx, cy).

  Returns:
    [..., 3] coordinates, transformed by scaling down by image size and
    applying the inverse of the intrinsics. z-coordinates are all 1.
  """
  # Shift to optical center and divide by focal length.
  # (These are element-wise operations on the x and y coords.)
  focal_length, optical_center = tf.split(intrinsics, [2, 2], axis=-1)
  xy_coords = (coords - optical_center) / focal_length
  return homogenize(xy_coords)
