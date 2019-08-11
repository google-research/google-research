# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""A set of TensorFlow functions for motion-warping depth maps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from depth_from_video_in_the_wild import transform_utils


class TransformedDepthMap(object):
  """A collection of tensors that described a transformed depth map.

  This class describes the result of a spatial transformation applied on a depth
  map. The initial depth map was defined on a regular pixel grid. Knowing the
  camera intrinsics, each pixel can be mapped to a point in space.

  However once the camera or the scene has moved, when the points are projected
  back onto the camera, they don't fall on a regular pixel grid anymore. To
  obtain a new depth map on a regular pixel grid, one needs to resample, taking
  into account occlusions, and leaving gaps at areas that were occluded before
  the movement.

  This class describes the transformed depth map on an IRREGULAR grid, before
  any resampling. The attributes are 4 tensors of shape [B, H, W]
  (batch, height, width): pixel_x, pixel_y, depth and mask.

  The given a triplet of indices, (b, i, j), the depth at the pixel location
  (pixel_y[b, i, j], pixel_x[b, i, j]) on the depth image is depth[b, i, j].
  As explained above, (pixel_y[b, i, j], pixel_x[b, i, j]) are not regular with
  respect to i and j. They are floating point numbers that generally fall in
  between pixels and can fall out of image boundaries (0, 0), (H - 1, W - 1).
  For all indices b, i, j where 0 <= pixel_y[b, i, j] <= H - 1 and
  0 <= pixel_x[b, i, j] < W - 1, mask[b, i, j] is True, otherwise it's False.

  For convenience, after we mark mask[b, i, j] as false for
  (pixel_y[b, i, j], pixel_x[b, i, j]) that are out of bounds, we clamp
  (pixel_y[b, i, j], pixel_x[b, i, j]) to be within the bounds. So, you're not
  supposed to look at (pixel_y[b, i, j], pixel_x[b, i, j], depth[b, i, j]) where
  mask[b, i, j] is False, but if you do, you'll find that they were clamped
  to be within the bounds. The motivation for this is that if we later use
  pixel_x and pixel_y for warping, the clamping will result in extrapolating
  from the boundary by replicating the boundary value, which is reasonable.
  """

  def __init__(self, pixel_x, pixel_y, depth, mask):
    """Initializes an instance. The arguments is explained above."""
    self._pixel_x = pixel_x
    self._pixel_y = pixel_y
    self._depth = depth
    self._mask = mask
    attrs = sorted(self.__dict__.keys())
    # Unlike equality, compatibility is not transitive, so we have to check all
    # pairs.
    for i in range(len(attrs)):
      for j in range(i):
        tensor_i = self.__dict__[attrs[i]]
        tensor_j = self.__dict__[attrs[j]]
        if not tensor_i.shape.is_compatible_with(tensor_j.shape):
          raise ValueError(
              'All tensors in TransformedDepthMap\'s constructor must have '
              'compatible shapes, however \'%s\' and \'%s\' have the '
              'incompatible shapes %s and %s.' %
              (attrs[i][1:], attrs[j][1:], tensor_i.shape, tensor_j.shape))
    self._pixel_xy = None

  @property
  def pixel_x(self):
    return self._pixel_x

  @property
  def pixel_y(self):
    return self._pixel_y

  @property
  def depth(self):
    return self._depth

  @property
  def mask(self):
    return self._mask

  @property
  def pixel_xy(self):
    if self._pixel_xy is None:
      name = self._pixel_x.op.name.rsplit('/', 1)[0]
      self._pixel_xy = tf.stack([self._pixel_x, self._pixel_y],
                                axis=3,
                                name='%s/pixel_xy' % name)
    return self._pixel_xy


def using_motion_vector(depth,
                        translation,
                        rotation_angles,
                        intrinsic_mat,
                        distortion_coeff=None,
                        name=None):
  """Transforms a depth map using a motion vector, or a motion vector field.

  This function receives a translation vector and rotation angles vector. They
  can be the same for the entire image, or different for each pixel.

  Args:
    depth: A tf.Tensor of shape [B, H, W]
    translation: A tf.Tensor of shape [B, 3] or [B, H, W, 3] representing a
      translation vector for the entire image or for every pixel respectively.
    rotation_angles: A tf.Tensor of shape [B, 3] or [B, H, W, 3] representing a
      set of rotation angles for the entire image or for every pixel
      respectively. We conform to the same convention as in inverse_warp above,
      but may need to reconsider, depending on the conventions tf.graphics and
      other users will converge to.
    intrinsic_mat: A tf.Tensor of shape [B, 3, 3].
    distortion_coeff: A scalar (python or tf.Tensor) of a floating point type,
      or None, the quadratic radial distortion coefficient. If 0.0 or None, a
      distortion-less implementation (which is simpler and maybe faster) will be
      used.
    name: A string or None, a name scope for the ops.

  Returns:
    A TransformedDepthMap object.
  """
  with tf.name_scope(
      name, 'TransformDepthMapUsingMotionVector',
      [depth, translation, rotation_angles, intrinsic_mat, distortion_coeff]):
    if distortion_coeff is not None and distortion_coeff != 0.0:
      pixel_x, pixel_y, z = _using_motion_vector_with_distortion(
          depth, translation, rotation_angles, intrinsic_mat, distortion_coeff)
    else:
      pixel_x, pixel_y, z = _using_motion_vector(
          depth, translation, rotation_angles, intrinsic_mat)
    pixel_x, pixel_y, mask = _clamp_and_filter_result(pixel_x, pixel_y, z)
    return TransformedDepthMap(pixel_x, pixel_y, z, mask)


def _using_motion_vector(depth, translation, rotation_angles, intrinsic_mat):
  """A helper for using_motion_vector. See docstring therein."""

  if translation.shape.ndims not in (2, 4):
    raise ValueError('\'translation\' should have rank 2 or 4, not %d' %
                     translation.shape.ndims)
  if translation.shape[-1] != 3:
    raise ValueError('translation\'s last dimension should be 3, not %d' %
                     translation.shape[1])
  if translation.shape.ndims == 2:
    translation = tf.expand_dims(tf.expand_dims(translation, 1), 1)

  _, height, width = tf.unstack(tf.shape(depth))
  grid = tf.squeeze(
      tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)
  grid = tf.to_float(grid)
  intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)

  rot_mat = transform_utils.matrix_from_angles(rotation_angles)
  # We have to treat separately the case of a per-image rotation vector and a
  # per-image rotation field, because the broadcasting capabilities of einsum
  # are limited.
  if rotation_angles.shape.ndims == 2:
    # The calculation here is identical to the one in inverse_warp above.
    # Howeverwe use einsum for better clarity. Under the hood, einsum performs
    # the reshaping and invocation of BatchMatMul, instead of doing it manually,
    # as in inverse_warp.
    projected_rotation = tf.einsum('bij,bjk,bkl->bil', intrinsic_mat, rot_mat,
                                   intrinsic_mat_inv)
    pcoords = tf.einsum('bij,jhw,bhw->bihw', projected_rotation, grid, depth)
  elif rotation_angles.shape.ndims == 4:
    # We push the H and W dimensions to the end, and transpose the rotation
    # matrix elements (as noted above).
    rot_mat = tf.transpose(rot_mat, [0, 3, 4, 1, 2])
    projected_rotation = tf.einsum('bij,bjkhw,bkl->bilhw', intrinsic_mat,
                                   rot_mat, intrinsic_mat_inv)
    pcoords = tf.einsum('bijhw,jhw,bhw->bihw', projected_rotation, grid, depth)

  projected_translation = tf.einsum('bij,bhwj->bihw', intrinsic_mat,
                                    translation)
  pcoords += projected_translation
  x, y, z = tf.unstack(pcoords, axis=1)
  return x / z, y / z, z


def _using_motion_vector_with_distortion(depth,
                                         translation,
                                         rotation_angles,
                                         intrinsic_mat,
                                         distortion_coeff=0.0):
  """A helper for using_motion_vector. See docstring therein."""

  if translation.shape.ndims not in (2, 4):
    raise ValueError('\'translation\' should have rank 2 or 4, not %d' %
                     translation.shape.ndims)
  if translation.shape[-1] != 3:
    raise ValueError('translation\'s last dimension should be 3, not %d' %
                     translation.shape[1])
  if translation.shape.ndims == 2:
    translation = tf.expand_dims(tf.expand_dims(translation, 1), 1)

  _, height, width = tf.unstack(tf.shape(depth))
  grid = tf.squeeze(
      tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))),
      axis=3)  # 3 x height x width
  grid = tf.to_float(grid)
  intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)

  normalized_grid = tf.einsum('bij,jhw->bihw', intrinsic_mat_inv, grid)
  radii_squared = tf.reduce_sum(tf.square(normalized_grid[:, :2, :, :]), axis=1)

  undistortion_factor = quadratic_inverse_distortion_scale(
      distortion_coeff, radii_squared)
  undistortion_factor = tf.stack([
      undistortion_factor, undistortion_factor,
      tf.ones_like(undistortion_factor)
  ],
                                 axis=1)
  normalized_grid *= undistortion_factor

  rot_mat = transform_utils.matrix_from_angles(rotation_angles)
  # We have to treat separately the case of a per-image rotation vector and a
  # per-image rotation field, because the broadcasting capabilities of einsum
  # are limited.
  if rotation_angles.shape.ndims == 2:
    # The calculation here is identical to the one in inverse_warp above.
    # Howeverwe use einsum for better clarity. Under the hood, einsum performs
    # the reshaping and invocation of BatchMatMul, instead of doing it manually,
    # as in inverse_warp.
    pcoords = tf.einsum('bij,bjhw,bhw->bihw', rot_mat, normalized_grid, depth)
  elif rotation_angles.shape.ndims == 4:
    # We push the H and W dimensions to the end, and transpose the rotation
    # matrix elements (as noted above).
    rot_mat = tf.transpose(rot_mat, [0, 3, 4, 1, 2])
    pcoords = tf.einsum('bijhw,bjhw,bhw->bihw', rot_mat, normalized_grid, depth)

  pcoords += tf.transpose(translation, [0, 3, 1, 2])

  x, y, z = tf.unstack(pcoords, axis=1)
  x /= z
  y /= z
  scale = quadraric_distortion_scale(distortion_coeff,
                                     tf.square(x) + tf.square(y))
  x *= scale
  y *= scale

  pcoords = tf.einsum('bij,bjhw->bihw', intrinsic_mat,
                      tf.stack([x, y, tf.ones_like(x)], axis=1))
  x, y, _ = tf.unstack(pcoords, axis=1)

  return x, y, z


def _clamp_and_filter_result(pixel_x, pixel_y, z):
  """Clamps and masks out out-of-bounds pixel coordinates.

  Args:
    pixel_x: a tf.Tensor containing x pixel coordinates in an image.
    pixel_y: a tf.Tensor containing y pixel coordinates in an image.
    z: a tf.Tensor containing the depth ar each (pixel_y, pixel_x)  All shapes
      are [B, H, W].

  Returns:
    pixel_x, pixel_y, mask, where pixel_x and pixel_y are the original ones,
    except:
    - Values that fall out of the image bounds, which are [0, W-1) in x and
      [0, H-1) in y, are clamped to the bounds
    - NaN values in pixel_x, pixel_y are replaced by zeros
    mask is False at allpoints where:
    - Clamping in pixel_x or pixel_y was performed
    - NaNs were replaced by zeros
    - z is non-positive,
    and True everywhere else, that is, where pixel_x, pixel_y are finite and
    fall within the frame.
  """
  with tf.name_scope('Clamp', values=[pixel_x, pixel_y, z]):
    _, height, width = tf.unstack(tf.shape(pixel_x))

    def _tensor(x):
      return tf.to_float(tf.convert_to_tensor(x))

    x_not_underflow = pixel_x >= 0.0
    y_not_underflow = pixel_y >= 0.0
    x_not_overflow = pixel_x < _tensor(width - 1)
    y_not_overflow = pixel_y < _tensor(height - 1)
    z_positive = z > 0.0
    x_not_nan = tf.math.logical_not(tf.is_nan(pixel_x))
    y_not_nan = tf.math.logical_not(tf.is_nan(pixel_y))
    not_nan = tf.logical_and(x_not_nan, y_not_nan)
    not_nan_mask = tf.to_float(not_nan)
    pixel_x *= not_nan_mask
    pixel_y *= not_nan_mask
    pixel_x = tf.clip_by_value(pixel_x, 0.0, _tensor(width - 1))
    pixel_y = tf.clip_by_value(pixel_y, 0.0, _tensor(height - 1))
    mask_stack = tf.stack([
        x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow,
        z_positive, not_nan
    ],
                          axis=0)
    mask = tf.reduce_all(mask_stack, axis=0)
    return pixel_x, pixel_y, mask


def quadraric_distortion_scale(distortion_coefficient, r_squared):
  """Calculates a quadratic distortion factor given squared radii.

  The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
  `distortion_coefficient` is negative (barrel distortion), the distorted radius
  is only monotonically increasing only when
  `r_squared` < r_squared_max = -1 / (3 * distortion_coefficient).

  Args:
    distortion_coefficient: A tf.Tensor of a floating point type. The rank can
      be from zero (scalar) to r_squared's rank. The shape of
      distortion_coefficient will be appended by ones until the rank equals that
      of r_squared.
    r_squared: A tf.Tensor of a floating point type, containing
      (x/z)^2 + (y/z)^2. We use r_squared rather than r to avoid an unnecessary
      sqrt, which may introduce gradient singularities. The non-negativity of
      r_squared only enforced in debug mode.

  Returns:
    A tf.Tensor of r_squared's shape, the correction factor that should
    multiply the projective coordinates (x/z) and (y/z) to apply the
    distortion.
  """
  return 1 + distortion_coefficient * r_squared


def quadratic_inverse_distortion_scale(distortion_coefficient,
                                       distorted_r_squared,
                                       newton_iterations=4):
  """Calculates the inverse quadratic distortion function given squared radii.

  The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
  `distortion_coefficient` is negative (barrel distortion), the distorted radius
  is monotonically increasing only when
  r < r_max = sqrt(-1 / (3 * distortion_coefficient)).
  max_distorted_r_squared is obtained by calculating the distorted_r_squared
  corresponding to r = r_max, and the result is
  max_distorted_r_squared = - 4 / (27.0 * distortion_coefficient)

  Args:
    distortion_coefficient: A tf.Tensor of a floating point type. The rank can
      be from zero (scalar) to r_squared's rank. The shape of
      distortion_coefficient will be appended by ones until the rank equals that
      of r_squared.
    distorted_r_squared: A tf.Tensor of a floating point type, containing
      (x/z)^2 + (y/z)^2. We use distorted_r_squared rather than distorted_r to
      avoid an unnecessary sqrt, which may introduce gradient singularities.
      The non-negativity of distorted_r_squared is only enforced in debug mode.
    newton_iterations: Number of Newton-Raphson iterations to calculate the
      inverse distprtion function. Defaults to 5, which is on the high-accuracy
      side.

  Returns:
    A tf.Tensor of distorted_r_squared's shape, containing the correction
    factor that should multiply the distorted the projective coordinates (x/z)
    and (y/z) to obtain the undistorted ones.
  """
  c = 1.0  # c for Correction
  # Newton-Raphson iterations for solving the inverse function of the
  # distortion.
  for _ in range(newton_iterations):
    c = (1.0 -
         (2.0 / 3.0) * c) / (1.0 + 3 * distortion_coefficient *
                             distorted_r_squared * c * c) + (2.0 / 3.0) * c
  return c


def _expand_last_dim_twice(x):
  return tf.expand_dims(tf.expand_dims(x, -1), -1)
