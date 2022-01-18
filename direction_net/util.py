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

"""Utilities for math and geometry operations."""
import math

from pano_utils import geometry
from pano_utils import math_utils
from pano_utils import transformation
import tensorflow.compat.v1 as tf
from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
import tensorflow_probability as tfp


def safe_sqrt(x):
  return tf.sqrt(tf.maximum(x, 1e-20))


def degrees_to_radians(degree):
  """Convert degrees to radians."""
  return math.pi * degree / 180.0


def radians_to_degrees(radians):
  """Convert radians to degrees."""
  return 180.0 * radians / math.pi


def angular_distance(v1, v2):
  dot = tf.reduce_sum(v1 * v2, -1)
  return tf.acos(tf.clip_by_value(dot, -1., 1.))


def equirectangular_area_weights(height):
  """Generate area weights for pixels in equirectangular images.

  This is to account for the area difference of pixels at different latitudes on
  equirectangular grids.

  Args:
    height: the height dimension of the equirectangular images.

  Returns:
    Area weighted with shape [1, HEIGHT, 1, 1].
  """
  with tf.name_scope(None, 'equirectangular_area_weights', [height]):
    pixel_h = math.pi / tf.cast(height, tf.float32)
    # Use half-integer pixel centre convention, and generate the spherical
    # coordinates for the centres of the pixels.
    colatitude = tf.lin_space(pixel_h / 2, math.pi - pixel_h / 2, height)
    colatitude = colatitude[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.sin(colatitude)


def spherical_normalization(x, rectify=True):
  """Apply area weights and normalization to spherical distributions.

  The sum of all pixel values over the spherical input will be one.

  Args:
    x: [BATCH, HEIGHT, WIDTH, CHANNELS] spherical raw distributions.
    rectify: apply softplus to the input x if true.

  Returns:
    [BATCH, HEIGHT, WIDTH, CHANNELS] normalized distributions.
  """
  with tf.name_scope(None, 'spherical_normalization', [x]):
    # Apply softplus to make the input non-negative.
    shape = x.shape.as_list()
    height = shape[1]
    if rectify:
      x = tf.nn.softplus(x)
    weighted = x * equirectangular_area_weights(height)
    # Return shape [BATCH, HEIGHT, WIDTH, CHANNELS].
    return tf.div_no_nan(x, tf.reduce_sum(weighted, axis=[1, 2], keepdims=True))


def spherical_expectation(spherical_probabilities):
  """Compute the expectation (a vector) from normalized spherical distribtuions.

  We define the spherical expectation as the integral of r*P(r)*dr where r is
  a unit direction vector in 2-sphere. We compute the discretized version on a
  spherical equirectangular map. To correctly use this function, the input has
  to be normalized properly using spherical_normalization().

  Args:
    spherical_probabilities: [BATCH, HEIGHT, WIDTH, N] spherical distributions
      in equirectangular form.

  Returns:
     expectation [BATCH, N, 3]
  """
  shape = spherical_probabilities.shape.as_list()
  height, width, channels = shape[1], shape[2], shape[3]
  spherical = tf.expand_dims(
      geometry.generate_equirectangular_grid([height, width]), 0)
  unit_directions = geometry.spherical_to_cartesian(spherical)
  axis_convert = tf.constant([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
  unit_directions = tf.squeeze(tf.matmul(
      axis_convert, tf.expand_dims(unit_directions, -1), transpose_a=True), -1)
  unit_directions = tf.tile(
      tf.expand_dims(unit_directions, -2), [1, 1, 1, channels, 1])
  weighted = spherical_probabilities * equirectangular_area_weights(height)
  expectation = tf.reduce_sum(
      unit_directions * tf.expand_dims(weighted, -1), [1, 2])
  return expectation


def von_mises_fisher(mean, concentration, shape):
  """Generate von Mises-Fisher distribution on spheres.

  This function samples probabilities from tensorflow_probability.VonMisesFisher
  on equirectangular grids of a sphere. The height dimension of the output
  ranges from pi/2 (top) to -pi/2 (bottom). The width dimension ranges from
  0 (left) to 2*pi (right).

  Args:
    mean: [BATCH, N, 3] a float tensor representing the unit direction of
      the mean.
    concentration: (float) a measure of concentration (a reciprocal measure of
      dispersion, so 1/kappa  is analogous to variance). concentration=0
      indicates a uniform distribution over the unit sphere,
      and concentration=+inf indicates a delta function at the mean direction.
    shape: a 2-d list represents the dimension (height, width) of the output.

  Returns:
    A 4-D tensor [BATCH, HEIGHT, WIDTH, N] represents the raw probabilities
    of the distribution. (surface integral != 1)

  Raises:
    ValueError: Input argument 'shape' is not valid.
    ValueError: Input argument 'mean' has wrong dimensions.
  """
  with tf.name_scope(None, 'von_mises_fisher', [mean, concentration, shape]):
    if not isinstance(shape, list) or len(shape) != 2:
      raise ValueError("Input argument 'shape' is not valid.")
    if mean.shape[-1] != 3:
      raise ValueError("Input argument 'mean' has wrong dimensions.")

    batch, channels = mean.shape[0], mean.shape[1]
    height, width = shape[0], shape[1]
    spherical_grid = geometry.generate_equirectangular_grid(shape)
    cartesian = geometry.spherical_to_cartesian(spherical_grid)
    axis_convert = tf.constant([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    cartesian = tf.squeeze(tf.matmul(
        axis_convert, tf.expand_dims(cartesian, -1), transpose_a=True), -1)

    cartesian = tf.tile(
        cartesian[tf.newaxis, tf.newaxis, :],
        [batch, channels, 1, 1, 1])
    mean = tf.tile(mean[:, :, tf.newaxis, tf.newaxis], [1, 1, height, width, 1])
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean, concentration=[concentration])
    spherical_gaussian = vmf.prob(cartesian)
    return tf.transpose(spherical_gaussian, [0, 2, 3, 1])


def rotation_geodesic(r1, r2):
  """Return the geodesic distance (angle in radians) between two rotations.

  Args:
    r1: [BATCH, 3, 3] rotation matrices.
    r2: [BATCH, 3, 3] rotation matrices.

  Returns:
    [BATCH] radian angular difference between rotation matrices.
  """
  diff = (tf.trace(tf.matmul(r1, r2, transpose_b=True)) - 1) / 2
  angular_diff = tf.acos(tf.clip_by_value(diff, -1., 1.))
  return angular_diff


def gram_schmidt(m):
  """Convert 6D representation to SO(3) using a partial Gram-Cchmidt process.

  Args:
    m: [BATCH, 2, 3] 2x3 matrices.

  Returns:
    [BATCH, 3, 3] SO(3) rotation matrices.
  """
  x = m[:, 0]
  y = m[:, 1]
  xn = tf.math.l2_normalize(x, axis=-1)
  z = tf.linalg.cross(xn, y)
  zn = tf.math.l2_normalize(z, axis=-1)
  y = tf.linalg.cross(zn, xn)
  r = tf.stack([xn, y, zn], 1)
  return r


def svd_orthogonalize(m):
  """Convert 9D representation to SO(3) using SVD orthogonalization.

  Args:
    m: [BATCH, 3, 3] 3x3 matrices.

  Returns:
    [BATCH, 3, 3] SO(3) rotation matrices.
  """
  m_transpose = tf.matrix_transpose(tf.math.l2_normalize(m, axis=-1))
  _, u, v = tf.svd(m_transpose)
  det = tf.linalg.det(tf.matmul(v, u, transpose_b=True))
  # Check orientation reflection.
  r = tf.matmul(
      tf.concat([v[:, :, :-1], v[:, :, -1:] * tf.reshape(det, [-1, 1, 1])], 2),
      u, transpose_b=True)
  return r


def perturb_rotation(r, perturb_limits):
  """Randomly perturb a 3d rotation with a normal distribution.

  Args:
    r: [BATCH, 3, 3] rotation matrices.
    perturb_limits: a 3d list containing the perturbing deviation limits
      (degrees) for each axis x, y, z.

  Returns:
    [BATCH, 3, 3] perturbed rotation matrices.
  """
  x, y, z = tf.split(r, [1, 1, 1], 1)
  x = math_utils.normal_sampled_vector_within_cone(
      tf.squeeze(x, 1), degrees_to_radians(perturb_limits[0]), 0.5)
  y = math_utils.normal_sampled_vector_within_cone(
      tf.squeeze(y, 1), degrees_to_radians(perturb_limits[1]), 0.5)
  z = math_utils.normal_sampled_vector_within_cone(
      tf.squeeze(z, 1), degrees_to_radians(perturb_limits[2]), 0.5)
  return svd_orthogonalize(tf.stack([x, y, z], 1))


def half_rotation(rotation):
  """Return half of the input rotation.

  Args:
    rotation: [BATCH, 3, 3] rotation matrices.

  Returns:
    [BATCH, 3, 3] rotation matrices.
  """
  axes, angles = axis_angle.from_rotation_matrix(rotation)
  return rotation_matrix_3d.from_axis_angle(axes, angles/2)


def distributions_to_directions(x):
  """Convert spherical distributions from the DirectionNet to directions."""
  distribution_pred = spherical_normalization(x)
  expectation = spherical_expectation(distribution_pred)
  expectation_normalized = tf.nn.l2_normalize(expectation, axis=-1)
  return expectation_normalized, expectation, distribution_pred


def derotation(src_img,
               trt_img,
               rotation,
               input_fov,
               output_fov,
               output_shape,
               derotate_both):
  """Transform a pair of images to cancel out the rotation.

  Args:
    src_img: [BATCH, HEIGHT, WIDTH, CHANNEL] input source images.
    trt_img: [BATCH, HEIGHT, WIDTH, CHANNEL] input target images.
    rotation: [BATCH, 3, 3] relative rotations between src_img and trt_img.
    input_fov: [BATCH] a 1-D tensor (float32) of input field of view in degrees.
    output_fov: (float) output field of view in degrees.
    output_shape: a 2-D list of output dimension [height, width].
    derotate_both: Derotate both input images to an intermediate frame using
      half of the relative rotation between them.

  Returns:
    transformed images [BATCH, height, width, CHANNELS].
  """
  batch = src_img.shape.as_list()[0]
  if derotate_both:
    half_derotation = half_rotation(rotation)
    transformed_src = transformation.rotate_image_in_3d(
        src_img,
        tf.matrix_transpose(half_derotation),
        input_fov,
        output_fov,
        output_shape)

    transformed_trt = transformation.rotate_image_in_3d(
        trt_img,
        half_derotation,
        input_fov,
        output_fov,
        output_shape)
  else:
    transformed_src = transformation.rotate_image_in_3d(
        src_img,
        tf.eye(3, batch_shape=[batch]),
        input_fov,
        output_fov,
        output_shape)

    transformed_trt = transformation.rotate_image_in_3d(
        trt_img,
        rotation,
        input_fov,
        output_fov,
        output_shape)

  return (transformed_src, transformed_trt)
