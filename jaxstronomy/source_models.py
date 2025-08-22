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

"""Implementations of light profiles for lensing.

Implementation of light profiles for lensing closely following implementations
in lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

import dm_pix
import jax.numpy as jnp

__all__ = ['Interpol', 'SersicElliptic']


class Interpol():
  """Interpolated light profile.

  Interpolated light profile functions, with calculation following those in
  Lenstronomy.
  """

  parameters = ('image', 'amp', 'center_x', 'center_y', 'angle', 'scale')

  @staticmethod
  def function(x, y, image, amp,
               center_x, center_y, angle,
               scale):
    """Calculate the brightness for the interpolated light profile.

    Args:
      x: X-coordinates at which to evaluate the profile.
      y: Y-coordinates at which to evaluate the profile.
      image: Source image as base for interpolation.
      amp: Normalization to source image.
      center_x: X-coordinate center of the light profile.
      center_y: Y-coordinate cetner of the light profile.
      angle: Clockwise rotation angle of simulated image with respect to
        simulation grid.
      scale: Pixel scale of the simulated image.

    Returns:
      Surface brightness at each coordinate.
    """
    x_image, y_image = Interpol._coord_to_image_pixels(x, y, center_x, center_y,
                                                       angle, scale)
    return amp * Interpol._image_interpolation(x_image, y_image, image)

  @staticmethod
  def _image_interpolation(x_image, y_image,
                           image):
    """Map coordinates to interpolated image brightness.

    Args:
      x_image: X-coordinates in the image plane.
      y_image: Y-coordinates in the image plane.
      image: Source image as base for interpolation.

    Returns:
      Interpolated image brightness.
    """
    # Interpolation in dm-pix expects (0,0) to be the upper left-hand corner,
    # whereas lensing calculation treat the image center as (0,0). Additionally,
    # interpolation treats x as rows and y as columns, whereas lensing does the
    # opposite. Finally, interpolation considers going down the rows as
    # increasing in the x-coordinate, whereas that's decreasing the
    # y-coordinate in lensing. We account for this with the offset, by
    # switching x and y in the interpolation input, and by negating y.
    offset = jnp.array([image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5])
    coordinates = jnp.concatenate(
        [jnp.expand_dims(coord, axis=0) for coord in [-y_image, x_image]],
        axis=0)
    coordinates += jnp.reshape(a=offset, shape=(*offset.shape, 1))
    return dm_pix.flat_nd_linear_interpolate_constant(
        image, coordinates, cval=0.0)

  @staticmethod
  def _coord_to_image_pixels(x, y, center_x, center_y, angle, scale):
    """Map from simulation coordinates to image coordinates.

    Args:
      x: X-coordinates at which to evaluate the profile.
      y: Y-coordinates at which to evaluate the profile.
      center_x: X-coordinate center of the light profile.
      center_y: Y-coordinate cetner of the light profile.
      angle: Clockwise rotation angle of simulated image with respect to
        simulation grid.
      scale: Pixel scale (in angular units) of the simulated image.

    Returns:
      X- and y-coordinates in the image plane.
    """
    x_image = (x - center_x) / scale
    y_image = (y - center_y) / scale
    # Lenstronomy uses clockwise rotation so we will stay consistent.
    complex_coords = jnp.exp(-1j * angle) * (x_image + 1j * y_image)

    return complex_coords.real, complex_coords.imag


class SersicElliptic():
  """Sersic light profile.

  Sersic light profile functions, with implementation closely following the
  Sersic class in Lenstronomy.
  """

  parameters = (
      'amp', 'sersic_radius', 'n_sersic', 'axis_ratio', 'angle', 'center_x',
      'center_y'
  )

  @staticmethod
  def function(x, y, amp, sersic_radius,
               n_sersic, axis_ratio, angle,
               center_x, center_y):
    """"Calculate the brightness for the elliptical Sersic light profile.

    Args:
      x: X-coordinates at which to evaluate the brightness.
      y: Y-coordinates at which to evaluate the derivative.
      amp: Amplitude of Sersic light profile.
      sersic_radius: Sersic radius.
      n_sersic: Sersic index.
      axis_ratio: Axis ratio of the major and minor axis of ellipticity.
      angle: Clockwise angle of orientation of major axis.
      center_x: X-coordinate center of the Sersic profile.
      center_y: Y-coordinate center of the Sersic profile.

    Returns:
      Brightness from elliptical Sersic profile.
    """
    radius = SersicElliptic._get_distance_from_center(x, y, axis_ratio, angle,
                                                      center_x, center_y)
    return amp * SersicElliptic._brightness(radius, sersic_radius, n_sersic)

  @staticmethod
  def _get_distance_from_center(x, y,
                                axis_ratio, angle,
                                center_x,
                                center_y):
    """Calculate the distance from the Sersic center, accounting for axis ratio.

    Args:
      x: X-coordinates at which to evaluate the brightness.
      y: Y-coordinates at which to evaluate the derivative
      axis_ratio: Axis ratio of the major and minor axis of ellipticity.
      angle: Clockwise angle of orientation of major axis.
      center_x: X-coordinate center of the Sersic profile.
      center_y: Y-coordinate center of the Sersic profile.

    Returns:
      Distance from Sersic center.
    """
    x_centered = x - center_x
    y_centered = y - center_y
    complex_coords = jnp.exp(-1j * angle) * (x_centered + 1j * y_centered)
    return jnp.sqrt((complex_coords.real * jnp.sqrt(axis_ratio))**2 +
                    (complex_coords.imag / jnp.sqrt(axis_ratio))**2)

  @staticmethod
  def _brightness(radius, sersic_radius,
                  n_sersic):
    """Return the sersic brightness.

    Args:
      radius: Radii at which to evaluate the brightness.
      sersic_radius: Sersic radius.
      n_sersic: Sersic index.

    Returns:
      Brightness values.
    """
    b_n = SersicElliptic._b_n(n_sersic)
    reduced_radius = radius / sersic_radius
    return jnp.nan_to_num(jnp.exp(-b_n * (reduced_radius**(1 / n_sersic) - 1.)))

  @staticmethod
  def _b_n(n_sersic):
    """Return approximation for Sersic b(n_sersic).

    Args:
      n_sersic: Sersic index.

    Returns:
      Approximate b(n_sersic).
    """
    return 1.9992 * n_sersic - 0.3271
