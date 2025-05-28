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

r"""Maps latent space to coordinates to pixel space.

This module will provide two primary sets of functions. One set of functions
will translate latent coordinates into the xy-coordinates of a curve, scaled so
the entire curve falls within [0, 1] \times [0, 1]. The other will translate a
list of points (xy-coordinates) into pixel intensity matrices. The
transformations are continuous and piecewise differentiable.
"""

import functools
from typing import Optional, Sequence

from absl import app
import jax
from jax import numpy as jnp
import numpy as np
from scipy import special


@functools.partial(jax.jit, static_argnames='num_points')
def derivs_to_path_points(derivs, num_points):
  """Computes taylor expansion of interval [-3, 3].

  Accepts a (batch of) 2 x n matrix as an input. Each row of this matrix
  represents the (1st, 2nd, 3rd, ..., nth) derivatives of a function. The
  taylor expansions (fx, fy) of these two functions are computed. The output of
  the layer is a list of points on the curve

      t |-> (fx(t), fy(t))

  where t assumes equally spaced values between -3 and 3 inclusive.

  Args:
    derivs: A batch of 2 x n matrices as described above.
    num_points: The number of points on the curve to produce in the output.

  Returns:
    A tensor of shape (batch_size, num_points, 2) giving the coordinates of
      points on the curves.
  """
  unused_batch_size, two, highest_deriv = derivs.shape
  if two != 2:
    raise ValueError('wrong shape: ', derivs.shape, ' should have second '
                     'dimension equal to 2')
  t = np.linspace(-3.0, 3.0, num_points)[:, np.newaxis]  # num_points x 1
  exponents = np.array(range(1, highest_deriv +
                             1))[np.newaxis, :]  # 1 x highest_deriv
  power_matrix = np.power(t, exponents)  # num_points x highest_deriv
  factorial_coeffs = 1.0 / special.factorial(exponents)[
      np.newaxis, :, :]  # 1 x 1 x highest_deriv
  coeffs = factorial_coeffs * derivs  # batch_size x 2 x highest_deriv
  points = jnp.einsum('ijk,lk->ilj', coeffs,
                      power_matrix)  # batch_size x num_points x 2
  return points


@functools.partial(jax.jit, static_argnames=('num_points', 't_scale'))
def sine_net_to_path_points(params,
                            num_points,
                            t_scale = 4.0):
  """Uses trig functions to compute a curve from a point in latent space.

  Accepts an n x 4 matrix as an input. The first two columns represent the
  frequency and phase (i.e., scale and shift) of n functions that are
  essentially [random Fourier features](
  https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) of a
  one-dimensional input t. The third and fourth column represent coefficients
  of these n functions in linear combinations to output x and y coordinates.
  The output is a list of points on the curve thus defined, where the input
  parameter t assumes equally spaced values.

  Args:
    params: The sin net parameters. Shape (batch_size, hidden_layer_width, 4).
    num_points: The number of points on the curve to produce in the output.
    t_scale: The length of the parametric interval. The points of the curve
      correspond to evenly spaced values of t ranging over [-t_scale/2,
      t_scale/2]. Larger values of t_scale yield more complex curves with more
      self-intersections on average.

  Returns:
    A (batch of) lists of (x,y) coordinates. Each list describes a curve.
    Shape (batch_size, num_points, 2).
  """
  t = np.linspace(-t_scale / 2, t_scale / 2, num_points)[np.newaxis,
                                                         np.newaxis, :]
  frequency_radians_per_second = params[:, :, 0:1]
  phase_offset = params[:, :, 1:2]
  xy_amplitude = params[:, :, 2:]
  hidden_preactivation = (frequency_radians_per_second * t) + phase_offset
  hidden_activation = jnp.sin(hidden_preactivation)
  return jnp.einsum('kil,kij->kjl', xy_amplitude, hidden_activation)


def rescale_points(points, margin = 0.1):
  """Rescales and translates points to lie inside unit square.

  Accepts a list of xy points as input, and applies a
  scalar multiplier and a translation so that the bounding box of the points
  becomes either [margin, 1 - margin] x (smaller interval centered on 0.5) or
  (smaller interval centered on 0.5) x [margin, 1 - margin].

  Args:
    points: The points to transform. A batch_size x n x 2 matrix.
    margin: How much empty space to leave on each side of the unit square.

  Returns:
    A new array consisting of the translated and rotates points.
  """
  min_corner = jnp.amin(points, axis=1, keepdims=True)  # num_images x 1 x 2
  points = points - min_corner  # bounding boxes: [0, x_size] x [0, y_size]
  rectangular_size = jnp.amax(
      points, axis=1, keepdims=True)  # num_images x 1 x 2
  points = points - 0.5 * rectangular_size
  # bounding boxes: [-x_size/2, -y_size/2] x [x_size/2, y_size/2]
  square_size = jnp.amax(
      rectangular_size, axis=2, keepdims=True)  # num_images x 1 x 1
  points = points * (1 - 2 * margin) / jnp.maximum(square_size, 1e-5)
  points = points + 0.5  # center on (0.5, 0.5)
  return points


@functools.partial(jax.jit, static_argnames=('x_pixels', 'y_pixels'))
def nearest_point_distance(points,
                           x_pixels = 60,
                           y_pixels = 60):
  """Computes the distances from grid points to the nearest input point.

  Accepts a (batch of) list of xy points as input. The output is a (batch of)
  matrix of distances, where each entry of the matrix represents the distance
  from the corresponding point in the unit square to the nearest input point.

  Args:
    points: A batch_size x n x 2 matrix of points.
    x_pixels: How many x coordinates the output should have.
    y_pixels: How many y coordinates the output should have.

  Returns:
    A batch of matrices of distances. It is a tensor of dimension
    batch_size x x_pixels x y_pixels.
  """
  x_coords = np.linspace(0.0, 1.0, x_pixels)[np.newaxis, np.newaxis, :]
  y_coords = np.linspace(0.0, 1.0, y_pixels)[np.newaxis, np.newaxis, :]

  x_points = points[:, :, 0:1]
  y_points = points[:, :, 1:2]

  x_diff_squared = jnp.square(x_points - x_coords)
  y_diff_squared = jnp.square(y_points - y_coords)

  distance_squared = jnp.amin(  # distance to *nearest* point only
      x_diff_squared[:, :, :, np.newaxis] + y_diff_squared[:, :, np.newaxis, :],
      axis=1)

  return jnp.sqrt(distance_squared)


@functools.partial(jax.jit, static_argnames=('spread',))
def gaussian_activation(distances,
                        spread = 1 / 60):
  if spread is None:
    spread = 1 / distances.shape[1]
  return jnp.exp(-jnp.square(distances * (1 / spread)))


@functools.partial(jax.jit, static_argnames=('x_pixels', 'y_pixels', 'spread'))
def coords_to_pixels(xy_points,
                     x_pixels = 60,
                     y_pixels = 60,
                     spread = 1 / 60):
  """Converts a list of coordinates into pixel intensities.

  Converts a list of coordinates as an input. The output is
  a matrix of pixel intensities. Each entry of the matrix is between 0 and 1 and
  represents the intensity of the pixel, based on how close the nearest point
  is. The pixels are assumed to lie in a [0,1] x [0,1] square; if this is not
  desired, rescale the coordinates before passing them into this layer.

  Args:
    xy_points: The (batch_size x num_points x 2) tensor of point coordinates.
    x_pixels: The number of pixels to fit horizontally into the interval [0, 1].
    y_pixels: The number of pixels to fit vertically into the interval [0, 1].
    spread: Morally, the width of the pen used to draw the curve (coordinates,
      not pixels). For details (including the behavior if not populated), see
      the `gaussian_activation` docstring.

  Returns:
    A batch of matrices of pixel intensities. Shape
    (batch_size, x_pixels, y_pixels, 1).
  """
  x = xy_points
  x = nearest_point_distance(x, x_pixels, y_pixels)
  x = gaussian_activation(x, spread)
  return x[Ellipsis, np.newaxis]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
