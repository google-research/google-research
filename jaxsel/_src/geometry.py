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

"""Tools for generating images to represent paths."""

import dataclasses
from typing import Sequence, Tuple, Any

import numpy as np

# Alias for readable type annotations
NDArray = Any


@dataclasses.dataclass
class ClipBehaviors:
  CLIP = 0  # Drop pixels outside Region.
  PROJECT = 1  # Project pixels to nearest pixel in Region.


class Region:
  """Region of interest, where Shapes will live."""

  def __init__(self, x_range, y_range):
    """Constructor.

    Args:
      x_range: Range of considered pixels on the x-axis
      y_range: Range of considered pixels on the y-axis
    """
    self.x_min = x_range[0]
    self.x_max = x_range[1]
    self.y_min = y_range[0]
    self.y_max = y_range[1]

  def height(self):
    return self.y_max - self.y_min

  def width(self):
    return self.x_max - self.x_min


class Shape:
  """A list of colored points."""

  def __init__(self, points, colors):
    """Constructor.

    Args:
      points: Set of points composing the shape.
      colors: An iterable of colors, each corresponding to a point.
    """
    self.points = points
    self.colors = colors


class Path(Shape):
  """A single color path from a start to end point."""

  def __init__(self, color, start_point,
               end_point, noise_probability = .5):
    """Makes path of color `color` from `start_point` to `end_point`.

    Args:
      color: Color of the path
      start_point: Coordinates of the start point
      end_point: Coordinates of the end point
      noise_probability: Probability of moving in a random direction
    """
    self.path = self._make_path(start_point, end_point, noise_probability)
    super().__init__(self.path, [color for _ in self.path])

  def _make_path(self, start_point, end_point,
                 noise_probability):
    """Makes a path from `start_point` to `end_point`.

    Args:
      start_point: Coordinates of the start point.
      end_point: Coordinates of the end point.
      noise_probability: Probability of taking a random direction instead of a
        step towards the end point.

    Returns:
      A list of points connecting `start_point` to `end_point`.
    """
    if np.random.rand() < noise_probability:
      # Random direction
      direction = np.random.randint(-1, 2, size=(2,), dtype=int)
    else:
      # go towards end_point
      direction = np.sign(end_point - start_point)
    next_point = start_point + direction

    if np.all(next_point == end_point):
      return [start_point, end_point]
    else:
      rest_of_path = self._make_path(next_point, end_point, noise_probability)
      return [start_point] + rest_of_path


class SmoothPath(Path):
  """A smooth path from `start_point` to `end_point`."""

  def _make_path(self, start_point, end_point,
                 noise_probability = 0.):
    """Makes a smooth path from `start_point` to `end_point`.

    Args:
      start_point: Coordinates of the start point.
      end_point: Coordinates of the end point.
      noise_probability: Not used here

    Returns:
      A list of points connecting `start_point` to `end_point`.
    """
    points = [start_point]

    cur_point = start_point.astype(float)
    momentum = 2 * np.random.randn(2)

    step = 1
    while not np.all(np.abs(cur_point - end_point) < 1):
      grad = np.sign(end_point - cur_point)
      momentum = .9 * momentum + .1 * grad / np.sqrt(step)
      momentum_norm = np.sum(np.abs(momentum))
      cur_point = cur_point + momentum / (1.2 * momentum_norm)
      points.append(np.round(cur_point).astype(int))
      step += 1
    return points


class FilledCircle(Shape):
  """A filled circle."""

  def __init__(self, center_point, center_color,
               surrounding_color):
    """Makes a `surrounding_color` circle around a `center_color` `center_point`.

    Args:
      center_point: Center of the circle.
      center_color: Color of the center point.
      surrounding_color: Color of the circle.

    Returns:
      points
      colors
    """
    super().__init__(
        *self._make_circle(center_point, center_color, surrounding_color))

  # pylint: disable=missing-function-docstring
  def _make_circle(self, center_point, center_color,
                   surrounding_color):
    x0, x1 = center_point[0], center_point[1]
    points = []
    colors = []
    for i in range(x0 - 1, x0 + 2):
      for j in range(x1 - 1, x1 + 2):
        if i != x0 or j != x1:
          points.append(np.array([i, j]))
          colors.append(surrounding_color)
    points.append(center_point)
    colors.append(center_color)
    return points, colors


class Scene:
  """A collection of shapes with z-order information."""

  def __init__(self, z_ordered_shapes):
    """A collection of shapes with z-order information.

    Args:
      z_ordered_shapes: A list of shapes z-ordered from front to back.
        Earlier shapes in the list will occlude later ones if they overlap.
    """
    self.shapes = z_ordered_shapes

  def to_image(self,
               background_color,
               roi,
               clip_behavior=ClipBehaviors.PROJECT):
    """Extracts a region of the scene and renders to image."""
    image = background_color * np.ones((roi.height(), roi.width()), dtype=int)

    for shape in reversed(self.shapes):
      for point, color in zip(shape.points, shape.colors):
        h = point[0] - roi.y_min
        w = point[1] - roi.x_min
        if clip_behavior == ClipBehaviors.PROJECT:
          h = np.maximum(h, 0)
          h = np.minimum(h, image.shape[0] - 1)
          w = np.maximum(w, 0)
          w = np.minimum(w, image.shape[1] - 1)

        if 0 <= h < image.shape[0] and 0 <= w < image.shape[1]:
          image[h, w] = color
    return image
