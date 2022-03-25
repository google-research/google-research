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

"""Synthetic data generation utilities.

This file provides utilities to generate Pathfinder like images and associated
graphs.
"""
import numpy as np

from jaxsel._src import geometry
from jaxsel._src import image_graph


BACKGROUND_COLOR_ID = 0  # Reserved color for background.
START_COLOR_ID = 1  # Reserved color, used only for specifying start point.


def _sample_end_point(grid_size, start_point, min_distance,
                      min_dist_from_edge=3):
  """Samples end point on grid at least min distance from `start_point`."""
  while True:
    end_point = np.random.randint(
        min_dist_from_edge, grid_size - min_dist_from_edge, size=(2,))
    if np.any(np.abs(end_point - start_point) >= min_distance):
      return end_point


def _generate_image(grid_size, num_paths, num_classes):
  """Constructs an image from scene elements z-ordered from back to front."""

  # Generate a bunch of paths starting from the same `start_point`.
  start_point = np.random.randint(2, grid_size - 2, size=(2,))
  paths = []
  path_color = START_COLOR_ID + 1  # Make all paths the same color
  for _ in range(num_paths):
    end_point = _sample_end_point(grid_size, start_point, 2)
    path = geometry.SmoothPath(
        path_color, start_point=start_point, end_point=end_point)
    paths.append(path)

  # Choose one path to be the "special path" that we need to follow to find the
  # label. Draw circles at the end points.
  label = np.random.randint(num_classes)
  label_color = label + START_COLOR_ID + 2
  special_path = np.random.choice(paths)
  special_path_color = special_path.colors[0]

  start_circle = geometry.FilledCircle(start_point, START_COLOR_ID,
                                       special_path_color)
  end_circle = geometry.FilledCircle(special_path.points[-1], label_color,
                                     label_color)

  scene_elements = [start_circle, end_circle] + paths
  scene = geometry.Scene(scene_elements)
  image = scene.to_image(BACKGROUND_COLOR_ID,
                         geometry.Region((0, grid_size), (0, grid_size)))

  return image, label


def generate(grid_size=3, num_paths=5, num_classes=10,
             patch_size=5,
             verbose=False, return_image=False):
  """Generates an image made of paths on a uniform background.

  Args:
    grid_size: Size of the image
    num_paths: Number of paths to generate
    num_classes: Number of possible end point colors
    patch_size: Size of patch to use as node features.
    verbose: If verbose, prints the image
    return_image: If True, additionally returns the image.

  Returns:
    graph: The graph corresponding to the generated image
    start_node_id: The ID of the node from which to start walks on the graph.
    label: Corresponds to the color of the end point to find.
    image: The image on which the graph is built.
  """
  if grid_size > 5:
    image, label = _generate_image(grid_size, num_paths, num_classes)
  else:
    # Make a trivial example with only a start pixel in a corner, and label 0.
    image = np.zeros((grid_size, grid_size), dtype=np.int)
    image[0, 0] = START_COLOR_ID
    label = 0

  if verbose:
    print('Image:')
    print(image)

  def _get_start_pixel(image):
    xs, ys = np.nonzero(image == START_COLOR_ID)
    return xs[0], ys[0]

  # Make a graph and start node.
  graph = image_graph.ImageGraph.create(
      image, _get_start_pixel, padding_value=BACKGROUND_COLOR_ID,
      patch_size=patch_size,
      num_colors=image.max())
  start_node_id = graph.sample_start_node_id()

  if return_image:
    return graph, start_node_id, label, image

  return graph, start_node_id, label
