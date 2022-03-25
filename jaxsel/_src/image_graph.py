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

"""GraphAPI implementation of graphs based on images."""

import functools
from typing import Tuple

import flax
import jax
from jax import numpy as jnp
import numpy as np

from jaxsel._src import graph_api


@functools.partial(jax.jit, static_argnums=(0,))
def _get_out_of_bounds_window(radius, padding_value):
  """Return a window full of padding_value."""
  return padding_value * np.ones((2 * radius + 1, 2 * radius + 1), dtype=int)


# TODO(gnegiar): Store the padded image in the graph to pad only once
@functools.partial(jax.jit, static_argnums=(2,))
def _get_in_bounds_window(point, image, radius, padding_value):
  padded_image = jnp.pad(
      image, radius, mode='constant', constant_values=padding_value)
  padded_image = padded_image.astype(int)

  # Compute window under shift caused by padding.
  window = jax.lax.dynamic_slice(padded_image, point,
                                 (2 * radius + 1) * np.ones(2, dtype=int))
  return window


@functools.partial(jax.jit, static_argnums=(2,))
def _get_window(point, image, radius, padding_value):
  """Extract a square 2d window around a given pixel in the image.

  If the point has coordinates (-1, -1), the window will be filled with
  `padding_value`.

  Args:
    point: The coordinates of the center point in the extracted window.
    image: The image to extract the patch from.
    radius: The extracted window will be a square of shape (2*radius x 2*radius)
      around `point`.
    padding_value: When the window goes beyond the borders of the initial image,
      we fill the corresponding values with `padding_value`.

  Returns:
    window: float tensor. It is a subtensor of the input `image` around `point`.
  """
  return jax.lax.cond(
      jnp.all(point == -1),
      lambda _: _get_out_of_bounds_window(radius, padding_value),
      lambda _: _get_in_bounds_window(point, image, radius, padding_value),
      None)


@flax.struct.dataclass
class ImageGraph(graph_api.GraphAPI):
  """An environment based on an input image."""

  image: np.ndarray
  _start_node_id: int

  image_shape: Tuple[int, int] = flax.struct.field(pytree_node=False)
  patch_size: int = flax.struct.field(pytree_node=False)
  padding_value: int = flax.struct.field(pytree_node=False)
  window_size: int = flax.struct.field(pytree_node=False)
  _num_colors: int = flax.struct.field(pytree_node=False)

  # pylint: disable=g-complex-comprehension
  RELATION_OFFSETS = [
      (i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not i == j == 0
  ]

  # pylint: enable=g-complex-comprehension

  @staticmethod
  def create(image,
             get_start_pixel_fn,
             patch_size = 5,
             num_colors = 10,
             padding_value = 0):
    """Constructor based on an integer image.

    Args:
      image: The image to be represented as a graph.
      get_start_pixel_fn: This function produces the pixel location of the start
        pixel as a pair of (h, w) coordinates.
      patch_size: Size of patch around each pixel corresponding to a node to use
        as node features.
      num_colors: Number of colors in the image.
      padding_value: Value to insert on the boundary of the pixel for computing
        patch features of pixels on the boundary.

    Returns:
      Corresponding graph
    """
    if image.ndim != 2:  # Only handle single-channel for now.
      raise ValueError(
          f'image must be a 2-d object. Got {image.ndim}D instead.')
    if patch_size % 2 != 1:  # Require odd-sized patches for symmetry.
      raise ValueError(
          f'Patch must be odd-sized for symmetry. Got {patch_size}.')
    window_size = (patch_size - 1) // 2
    start_node_id = np.ravel_multi_index(get_start_pixel_fn(image), image.shape)

    return ImageGraph(
        image=image,
        image_shape=image.shape,
        patch_size=patch_size,
        padding_value=padding_value,
        window_size=window_size,
        _start_node_id=start_node_id,
        _num_colors=num_colors)

  def graph_parameters(self):
    return graph_api.GraphParameters(
        node_vocab_size=self._num_colors,
        num_relation_types=len(self.RELATION_OFFSETS),
        node_feature_dim=self.patch_size**2,
        node_feature_kind=graph_api.NodeFeatureKind.CATEGORICAL,
        task_vocab_size=self._num_colors,
        task_feature_dim=self.patch_size**2,
        task_feature_kind=graph_api.NodeFeatureKind.CATEGORICAL)

  def sample_start_node_id(self, seed=0):
    del seed  # Currently assuming there's only a single start node.
    return self._start_node_id

  def _pixel_coordinates_to_id(self, pixel_hw):
    node_id = self.image_shape[1] * pixel_hw[0] + pixel_hw[1]
    return jnp.where(node_id < 0, -1, node_id)

  def _pixel_id_to_coordinates(self, pixel_id):
    """Converts pixel id to coordinates.

    Args:
      pixel_id: id of the pixel.

    Returns:
      coordinates: coordinates of the pixel in the image. (-1, -1) for the
        'pixel' with id -1.
    """
    # Can't use jnp.unravel_index due to the -1 node.
    coordinates = jnp.array(
        (pixel_id // self.image_shape[1], pixel_id % self.image_shape[1]))
    coordinates = jnp.where(
        jnp.all(coordinates >= 0, axis=-1, keepdims=True), coordinates, -1)
    return coordinates

  @property
  def start_node_coords(self):
    return self.node_metadata(self._start_node_id)['coordinates']

  def node_metadata(self, node_id):
    return {'coordinates': self._pixel_id_to_coordinates(node_id)}

  def node_patch(self, node_id):
    """Gets patch surrounding `node_id`."""
    node_coordinates = self._pixel_id_to_coordinates(node_id)
    return _get_window(node_coordinates, self.image, self.window_size,
                       self.padding_value)

  def node_features(self, node_id):
    """Concatenates the patch surrounding `nocde_id` to represent it."""
    return self.node_patch(node_id).reshape(-1)

  def task_features(self):
    """Describe the task just in terms of the start location.

    # TODO(dtarlow): Consider alternatives like including pixel coordinates as
    features, so each node can know its relation to the start node.

    Returns:
      patch: Flattened window around the start node, used as node features.
    """
    return self.node_patch(self._start_node_id).reshape(-1)

  def _outgoing_edges_out_of_bounds_pixel(self, node_id, relation_ids):
    del node_id
    return jnp.full_like(relation_ids, self._start_node_id)

  def _outgoing_edges_in_bounds_pixel(self, node_id, relation_ids):
    """Returns the outgoing edges from `node_id`.

    Pixels outside of the image are mapped to node_id -1.
    This "out of bounds pixel" node points to the start node.
    The considered initial node must be inside the image bounds.

    Args:
      node_id: ID of start node
      relation_ids: IDs of the relations to neighbors.

    Returns:
      A list of neighbor ids, in the order given by relation_ids.
    """
    node_coordinates = self._pixel_id_to_coordinates(node_id)
    neighbor_coordinates = jnp.repeat(
        jnp.array(node_coordinates)[jnp.newaxis, :], len(self.RELATION_OFFSETS),
        0)
    offsets = jnp.array(self.RELATION_OFFSETS)
    neighbor_coordinates = neighbor_coordinates + offsets
    # make sure coords are within bounds, or -1
    neighbor_coordinates = jnp.where(
        jnp.less_equal(neighbor_coordinates,
                       jnp.array(self.image.shape) - 1), neighbor_coordinates,
        -1)
    # make sure coordinates are >= 0, or -1
    neighbor_coordinates = jnp.where(
        jnp.all(neighbor_coordinates >= 0, axis=-1, keepdims=True),
        neighbor_coordinates, -1)

    neighbor_ids = jax.vmap(self._pixel_coordinates_to_id)(neighbor_coordinates)
    return neighbor_ids

  def outgoing_edges(self, node_id):
    """Returns the outgoing edges from `node_id`.

    Pixels outside of the image are mapped to node_id -1.
    This "out of bounds pixel" node points to the start node.

    Args:
      node_id: ID of start node

    Returns:
       A list of relation ids and neighbor ids. The relation id is that of
       the edge from the initial node to the neighbor.
    """
    relation_ids = jnp.arange(len(self.RELATION_OFFSETS))

    neighbor_ids = jax.lax.cond(
        node_id == -1,
        lambda _: self._outgoing_edges_out_of_bounds_pixel(  # pylint: disable=g-long-lambda
            node_id, relation_ids),
        lambda _: self._outgoing_edges_in_bounds_pixel(node_id, relation_ids),
        None)

    return relation_ids, neighbor_ids
