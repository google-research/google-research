# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utility functions for point clouds."""

import gin
import gin.tf
import numpy as np
from scipy import spatial
import tensorflow as tf


def flip_normals_towards_viewpoint(points, normals, viewpoint):
  """Flips the normals to face towards the view point.

  Args:
    points: A tf.float32 tensor of size [N, 3].
    normals: A tf.float32 tensor of size [N, 3].
    viewpoint: A tf.float32 tensor of size [3].

  Returns:
    flipped_normals: A tf.float32 tensor of size [N, 3].
  """
  # (viewpoint - point)
  view_vector = tf.expand_dims(viewpoint, axis=0) - points
  # Dot product between the (viewpoint - point) and the plane normal
  cos_theta = tf.expand_dims(
      tf.reduce_sum(view_vector * normals, axis=1), axis=1)
  # Revert normals where cos is negative.
  normals *= tf.sign(tf.tile(cos_theta, [1, 3]))
  return normals


def points_to_normals_unbatched(points,
                                k,
                                distance_upper_bound,
                                viewpoint=None,
                                noise_magnitude=1e-4,
                                method='pca'):
  """Computes normals for the points in a point cloud.

  Args:
    points: A tf.float32 tensor of size [N, 3].
    k: An integer determining the size of the neighborhood.
    distance_upper_bound: Maximum distance of the neighbor points. If None, it
      will not add a cap on the distance.
    viewpoint: A tf.float32 tensor of size [3]. Normals will be flipped to point
      towards view point. If None, it won't be used.
    noise_magnitude: Noise magnitude to be added to the input of svd. If None,
      it won't add noise.
    method: The normal prediction method, options are `pca` and `cross` (cross
      product).

  Returns:
    normals: A tf.float32 tensor of size [N, 3].
  """
  if method == 'pca':
    if k <= 3:
      raise ValueError('At least 3 neighbors are required for computing PCA.')
  elif method == 'cross':
    if k <= 2:
      raise ValueError('At least 2 neighbors are required for computing cross.')
  else:
    raise ValueError(('Unknown method of normal prediction %s' % method))
  n = tf.shape(points)[0]
  d = points.get_shape().as_list()[1]
  if d != 3:
    raise ValueError('Points dimension is not 3.')
  _, knn_adjacencies = knn_graph_from_points_unbatched(
      points=points, k=k, distance_upper_bound=distance_upper_bound)
  knn_adjacencies = knn_adjacencies[:, 1:]
  knn_adjacencies = tf.reshape(knn_adjacencies, [n * (k - 1)])
  adjacency_points = tf.gather(points, indices=knn_adjacencies)
  adjacency_points = tf.reshape(adjacency_points, [n, (k - 1), d])
  if method == 'pca':
    adjacency_relative_points = adjacency_points - tf.expand_dims(
        points, axis=1)
    if noise_magnitude is not None:
      adjacency_relative_points += tf.random.uniform(
          tf.shape(adjacency_relative_points),
          minval=-noise_magnitude,
          maxval=noise_magnitude,
          dtype=tf.float32)
    _, _, v = tf.linalg.svd(adjacency_relative_points)
    normals = v[:, 2, :]
  elif method == 'cross':
    v1 = adjacency_points[:, 0, :] - points
    v2 = adjacency_points[:, 1, :] - points
    normals = tf.linalg.cross(v1, v2)
    normals_length = tf.expand_dims(tf.norm(normals, axis=1), axis=1)
    if noise_magnitude is not None:
      normals_length += noise_magnitude
    normals /= normals_length
  else:
    raise ValueError(('Unknown method of normal prediction %s' % method))
  if viewpoint is not None:
    normals = flip_normals_towards_viewpoint(
        points=points, normals=normals, viewpoint=viewpoint)
  return normals


@gin.configurable
def points_to_normals(points,
                      num_valid_points,
                      k=10,
                      distance_upper_bound=0.5,
                      viewpoints=None,
                      noise_magnitude=1e-4,
                      method='pca'):
  """Computes normals for the points in a point cloud.

  Args:
    points: A tf.float32 tensor of size [batch_size, N, 3].
    num_valid_points: A tf.int32 tensor of size [batch_size] representing the
      number of valid points in each example.
    k: An integer determining the size of the neighborhood.
    distance_upper_bound: Maximum distance of the neighbor points. If None, it
      will not add a cap on the distance.
    viewpoints: A tf.float32 tensor of size [batch_size, 3]. Normals will be
      flipped to point towards view point. If None, it won't be used.
    noise_magnitude: Noise magnitude to be added to the input of svd. If None,
      it won't add noise.
    method: The normal prediction method, options are `pca` and `cross` (cross
      product).

  Returns:
    normals: A tf.float32 tensor of size [batch_size, N, 3].
  """
  batch_size = points.get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('batch_size is unknown at graph construction time.')
  num_points = tf.shape(points)[1]

  def fn_normals_single_batch(i):
    """Function for computing normals for a single batch."""
    num_valid_points_i = num_valid_points[i]
    points_i = points[i, 0:num_valid_points_i, :]
    if viewpoints is None:
      viewpoint_i = None
    else:
      viewpoint_i = viewpoints[i, :]
    normals_i = points_to_normals_unbatched(
        points=points_i,
        k=k,
        distance_upper_bound=distance_upper_bound,
        viewpoint=viewpoint_i,
        noise_magnitude=noise_magnitude,
        method=method)
    return tf.pad(
        normals_i, paddings=[[0, num_points - num_valid_points_i], [0, 0]])

  normals = []
  for i in range(batch_size):
    normals.append(fn_normals_single_batch(i))
  return tf.stack(normals, axis=0)


def np_knn_graph_from_points_unbatched(points,
                                       k,
                                       distance_upper_bound,
                                       mask=None):
  """Returns the distances and indices of the neighbors of each point.

  Args:
    points: A np.float32 numpy array of [N, D] where D is the point dimensions.
    k: Number of neighbors for each point.
    distance_upper_bound: Only build the graph using points that are closer than
      this distance.
    mask: If None, will be ignored. If not None, A np.bool numpy array of
      size [N]. knn will be applied to only points where the mask is True. The
      points where the mask is False will have themselves as their neighbors.

  Returns:
    distances: A np.float32 numpy array of [N, k].
    indices: A np.int32 numpy array of [N, k].
  """
  num_points = points.shape[0]
  if mask is None:
    mask = np.ones([num_points], dtype=np.bool)
  num_masked_points = np.sum(mask.astype(np.int32))
  indices = np.expand_dims(np.arange(num_points), axis=1)
  indices = np.tile(indices, [1, k])
  distances = np.zeros([num_points, k], dtype=np.float32)
  if num_masked_points >= k:
    masked_points = points[mask, :]
    tree = spatial.cKDTree(masked_points)
    masked_distances, masked_indices = tree.query(
        masked_points, k=k, distance_upper_bound=distance_upper_bound)
    placeholder = np.tile(
        np.expand_dims(np.arange(num_masked_points), axis=1), [1, k])
    valid_mask = np.greater_equal(masked_indices,
                                  num_masked_points).astype(np.int32)
    masked_indices = masked_indices * (1 -
                                       valid_mask) + placeholder * valid_mask
    masked_distances = np.nan_to_num(masked_distances)
    masked_distances *= (1.0 - valid_mask)
    masked_indices_shape = masked_indices.shape
    masked_indices = np.arange(num_points)[mask][np.reshape(
        masked_indices, [-1])]
    masked_indices = np.reshape(masked_indices, masked_indices_shape)
    indices[mask, :] = masked_indices
    distances[mask, :] = masked_distances
  return distances.astype(np.float32), indices.astype(np.int32)


@gin.configurable
def knn_graph_from_points_unbatched(points, k, distance_upper_bound, mask=None):
  """Returns the distances and indices of the neighbors of each point.

  Note that each point will have at least k neighbors unless the number of
  points is less than k. In that case, the python function that is wrapped in
  py_function will raise a value error.

  Args:
    points: A tf.float32 tensor of size [N, D] where D is the point dimensions.
    k: Number of neighbors for each point.
    distance_upper_bound: Only build the graph using points that are closer than
      this distance.
    mask: If not None, A tf.bool tensor of size [N]. If None, it is ignored.
      If not None, knn will be applied to only points where the mask is True.
      The points where the mask is False will have themselves as their
      neighbors.

  Returns:
    distances: A tf.float32 tensor of size [N, k].
    indices: A tf.int32 tensor of [N, k].
  """

  def fn(np_points, np_mask):
    return np_knn_graph_from_points_unbatched(
        points=np_points,
        k=k,
        distance_upper_bound=distance_upper_bound,
        mask=np_mask)

  num_points = tf.shape(points)[0]
  if mask is None:
    mask = tf.cast(tf.ones([num_points], dtype=tf.int32), dtype=tf.bool)
  else:
    mask = tf.reshape(mask, [num_points])
  distances, indices = tf.compat.v1.py_func(fn, [points, mask],
                                            [tf.float32, tf.int32])
  distances = tf.reshape(distances, [num_points, k])
  indices = tf.reshape(indices, [num_points, k])
  return distances, indices


@gin.configurable
def knn_graph_from_points(points, num_valid_points, k,
                          distance_upper_bound, mask=None):
  """Returns the distances and indices of the neighbors of each point.

  Note that each point will have at least k neighbors unless the number of
  points is less than k. In that case, the python function that is wrapped in
  py_function will raise a value error.

  Args:
    points: A tf.float32 tensor of size [batch_size, N, D] where D is the point
      dimensions.
    num_valid_points: A tf.int32 tensor of size [batch_size] containing the
      number of valid points in each batch example.
    k: Number of neighbors for each point.
    distance_upper_bound: Only build the graph using points that are closer than
      this distance.
    mask: If not None, A tf.bool tensor of size [batch_size, N]. If None, it is
      ignored. If not None, knn will be applied to only points where the mask is
      True. The points where the mask is False will have themselves as their
      neighbors.

  Returns:
    distances: A tf.float32 tensor of size [batch_size, N, k].
    indices: A tf.int32 tensor of size [batch_size, N, k].

  Raises:
    ValueError: If batch_size is unknown.
  """
  if points.get_shape().as_list()[0] is None:
    raise ValueError('Batch size is unknown.')
  batch_size = points.get_shape().as_list()[0]
  num_points = tf.shape(points)[1]

  def fn_knn_graph_from_points_unbatched(i):
    """Computes knn graph for example i in the batch."""
    num_valid_points_i = num_valid_points[i]
    points_i = points[i, :num_valid_points_i, :]
    if mask is None:
      mask_i = None
    else:
      mask_i = mask[i, :num_valid_points_i]
    distances_i, indices_i = knn_graph_from_points_unbatched(
        points=points_i,
        k=k,
        distance_upper_bound=distance_upper_bound,
        mask=mask_i)
    distances_i = tf.pad(
        distances_i, paddings=[[0, num_points - num_valid_points_i], [0, 0]])
    indices_i = tf.pad(
        indices_i, paddings=[[0, num_points - num_valid_points_i], [0, 0]])
    return distances_i, indices_i

  distances, indices = tf.map_fn(
      fn=fn_knn_graph_from_points_unbatched,
      elems=tf.range(batch_size),
      dtype=(tf.float32, tf.int32))

  return distances, indices


@gin.configurable
def identity_knn_graph_unbatched(points, k):
  """Returns each points as its own neighbor k times.

  Args:
    points: A tf.float32 tensor of [N, D] where D is the point dimensions.
    k: Number of neighbors for each point.

  Returns:
    distances: A tf.float32 tensor of [N, k]. Distances is all zeros since
      each point is returned as its own neighbor.
    indices: A tf.int32 tensor of [N, k]. Each row will contain values that
      are identical to the index of that row.
  """
  num_points = tf.shape(points)[0]
  indices = tf.expand_dims(tf.range(num_points), axis=1)
  indices = tf.tile(indices, [1, k])
  distances = tf.zeros([num_points, k], dtype=tf.float32)
  return distances, indices


@gin.configurable
def identity_knn_graph(points, num_valid_points, k):  # pylint: disable=unused-argument
  """Returns each points as its own neighbor k times.

  Args:
    points: A tf.float32 tensor of size [num_batches, N, D] where D is the point
      dimensions.
    num_valid_points: A tf.int32 tensor of size [num_batches] containing the
      number of valid points in each batch example.
    k: Number of neighbors for each point.

  Returns:
    distances: A tf.float32 tensor of [num_batches, N, k]. Distances is all
      zeros since each point is returned as its own neighbor.
    indices: A tf.int32 tensor of [num_batches, N, k]. Each row will contain
      values that are identical to the index of that row.
  """
  num_batches = points.get_shape()[0]
  num_points = tf.shape(points)[1]
  indices = tf.expand_dims(tf.range(num_points), axis=1)
  indices = tf.tile(tf.expand_dims(indices, axis=0), [num_batches, 1, k])
  distances = tf.zeros([num_batches, num_points, k], dtype=tf.float32)
  return distances, indices
