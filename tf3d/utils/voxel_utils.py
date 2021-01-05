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

"""Utility function for voxels."""

import gin
import gin.tf
import tensorflow as tf

from tf3d.layers import sparse_voxel_net_utils
from object_detection.utils import shape_utils


compute_pooled_voxel_indices = sparse_voxel_net_utils.compute_pooled_voxel_indices
pool_features_given_indices = sparse_voxel_net_utils.pool_features_given_indices


def crop_and_pad_voxels(voxels, start_coordinates, end_coordinates):
  """Crops a voxel region and pads past the boundaries with zeros.

  This accepts start and end coordinates past the limits of the voxel grid,
  and uses it to calculate how much top/left/right/bottom padding to add.

  Args:
    voxels: A tf.float32 tensor of shape [x, y, z, f] to crop
    start_coordinates: A list of len 4 with the [x, y, z, f] starting location
      of our crop. This can be negative, which indicates left/top padding.
    end_coordinates: A list of len 4 with the [x, y, z, f] ending location of
      our crop. This can be beyond the size of the voxel tensor, which indicates
      padding.

  Returns:
    cropped_and_padded_voxels: A voxel grid with shape
      [end_coordinates[0] - start_coordinates[0],
       end_coordinates[1] - start_coordinates[1],
       end_coordinates[2] - start_coordinates[2],
       end_coordinates[3] - start_coordinates[3]]
  Raises:
    ValueError: If requested crop and pad is outside the bounds of what the
      function supports.
  """
  if len(start_coordinates) != 4:
    raise ValueError('start_coordinates should be of length 4')
  if len(end_coordinates) != 4:
    raise ValueError('end_coordinates should be of length 4')
  if any([coord <= 0 for coord in end_coordinates]):
    raise ValueError('Requested end coordinates should be > 0')

  start_coordinates = tf.convert_to_tensor(start_coordinates, tf.int32)
  end_coordinates = tf.convert_to_tensor(end_coordinates, tf.int32)

  # Clip the coordinates to within the voxel grid
  clipped_start_coordinates = tf.maximum(0, start_coordinates)
  clipped_end_coordinates = tf.minimum(voxels.shape, end_coordinates)

  cropped_voxels = tf.slice(voxels,
                            begin=clipped_start_coordinates,
                            size=(clipped_end_coordinates -
                                  clipped_start_coordinates))

  top_and_left_padding = tf.maximum(0, -start_coordinates)
  bottom_and_right_padding = tf.maximum(0, end_coordinates - voxels.shape)

  padding = tf.stack([top_and_left_padding, bottom_and_right_padding], axis=1)
  return tf.pad(cropped_voxels, padding)


def pointcloud_to_voxel_grid(points,
                             features,
                             grid_cell_size,
                             start_location,
                             end_location,
                             segment_func=tf.math.unsorted_segment_mean):
  """Converts a pointcloud into a voxel grid.

  Args:
    points: A tf.float32 tensor of size [N, 3].
    features: A tf.float32 tensor of size [N, F].
    grid_cell_size: A tf.float32 tensor of size [3].
    start_location: A tf.float32 tensor of size [3].
    end_location: A tf.float32 tensor of size [3].
    segment_func: A tensorflow function that operates on segments. Expect one
      of tf.math.unsorted_segment_{min/max/mean/prod/sum}. Defaults to
      tf.math.unsorted_segment_mean

  Returns:
    voxel_features: A tf.float32 tensor of
      size [grid_x_len, grid_y_len, grid_z_len, F].
    segment_ids: A tf.int32 tensor of IDs for each point indicating
      which (flattened) voxel cell its data was mapped to.
    point_indices: A tf.int32 tensor of size [num_points, 3] containing the
      location of each point in the 3d voxel grid.
  """
  grid_cell_size = tf.convert_to_tensor(grid_cell_size, dtype=tf.float32)
  start_location = tf.convert_to_tensor(start_location, dtype=tf.float32)
  end_location = tf.convert_to_tensor(end_location, dtype=tf.float32)
  point_indices = tf.cast(
      (points - tf.expand_dims(start_location, axis=0)) /
      tf.expand_dims(grid_cell_size, axis=0),
      dtype=tf.int32)
  grid_size = tf.cast(
      tf.math.ceil((end_location - start_location) / grid_cell_size),
      dtype=tf.int32)
  # Note: all points outside the grid are added to the edges
  # Cap index at grid_size - 1 (so a 10x10x10 grid's max cell is (9,9,9))
  point_indices = tf.minimum(point_indices, tf.expand_dims(grid_size - 1,
                                                           axis=0))
  # Don't allow any points below index (0, 0, 0)
  point_indices = tf.maximum(point_indices, 0)
  segment_ids = tf.reduce_sum(
      point_indices * tf.stack(
          [grid_size[1] * grid_size[2], grid_size[2], 1], axis=0),
      axis=1)
  voxel_features = segment_func(
      data=features,
      segment_ids=segment_ids,
      num_segments=(grid_size[0] * grid_size[1] * grid_size[2]))
  return (tf.reshape(voxel_features,
                     [grid_size[0],
                      grid_size[1],
                      grid_size[2],
                      features.get_shape().as_list()[1]]),
          segment_ids,
          point_indices)


def voxels_to_points(voxels, segment_ids):
  """Convert voxels back to points given their segment id.

  Args:
    voxels: A tf.float32 tensor representing a voxel grid. Expect shape
      [x, y, z, f].
    segment_ids: A tf.int32 tensor representing the segment id of each point
      in the original pointcloud we want to project voxel features back to.
  Returns:
    point_features: A tf.float32 tensor of shape [N, f] where each point
      now has the features in the associated voxel cell.
  """
  flattened_voxels = tf.reshape(voxels, shape=(-1, voxels.shape[-1]))
  return tf.gather(flattened_voxels, segment_ids)


def _points_offset_in_voxels_unbatched(points, grid_cell_size):
  """Converts points into offsets in voxel grid for a single batch.

  The values range from -0.5 to 0.5

  Args:
    points: A tf.float32 tensor of size [N, 3].
    grid_cell_size: The size of the grid cells in x, y, z dimensions in the
      voxel grid. It should be either a tf.float32 tensor, a numpy array or a
      list of size [3].

  Returns:
    voxel_xyz_offsets: A tf.float32 tensor of size [N, 3].
  """
  min_points = tf.reduce_min(points, axis=0)
  points_index = tf.math.floordiv(points - min_points, grid_cell_size)
  points_offset = points - min_points - (points_index * grid_cell_size)
  return (points_offset / grid_cell_size) - 0.5


def points_offset_in_voxels(points, grid_cell_size):
  """Converts points into offsets in voxel grid.

  Args:
    points: A tf.float32 tensor of size [batch_size, N, 3].
    grid_cell_size: The size of the grid cells in x, y, z dimensions in the
      voxel grid. It should be either a tf.float32 tensor, a numpy array or a
      list of size [3].

  Returns:
    voxel_xyz_offsets: A tf.float32 tensor of size [batch_size, N, 3].
  """
  batch_size = points.get_shape().as_list()[0]

  def fn(i):
    return _points_offset_in_voxels_unbatched(
        points=points[i, :, :], grid_cell_size=grid_cell_size)

  return tf.map_fn(fn=fn, elems=tf.range(batch_size), dtype=tf.float32)


def _points_to_voxel_indices(points, grid_cell_size):
  """Converts points into corresponding voxel indices.

  Maps each point into a voxel grid with cell size given by grid_cell_size.
  For each voxel, it computes a x, y, z index. Also converts the x, y, z index
  to a single number index where there is a one-on-one mapping between
  each x, y, z index value and its corresponding single number index value.

  Args:
    points: A tf.float32 tensor of size [N, 3].
    grid_cell_size: The size of the grid cells in x, y, z dimensions in the
      voxel grid. It should be either a tf.float32 tensor, a numpy array or a
      list of size [3].

  Returns:
    voxel_xyz_indices: A tf.int32 tensor of size [N, 3] containing the x, y, z
      index of the voxel corresponding to each given point.
    voxel_single_number_indices: A tf.int32 tensor of size [N] containing the
      single number index of the voxel corresponding to each given point.
    voxel_start_location: A tf.float32 tensor of size [3] containing the start
      location of the voxels.
  """
  voxel_start_location = tf.reduce_min(points, axis=0)
  voxel_xyz_indices = tf.cast(
      tf.math.floordiv(points - voxel_start_location, grid_cell_size),
      dtype=tf.int32)
  voxel_xyz_indices, voxel_single_number_indices = compute_pooled_voxel_indices(
      voxel_xyz_indices=voxel_xyz_indices, pooling_size=(1, 1, 1))
  return voxel_xyz_indices, voxel_single_number_indices, voxel_start_location


def pointcloud_to_sparse_voxel_grid_unbatched(points, features, grid_cell_size,
                                              segment_func):
  """Converts a pointcloud into a voxel grid.

  This function does not handle batch size and only works for a single batch
  of points. The function `pointcloud_to_sparse_voxel_grid` below calls this
  function in a while loop to map a batch of points to a batch of voxels.

  A sparse voxel grid is represented by only keeping the voxels that
  have points in them in memory. Assuming that N' voxels have points in them,
  we represent a sparse voxel grid by
    (a) voxel_features, a [N', F] or [N', G, F] tensor containing the feature
          vector for each voxel.
    (b) voxel_indices, a [N', 3] tensor containing the x, y, z index of each
          voxel.

  Args:
    points: A tf.float32 tensor of size [N, 3].
    features: A tf.float32 tensor of size [N, F].
    grid_cell_size: The size of the grid cells in x, y, z dimensions in the
      voxel grid. It should be either a tf.float32 tensor, a numpy array or a
      list of size [3].
    segment_func: A tensorflow function that operates on segments. Examples are
      one of tf.math.unsorted_segment_{min/max/mean/prod/sum}.

  Returns:
    voxel_features: A tf.float32 tensor of size [N', F] or [N', G, F] where G is
      the number of points sampled per voxel.
    voxel_indices: A tf.int32 tensor of size [N', 3].
    segment_ids: A size [N] tf.int32 tensor of IDs for each point indicating
      which (flattened) voxel cell its data was mapped to.
    voxel_start_location: A tf.float32 tensor of size [3] containing the start
      location of the voxels.

  Raises:
    ValueError: If pooling method is unknown.
  """
  grid_cell_size = tf.convert_to_tensor(grid_cell_size, dtype=tf.float32)
  voxel_xyz_indices, voxel_single_number_indices, voxel_start_location = (
      _points_to_voxel_indices(points=points, grid_cell_size=grid_cell_size))
  voxel_features, segment_ids, num_segments = pool_features_given_indices(
      features=features,
      indices=voxel_single_number_indices,
      segment_func=segment_func)
  voxel_xyz_indices = tf.math.unsorted_segment_max(
      data=voxel_xyz_indices,
      segment_ids=segment_ids,
      num_segments=num_segments)
  return voxel_features, voxel_xyz_indices, segment_ids, voxel_start_location


def _pad_or_clip_voxels(voxel_features, voxel_indices, num_valid_voxels,
                        segment_ids, voxels_pad_or_clip_size):
  """Pads or clips voxels."""
  if voxels_pad_or_clip_size:
    num_valid_voxels = tf.minimum(num_valid_voxels, voxels_pad_or_clip_size)
    num_channels = voxel_features.get_shape().as_list()[-1]
    if len(voxel_features.shape.as_list()) == 2:
      output_shape = [voxels_pad_or_clip_size, num_channels]
    elif len(voxel_features.shape.as_list()) == 3:
      num_samples_per_voxel = voxel_features.get_shape().as_list()[1]
      if num_samples_per_voxel is None:
        num_samples_per_voxel = tf.shape(voxel_features)[1]
      output_shape = [
          voxels_pad_or_clip_size, num_samples_per_voxel, num_channels
      ]
    else:
      raise ValueError('voxel_features should be either rank 2 or 3.')
    voxel_features = shape_utils.pad_or_clip_nd(
        tensor=voxel_features, output_shape=output_shape)
    voxel_indices = shape_utils.pad_or_clip_nd(
        tensor=voxel_indices, output_shape=[voxels_pad_or_clip_size, 3])
    valid_segment_ids_mask = tf.cast(
        tf.less(segment_ids, num_valid_voxels), dtype=tf.int32)
    segment_ids *= valid_segment_ids_mask
  return voxel_features, voxel_indices, num_valid_voxels, segment_ids


def pointcloud_to_sparse_voxel_grid(points, features, num_valid_points,
                                    grid_cell_size, voxels_pad_or_clip_size,
                                    segment_func):
  """Converts a pointcloud into a voxel grid.

  This function calls the `pointcloud_to_sparse_voxel_grid_unbatched`
  function avove in a while loop to map a batch of points to a batch of voxels.

  Args:
    points: A tf.float32 tensor of size [batch_size, N, 3].
    features: A tf.float32 tensor of size [batch_size, N, F].
    num_valid_points: A tf.int32 tensor of size [num_batches] containing the
      number of valid points in each batch example.
    grid_cell_size: A tf.float32 tensor of size [3].
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the padding.
    segment_func: A tensorflow function that operates on segments. Examples are
      one of tf.math.unsorted_segment_{min/max/mean/prod/sum}.

  Returns:
    voxel_features: A tf.float32 tensor of size [batch_size, N', F]
      or [batch_size, N', G, F] where G is the number of points sampled per
      voxel.
    voxel_indices: A tf.int32 tensor of size [batch_size, N', 3].
    num_valid_voxels: A tf.int32 tensor of size [batch_size].
    segment_ids: A size [batch_size, N] tf.int32 tensor of IDs for each point
      indicating which (flattened) voxel cell its data was mapped to.
    voxel_start_location: A size [batch_size, 3] tf.float32 tensor of voxel
      start locations.

  Raises:
    ValueError: If pooling method is unknown.
  """
  batch_size = points.get_shape().as_list()[0]
  if batch_size is None:
    batch_size = tf.shape(points)[0]
  num_points = tf.shape(points)[1]

  def fn(i):
    """Map function."""
    num_valid_points_i = num_valid_points[i]
    points_i = points[i, :num_valid_points_i, :]
    features_i = features[i, :num_valid_points_i, :]
    voxel_features_i, voxel_indices_i, segment_ids_i, voxel_start_location_i = (
        pointcloud_to_sparse_voxel_grid_unbatched(
            points=points_i,
            features=features_i,
            grid_cell_size=grid_cell_size,
            segment_func=segment_func))
    num_valid_voxels_i = tf.shape(voxel_features_i)[0]
    (voxel_features_i, voxel_indices_i, num_valid_voxels_i,
     segment_ids_i) = _pad_or_clip_voxels(
         voxel_features=voxel_features_i,
         voxel_indices=voxel_indices_i,
         num_valid_voxels=num_valid_voxels_i,
         segment_ids=segment_ids_i,
         voxels_pad_or_clip_size=voxels_pad_or_clip_size)
    segment_ids_i = tf.pad(
        segment_ids_i, paddings=[[0, num_points - num_valid_points_i]])
    return (voxel_features_i, voxel_indices_i, num_valid_voxels_i,
            segment_ids_i, voxel_start_location_i)

  return tf.map_fn(
      fn=fn,
      elems=tf.range(batch_size),
      dtype=(tf.float32, tf.int32, tf.int32, tf.int32, tf.float32))


def sparse_voxel_grid_to_pointcloud(voxel_features, segment_ids,
                                    num_valid_voxels, num_valid_points):
  """Convert voxel features back to points given their segment ids.

  Args:
    voxel_features: A tf.float32 tensor of size [batch_size, N', F].
    segment_ids: A size [batch_size, N] tf.int32 tensor of IDs for each point
      indicating which (flattened) voxel cell its data was mapped to.
    num_valid_voxels: A tf.int32 tensor of size [batch_size] containing the
      number of valid voxels in each batch example.
    num_valid_points: A tf.int32 tensor of size [batch_size] containing the
      number of valid points in each batch example.

  Returns:
    point_features: A tf.float32 tensor of size [batch_size, N, F].

  Raises:
    ValueError: If batch_size is unknown at graph construction time.
  """
  batch_size = voxel_features.shape[0]
  if batch_size is None:
    raise ValueError('batch_size is unknown at graph construction time.')
  num_points = tf.shape(segment_ids)[1]

  def fn(i):
    num_valid_voxels_i = num_valid_voxels[i]
    num_valid_points_i = num_valid_points[i]
    voxel_features_i = voxel_features[i, :num_valid_voxels_i, :]
    segment_ids_i = segment_ids[i, :num_valid_points_i]
    point_features = tf.gather(voxel_features_i, segment_ids_i)
    point_features_rank = len(point_features.get_shape().as_list())
    point_features_paddings = [[0, num_points - num_valid_points_i]]
    for _ in range(point_features_rank - 1):
      point_features_paddings.append([0, 0])
    point_features = tf.pad(point_features, paddings=point_features_paddings)
    return point_features

  return tf.map_fn(fn=fn, elems=tf.range(batch_size), dtype=tf.float32)


@gin.configurable
def per_voxel_point_sample_segment_func(data, segment_ids, num_segments,
                                        num_samples_per_voxel):
  """Samples features from the points within each voxel.

  Args:
    data: A tf.float32 tensor of size [N, F].
    segment_ids: A tf.int32 tensor of size [N].
    num_segments: Number of segments.
    num_samples_per_voxel: Number of features to sample per voxel. If the voxel
      has less number of points in it, the point features will be padded by 0.

  Returns:
    A tf.float32 tensor of size [num_segments, num_samples_per_voxel, F].
    A tf.int32 indices of size [N, num_samples_per_voxel].
  """
  num_channels = data.get_shape().as_list()[1]
  if num_channels is None:
    raise ValueError('num_channels is None.')
  n = tf.shape(segment_ids)[0]

  def _body_fn(i, indices_range, indices):
    """Computes the indices of the i-th point feature in each segment."""
    indices_i = tf.math.unsorted_segment_max(
        data=indices_range, segment_ids=segment_ids, num_segments=num_segments)
    indices_i_positive_mask = tf.greater(indices_i, 0)
    indices_i_positive = tf.boolean_mask(indices_i, indices_i_positive_mask)
    boolean_mask = tf.scatter_nd(
        indices=tf.cast(
            tf.expand_dims(indices_i_positive - 1, axis=1), dtype=tf.int64),
        updates=tf.ones_like(indices_i_positive, dtype=tf.int32),
        shape=(n,))
    indices_range *= (1 - boolean_mask)
    indices_i *= tf.cast(indices_i_positive_mask, dtype=tf.int32)
    indices_i = tf.pad(
        tf.expand_dims(indices_i, axis=1),
        paddings=[[0, 0], [i, num_samples_per_voxel - i - 1]])
    indices += indices_i
    i = i + 1
    return i, indices_range, indices

  cond = lambda i, indices_range, indices: i < num_samples_per_voxel

  (_, _, indices) = tf.while_loop(
      cond=cond,
      body=_body_fn,
      loop_vars=(tf.constant(0, dtype=tf.int32), tf.range(n) + 1,
                 tf.zeros([num_segments, num_samples_per_voxel],
                          dtype=tf.int32)))

  data = tf.pad(data, paddings=[[1, 0], [0, 0]])
  voxel_features = tf.gather(data, tf.reshape(indices, [-1]))
  return tf.reshape(voxel_features,
                    [num_segments, num_samples_per_voxel, num_channels])


def compute_pointcloud_weights_based_on_voxel_density(points, grid_cell_size):
  """Computes pointcloud weights based on voxel density.

  Args:
    points: A tf.float32 tensor of size [num_points, 3].
    grid_cell_size: The size of the grid cells in x, y, z dimensions in the
      voxel grid. It should be either a tf.float32 tensor, a numpy array or a
      list of size [3].

  Returns:
    A tf.float32 tensor of size [num_points, 1] containing weights that are
      inverse proportional to the denisty of the points in voxels.
  """
  num_points = tf.shape(points)[0]
  features = tf.ones([num_points, 1], dtype=tf.float32)
  voxel_features, _, segment_ids, _ = (
      pointcloud_to_sparse_voxel_grid_unbatched(
          points=points,
          features=features,
          grid_cell_size=grid_cell_size,
          segment_func=tf.math.unsorted_segment_sum))
  num_voxels = tf.shape(voxel_features)[0]
  point_features = sparse_voxel_grid_to_pointcloud(
      voxel_features=tf.expand_dims(voxel_features, axis=0),
      segment_ids=tf.expand_dims(segment_ids, axis=0),
      num_valid_voxels=tf.expand_dims(num_voxels, axis=0),
      num_valid_points=tf.expand_dims(num_points, axis=0))
  inverse_point_densities = 1.0 / tf.squeeze(point_features, axis=0)
  total_inverse_density = tf.reduce_sum(inverse_point_densities)
  return (inverse_point_densities * tf.cast(num_points, dtype=tf.float32) /
          total_inverse_density)
