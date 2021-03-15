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

"""Sparse conv utility functions."""

import tensorflow as tf

ops_imported = False


if not ops_imported:
  try:
    import tensorflow_sparse_conv_ops as sparse_conv_ops  # pylint: disable=g-import-not-at-top
  except ImportError:
    import tf3d.ops.tensorflow_sparse_conv_ops as sparse_conv_ops  # pylint: disable=g-import-not-at-top


def compute_pooled_voxel_indices(voxel_xyz_indices, pooling_size):
  """Computes and returns the pooled voxel indices.

  Applies the pooling based on the given `pooling_size` and computes
  x, y, z indices for the pooled voxels. Also converts the x, y, z index
  to a single number index where there is a one-on-one mapping between
  each x, y, z index value and its corresponding single number index value.

  Args:
    voxel_xyz_indices: A tf.int32 tensor of size [N, 3] containing voxel
      x, y, z indices.
    pooling_size: The size of the pooling window in x, y, z dimensions in the
      voxel grid. It should be either a tf.int32 tensor, a numpy array or a
      list of size [3].

  Returns:
    pooled_xyz_indices: A tf.int32 tensor of size [N, 3] containing the x, y, z
      index of the pooled voxel corresponding to each given voxel.
    pooled_single_number_indices: A tf.int32 tensor of size [N] containing the
      single number index of the pooled voxel corresponding to each given voxel.
  """
  pooling_size = tf.convert_to_tensor(pooling_size, dtype=tf.int32)
  pooled_xyz_indices = tf.cast(
      tf.math.floordiv(voxel_xyz_indices, pooling_size), dtype=tf.int32)
  pooled_xyz_indices_min = tf.reduce_min(pooled_xyz_indices, axis=0)
  pooled_xyz_indices -= pooled_xyz_indices_min
  pooled_xyz_indices_max = tf.reduce_max(pooled_xyz_indices, axis=0)
  xyz_to_single_number_mapping_coefs = [
      (pooled_xyz_indices_max[1] + 1) * (pooled_xyz_indices_max[2] + 1),
      (pooled_xyz_indices_max[2] + 1), 1
  ]
  pooled_single_number_indices = tf.reduce_sum(
      pooled_xyz_indices * tf.expand_dims(
          tf.stack(xyz_to_single_number_mapping_coefs), axis=0),
      axis=1)
  pooled_xyz_indices += pooled_xyz_indices_min
  return pooled_xyz_indices, pooled_single_number_indices


def pool_features_given_indices(features, indices, segment_func):
  """Pools the features based on their indices.

  If more than one feature have the same index, it will use the pooling method
  to aggregate the features.

  Args:
    features: A tensor of size [N, F].
    indices: A tf.int32 tensor of size [N].
    segment_func: A tensorflow function that operates on segments. Examples
      are one of tf.math.unsorted_segment_{min/max/mean/prod/sum}.

  Returns:
    pooled_features: A tensor of size [N', F] or [N', G, F] where G is the
      number of points sampled per voxel.
    segment_ids: A tf.int32 tensor of size [N].
    num_segments: A tf.int32 scalar corresponding to number of segments.

  Raises:
    ValueError: If pooling method is unknown.
  """
  segment_ids = None


  if segment_ids is None:  # using external implementation
    _, segment_ids = tf.unique(indices)

  num_segments = tf.reduce_max(segment_ids) + 1
  # Each voxel might contain more than one point. Here we pool the point
  # features either using mean or max.
  pooled_features = segment_func(
      data=features, segment_ids=segment_ids, num_segments=num_segments)
  return pooled_features, segment_ids, num_segments


def voxel_pooling(voxel_features, voxel_xyz_indices, num_valid_voxels,
                  pooling_size, segment_func=tf.math.unsorted_segment_max):
  """Pools voxel features.

  Args:
    voxel_features: A tf.float32 tensor of size [batch_size, N, fd] where
      fd is the feature size.
    voxel_xyz_indices: A tf.int32 tensor of size [batch_size, N, 3] containing
      the voxel index in each of the x, y, z dimensions.
    num_valid_voxels: A tf.int32 tensor of size [batch_size].
    pooling_size: A tf.int32 tensor of size [3] containing the pooling size.
    segment_func: A function defining the pooling method.

  Returns:
    pooled_voxel_features: A tf.float32 tensor of size [batch_size, N', fd].
    pooled_voxel_indices: A tf.int32 tensor of size [batch_size, N', 3].
    num_valid_pooled_voxels: A tf.int32 tensor of size [batch_size].
    index_mapping: A tf.int32 tensor of size [batch_size, N] containing the
      mapping from voxel indices to pooled voxel indices.

  Raises:
    ValueError: If pooling method is unknown.
    ValueError: If batch size or feature dimensions are unknown at graph
      construction time.
  """
  batch_size = voxel_xyz_indices.get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError("batch_size is unknown at graph construction time.")
  feature_dims = voxel_features.get_shape().as_list()[2]
  if feature_dims is None:
    raise ValueError("Feature dimension is unknown at graph construction time.")
  num_voxels = tf.shape(voxel_features)[1]

  def _slice_valid_voxels(i, num_valid_voxels_i):
    """Slices valid voxels and indices."""
    voxel_features_i = tf.slice(
        voxel_features,
        begin=[i, 0, 0],
        size=[1, num_valid_voxels_i, feature_dims])
    voxel_features_i = tf.squeeze(voxel_features_i, axis=0)
    voxel_xyz_indices_i = tf.slice(
        voxel_xyz_indices, begin=[i, 0, 0], size=[1, num_valid_voxels_i, 3])
    voxel_xyz_indices_i = tf.squeeze(voxel_xyz_indices_i, axis=0)
    return voxel_features_i, voxel_xyz_indices_i

  def _voxel_pooling_unbatched(voxel_features, voxel_xyz_indices, pooling_size,
                               segment_func):
    """Pools voxel features.

    Args:
      voxel_features: A tf.float32 tensor of size [N, fd] where fd is the
        feature size.
      voxel_xyz_indices: A tf.int32 tensor of size [N, 3] containing the voxel
        index in each of the x, y, z dimensions.
      pooling_size: A tf.int32 tensor of size [3] containing the pooling size.
      segment_func: A function defining the pooling method.

    Returns:
      pooled_voxel_features: A tf.float32 tensor of size [N', fd].
      pooled_voxel_indices: A tf.int32 tensor of size [N', 3].
      index_mapping: A tf.int32 tensor of size [N] containing the mapping from
        voxel indices to pooled voxel indices.

    Raises:
      ValueError: If pooling method is unknown.
    """
    pooled_xyz_indices, pooled_single_number_indices = (
        compute_pooled_voxel_indices(
            voxel_xyz_indices=voxel_xyz_indices, pooling_size=pooling_size))
    (pooled_voxel_features, segment_ids,
     num_segments) = pool_features_given_indices(
         features=voxel_features,
         indices=pooled_single_number_indices,
         segment_func=segment_func)
    # The original pooled_xyz_indices where zeroed out so it becomes easier to
    # compute a single index number that uniquely maps to each x, y, z index.
    # Here we add the pooled_xyz_indices_min back to pooled_xyz_indices to
    # offset that.
    pooled_voxel_indices = tf.math.unsorted_segment_max(
        data=pooled_xyz_indices,
        segment_ids=segment_ids,
        num_segments=num_segments)
    return pooled_voxel_features, pooled_voxel_indices, segment_ids

  def _pad_pooled_voxels(pooled_voxel_features_i, pooled_voxel_indices_i,
                         index_mapping_i, num_valid_voxels_i, num_voxels):
    """Pad pooled voxels helper function."""
    num_valid_pooled_voxels_i = tf.shape(pooled_voxel_features_i)[0]
    pooled_voxel_features_i = tf.pad(
        pooled_voxel_features_i,
        paddings=[[0, num_voxels - num_valid_pooled_voxels_i], [0, 0]])
    pooled_voxel_indices_i = tf.pad(
        pooled_voxel_indices_i,
        paddings=[[0, num_voxels - num_valid_pooled_voxels_i], [0, 0]])
    index_mapping_i = tf.pad(
        index_mapping_i, paddings=[[0, num_voxels - num_valid_voxels_i]])
    return (pooled_voxel_features_i, pooled_voxel_indices_i,
            num_valid_pooled_voxels_i, index_mapping_i)

  def fn(i):
    """Map function."""
    num_valid_voxels_i = num_valid_voxels[i]
    voxel_features_i, voxel_xyz_indices_i = _slice_valid_voxels(
        i=i, num_valid_voxels_i=num_valid_voxels_i)
    (pooled_voxel_features_i,
     pooled_voxel_indices_i,
     index_mapping_i) = _voxel_pooling_unbatched(
         voxel_features=voxel_features_i,
         voxel_xyz_indices=voxel_xyz_indices_i,
         pooling_size=pooling_size,
         segment_func=segment_func)
    return _pad_pooled_voxels(
        pooled_voxel_features_i=pooled_voxel_features_i,
        pooled_voxel_indices_i=pooled_voxel_indices_i,
        index_mapping_i=index_mapping_i,
        num_valid_voxels_i=num_valid_voxels_i,
        num_voxels=num_voxels)

  (pooled_voxel_features, pooled_voxel_indices, num_valid_pooled_voxels,
   index_mapping) = tf.map_fn(
       fn=fn,
       elems=tf.range(batch_size),
       dtype=(tf.float32, tf.int32, tf.int32, tf.int32))

  # Maximum number of valid_pooled_voxels across the batch.
  max_num_valid_pooled_voxels = tf.reduce_max(num_valid_pooled_voxels)
  pooled_voxel_features = tf.slice(
      pooled_voxel_features,
      begin=[0, 0, 0],
      size=[batch_size, max_num_valid_pooled_voxels, feature_dims])
  pooled_voxel_indices = tf.slice(
      pooled_voxel_indices,
      begin=[0, 0, 0],
      size=[batch_size, max_num_valid_pooled_voxels, 3])
  return (pooled_voxel_features, pooled_voxel_indices, num_valid_pooled_voxels,
          index_mapping)


def voxel_upsampling(pooled_voxel_features, index_mapping):
  """Upsamples voxel features.

  Args:
    pooled_voxel_features: A tf.float32 tensor of size [batch_size, N', fd].
    index_mapping: A tf.int32 tensor of size [batch_size, N] containing the
      mapping from the original voxel indices to pooled voxel indices.

  Returns:
    voxel_features: A tf.float32 tensor of size [batch_size, N, fd] where
      fd is the feature size.
  """
  return tf.gather(pooled_voxel_features, index_mapping, batch_dims=1)


class MaskedBatchNorm(tf.keras.layers.Layer):
  """Applies batch norm to only valid input features."""

  def __init__(self):
    super().__init__()
    self.bn = tf.keras.layers.BatchNormalization()
    self.input_spec = [
        tf.keras.layers.InputSpec(shape=(None, None, None), dtype=tf.float32),
        tf.keras.layers.InputSpec(shape=(None,), dtype=tf.int32)
    ]

  def build(self, input_shapes):
    """Masked batch norm build function."""
    voxel_features_shape = input_shapes[0]
    self.batch_size = voxel_features_shape[0]

  def call(self, inputs, training=None):
    """Masked batch norm call function.

    Args:
      inputs: A list of tensors containing
        [voxel_features] A tf.float32 tensor of size [batch_size, N, fd] where
          fd is the feature size.
        [num_valid_voxels] A tf.int32 tensor of size [batch_size] containing the
          number of valid voxels for each example in the batch.
      training: If the layer is being executed in training mode (useful for
        the traditional batch normalization layer inside).

    Returns:
      voxel_features after applying batch norm.
    """
    if len(inputs) != 2:
      raise ValueError("inputs should have a length of 2.")
    voxel_features, num_valid_voxels = inputs
    if num_valid_voxels is None:
      return self.bn(voxel_features, training=training)
    num_voxels = tf.shape(voxel_features)[1]
    unpadded_features = []
    for i in range(self.batch_size):
      unpadded_features.append(voxel_features[i, 0:num_valid_voxels[i], :])
    unpadded_features = tf.concat(unpadded_features, axis=0)
    unpadded_features = self.bn(unpadded_features, training=training)
    num_valid_voxels_cumsum = tf.math.cumsum(num_valid_voxels)
    num_valid_voxels_cumsum = tf.concat([
        tf.constant([0], dtype=num_valid_voxels_cumsum.dtype),
        num_valid_voxels_cumsum
    ],
                                        axis=0)
    padded_features = []
    for i in range(self.batch_size):
      unpadded_features_i = unpadded_features[
          num_valid_voxels_cumsum[i]:num_valid_voxels_cumsum[i + 1], :]
      padded_features_i = tf.pad(
          unpadded_features_i,
          paddings=[[0, num_voxels - num_valid_voxels[i]], [0, 0]])
      padded_features.append(padded_features_i)
    return tf.stack(padded_features, axis=0)


class SparseConvBlock3D(tf.keras.layers.Layer):
  """Applies a series of 3d sparse convolutions to the voxel features."""

  def __init__(self,
               num_convolution_channels_list,
               conv_filter_size=3,
               use_batch_norm=True,
               dropout_prob=0.0,
               apply_relu_to_last_conv=True,
               normalize_sparse_conv=True):
    """3D sparse conv block constructor.

    The block contains a sequence of 3d sparse convolutions.

    Args:
      num_convolution_channels_list: A list that contains the number of output
        channels of the convolutions in the block. The length of
        the list identifies the number of convolutions in the block.
      conv_filter_size: The 3d convolution filter size. The 3d sparse
        convolution op is highly optimized for a filter of size 3.
      use_batch_norm: If True, it will train with batch norm.
      dropout_prob: Dropout probability.
      apply_relu_to_last_conv: If True, will apply relu to the last convolution
        of the block.
      normalize_sparse_conv: If True, performs a convolution on the 0-1 voxel
        occupancy grid and normalizes the sparse conv output with that.
    """
    super().__init__()

    self.num_convolution_channels_list = num_convolution_channels_list
    self.num_convolutions = len(num_convolution_channels_list)
    self.conv_filter_size = conv_filter_size
    self.use_batch_norm = use_batch_norm
    self.dropout_prob = dropout_prob
    self.apply_relu_to_last_conv = apply_relu_to_last_conv
    self.normalize_sparse_conv = normalize_sparse_conv
    self.use_tpu = False
    self.batch_norm_fns = []
    for _ in range(self.num_convolutions):
      self.batch_norm_fns.append(MaskedBatchNorm())
    self.dropout_fn = tf.keras.layers.Dropout(dropout_prob)
    self.input_spec = [
        tf.keras.layers.InputSpec(shape=(None, None, None), dtype=tf.float32),
        tf.keras.layers.InputSpec(shape=(None, None, 3), dtype=tf.int32),
        tf.keras.layers.InputSpec(shape=(None,), dtype=tf.int32)
    ]

  def build(self, input_shapes):
    """Building layer weights."""
    if len(input_shapes) != 3:
      raise ValueError("input_shapes should have a length of 3.")
    voxel_features_shape = input_shapes[0]
    self.batch_size = voxel_features_shape[0]
    num_channels = voxel_features_shape[2]
    self.ws = []
    self.normalizer_ws = []
    for i in range(self.num_convolutions):
      self.ws.append(
          self.add_weight(
              shape=(self.conv_filter_size, self.conv_filter_size,
                     self.conv_filter_size, num_channels,
                     self.num_convolution_channels_list[i]),
              name=("conv_kernel_{}".format(i)),
              initializer="random_uniform",
              trainable=True))
      num_channels = self.num_convolution_channels_list[i]
      if self.normalize_sparse_conv:
        self.normalizer_ws.append(
            self.add_weight(
                shape=(self.conv_filter_size, self.conv_filter_size,
                       self.conv_filter_size, 1, 1),
                name=("normalizer_conv_kernel_{}".format(i)),
                initializer="random_uniform",
                trainable=True))

  def call(self, inputs, training=None):
    """3D sparse conv block call function.

    Args:
      inputs: A list of tensors containing
        [voxel_features] A tf.float32 tensor of size [batch_size, N, fd] where
          fd is the feature size.
        [voxel_xyz_indices] A tf.int32 tensor of size [batch_size, N, 3]
          containing the voxel index in each of the x, y, z dimensions.
        [num_valid_voxels] A tf.int32 tensor of size [batch_size] containing the
          number of valid voxels for each example in the batch.
      training: If the layer is being executed in training mode (useful for
        the traditional batch normalization layer inside).

    Returns:
      convolved voxel features of size [batch_size, N, fd'] where fd' is the
        number of output channels of last convolution in the block.
    """
    if len(inputs) != 3:
      raise ValueError("inputs should have a length of 3.")
    voxel_features, voxel_xyz_indices, num_valid_voxels = inputs
    num_voxels = tf.shape(voxel_features)[1]
    rules = None


    net = voxel_features
    for i in range(self.num_convolutions):


      if rules is None:  # using external implementation
        net = sparse_conv_ops.submanifold_sparse_conv3d(voxel_xyz_indices,
                                                        num_valid_voxels, net,
                                                        self.ws[i])

      if self.normalize_sparse_conv:
        net_normalizer = tf.ones(
            tf.stack([self.batch_size, num_voxels, 1]), dtype=tf.float32)


        if rules is None:  # using external implementation
          net_normalizer = sparse_conv_ops.submanifold_sparse_conv3d(
              voxel_xyz_indices, num_valid_voxels, net_normalizer,
              self.normalizer_ws[i])

        net = tf.math.truediv(net, 1.0 + tf.math.abs(net_normalizer))
      if self.use_batch_norm:
        net = self.batch_norm_fns[i]([net, num_valid_voxels], training)
      if self.apply_relu_to_last_conv or i < (self.num_convolutions - 1):
        net = tf.nn.relu(net)
      if self.dropout_prob > 0:
        net = self.dropout_fn(net)
    return net
