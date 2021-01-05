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

"""Implements a 3D Sparse Voxel HourGlass model."""

import gin
import gin.tf
import tensorflow as tf

from tf3d.layers import sparse_voxel_unet


@gin.configurable
class SparseConvHourGlass(tf.keras.layers.Layer):
  """3D HourGlass sparse voxel network."""

  def __init__(self,
               num_stacked_networks=1,
               task_names_to_num_output_channels=None,
               task_names_to_use_relu_last_conv=None,
               task_names_to_use_batch_norm_in_last_layer=None,
               conv_filter_size=3,
               encoder_dimensions=((32, 64), (64, 128), (128, 256)),
               bottleneck_dimensions=(256, 256),
               decoder_dimensions=((256, 256), (128, 128), (64, 64)),
               dropout_prob=0.0,
               use_batch_norm=True,
               network_pooling_segment_func=tf.math.unsorted_segment_max,
               normalize_sparse_conv=True):
    """3D HourGlass sparse voxel network.

    Args:
      num_stacked_networks: Number of stacked networks that build the hour-glass
        structure.
      task_names_to_num_output_channels: A dictionary containing the mapping
        between task names to number of prediction channels for each task.
      task_names_to_use_relu_last_conv: A dictionary containing the mapping
        between task names to whether relu should be applied at the last
        convolution or not. If None, by default relu will not be applied.
      task_names_to_use_batch_norm_in_last_layer: A dictionary containing the
        mapping between task names to whether batch norm is applied to the last
        convolution of the tasks.
      conv_filter_size: The 3d convolution filter size. Currently the 3d
        convolution op is optimized for a filter size of 3.
      encoder_dimensions: A tuple of tuples, where each nested tuple is a list
        of ints describing the output feature dimensionality of each 3x3x3
        convolution. After every nested tuple we do a 2x2x2 3D Max Pooling.
      bottleneck_dimensions: A tuple of ints describing the output feature
        dimensionality of each 3x3x3 convolution in the middle of the network,
        which is after we have finished downsampling but before upsampling.
      decoder_dimensions: A tuple of tuples, where each nested tuple is a list
        of ints describing the output feature dimensionality of each 3x3x3
        convolution. Before every new nested tuple we do a 2x2x2 upsampling
        operation, and then concatenate encoder features in a UNet fashion.
      dropout_prob: A float indicating the probability of dropout.
      use_batch_norm: Whether to use batch normalization or not.
      network_pooling_segment_func: Function used to pool voxel features in the
        network.
      normalize_sparse_conv: If True, applies normalization to 3d sparse convs.

    Returns:
      A dictionary containing a predicted tensor per task. The predicted tensors
        are of size [batch_size, num_voxels, num_task_channels].

    Raises:
      ValueError: If task_names_to_num_output_channels is None.
      ValueError: If the encoder and decoder have a different number of
        downsampling/upsampling levels.
    """
    super().__init__()

    if task_names_to_num_output_channels is None:
      raise ValueError('task_names_to_num_output_channels cannot be None')

    if len(encoder_dimensions) != len(decoder_dimensions):
      raise ValueError(
          'The number of encoder and decoder blocks should be equal')

    if task_names_to_use_relu_last_conv is None:
      task_names_to_use_relu_last_conv = {}
      for key in sorted(task_names_to_num_output_channels):
        task_names_to_use_relu_last_conv[key] = False

    if task_names_to_use_batch_norm_in_last_layer is None:
      task_names_to_use_batch_norm_in_last_layer = {}
      for key in sorted(task_names_to_num_output_channels):
        task_names_to_use_batch_norm_in_last_layer[key] = False

    self.num_stacked_networks = num_stacked_networks
    self.input_spec = [
        tf.keras.layers.InputSpec(shape=(None, None, None), dtype=tf.float32),
        tf.keras.layers.InputSpec(shape=(None, None, 3), dtype=tf.int32),
        tf.keras.layers.InputSpec(shape=(None,), dtype=tf.int32)
    ]

    self.networks = []
    decoder_dimensions_last = decoder_dimensions[-1][-1]
    for i in range(num_stacked_networks):
      if i == num_stacked_networks - 1:
        task_channels = task_names_to_num_output_channels
        task_relu = task_names_to_use_relu_last_conv
        task_batch_norm = task_names_to_use_batch_norm_in_last_layer
      else:
        task_channels = {'intermediate_output': decoder_dimensions_last}
        task_relu = {'intermediate_output': True}
        task_batch_norm = {'intermediate_output': use_batch_norm}
      self.networks.append(
          sparse_voxel_unet.SparseConvUNet(
              task_names_to_num_output_channels=task_channels,
              task_names_to_use_relu_last_conv=task_relu,
              task_names_to_use_batch_norm_in_last_layer=task_batch_norm,
              conv_filter_size=conv_filter_size,
              encoder_dimensions=encoder_dimensions,
              bottleneck_dimensions=bottleneck_dimensions,
              decoder_dimensions=decoder_dimensions,
              dropout_prob=dropout_prob,
              use_batch_norm=use_batch_norm,
              network_pooling_segment_func=network_pooling_segment_func,
              normalize_sparse_conv=normalize_sparse_conv))

  def call(self, inputs, training=True):
    voxel_features, voxel_xyz_indices, num_valid_voxels = inputs
    for i in range(self.num_stacked_networks):
      outputs = self.networks[i](
          inputs=[voxel_features, voxel_xyz_indices, num_valid_voxels],
          training=training)
      if i != self.num_stacked_networks - 1:
        voxel_features = outputs['intermediate_output']
    return outputs
