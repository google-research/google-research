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

"""Implements a 3D Sparse Voxel UNet model.

See Najibi et al., CVPR 2020
"""

import gin
import gin.tf
import tensorflow as tf

from tf3d.layers import sparse_voxel_net_utils


@gin.configurable
class SparseConvUNet(tf.keras.layers.Layer):
  """3D UNet sparse voxel network.

  Please refer to the following paper for more details:
  M. Najibi, G. Lai, A. Kundu, Z. Lu, V. Rathod, T. Funkhouser, C. Pantofaru,
  D. Ross, L. S. Davis, A. Fathi,
  'DOPS: Learning to Detect 3D Objects and Predict Their 3D Shapes', CVPR 2020.
  """

  def __init__(self,
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
    """3D UNet sparse voxel network.

    Args:
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

    self.num_levels = len(encoder_dimensions)
    self.network_pooling_segment_func = network_pooling_segment_func
    self.task_names_to_num_output_channels = task_names_to_num_output_channels
    self.input_spec = [
        tf.keras.layers.InputSpec(shape=(None, None, None), dtype=tf.float32),
        tf.keras.layers.InputSpec(shape=(None, None, 3), dtype=tf.int32),
        tf.keras.layers.InputSpec(shape=(None,), dtype=tf.int32)
    ]

    for level in range(self.num_levels):
      conv_block_i = sparse_voxel_net_utils.SparseConvBlock3D(
          num_convolution_channels_list=encoder_dimensions[level],
          conv_filter_size=conv_filter_size,
          use_batch_norm=use_batch_norm,
          dropout_prob=dropout_prob,
          apply_relu_to_last_conv=True,
          normalize_sparse_conv=normalize_sparse_conv)
      setattr(self, 'encoder_' + str(level), conv_block_i)

    self.middle_layer = sparse_voxel_net_utils.SparseConvBlock3D(
        num_convolution_channels_list=bottleneck_dimensions,
        conv_filter_size=conv_filter_size,
        use_batch_norm=use_batch_norm,
        dropout_prob=dropout_prob,
        apply_relu_to_last_conv=True,
        normalize_sparse_conv=normalize_sparse_conv)

    for level in reversed(range(self.num_levels)):
      conv_block_i = sparse_voxel_net_utils.SparseConvBlock3D(
          num_convolution_channels_list=decoder_dimensions[self.num_levels -
                                                           level - 1],
          conv_filter_size=conv_filter_size,
          use_batch_norm=use_batch_norm,
          dropout_prob=dropout_prob,
          apply_relu_to_last_conv=True,
          normalize_sparse_conv=normalize_sparse_conv)
      setattr(self, 'decoder_' + str(level), conv_block_i)

    for task_name in sorted(task_names_to_num_output_channels):
      num_task_channels = task_names_to_num_output_channels[task_name]
      num_channels = encoder_dimensions[0][-1]
      conv_block_task_1 = sparse_voxel_net_utils.SparseConvBlock3D(
          num_convolution_channels_list=[num_channels, num_channels],
          conv_filter_size=conv_filter_size,
          use_batch_norm=True,
          dropout_prob=dropout_prob,
          apply_relu_to_last_conv=True,
          normalize_sparse_conv=normalize_sparse_conv)
      setattr(self, f'{task_name}/final_conv1_block', conv_block_task_1)

      conv_block_task_2 = sparse_voxel_net_utils.SparseConvBlock3D(
          num_convolution_channels_list=[num_task_channels],
          conv_filter_size=conv_filter_size,
          use_batch_norm=task_names_to_use_batch_norm_in_last_layer[task_name],
          dropout_prob=0.0,
          apply_relu_to_last_conv=task_names_to_use_relu_last_conv[task_name],
          normalize_sparse_conv=normalize_sparse_conv)
      setattr(self, f'{task_name}/final_conv2_block', conv_block_task_2)

  def call(self, inputs, training=True):
    voxel_features, voxel_xyz_indices, num_valid_voxels = inputs
    voxel_features_list = [voxel_features]
    voxel_xyz_indices_list = [voxel_xyz_indices]
    num_valid_voxels_list = [num_valid_voxels]
    index_mapping_list = []

    if training:
      tf.print('num_valid_voxels', num_valid_voxels)

    # Encoder
    for level in range(self.num_levels):
      inputs_i = (voxel_features_list[-1], voxel_xyz_indices_list[-1],
                  num_valid_voxels_list[-1])
      conv_block_i = getattr(self, 'encoder_' + str(level))
      outputs_i = conv_block_i(inputs_i, training)
      (pooled_voxel_features, pooled_voxel_indices, num_valid_pooled_voxels,
       index_mapping) = sparse_voxel_net_utils.voxel_pooling(
           voxel_features=outputs_i,
           voxel_xyz_indices=inputs_i[1],
           num_valid_voxels=inputs_i[2],
           pooling_size=(2, 2, 2),
           segment_func=self.network_pooling_segment_func)
      voxel_features_list.append(pooled_voxel_features)
      voxel_xyz_indices_list.append(pooled_voxel_indices)
      num_valid_voxels_list.append(num_valid_pooled_voxels)
      index_mapping_list.append(index_mapping)

    # Bottleneck
    outputs_midl = self.middle_layer((
        voxel_features_list[-1],
        voxel_xyz_indices_list[-1],
        num_valid_voxels_list[-1],
    ), training)
    voxel_features_list[-1] = outputs_midl

    # Decoder
    for level in reversed(range(self.num_levels)):
      unpooled_features = sparse_voxel_net_utils.voxel_upsampling(
          pooled_voxel_features=voxel_features_list[level + 1],
          index_mapping=index_mapping_list[level])
      concatenated_features = tf.concat(
          [voxel_features_list[level], unpooled_features], axis=2)
      inputs_i = (concatenated_features, voxel_xyz_indices_list[level],
                  num_valid_voxels_list[level])
      conv_block_i = getattr(self, 'decoder_' + str(level))
      outputs_i = conv_block_i(inputs_i, training)
      voxel_features_list[level] = outputs_i

    outputs = {}
    # Output head convolutions
    for task_name in sorted(self.task_names_to_num_output_channels):
      conv_block_task_1 = getattr(self, f'{task_name}/final_conv1_block')
      conv_block_task_2 = getattr(self, f'{task_name}/final_conv2_block')
      net = conv_block_task_1(
          (voxel_features_list[0], voxel_xyz_indices, num_valid_voxels_list[0]),
          training)
      outputs[task_name] = conv_block_task_2(
          (net, voxel_xyz_indices, num_valid_voxels_list[0]), training)

    return outputs
