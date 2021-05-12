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

"""Implementation of RAFT."""
# pylint:skip-file

import gin
import tensorflow as tf

from smurf import smurf_utils
from smurf.smurf_models.raft_corr import corr_block, corr_pyramid
from smurf.smurf_models.raft_extractor import BasicEncoder, SmallEncoder
from smurf.smurf_models.raft_update import BasicUpdateBlock, SmallUpdateBlock
from smurf.smurf_models.raft_utils import compute_upsample_flow, initialize_flow


class RAFTFeatureSiamese(tf.keras.Model):
  """Computes the correlation pyramid and context features for RAFT."""

  def __init__(self, args=None, **kwargs):
    super(RAFTFeatureSiamese, self).__init__(**kwargs)

    # Initialized this way to work with gin configurable RAFTArgs class.
    if args is None:
      args = RAFTArgs()
    self._args = args

    if self._args.small:
      if self._args.use_norms:
        self.fnet = SmallEncoder(
            output_dim=128, norm_fn='instance', dropout=self._args.dropout)
        self.cnet = SmallEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)
      else:
        self.fnet = SmallEncoder(
            output_dim=128, norm_fn='none', dropout=self._args.dropout)
        self.cnet = SmallEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)
    else:
      if self._args.use_norms:
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='instance', dropout=self._args.dropout)
        self.cnet = BasicEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='instance',
            dropout=self._args.dropout)
      else:
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='none', dropout=self._args.dropout)
        self.cnet = BasicEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)

  def call(self,
           image1,
           image2,
           training=False,
           bidirectional=False):
    """Runs the model.

    Args:
      image1: First/reference image batch [b, h, w, c].
      image2: Second image batch [b, h, w, c].
      training: Flag indicating if the model is being trained or not.
      bidirectional: Flag indicating if features should also be computed for
        the reversed image order of the pair.

    Returns:
      Dictionary holding the correlation pyramid (potentially also for the
      reversed pair order) and the context net features.
    """
    # Rescale input images from [0,1] to [-1, 1].
    image1 = image1 * 2. - 1.
    image2 = image2 * 2. - 1.

    # Run feature network.
    image_pair = tf.concat((image1, image2), axis=0)
    fmap_pair = self.fnet(image_pair, training=training)
    fmap1, fmap2 = tf.split(fmap_pair, num_or_size_splits=2, axis=0)

    # Compute correlation pyarmid (potentially forward and backward).
    pyramids = corr_pyramid([fmap1, fmap2],
                            num_levels=self._args.corr_levels - 1,
                            bidirectional=bidirectional)

    # Run context network.
    cnet = self.cnet(image1, training=training)
    net_1, inp_1 = tf.split(
        cnet, [self._args.hidden_dim, self._args.context_dim], axis=-1)
    net_1, inp_1 = tf.math.tanh(net_1), tf.nn.relu(inp_1)

    # Store size information.
    original_bs = tf.shape(image1)[-4]
    original_ht = tf.shape(image1)[-3]
    original_wd = tf.shape(image1)[-2]
    original_size = (original_ht, original_wd)

    # Prepare output dictionary.
    output = {
        'correlation_pyarmid_fw': pyramids['fw'],
        'net_1': net_1,
        'inp_1': inp_1,
        'original_size': original_size,
        'batch_size': original_bs
    }

    # Add features for the reversed image pair order (backward direction) if
    # required.
    if bidirectional:
    # Run context network.
      cnet = self.cnet(image2, training=training)
      net_2, inp_2 = tf.split(
          cnet, [self._args.hidden_dim, self._args.context_dim], axis=-1)
      net_2, inp_2 = tf.math.tanh(net_2), tf.nn.relu(inp_2)
      output.update({
          'correlation_pyarmid_bw': pyramids['bw'],
          'net_2': net_2,
          'inp_2': inp_2
      })
    return output


@gin.configurable('raft_model_parameters')
class RAFTArgs(object):
  """RAFT arguments."""

  def __init__(self,
               small=False,
               use_norms=True,
               corr_levels=None,
               corr_radius=None,
               convex_upsampling=True,
               dropout=0.0,
               max_rec_iters=12):
    self.small = small
    self.use_norms = use_norms
    self.convex_upsampling = convex_upsampling
    self.dropout = dropout
    self.max_rec_iters = max_rec_iters

    if self.small:
      self.hidden_dim = 96
      self.context_dim = 64
      self.corr_levels = 4 if corr_levels is None else corr_levels
      self.corr_radius = 3 if corr_radius is None else corr_radius
    else:
      self.hidden_dim = 128
      self.context_dim = 128
      self.corr_levels = 4 if corr_levels is None else corr_levels
      self.corr_radius = 4 if corr_radius is None else corr_radius

    if small and convex_upsampling:
      raise ValueError('Convex upsampling is not implemented for the small '
                       'setting of raft.')


class RAFTFeatureSiamese(tf.keras.Model):
  """Computes the correlation pyramid and context features for RAFT."""

  def __init__(self, args=None, **kwargs):
    super(RAFTFeatureSiamese, self).__init__(**kwargs)

    # Initialized this way to work with gin configurable RAFTArgs class.
    if args is None:
      args = RAFTArgs()
    self._args = args

    if self._args.small:
      if self._args.use_norms:
        self.fnet = SmallEncoder(
            output_dim=128, norm_fn='instance', dropout=self._args.dropout)
        self.cnet = SmallEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)
      else:
        self.fnet = SmallEncoder(
            output_dim=128, norm_fn='none', dropout=self._args.dropout)
        self.cnet = SmallEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)
    else:
      if self._args.use_norms:
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='instance', dropout=self._args.dropout)
        self.cnet = BasicEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='instance',
            dropout=self._args.dropout)
      else:
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='none', dropout=self._args.dropout)
        self.cnet = BasicEncoder(
            output_dim=self._args.hidden_dim + self._args.context_dim,
            norm_fn='none',
            dropout=self._args.dropout)

  def call(self,
           image1,
           image2,
           training=False,
           bidirectional=False):
    """Runs the model.

    Args:
      image1: First/reference image batch [b, h, w, c].
      image2: Second image batch [b, h, w, c].
      training: Flag indicating if the model is being trained or not.
      bidirectional: Flag indicating if features should also be computed for
        the reversed image order of the pair.

    Returns:
      Dictionary holding the correlation pyramid (potentially also for the
      reversed pair order) and the context net features.
    """
    # Rescale input images from [0,1] to [-1, 1].
    image1 = image1 * 2. - 1.
    image2 = image2 * 2. - 1.

    # Run feature network.
    image_pair = tf.concat((image1, image2), axis=0)
    fmap_pair = self.fnet(image_pair, training=training)
    fmap1, fmap2 = tf.split(fmap_pair, num_or_size_splits=2, axis=0)

    # Compute correlation pyarmid (potentially forward and backward).
    pyramids = corr_pyramid([fmap1, fmap2],
                            num_levels=self._args.corr_levels - 1,
                            bidirectional=bidirectional)

    # Run context network.
    cnet = self.cnet(image1, training=training)
    net_1, inp_1 = tf.split(
        cnet, [self._args.hidden_dim, self._args.context_dim], axis=-1)
    net_1, inp_1 = tf.math.tanh(net_1), tf.nn.relu(inp_1)

    # Store size information.
    original_bs = tf.shape(image1)[-4]
    original_ht = tf.shape(image1)[-3]
    original_wd = tf.shape(image1)[-2]
    original_size = (original_ht, original_wd)

    # Prepare output dictionary.
    output = {
        'correlation_pyarmid_fw': pyramids['fw'],
        'net_1': net_1,
        'inp_1': inp_1,
        'original_size': original_size,
        'batch_size': original_bs
    }

    # Add features for the reversed image pair order (backward direction) if
    # required.
    if bidirectional:
    # Run context network.
      cnet = self.cnet(image2, training=training)
      net_2, inp_2 = tf.split(
          cnet, [self._args.hidden_dim, self._args.context_dim], axis=-1)
      net_2, inp_2 = tf.math.tanh(net_2), tf.nn.relu(inp_2)
      output.update({
          'correlation_pyarmid_bw': pyramids['bw'],
          'net_2': net_2,
          'inp_2': inp_2
      })
    return output


class RAFT(tf.keras.Model):
  """Implements a RAFT optical flow model as a keras model."""

  def __init__(self, args=None, **kwargs):
    super(RAFT, self).__init__(**kwargs)

    # Initialized this way to work with gin configurable RAFTArgs class.
    if args is None:
      args = RAFTArgs()
    self.args = args

    if self.args.small:
      self.update_block = SmallUpdateBlock(
          self.args, hidden_dim=self.args.hidden_dim)
    else:
      self.update_block = BasicUpdateBlock(
          self.args, hidden_dim=self.args.hidden_dim)

  def build(self, input_shape):
    del input_shape  # unused

  def _upsample_flow(self, flow, mask):
    """Upsample flow [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
    bs, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.transpose(mask, perm=[0, 3, 1, 2])
    mask = tf.reshape(mask, [bs, 1, 9, 8, 8, height, width])
    mask = tf.nn.softmax(mask, axis=2)

    up_flow = tf.image.extract_patches(
        images=tf.pad(8 * flow, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]]),
        sizes=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    up_flow = tf.reshape(up_flow, [bs, height, width, 1, 1, 9, 2])
    up_flow = tf.transpose(up_flow, [0, 6, 5, 4, 3, 1, 2])

    up_flow = tf.math.reduce_sum(mask * up_flow, axis=2)
    up_flow = tf.transpose(up_flow, perm=[0, 4, 2, 5, 3, 1])
    up_flow = tf.reshape(up_flow, [bs, height * 8, width * 8, 2])
    return up_flow

  def call(self,
           feature_dict,
           training=False,
           backward=False,
           max_rec_iters=None):
    if max_rec_iters is None:
      max_rec_iters = self.args.max_rec_iters

    original_size = feature_dict['original_size']
    batch_size = feature_dict['batch_size']
    if backward:
      pyramid = feature_dict['correlation_pyarmid_bw']
      net = feature_dict['net_2']
      inp = feature_dict['inp_2']
    else:
      pyramid = feature_dict['correlation_pyarmid_fw']
      net = feature_dict['net_1']
      inp = feature_dict['inp_1']

    corr_fn = lambda coords: corr_block(
        pyramid, coords, radius=self.args.corr_radius)

    def cond(iter_num, net, inp, coords0, coords1, flow, flow_list):
      return iter_num < max_rec_iters

    def body(iter_num, net, inp, coords0, coords1, flow, flow_list):
      coords1 = tf.stop_gradient(coords1)

      corr = corr_fn(coords1)
      flow = coords1 - coords0
      net, up_mask, delta_flow = self.update_block(
          [net, inp, corr, flow], training=training)

      # F(t+1) = F(t) + \Delta(t)
      coords1 = coords1 + delta_flow
      saved_flow = coords1 - coords0
      if up_mask is not None:
        saved_flow = self._upsample_flow(saved_flow, up_mask)
      flow_list = flow_list.write(iter_num, saved_flow)
      iter_num += 1
      return [
          iter_num, net, inp, coords0, coords1, coords1 - coords0, flow_list
      ]

    iter_num = 0
    coords0, coords1 = initialize_flow(batch_size, original_size[0],
                                       original_size[1])
    flow = coords1 - coords0
    flow_list = tf.TensorArray(
        dtype=tf.float32, size=max_rec_iters, dynamic_size=False)

    loop_vars = [iter_num, net, inp, coords0, coords1, flow, flow_list]
    iter_num, net, inp, coords0, coords1, flow, flow_list = tf.while_loop(
        cond,
        body,
        loop_vars,
        parallel_iterations=1,
        maximum_iterations=max_rec_iters)

    self.flow_preds = flow_list.stack()
    upsampled_flow_preds = [
        compute_upsample_flow(flow_pred, original_size)
        for flow_pred in tf.unstack(self.flow_preds)
    ]
    self.upsampled_flow_preds = tf.stack(upsampled_flow_preds)

    # Reverse to match (height, width)-displacement order.
    self.upsampled_flow_preds = self.upsampled_flow_preds[Ellipsis, ::-1]

    if self.args.convex_upsampling:
      # With convex upsampling the flow field is already at the full input
      # resolution, no further upsampling is required. Hence, we only output the
      # full resolution flow.
      flow_res_1 = self.flow_preds[-1, :, :, :, ::-1]
      flows = [flow_res_1]
    else:
      # If no convex upsampling is used the flow field is at a 1/8 of the input
      # resolution. Hence we perform a stepwise upsampling to output multiple
      # flow resolutions.
      flow_res_8 = self.flow_preds[-1, :, :, :, ::-1]
      flow_res_4 = smurf_utils.upsample(flow_res_8, is_flow=True)
      flow_res_2 = smurf_utils.upsample(flow_res_4, is_flow=True)
      flow_res_1 = smurf_utils.upsample(flow_res_2, is_flow=True)
      flows = [flow_res_1, flow_res_2, flow_res_4, flow_res_8]
    return flows

  def get_flow_sequence(self):
    return self.upsampled_flow_preds

  def get_flow_sequence_length(self):
    return self.args.max_rec_iters
