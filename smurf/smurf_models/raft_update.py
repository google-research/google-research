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

"""Implementation of RAFT."""

# pylint:skip-file
import tensorflow as tf


def create_update_Conv2d(c_in, c_out, k_size):
  kernel_scale = 1.0 / 3.0
  if isinstance(k_size, list) or isinstance(k_size, tuple):
    bias_scale = c_out / (3.0 * c_in * k_size[0] * k_size[1])
  else:
    bias_scale = c_out / (3.0 * c_in * k_size * k_size)
  return tf.keras.layers.Conv2D(
      filters=c_out,
      kernel_size=k_size,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          distribution='uniform', scale=kernel_scale, mode='fan_in'),
      bias_initializer=tf.keras.initializers.VarianceScaling(
          distribution='uniform', scale=bias_scale, mode='fan_in'))


class ConvGRU(tf.keras.layers.Layer):

  def __init__(self, hidden_dim=128, input_dim=192 + 128, **kwargs):
    super(ConvGRU, self).__init__(**kwargs)

    self.convz = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=3)
    self.convr = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=3)
    self.convq = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=3)

  def call(self, input_tensor):
    h, x = input_tensor

    hx = tf.concat([h, x], axis=3)

    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    pad_hx = tf.pad(hx, paddings)
    z = tf.math.sigmoid(self.convz(pad_hx))
    r = tf.math.sigmoid(self.convr(pad_hx))

    pad_q = tf.pad(tf.concat([r * h, x], axis=3), paddings)
    q = tf.math.tanh(self.convq(pad_q))

    h = (1 - z) * h + z * q
    return h


class SepConvGRU(tf.keras.layers.Layer):

  def __init__(self, hidden_dim=128, input_dim=192 + 128):
    super(SepConvGRU, self).__init__()
    self.convz1 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(1, 5))
    self.convr1 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(1, 5))
    self.convq1 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(1, 5))

    self.convz2 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(5, 1))
    self.convr2 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(5, 1))
    self.convq2 = create_update_Conv2d(
        c_in=hidden_dim + input_dim, c_out=hidden_dim, k_size=(5, 1))

  def call(self, input_tensor):
    h, x = input_tensor

    # horizontal
    hx = tf.concat([h, x], axis=3)
    paddings = [[0, 0], [0, 0], [2, 2], [0, 0]]
    pad_hx = tf.pad(hx, paddings)
    z = tf.math.sigmoid(self.convz1(pad_hx))
    r = tf.math.sigmoid(self.convr1(pad_hx))
    pad_q = tf.pad(tf.concat([r * h, x], axis=3), paddings)
    q = tf.math.tanh(self.convq1(pad_q))
    h = (1 - z) * h + z * q

    # vertical
    hx = tf.concat([h, x], axis=3)
    paddings = [[0, 0], [2, 2], [0, 0], [0, 0]]
    pad_hx = tf.pad(hx, paddings)
    z = tf.math.sigmoid(self.convz2(pad_hx))
    r = tf.math.sigmoid(self.convr2(pad_hx))
    pad_q = tf.pad(tf.concat([r * h, x], axis=3), paddings)
    q = tf.math.tanh(self.convq2(pad_q))
    h = (1 - z) * h + z * q

    return h


class FlowHead(tf.keras.layers.Layer):

  def __init__(self, hidden_dim=256, input_dim=128, **kwargs):
    super(FlowHead, self).__init__(**kwargs)
    self.conv1 = create_update_Conv2d(
        c_in=input_dim, c_out=hidden_dim, k_size=3)
    self.conv2 = create_update_Conv2d(c_in=hidden_dim, c_out=2, k_size=3)

  def call(self, x):
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    conv = tf.nn.relu(self.conv1(tf.pad(x, paddings)))
    return self.conv2(tf.pad(conv, paddings))


class BasicMotionEncoder(tf.keras.layers.Layer):

  def __init__(self, args, **kwargs):
    super(BasicMotionEncoder, self).__init__(**kwargs)
    cor_planes = args.corr_levels * (2 * args.corr_radius + 1)**2
    self.convc1 = create_update_Conv2d(c_in=cor_planes, c_out=256, k_size=1)
    self.convc2 = create_update_Conv2d(c_in=256, c_out=192, k_size=3)
    self.convf1 = create_update_Conv2d(c_in=2, c_out=128, k_size=7)
    self.convf2 = create_update_Conv2d(c_in=128, c_out=64, k_size=3)
    self.conv = create_update_Conv2d(c_in=64 + 192, c_out=128 - 2, k_size=3)

  def call(self, input_tensor):
    flow, corr = input_tensor

    cor = tf.nn.relu(self.convc1(corr))
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    cor = tf.nn.relu(self.convc2(tf.pad(cor, paddings)))
    paddings7 = [[0, 0], [3, 3], [3, 3], [0, 0]]
    flo = tf.nn.relu(self.convf1(tf.pad(flow, paddings7)))
    flo = tf.nn.relu(self.convf2(tf.pad(flo, paddings)))

    cor_flo = tf.concat([cor, flo], axis=-1)
    out = tf.nn.relu(self.conv(tf.pad(cor_flo, paddings)))
    return tf.concat([out, flow], axis=-1)


class SmallMotionEncoder(tf.keras.layers.Layer):

  def __init__(self, args, **kwargs):
    super(SmallMotionEncoder, self).__init__(**kwargs)
    cor_planes = args.corr_levels * (2 * args.corr_radius + 1)**2
    self.convc1 = create_update_Conv2d(c_in=cor_planes, c_out=96, k_size=1)
    self.convf1 = create_update_Conv2d(c_in=96, c_out=64, k_size=7)
    self.convf2 = create_update_Conv2d(c_in=64, c_out=32, k_size=3)
    self.conv = create_update_Conv2d(c_in=32, c_out=80, k_size=3)

  def call(self, input_tensor):
    flow, corr = input_tensor

    cor = tf.nn.relu(self.convc1(corr))
    paddings7 = [[0, 0], [3, 3], [3, 3], [0, 0]]
    flo = tf.nn.relu(self.convf1(tf.pad(flow, paddings7)))
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    flo = tf.nn.relu(self.convf2(tf.pad(flo, paddings)))

    cor_flo = tf.concat([cor, flo], axis=-1)
    out = tf.nn.relu(self.conv(tf.pad(cor_flo, paddings)))
    return tf.concat([out, flow], axis=-1)


class BasicUpdateBlock(tf.keras.layers.Layer):

  def __init__(self, args, hidden_dim=128, **kwargs):
    super(BasicUpdateBlock, self).__init__(**kwargs)

    self.args = args
    self.encoder = BasicMotionEncoder(args)
    self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
    self.flow_head = FlowHead(hidden_dim=256, input_dim=hidden_dim)
    if args.convex_upsampling:
      self.mask = tf.keras.Sequential(
          [create_update_Conv2d(c_in=128, c_out=256, k_size=3),
           tf.keras.layers.ReLU(),
           create_update_Conv2d(c_in=256, c_out=64 * 9, k_size=1)
          ])

  def call(self, input_tensor, training):
    net, inp, corr, flow = input_tensor

    motion_features = self.encoder([flow, corr])
    inp = tf.concat([inp, motion_features], axis=-1)
    net = self.gru([net, inp])
    delta_flow = self.flow_head(net)

    if self.args.convex_upsampling:
      # Scale mask to balance gradients.
      paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
      pad_net = tf.pad(net, paddings)
      mask = .25 * self.mask(pad_net)
    else:
      mask = None

    return net, mask, delta_flow


class SmallUpdateBlock(tf.keras.layers.Layer):

  def __init__(self, args, hidden_dim=96, **kwargs):
    super(SmallUpdateBlock, self).__init__(**kwargs)

    self.encoder = SmallMotionEncoder(args)
    self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
    self.flow_head = FlowHead(hidden_dim=128, input_dim=hidden_dim)

  def call(self, input_tensor, training):
    net, inp, corr, flow = input_tensor

    motion_features = self.encoder([flow, corr])
    inp = tf.concat([inp, motion_features], axis=-1)
    net = self.gru([net, inp])
    delta_flow = self.flow_head(net)

    return net, None, delta_flow
