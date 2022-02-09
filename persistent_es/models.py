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

"""Models for meta-learning experiments (MLP and ConvNets)."""
import pdb

import jax
import jax.numpy as jnp

import haiku as hk


class MLP(hk.Module):

  def __init__(self,
               nlayers=2,
               nhid=100,
               with_bias=False,
               batch_norm=False,
               activation='relu'):
    super().__init__()
    self.nlayers = nlayers
    self.nhid = nhid
    self.with_bias = with_bias
    self.batch_norm = batch_norm
    self.activation = activation

  def __call__(self, x, mask_props=None, is_training=True):
    out = hk.Flatten()(x)

    for l in range(self.nlayers):
      out = hk.Linear(self.nhid, with_bias=self.with_bias)(out)
      if self.batch_norm:
        out = hk.BatchNorm(
            create_scale=False, create_offset=False,
            decay_rate=0.9)(out, is_training)

      if mask_props is not None:
        num_units = jnp.floor(mask_props[l] * out.shape[1]).astype(jnp.int32)
        mask = jnp.arange(out.shape[1]) < num_units
        out = jnp.where(mask, out, jnp.zeros(out.shape))

      if self.activation == 'relu':
        out = jax.nn.relu(out)
      elif self.activation == 'sigmoid':
        out = jax.nn.sigmoid(out)
      elif self.activation == 'tanh':
        out = jnp.tanh(out)
      elif self.activation == 'linear':
        out = out

    out = hk.Linear(10, with_bias=self.with_bias)(out)
    return out


class ConvBN(hk.Module):

  def __init__(self, c_in, c_out):
    super().__init__()
    self.conv = hk.Conv2D(
        output_channels=c_out,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding='SAME',
        data_format='NCHW')
    self.bn = hk.BatchNorm(
        create_scale=True,
        create_offset=True,
        decay_rate=0.9,
        data_format='NC...')

  def __call__(self, x, is_training):
    out = self.conv(x)
    out = self.bn(out, is_training=is_training)
    return jax.nn.relu(out)


class Residual(hk.Module):

  def __init__(self, c):
    super().__init__()
    self.res1 = ConvBN(c, c)
    self.res2 = ConvBN(c, c)

  def __call__(self, x, is_training):
    out1 = self.res1(x, is_training=is_training)
    out2 = self.res2(out1, is_training=is_training)
    return x + out2


class Net(hk.Module):

  def __init__(self, model_size):
    super().__init__()
    if model_size == 'large':
      channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    elif model_size == 'med':
      channels = {'prep': 32, 'layer1': 64, 'layer2': 128, 'layer3': 256}
    elif model_size == 'small':
      channels = {'prep': 16, 'layer1': 32, 'layer2': 64, 'layer3': 128}
    elif model_size == 'tiny':
      channels = {'prep': 8, 'layer1': 16, 'layer2': 32, 'layer3': 64}
    self.prep = ConvBN(3, channels['prep'])

    # Layer 1
    self.conv1 = ConvBN(channels['prep'], channels['layer1'])
    self.pool1 = hk.MaxPool(
        window_shape=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding='SAME')
    self.residual1 = Residual(channels['layer1'])

    # Layer 2
    self.conv2 = ConvBN(channels['layer1'], channels['layer2'])
    self.pool2 = hk.MaxPool(
        window_shape=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding='SAME')

    # Layer 3
    self.conv3 = ConvBN(channels['layer2'], channels['layer3'])
    self.pool3 = hk.MaxPool(
        window_shape=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding='SAME')
    self.residual3 = Residual(channels['layer3'])

    self.pool4 = hk.MaxPool(
        window_shape=(1, 1, 4, 4), strides=(1, 1, 4, 4), padding='SAME')

    self.fc = hk.Linear(10, with_bias=False)
    self.logit_weight = 0.125

  def __call__(self, x, is_training):
    out = self.prep(x, is_training=is_training)  # (512, 64, 32, 32)
    out = self.conv1(out, is_training=is_training)  # (512, 128, 32, 32)
    out = self.pool1(out)  # (512, 128, 16, 16)
    out = self.residual1(out, is_training=is_training)  # (512, 128, 16, 16)
    out = self.conv2(out, is_training=is_training)  # (512, 256, 16, 16)
    out = self.pool2(out)  # (512, 256, 8, 8)
    out = self.conv3(out, is_training=is_training)  # (512, 512, 8, 8)
    out = self.pool3(out)  # (512, 512, 4, 4)
    out = self.residual3(out, is_training=is_training)  # (512, 512, 4, 4)
    out = self.pool4(out)  # (512, 512, 1, 1)
    out = self.fc(out.reshape(out.shape[0], -1))  # (512, 10)
    out = out * self.logit_weight
    return out


class BasicBlock(hk.Module):

  def __init__(self, in_planes, planes, stride=1):
    super().__init__()

    self.conv1 = hk.Conv2D(
        output_channels=planes,
        kernel_shape=3,
        stride=stride,
        padding='SAME',
        with_bias=False,
        data_format='NCHW')
    self.bn1 = hk.BatchNorm(
        create_scale=True,
        create_offset=True,
        decay_rate=0.9,
        data_format='NC...')

    self.conv2 = hk.Conv2D(
        output_channels=planes,
        kernel_shape=3,
        stride=1,
        padding='SAME',
        with_bias=False,
        data_format='NCHW')
    self.bn2 = hk.BatchNorm(
        create_scale=True,
        create_offset=True,
        decay_rate=0.9,
        data_format='NC...')

    self.in_planes = in_planes
    self.planes = planes
    self.stride = stride

  def __call__(self, x, is_training):
    out = jax.nn.relu(self.bn1(self.conv1(x), is_training=is_training))
    out = self.bn2(self.conv2(out), is_training=is_training)

    if self.stride != 1 or self.in_planes != self.planes:
      out += jnp.pad(
          x[:, :, ::2, ::2],
          ((0, 0), (self.planes // 4, self.planes // 4), (0, 0), (0, 0)),
          mode='constant',
          constant_values=0)
    else:
      out += x

    out = jax.nn.relu(out)
    return out


class MultiBlock(hk.Module):

  def __init__(self, in_planes, out_planes, strides):
    super().__init__()
    self.layers = []
    for stride in strides:
      self.layers.append(BasicBlock(in_planes, out_planes, stride))
      in_planes = out_planes

  def __call__(self, x, is_training):
    out = x
    for layer in self.layers:
      out = layer(out, is_training=is_training)
    return out


class ResNet(hk.Module):

  def __init__(self, num_blocks=[5, 5, 5], num_classes=10):
    super().__init__()

    self.conv1 = hk.Conv2D(
        output_channels=16,
        # output_channels=64,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding='SAME',
        data_format='NCHW')
    self.bn1 = hk.BatchNorm(
        create_scale=True,
        create_offset=True,
        decay_rate=0.9,
        data_format='NC...')
    self.layer1 = MultiBlock(16, 16, [1] + [1] * (num_blocks[0] - 1))
    self.layer2 = MultiBlock(16, 32, [2] + [1] * (num_blocks[1] - 1))
    self.layer3 = MultiBlock(32, 64, [2] + [1] * (num_blocks[2] - 1))
    self.linear = hk.Linear(num_classes)

  def __call__(self, x, is_training):
    out = jax.nn.relu(self.bn1(self.conv1(x), is_training=is_training))
    out = self.layer1(out, is_training=is_training)
    out = self.layer2(out, is_training=is_training)
    out = self.layer3(out, is_training=is_training)
    out = hk.avg_pool(
        out,
        window_shape=(1, 1, out.shape[2], out.shape[3]),
        strides=(1, 1, out.shape[2], out.shape[3]),
        padding='SAME')
    out = out.reshape(out.shape[0], -1)
    out = self.linear(out)
    return out


class TinyConv(hk.Module):
  def __init__(self):
    super().__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = hk.Conv2D(
        output_channels=32, kernel_shape=5, padding='SAME', data_format='NCHW')
    self.pool1 = hk.MaxPool(
        window_shape=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding='SAME')

    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = hk.Conv2D(
        output_channels=64, kernel_shape=5, padding='SAME', data_format='NCHW')
    self.pool2 = hk.MaxPool(
        window_shape=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding='SAME')
    # feature map size is 7*7 by pooling
    self.fc = hk.Linear(10)

  def __call__(self, x):
    out = self.pool1(jax.nn.relu(self.conv1(x)))    # (500, 32, 14, 14)
    out = self.pool2(jax.nn.relu(self.conv2(out)))  # (500, 64, 7, 7)
    out = out.reshape(-1, 64*7*7)                   # (500, 3136)
    out = self.fc(out)                              # (500, 10)
    return out
