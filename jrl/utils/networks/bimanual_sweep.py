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

"""Networks used in the Bimanual Sweep Task."""

import haiku as hk
import acme
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import numpy as np
import gin
from collections import OrderedDict

from typing import Optional


class ResNetDenseBlock(hk.Module):
  """ResNet block with dense layers."""
  def __init__(self, features, name='resnet_dense_block'):
    super().__init__(name)
    self.features = features

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)

    self.dense0 = hk.Linear(self.features // 4, w_init=w_init, b_init=b_init)
    self.dense1 = hk.Linear(self.features // 4, w_init=w_init, b_init=b_init)
    self.dense2 = hk.Linear(self.features, w_init=w_init, b_init=b_init)
    self.dense3 = hk.Linear(self.features, w_init=w_init, b_init=b_init)

  def __call__(self, x):
    y = self.dense0(jax.nn.relu(x))
    y = self.dense1(jax.nn.relu(y))
    y = self.dense2(jax.nn.relu(y))
    if x.shape != y.shape:
      x = self.dense3(jax.nn.relu(x))
    return x + y


def build_output_dist(action_dim):
  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
  b_init = hk.initializers.Constant(0.)
  return networks_lib.NormalTanhDistribution(
      action_dim,
      w_init=w_init,
      b_init=b_init)


class ResNetDense(hk.Module):
  """Dense Resnet"""
  # features: int  # Number of features for each layer.
  # depth: int = 8  # Number of layers, equivalent to (N-2)//3 blocks.
  def __init__(self, features, depth, name='dense_resnet'):
    super().__init__(name)
    self.features = features
    self.depth = depth

    assert (self.depth - 2) % 3 == 0
    self.num_blocks = (self.depth - 2) // 3
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    self.dense0 = hk.Linear(self.features, w_init=w_init, b_init=b_init)
    self.blocks = [
        ResNetDenseBlock(self.features) for i in range(self.num_blocks)
    ]
    self.dense1 = hk.Linear(self.features, w_init=w_init, b_init=b_init)

    # self.output_dist = networks_lib.NormalTanhDistribution(
    #     action_dim,
    #     w_init=w_init,
    #     b_init=b_init)

  def __call__(self, x):
    x = self.dense0(x)
    for i in range(self.num_blocks):
      x = self.blocks[i](x)
    x = self.dense1(x)
    x = jax.nn.relu(x)
    return x
    # return self.output_dist(x)


class SpatialSoftmax(hk.Module):
  def __init__(self, height, width, K, name='spatial_softmax'):
    super().__init__(name)
    self.K = K # num attention heads

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    self.conv = hk.Conv2D(
        output_channels=K,
        kernel_shape=(1, 1),
        stride=(1, 1),
        w_init=w_init,
        b_init=b_init,)

    xspace = np.linspace(-1, 1, width)
    yspace = np.linspace(-1, 1, height)
    iv, jv = np.meshgrid(yspace, xspace, sparse=False, indexing='ij')
    coords = np.concatenate((iv[Ellipsis, None], jv[Ellipsis, None]), axis=2)
    coords = coords.reshape((1, height, width, 1, 2))
    self.coords = jnp.float32(coords)

  def __call__(self, x):
    B, H, W, C = x.shape
    x = self.conv(x)
    x = x.transpose((0, 3, 1, 2)) # B x C x H x W
    x = x.reshape((B, self.K, H * W))  # Flatten image plane.
    x = jax.nn.softmax(x, axis=-1)  # Image-wide softmax.
    x = x.reshape((B, self.K, H, W, 1))
    x = x.transpose((0, 2, 3, 1, 4)) # B x H x W x K x 1
    x = jnp.broadcast_to(x, (B, H, W, self.K, 2))
    coords = jnp.broadcast_to(self.coords, x.shape)
    x = jnp.sum(x * coords, axis=(1, 2))
    return x


class ResNetConvBlock(hk.Module):
  """ResNet block with convolution layers."""
  def __init__(
      self,
      features,
      stride = 1,
      with_layer_norm = False,
      name = 'resnetblock'):
    super().__init__(name)
    self.with_layer_norm = with_layer_norm
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    self.conv0 = hk.Conv2D(
        output_channels=features // 4,
        kernel_shape=(1, 1),
        stride=(stride, stride),
        w_init=w_init,
        b_init=b_init,)
    self.conv1 = hk.Conv2D(
        output_channels=features // 4,
        kernel_shape=(3, 3),
        w_init=w_init,
        b_init=b_init,)
    self.conv2 = hk.Conv2D(
        output_channels=features,
        kernel_shape=(1, 1),
        w_init=w_init,
        b_init=b_init,)
    self.conv3 = hk.Conv2D(
        output_channels=features,
        kernel_shape=(1, 1),
        stride=(stride, stride),
        w_init=w_init,
        b_init=b_init,)

    if with_layer_norm:
      self.ln0 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      self.ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      self.ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      self.ln3 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

  def __call__(self, x):
    if self.with_layer_norm:
      y = self.conv0(jax.nn.relu(self.ln0(x)))
      y = self.conv1(jax.nn.relu(self.ln1(y)))
      y = self.conv2(jax.nn.relu(self.ln2(y)))
      if x.shape != y.shape:
        x = self.conv3(jax.nn.relu(self.ln3(x)))
    else:
      y = self.conv0(jax.nn.relu(x))
      y = self.conv1(jax.nn.relu(y))
      y = self.conv2(jax.nn.relu(y))
      if x.shape != y.shape:
        x = self.conv3(jax.nn.relu(x))
    return x + y


class ResNetImageEncoder(hk.Module):
  """Encoder using resnet blocks, input has coordinates concatenated to it."""
  # features: int  # Number of convolution kernels for each layer.
  # height: int  # Image height.
  # width: int  # Image width.
  # depth: int = 26  # Number of layers, equivalent to (N-2)//3 blocks.

  def __init__(self, features, height, width, depth, with_layer_norm=False, name='ResNetImageEncoder'):
    super().__init__(name)

    self.features = features
    self.height = height
    self.width = width
    self.depth = depth

    assert (self.depth - 2) % 3 == 0
    self.num_blocks = (self.depth - 2) // 3
    # Coordinates.
    xspace = np.linspace(-1, 1, self.width)
    yspace = np.linspace(-1, 1, self.height)
    iv, jv = np.meshgrid(yspace, xspace, sparse=False, indexing='ij')
    coords = np.concatenate((iv[Ellipsis, None], jv[Ellipsis, None]), axis=2)
    # coords = coords.reshape((1, self.height, self.width, 1, 2))
    coords = coords.reshape((1, self.height, self.width, 2))
    self.coords = jnp.float32(coords)
    self.coords.reshape((1, height, width, 2))

    # Encoder.
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    self.conv0 = hk.Conv2D(self.features, (3,3), (1, 1), w_init=w_init, b_init=b_init)
    self.blocks = [
        ResNetConvBlock(self.features, with_layer_norm=with_layer_norm) for i in range(self.num_blocks)
    ]

  def __call__(self, x):
    c = self.coords
    # print(c.shape, x.shape)
    c = jnp.broadcast_to(c, (x.shape[0], x.shape[1], x.shape[2], 2))
    x = jnp.concatenate((x, c), axis=3)
    x = self.conv0(x)
    for i in range(self.num_blocks):
      x = self.blocks[i](x)
    return x


@gin.register
def build_bimanual_sweep_actor_fn_v0(action_dim):
  def actor_fn(x):
    imgs = x['state_image'] # (B, T, 96, 96, 3)
    B, T, H, W, C = imgs.shape
    dense_states = x['state_dense'] # (B, T, 12)

    imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
    imgs = jnp.reshape(imgs, [B, H, W, T*C])
    dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])

    img_encoder = ResNetImageEncoder(64, 96, 96, 8)
    img_embeddings = img_encoder(imgs)
    img_embeddings = SpatialSoftmax(
        img_embeddings.shape[1], img_embeddings.shape[2], K=64
        )(img_embeddings)
    # print('spatial', img_embeddings.shape)
    img_embeddings = jnp.reshape(img_embeddings, [B, -1])

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    # fc_network = hk.Sequential([
    #     hk.nets.MLP(
    #         [256, 256, 256],
    #         w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
    #         activation=jax.nn.relu,
    #         activate_final=True),
    #         networks_lib.NormalTanhDistribution(action_dim),
    # ])
    # dist = fc_network(flat_features)

    resnet_dense_network = ResNetDense(features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


@gin.register
def tiny_build_bimanual_sweep_actor_fn_v0(action_dim):
  # for CPU debugging
  def actor_fn(x):
    imgs = x['state_image'] # (B, T, 96, 96, 3)
    B, T, H, W, C = imgs.shape
    dense_states = x['state_dense'] # (B, T, 12)

    imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
    imgs = jnp.reshape(imgs, [B, H, W, T*C])
    dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])

    img_encoder = ResNetImageEncoder(4, 96, 96, 5)
    img_embeddings = img_encoder(imgs)
    img_embeddings = SpatialSoftmax(
        img_embeddings.shape[1], img_embeddings.shape[2], K=4
        )(img_embeddings)
    # print('spatial', img_embeddings.shape)
    img_embeddings = jnp.reshape(img_embeddings, [B, -1])

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    # fc_network = hk.Sequential([
    #     hk.nets.MLP(
    #         [256, 256, 256],
    #         w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
    #         activation=jax.nn.relu,
    #         activate_final=True),
    #         networks_lib.NormalTanhDistribution(action_dim),
    # ])
    # dist = fc_network(flat_features)

    resnet_dense_network = ResNetDense(features=16, depth=5)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


@gin.register
def debug_actor_fn(action_dim):
  # for CPU debugging
  def actor_fn(x):
#     imgs = x['state_image'] # (B, T, 96, 96, 3)
#     B, T, H, W, C = imgs.shape
#     dense_states = x['state_dense'] # (B, T, 12)

#     imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
#     imgs = jnp.reshape(imgs, [B, -1])
#     dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])
#     full_input = jnp.concatenate([imgs, dense_states], axis=-1)

    full_input = jnp.zeros(shape=(32, 8))

    dist = networks_lib.NormalTanhDistribution(action_dim)(full_input)
    return dist
  return actor_fn


@gin.register
def largest_build_bimanual_sweep_actor_fn_v0(action_dim):
  def actor_fn(x):
    imgs = x['state_image'] # (B, T, 96, 96, 3)
    B, T, H, W, C = imgs.shape
    dense_states = x['state_dense'] # (B, T, 12)

    imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
    imgs = jnp.reshape(imgs, [B, H, W, T*C])
    dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])

    img_encoder = ResNetImageEncoder(64, 96, 96, 26)
    img_embeddings = img_encoder(imgs)
    img_embeddings = SpatialSoftmax(
        img_embeddings.shape[1], img_embeddings.shape[2], K=64
        )(img_embeddings)
    # print('spatial', img_embeddings.shape)
    img_embeddings = jnp.reshape(img_embeddings, [B, -1])

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    # fc_network = hk.Sequential([
    #     hk.nets.MLP(
    #         [256, 256, 256],
    #         w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
    #         activation=jax.nn.relu,
    #         activate_final=True),
    #         networks_lib.NormalTanhDistribution(action_dim),
    # ])
    # dist = fc_network(flat_features)

    resnet_dense_network = ResNetDense(
        features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


@gin.register
def largest_with_layer_norm_actor_fn_v0(action_dim):
  def actor_fn(x):
    imgs = x['state_image'] # (B, T, 96, 96, 3)
    B, T, H, W, C = imgs.shape
    dense_states = x['state_dense'] # (B, T, 12)

    imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
    imgs = jnp.reshape(imgs, [B, H, W, T*C])
    dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])

    img_encoder = ResNetImageEncoder(64, 96, 96, 26, with_layer_norm=True)
    img_embeddings = img_encoder(imgs)
    img_embeddings = SpatialSoftmax(
        img_embeddings.shape[1], img_embeddings.shape[2], K=64
        )(img_embeddings)
    # print('spatial', img_embeddings.shape)
    img_embeddings = jnp.reshape(img_embeddings, [B, -1])

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    # fc_network = hk.Sequential([
    #     hk.nets.MLP(
    #         [256, 256, 256],
    #         w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
    #         activation=jax.nn.relu,
    #         activate_final=True),
    #         networks_lib.NormalTanhDistribution(action_dim),
    # ])
    # dist = fc_network(flat_features)

    resnet_dense_network = ResNetDense(
        features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


@gin.register
def largest_with_average_pooling(action_dim):
  def actor_fn(x):
    imgs = x['state_image'] # (B, T, 96, 96, 3)
    B, T, H, W, C = imgs.shape
    dense_states = x['state_dense'] # (B, T, 12)

    imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
    imgs = jnp.reshape(imgs, [B, H, W, T*C])
    dense_states = jnp.reshape(dense_states, [B, T*dense_states.shape[2]])

    img_encoder = ResNetImageEncoder(64, 96, 96, 26)
    img_embeddings = img_encoder(imgs)

    # img_embeddings = hk.avg_pool(
    #     value=img_embeddings,
    #     window_shape=img_embeddings.shape[1],
    #     strides=1,
    #     padding='VALID',
    #     channel_axis=-1)(img_embeddings)
    img_embeddings = jnp.mean(img_embeddings, axis=(1, 2))
    # print('IMG_EMBEDDINGS_SHAPE', img_embeddings.shape)
    img_embeddings = jnp.reshape(img_embeddings, [B, -1])

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    # fc_network = hk.Sequential([
    #     hk.nets.MLP(
    #         [256, 256, 256],
    #         w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
    #         activation=jax.nn.relu,
    #         activate_final=True),
    #         networks_lib.NormalTanhDistribution(action_dim),
    # ])
    # dist = fc_network(flat_features)

    resnet_dense_network = ResNetDense(
        features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn

############ ENCODERS
@gin.register
def encoder_largest_with_average_pooling(x):
  imgs = x # (B, T, 96, 96, 3)
  B, T, H, W, C = imgs.shape

  imgs = jnp.transpose(imgs, axes=[0, 2, 3, 1, 4]) # (B x 96 x 96 x T x 3)
  imgs = jnp.reshape(imgs, [B, H, W, T*C])

  img_encoder = ResNetImageEncoder(64, 96, 96, 26)
  img_embeddings = img_encoder(imgs)
  return img_embeddings

############ PROJECTORS
@gin.register
def projector_v0(x):
  x = jax.nn.relu(x)

  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
  b_init = hk.initializers.Constant(0.)
  conv = hk.Conv2D(
      output_channels=1024,
      kernel_shape=(1, 1),
      stride=(1, 1),
      w_init=w_init,
      b_init=b_init,)
  spatial_softmax = SpatialSoftmax(x.shape[1], x.shape[2], K=1024)

  conv_embed = jnp.mean(
      conv(x),
      axis=(1, 2))
  conv_embed = hk.Flatten()(conv_embed)

  spatial_embed = hk.Flatten()(spatial_softmax(x))

  feats = jnp.concatenate([conv_embed, spatial_embed], axis=-1)
  feats = hk.Linear(2048, w_init=w_init, b_init=b_init)(feats)
  feats = jax.nn.relu(feats)
  feats = hk.Linear(1024, w_init=w_init, b_init=b_init)(feats)

  return feats

############## POLICIES ON TOP OF ENCODERS
@gin.register
def policy_on_encoder_v0(action_dim):
  # for on top of encoder_largest_with_average_pooling
  def actor_fn(x):
    img_embeddings = jax.nn.relu(x['state_image'])
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    conv = hk.Conv2D(
        output_channels=1024,
        kernel_shape=(1, 1),
        stride=(1, 1),
        w_init=w_init,
        b_init=b_init,)
    img_embeddings = conv(img_embeddings)
    img_embeddings = jnp.mean(img_embeddings, axis=(1, 2))
    # print('IMG_EMBEDDINGS_SHAPE', img_embeddings.shape)
    img_embeddings = hk.Flatten()(img_embeddings)

    dense_states = x['state_dense'] # (B, T, 12)
    dense_states = hk.Flatten()(dense_states)

    flat_features = jnp.concatenate([img_embeddings, dense_states], axis=1)

    resnet_dense_network = ResNetDense(
        features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


@gin.register
def policy_on_encoder_v1(action_dim):
  # for on top of encoder_largest_with_average_pooling
  def actor_fn(x):
    img_embeddings = jax.nn.relu(x['state_image'])

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)
    conv = hk.Conv2D(
        output_channels=64,
        kernel_shape=(1, 1),
        stride=(1, 1),
        w_init=w_init,
        b_init=b_init,)
    conv_img_embeddings = conv(img_embeddings)
    conv_img_embeddings = jnp.mean(conv_img_embeddings, axis=(1, 2))
    # print('IMG_EMBEDDINGS_SHAPE', img_embeddings.shape)
    conv_img_embeddings = hk.Flatten()(conv_img_embeddings)

    spatial_softmax = SpatialSoftmax(
        img_embeddings.shape[1], img_embeddings.shape[2], K=64)
    ssmax_img_embeddings = spatial_softmax(img_embeddings)
    ssmax_img_embeddings = hk.Flatten()(ssmax_img_embeddings)

    dense_states = x['state_dense'] # (B, T, 12)
    dense_states = hk.Flatten()(dense_states)

    flat_features = jnp.concatenate(
        [conv_img_embeddings, ssmax_img_embeddings, dense_states], axis=1)

    resnet_dense_network = ResNetDense(
        features=512, depth=8)
    output_dist = build_output_dist(action_dim)
    dist = output_dist(resnet_dense_network(flat_features))
    return dist
  return actor_fn


############## Q-FNS ON TOP OF ENCODERS
@gin.register
def critic_on_encoder_v0(use_double_q=True):
  def _critic_fn(obs, action):
    # for on top of encoder_largest_with_average_pooling
    img_embeddings = jax.nn.relu(obs['state_image'])

    if use_double_q:
      num_critics = 2
    else:
      num_critics = 1

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)

    critic_preds = []
    for _ in range(num_critics):
      conv = hk.Conv2D(
          output_channels=64,
          kernel_shape=(1, 1),
          stride=(1, 1),
          w_init=w_init,
          b_init=b_init,)
      img_embeddings = conv(img_embeddings)
      img_embeddings = jnp.mean(img_embeddings, axis=(1, 2))
      # print('IMG_EMBEDDINGS_SHAPE', img_embeddings.shape)
      img_embeddings = hk.Flatten()(img_embeddings)

      dense_states = obs['state_dense'] # (B, T, 12)
      dense_states = hk.Flatten()(dense_states)

      flat_features = jnp.concatenate(
          [img_embeddings, dense_states, action], axis=1)

      resnet_dense_network = ResNetDense(features=512, depth=8)
      resnet_output = resnet_dense_network(flat_features)

      preds = hk.Linear(1, w_init=w_init, b_init=b_init)(
          jax.nn.relu(
              hk.Linear(512, w_init=w_init, b_init=b_init)(
                  resnet_output)
              )
          )
      critic_preds.append(preds)

    return jnp.concatenate(critic_preds, axis=-1)

  return _critic_fn


@gin.register
def critic_on_encoder_v1(use_double_q=True):
  def _critic_fn(obs, action):
    # for on top of encoder_largest_with_average_pooling
    img_embeddings = jax.nn.relu(obs['state_image'])

    if use_double_q:
      num_critics = 2
    else:
      num_critics = 1

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg",  "truncated_normal")
    b_init = hk.initializers.Constant(0.)

    critic_preds = []
    for _ in range(num_critics):
      conv = hk.Conv2D(
          output_channels=64,
          kernel_shape=(1, 1),
          stride=(1, 1),
          w_init=w_init,
          b_init=b_init,)
      conv_img_embeddings = conv(img_embeddings)
      conv_img_embeddings = jnp.mean(conv_img_embeddings, axis=(1, 2))
      # print('IMG_EMBEDDINGS_SHAPE', img_embeddings.shape)
      conv_img_embeddings = hk.Flatten()(conv_img_embeddings)

      spatial_softmax = SpatialSoftmax(
          img_embeddings.shape[1], img_embeddings.shape[2], K=64)
      ssmax_img_embeddings = spatial_softmax(img_embeddings)
      ssmax_img_embeddings = hk.Flatten()(ssmax_img_embeddings)

      dense_states = obs['state_dense'] # (B, T, 12)
      dense_states = hk.Flatten()(dense_states)

      flat_features = jnp.concatenate(
          [conv_img_embeddings, ssmax_img_embeddings, dense_states, action],
          axis=1)

      resnet_dense_network = ResNetDense(features=512, depth=8)
      resnet_output = resnet_dense_network(flat_features)

      preds = hk.Linear(1, w_init=w_init, b_init=b_init)(
          jax.nn.relu(
              hk.Linear(512, w_init=w_init, b_init=b_init)(
                  resnet_output)
              )
          )
      critic_preds.append(preds)

    return jnp.concatenate(critic_preds, axis=-1)

  return _critic_fn
