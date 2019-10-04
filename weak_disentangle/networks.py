# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# python3
"""Models."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
# pylint: disable=g-wrong-blank-lines, missing-super-argument
import gin
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from collections import OrderedDict
import numpy as np

from weak_disentangle import tensorsketch as ts
from weak_disentangle import utils as ut

tfd = tfp.distributions
dense = gin.external_configurable(ts.Dense)
conv = gin.external_configurable(ts.Conv2d)
deconv = gin.external_configurable(ts.ConvTranspose2d)
add_wn = gin.external_configurable(ts.WeightNorm.add)
add_bn = gin.external_configurable(ts.BatchNorm.add)


@gin.configurable
class Encoder(ts.Module):
  def __init__(self, x_shape, z_dim, width=1, spectral_norm=True):
    super().__init__()
    self.net = ts.Sequential(
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        ts.Flatten(),
        dense(128 * width), ts.LeakyReLU(),
        dense(2 * z_dim)
        )

    if spectral_norm:
      self.net.apply(ts.SpectralNorm.add, targets=ts.Affine)

    ut.log("Building encoder...")
    self.build([1] + x_shape)
    self.apply(ut.reset_parameters)

  def forward(self, x):
    h = self.net(x)
    a, b = tf.split(h, 2, axis=-1)
    return tfd.MultivariateNormalDiag(
        loc=a,
        scale_diag=tf.nn.softplus(b) + 1e-8)


@gin.configurable
class LabelDiscriminator(ts.Module):
  def __init__(self, x_shape, y_dim, width=1, share_dense=False,
               uncond_bias=False):
    super().__init__()
    self.y_dim = y_dim
    self.body = ts.Sequential(
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        ts.Flatten(),
        )

    self.aux = ts.Sequential(
        dense(128 * width), ts.LeakyReLU(),
        )

    if share_dense:
      self.body.append(dense(128 * width), ts.LeakyReLU())
      self.aux.append(dense(128 * width), ts.LeakyReLU())

    self.head = ts.Sequential(
        dense(128 * width), ts.LeakyReLU(),
        dense(128 * width), ts.LeakyReLU(),
        dense(1, bias=uncond_bias)
        )

    for m in (self.body, self.aux, self.head):
      m.apply(ts.SpectralNorm.add, targets=ts.Affine)

    ut.log("Building label discriminator...")
    x_shape, y_shape = [1] + x_shape, (1, y_dim)
    self.build(x_shape, y_shape)
    self.apply(ut.reset_parameters)

  def forward(self, x, y):
    hx = self.body(x)
    hy = self.aux(y)
    o = self.head(tf.concat((hx, hy), axis=-1))
    return o


@gin.configurable
class Discriminator(ts.Module):
  def __init__(self, x_shape, y_dim, width=1, share_dense=False,
               uncond_bias=False, cond_bias=False, mask_type="match"):
    super().__init__()
    self.y_dim = y_dim
    self.mask_type = mask_type
    self.body = ts.Sequential(
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(32 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        conv(64 * width, 4, 2, "same"), ts.LeakyReLU(),
        ts.Flatten(),
        )

    if share_dense:
      self.body.append(dense(128 * width), ts.LeakyReLU())

    if mask_type == "match":
      self.neck = ts.Sequential(
          dense(128 * width), ts.LeakyReLU(),
          dense(128 * width), ts.LeakyReLU(),
          )

      self.head_uncond = dense(1, bias=uncond_bias)
      self.head_cond = dense(128 * width, bias=cond_bias)

      for m in (self.body, self.neck, self.head_uncond):
        m.apply(ts.SpectralNorm.add, targets=ts.Affine)
      add_wn(self.head_cond)
      x_shape, y_shape = [1] + x_shape, ((1,), tf.int32)

    elif mask_type == "rank":
      self.body.append(
          dense(128 * width), ts.LeakyReLU(),
          dense(128 * width), ts.LeakyReLU(),
          dense(1 + y_dim, bias=uncond_bias)
          )

      self.body.apply(ts.SpectralNorm.add, targets=ts.Affine)
      x_shape, y_shape = [1] + x_shape, (1, y_dim)

    ut.log("Building {} discriminator...".format(mask_type))
    self.build(x_shape, x_shape, y_shape)
    self.apply(ut.reset_parameters)

  def forward(self, x1, x2, y):
    if self.mask_type == "match":
      h = self.body(tf.concat((x1, x2), axis=0))
      h1, h2 = tf.split(h, 2, axis=0)
      h = self.neck(tf.concat((h1, h2), axis=-1))
      o_uncond = self.head_uncond(h)

      w = self.head_cond(tf.one_hot(y, self.y_dim))
      o_cond = tf.reduce_sum(h * w, axis=-1, keepdims=True)
      return o_uncond + o_cond

    elif self.mask_type == "rank":
      h = self.body(tf.concat((x1, x2), axis=0))
      h1, h2 = tf.split(h, 2, axis=0)
      o1, z1 = tf.split(h1, (1, self.y_dim), axis=-1)
      o2, z2 = tf.split(h2, (1, self.y_dim), axis=-1)
      y_pm = y * 2 - 1  # convert from {0, 1} to {-1, 1}
      diff = (z1 - z2) * y_pm
      o_diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
      return o1 + o2 + o_diff

  def expose_encoder(self, x):
    h = self.body(x)
    _, z = tf.split(h, (1, self.y_dim), axis=-1)
    return z


@gin.configurable
class Generator(ts.Module):
  def __init__(self, x_shape, z_dim, batch_norm=True):
    super().__init__()
    ch = x_shape[-1]
    self.net = ts.Sequential(
        dense(128), ts.ReLU(),
        dense(4 * 4 * 64), ts.ReLU(), ts.Reshape((-1, 4, 4, 64)),
        deconv(64, 4, 2, "same"), ts.LeakyReLU(),
        deconv(32, 4, 2, "same"), ts.LeakyReLU(),
        deconv(32, 4, 2, "same"), ts.LeakyReLU(),
        deconv(ch, 4, 2, "same"), ts.Sigmoid(),
        )

    # Add batchnorm post-activation (attach to activation out_hook)
    if batch_norm:
      self.net.apply(add_bn, targets=(ts.ReLU, ts.LeakyReLU))

    ut.log("Building generator...")
    self.build((1, z_dim))
    self.apply(ut.reset_parameters)

  def forward(self, z):
    return self.net(z)
