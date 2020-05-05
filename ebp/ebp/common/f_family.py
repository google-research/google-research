# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow.compat.v1 as tf
import sonnet as snt
from ebp.common.tf_utils import MLP


class DeepsetEncoder(snt.AbstractModule):

  def __init__(self,
               dim=2,
               n_filters=(128, 256),
               act=tf.nn.relu,
               symmetry=tf.reduce_max,
               name='deepset'):
    super(DeepsetEncoder, self).__init__(name=name)
    self.dim = dim
    self.act = act
    self.n_filters = n_filters
    self.symmetry = symmetry

    with self._enter_variable_scope():
      self.mlp_deepsets = snt.nets.MLP(
          output_sizes=n_filters,
          activation=act,
          activate_final=True,
          name='mlp_deepsets')

      self.linear_anchor = snt.Linear(n_filters[-1], name='linear_anchor')

  def _build(self, pts):
    q = tf.reshape(pts, [-1, self.dim])
    q = self.mlp_deepsets(q)
    q = tf.reshape(q, [tf.shape(pts)[0], -1, self.n_filters[-1]])

    k = tf.reshape(q, [-1, self.n_filters[-1]])
    k = self.linear_anchor(k)
    k = self.act(k)
    k = tf.reshape(k, [tf.shape(pts)[0], -1, self.n_filters[-1]])
    k = self.symmetry(k, axis=1, keepdims=False)  # [B, d]
    return k


class VAE(snt.AbstractModule):

  def __init__(self,
               dim=2,
               n_filters=(64, 32),
               act=tf.nn.relu,
               symmetry=tf.reduce_max,
               sigma_eps=1e-1,
               prior_sigma=1,
               name='vae'):
    super(VAE, self).__init__(name=name)

    self.prior_var = prior_sigma**2
    self.sigma_eps = sigma_eps
    with self._enter_variable_scope():
      self.deepsets = DeepsetEncoder(dim, n_filters, act, symmetry)

      self.mlp_gauss = snt.nets.MLP(
          output_sizes=[n_filters[-1], n_filters[-1] * 2],
          activation=act,
          activate_final=False,
          name='mlp_xpx')

  def _build(self, pts):
    set_embed = self.deepsets(pts)
    params = self.mlp_gauss(set_embed)
    mu, logit_sigma = tf.split(params, num_or_size_splits=2, axis=-1)
    #sigma = tf.exp(0.5 * logvar)
    sigma = tf.sigmoid(logit_sigma) * self.sigma_eps
    eps = tf.random.normal(
        shape=tf.shape(mu), mean=0, stddev=1, dtype=tf.float32)
    x = mu + sigma * eps
    neg_KL = 0.5 * tf.reduce_sum(
        1 + 2 * tf.log(sigma) - mu**2 / self.prior_var -
        sigma**2 / self.prior_var,
        axis=1,
        keepdims=True)
    return x, mu, sigma, neg_KL


class ScoreFunc(snt.AbstractModule):

  def __init__(self,
               dim=2,
               embed_dim=256,
               fc_dims=(128, 128),
               act=tf.nn.relu,
               symmetry=tf.reduce_max,
               name='ScoreFunc'):
    super(ScoreFunc, self).__init__(name=name)

    self.dim = dim
    self.embed_dim = embed_dim
    self.fc_dims = fc_dims
    self.symmetry = symmetry

    with self._enter_variable_scope():
      self.mlp_xpx = snt.nets.MLP(
          output_sizes=fc_dims,
          activation=act,
          activate_final=True,
          name='mlp_xpx')

      self.mlp_final = snt.Linear(1, name='mlp_final')

  def _build(self, pts, set_embed):
    px = tf.expand_dims(set_embed, 1)
    px = tf.tile(px, [1, tf.shape(pts)[1], 1])

    px_x = tf.concat([px, pts], -1)
    px_x = tf.reshape(px_x, [-1, self.embed_dim + self.dim])
    px_x = self.mlp_xpx(px_x)
    px_x = tf.reshape(
        px_x, [tf.shape(pts)[0], -1, self.fc_dims[-1]], name='xpx_out')
    px_x = tf.reduce_sum(px_x, axis=1)
    px_x = self.mlp_final(px_x)
    return px_x


class DeepsetScoreFunc(snt.AbstractModule):

  def __init__(self,
               dim=2,
               bsize=64,
               n_filters=(128, 256),
               fc_dims=(128, 128),
               act=tf.nn.relu,
               symmetry=tf.reduce_max,
               name='DeepsetScoreFunc'):
    super(DeepsetScoreFunc, self).__init__(name=name)

    with self._enter_variable_scope():
      self.deepsets = DeepsetEncoder(dim, bsize, n_filters, act, symmetry)
      self.score_func = ScoreFunc(dim, bsize, n_filters[-1], fc_dims, act,
                                  symmetry)

  def _build(self, pts):
    set_embed = self.deepsets(pts)
    return self.score_func(pts, set_embed)


class DeepVAEScoreFunc(snt.AbstractModule):

  def __init__(self,
               dim=2,
               bsize=64,
               n_filters=(128, 256),
               fc_dims=(128, 128),
               act=tf.nn.relu,
               symmetry=tf.reduce_max,
               prior_sigma=1.0,
               name='DeepsetScoreFunc'):
    super(DeepVAEScoreFunc, self).__init__(name=name)

    with self._enter_variable_scope():
      self.deepsets = VAE(dim, bsize, n_filters, act, symmetry, prior_sigma)
      self.score_func = ScoreFunc(dim, bsize, n_filters[-1], fc_dims, act,
                                  symmetry)

  def _build(self, pts):
    set_embed, _, _, neg_KL = self.deepsets(pts)
    raw_score = self.score_func(pts, set_embed)
    score = raw_score + neg_KL
    return score


class MLPEnergy(snt.AbstractModule):

  def __init__(self,
               dim,
               hidden_dim,
               depth,
               output_dim=1,
               act_hidden=tf.nn.relu,
               act_out=None,
               sp_iters=0,
               mlp=None,
               name='mlp_energy'):
    super(MLPEnergy, self).__init__(name=name)
    self.act_out = act_out
    self.dim = dim
    with self._enter_variable_scope():
      if mlp is None:
        self.mlp = MLP(dim, hidden_dim, depth, output_dim, act_hidden, sp_iters)
      else:
        self.mlp = mlp

  def _build(self, raw_pts):
    x = tf.reshape(raw_pts, [-1, self.dim])
    score = self.mlp(x)
    if self.act_out is not None:
      score = self.act_out(score)
    score = tf.reshape(score, [-1, 1])
    return score
