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
from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v1 as tf
import sonnet as snt
import numpy as np
import scipy
from scipy.spatial.distance import cdist


def get_gamma(X, bandwidth):
  x_norm = np.sum(X**2, axis=1, keepdims=True)
  x_t = X.transpose()
  x_norm_t = np.reshape(x_norm, (1, -1))
  t = x_norm + x_norm_t - 2.0 * np.matmul(X, x_t)
  d = np.maximum(t, 0)

  d = d[np.isfinite(d)]
  d = d[d > 0]
  median_dist2 = float(np.median(d))
  print('median_dist2:', median_dist2)
  gamma = 0.5 / median_dist2 / bandwidth
  return gamma


def get_kernel_mat(x, landmarks, gamma):
  feat_dim = x.shape[1]
  batch_size = x.shape[0]
  d = cdist(x, landmarks, metric='sqeuclidean')

  # get kernel matrix
  k = np.exp(d * -gamma)
  k = np.reshape(k, (batch_size, -1))
  return k


def MMD(y, x, gamma):
  kxx = get_kernel_mat(x, x, gamma)
  np.fill_diagonal(kxx, 0)
  kxx = np.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

  kyy = get_kernel_mat(y, y, gamma)
  np.fill_diagonal(kyy, 0)
  kyy = np.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
  kxy = np.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
  mmd = kxx + kyy - 2 * kxy
  return mmd


class SpectralNormLinear(snt.AbstractModule):

  def __init__(self, sp_iters, name):
    super(SpectralNormLinear, self).__init__(name=name)
    self.sp_iters = sp_iters
    self.name = name

  def spectral_norm(self, w, l):
    """
            https://github.com/taki0112/Spectral_Normalization-Tensorflow
        """
    iteration = self.sp_iters
    if iteration == 0:
      return w
    eps = 1e-6
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        'u-%d' % l, [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(),
        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
      """
           power iteration
           Usually iteration = 1 will be enough
           """
      v_ = tf.matmul(u_hat, tf.transpose(w))
      v_hat = tf.nn.l2_normalize(v_, dim=-1)

      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_, dim=-1)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
      w_norm = w / (sigma + eps)
      w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


class MLP(SpectralNormLinear):

  def __init__(self,
               input_dim,
               hidden_dim,
               depth,
               output_dim=1,
               act_hidden=tf.nn.relu,
               sp_iters=0,
               vars=None,
               name='mlp'):
    super(MLP, self).__init__(sp_iters, name=name)
    dims = [input_dim] + [hidden_dim] * depth + [output_dim]
    self.act_hidden = act_hidden
    self.list_linear = []
    with self._enter_variable_scope():
      for t in range(len(dims) - 1):
        in_dim, out_dim = dims[t], dims[t + 1]
        if vars is None:
          w = tf.get_variable('w%d' % t, shape=(in_dim, out_dim))
          b = tf.get_variable('b%d' % t, shape=(1, out_dim))
        else:
          w = vars[t * 2]
          b = vars[t * 2 + 1]
          print(w, b)
        self.list_linear.append((w, b))

  def _build(self, x):
    h = x
    for l in range(len(self.list_linear)):
      w, b = self.list_linear[l]
      w = self.spectral_norm(w, l)
      h = tf.matmul(h, w) + b
      if l + 1 < len(self.list_linear):
        h = self.act_hidden(h)
    return h
