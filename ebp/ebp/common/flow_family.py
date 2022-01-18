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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v1 as tf
import sonnet as snt
import numpy as np
from ebp.common.tf_utils import MLP, SpectralNormLinear
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian


class AbstractNormFlow(snt.AbstractModule):
  """Normalizing flow abstract"""

  def __init__(self, dim, num_layers, flow_type, name='abs_norm_flow'):
    super(AbstractNormFlow, self).__init__(name=name)
    self.name = name
    self.dim = dim
    self.num_layers = num_layers

    self.flow_blocks = []
    with self._enter_variable_scope():
      for i in range(self.num_layers):
        if isinstance(flow_type, str):
          if flow_type == 'planar':
            flow = PlanarFlow(self.dim, name='planar_flow_layer_%d' % i)
          elif flow_type == 'ires':
            flow = iResFlow(self.dim, name='ires_flow_layer_%d' % i)
          else:
            raise NotImplementedError
        else:
          flow = flow_type(i)
        self.flow_blocks.append(flow)


class PlanarFlow(snt.AbstractModule):
  """PlanarFlow"""

  def __init__(self, dim, name='planar_flow_layer'):
    super(PlanarFlow, self).__init__(name=name)
    self.name = name
    self.dim = dim
    self.h = tf.tanh
    self.num_params = self.dim * 2 + 1

  def _build(self, z, logp, params):
    self.u, self.w, self.b = tf.split(params, [self.dim, self.dim, 1], axis=1)
    self.u = tf.expand_dims(self.u, -1)
    self.w = tf.expand_dims(self.w, -1)
    self.b = tf.expand_dims(self.b, 1)

    a = self.h(tf.matmul(z, self.w) + self.b)
    psi = tf.matmul(1 - a**2, tf.transpose(self.w, perm=[0, 2, 1]))

    x = tf.matmul(tf.transpose(self.w, perm=[0, 2, 1]), self.u)
    m = -1 + tf.nn.softplus(x)
    u_h = self.u + (m - x) * self.w / (
        tf.matmul(tf.transpose(self.w, perm=[0, 2, 1]), self.w))
    logp = logp - tf.reduce_sum(
        tf.log(tf.maximum(1 + tf.matmul(psi, u_h), 1e-6)), axis=[1, 2])
    z = z + tf.matmul(a, tf.transpose(u_h, perm=[0, 2, 1]))
    return z, logp


class iResFlow(SpectralNormLinear):
  """Invertable Residual Flow"""

  def __init__(self,
               dim,
               act_hidden='tanh',
               n_layer=2,
               sp_iters=5,
               name='ires_flow_layer'):
    super(iResFlow, self).__init__(sp_iters, name=name)
    self.name = name
    self.dim = dim
    self.n_layer = n_layer
    self.act_hidden = act_hidden
    self.param_dims = []
    for _ in range(self.n_layer):
      self.param_dims.append(self.dim**2)
      self.param_dims.append(self.dim)
    if act_hidden == 'lipswish':
      for _ in range(self.n_layer - 1):
        self.param_dims.append(1)
      self.act_hidden = lambda x, p: x * tf.sigmoid(tf.nn.softplus(p) * x) / 1.1
    elif act_hidden == 'tanh':
      self.act_hidden = lambda x, p: tf.tanh(x)
    elif act_hidden == 'elu':
      self.act_hidden = lambda x, p: tf.nn.elu(x)
    elif act_hidden == 'softplus':
      self.act_hidden = lambda x, p: tf.nn.softplus(x)
    else:
      raise NotImplementedError
    self.num_params = sum(self.param_dims)

  def _build(self, z, logp, params):
    params = tf.split(params, self.param_dims, axis=0)

    h = z
    for l in range(self.n_layer):
      w, b = params[l * 2], params[l * 2 + 1]
      w = tf.reshape(w, [self.dim, self.dim])
      w = self.spectral_norm(w, l)
      h = tf.matmul(h, w) + b
      if l + 1 < self.n_layer:
        h = self.act_hidden(h, params[-l - 1])
    x = z + h
    logp = logp - tf.log(tf.abs(tf.linalg.det(batch_jacobian(x, z))))
    return x, logp


class iCondResBlock(snt.AbstractModule):
  """block of Conditional Invertable Residual Flow"""

  def __init__(self,
               dim,
               cond_dim,
               act_hidden='tanh',
               sp_iters=1,
               name='icondres_flow_layer'):
    super(iCondResBlock, self).__init__(name=name)
    self.name = name
    self.dim = dim
    self.cond_dim = cond_dim
    if act_hidden == 'tanh':
      act = tf.tanh
    elif act_hidden == 'elu':
      act = tf.nn.elu
    elif act_hidden == 'relu':
      act = tf.nn.relu
    else:
      raise NotImplementedError
    with self._enter_variable_scope():
      self.mlp = MLP(
          dim + cond_dim,
          dim + cond_dim,
          1,
          output_dim=dim,
          act_hidden=act,
          sp_iters=sp_iters)

  def _build(self, z, z_cond, logp):
    bsize = tf.shape(z)[0]
    z_cond = tf.reshape(z_cond, (-1, self.cond_dim))
    z = tf.reshape(z, (-1, self.dim))
    z_input = tf.concat([z, z_cond], axis=1)

    x = self.mlp(z_input) + z
    if isinstance(logp, tf.Tensor):
      logp = tf.reshape(logp, shape=[-1])
    logp = logp - tf.log(tf.abs(tf.linalg.det(batch_jacobian(x, z))))

    x = tf.reshape(x, (bsize, -1, self.dim))
    logp = tf.reshape(logp, (bsize, -1))
    return x, logp


class iCondResFlow(AbstractNormFlow):
  """Conditional Invertable Residual Flow"""

  def __init__(self,
               dim,
               cond_dim,
               num_layers,
               act_hidden='tanh',
               sp_iters=5,
               name='icondres_flow'):
    fn_flow = lambda i: iCondResBlock(
        dim, cond_dim, act_hidden, sp_iters, name='icond-block-%d' % i)
    super(iCondResFlow, self).__init__(dim, num_layers, fn_flow, name=name)

  def _build(self, z, z_cond, logp):
    """
        Args:
            z: B x resolution (2048) x 3
            z_cond: B x cond_dim (128)
            logp: B x resolution (2048)
        Return:
            x: B x resolution (2048) x 3
            logp: B x resolution (2048)
    """
    if isinstance(z_cond, np.ndarray):
      z_cond = tf.constant(z_cond)
    for flow in self.flow_blocks:
      z, logp = flow(z, z_cond, logp)
    return z, logp


class ParamNormFlow(AbstractNormFlow):
  """Normalizing flow with params"""

  def __init__(self, dim, num_layers, flow_type, name='param_norm_flow'):
    super(ParamNormFlow, self).__init__(dim, num_layers, flow_type, name)
    with self._enter_variable_scope():
      self.params = []
      for i in range(self.num_layers):
        param = tf.get_variable(
            'param_%d' % i, shape=(self.flow_blocks[i].num_params,))
        self.params.append(param)

  def _build(self, z, logp):
    for i, flow in enumerate(self.flow_blocks):
      z, logp = flow(z, logp, self.params[i])
    return z, logp


class NormFlow(AbstractNormFlow):
  """Normalizing flow that needs external params"""

  def __init__(self, dim, num_layers, flow_type, name='norm_flow'):
    super(NormFlow, self).__init__(dim, num_layers, flow_type, name)
    self.num_params = sum([f.num_params for f in self.flow_blocks])

  def _build(self, z, logp, params):
    param_list = tf.split(params, num_or_size_splits=self.num_layers, axis=1)

    for i, flow in enumerate(self.flow_blocks):
      z, logp = flow(z, logp, param_list[i])
    return z, logp


class HyperNet(snt.AbstractModule):
  """Normalizing flow abstract"""

  def __init__(self,
               input_dim,
               hidden_dim,
               output_dim,
               depth,
               act_out=tf.nn.tanh,
               name='hypernet'):
    super(HyperNet, self).__init__(name=name)
    self.mlp = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        output_dim=output_dim)

  def _build(self, z):
    if isinstance(z, np.ndarray):
      z = tf.constant(z)
    return self.mlp(z)
