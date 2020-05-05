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

from ebp.common.tf_utils import MLP
from ebp.common.flow_family import iCondResFlow
from ebp.common.flow_family import HyperNet, NormFlow


class Generator(snt.AbstractModule):

  def __init__(self,
               pc_dim=(2048, 3),
               fc_dims=(64, 128, 512, 1024),
               act=tf.nn.relu,
               entropy_reg=True,
               batch_norm=False,
               name='gen'):
    super(Generator, self).__init__(name=name)

    self.pc_dim = pc_dim
    self.act = act
    self.batch_norm = batch_norm
    self.entropy_reg = entropy_reg

    self.fc_body = []
    self.fc_sigma_body = []
    self.bn_body = []
    self.bn_sigma_body = []

    with self._enter_variable_scope():
      for i, fc_dim in enumerate(fc_dims):
        fc = snt.Linear(fc_dim, name='fc_%d' % i)
        self.fc_body.append(fc)
        self.bn_body.append(
            snt.BatchNorm(offset=True, scale=True, name='bn_%d' % i))
      self.fc_final = snt.Linear(np.prod(pc_dim), name='fc_final')
      for i, fc_dim in enumerate(fc_dims):
        fc = snt.Linear(fc_dim, name='fc_sigma_%d' % i)
        self.fc_sigma_body.append(fc)
        self.bn_sigma_body.append(
            snt.BatchNorm(offset=True, scale=True, name='bn_sigma_%d' % i))
      self.fc_sigma_final = snt.Linear(np.prod(pc_dim), name='fc_sigma_final')

  def _build(self, z, is_training=True):
    x = self.fc_body[0](z)
    if self.batch_norm:
      x = self.bn_body[0](x, is_training)
    for i in range(1, len(self.fc_body)):
      x = self.act(x)
      x = self.fc_body[i](x)
      if self.batch_norm:
        x = self.bn_body[i](x, is_training)
    x = self.act(x)
    x = self.fc_final(x)

    logprob = None
    if self.entropy_reg:
      sigma = self.fc_sigma_body[0](z)
      for fc in self.fc_sigma_body[1:]:
        sigma = self.act(sigma)
        sigma = fc(sigma)
      sigma = self.act(sigma)
      sigma = self.fc_sigma_final(sigma)
      sigma = tf.sigmoid(sigma)
      #sigma = tf.abs(1e-3 * tf.sigmoid(sigma))
      logprob = tf.reduce_sum(-tf.log(sigma + 1e-6), axis=1)
      x = x + sigma * tf.random_normal(tf.shape(sigma))

    x = tf.reshape(x, (-1,) + self.pc_dim)
    #with tf.control_dependencies([tf.print('ent', tf.reduce_mean(logprob))]):
    return x, tf.identity(logprob)

  def generate_noise(self, num_samples, z_dim=128, mu=0, sigma=0.2):
    return np.random.normal(mu, sigma, (num_samples, *z_dim))


class LVMBlock(snt.AbstractModule):

  def __init__(self,
               gauss_dim,
               depth=3,
               act_hidden=tf.nn.relu,
               name='lvm_block'):
    super(LVMBlock, self).__init__(name=name)

    hidden_dims = [min(gauss_dim, 256)] * depth
    with self._enter_variable_scope():
      self.mlp = snt.nets.MLP(
          output_sizes=hidden_dims, activation=act_hidden, activate_final=True)
      self.w_mu = tf.get_variable('w_mu', shape=[hidden_dims[-1], gauss_dim])
      self.b_mu = tf.get_variable('b_mu', shape=[1, gauss_dim])
      self.w_logsig = tf.get_variable(
          'w_logsig', shape=[hidden_dims[-1], gauss_dim])
      self.b_logsig = tf.get_variable('b_logsig', shape=[1, gauss_dim])

  def _build(self, inputs):
    z = self.mlp(inputs)
    mu = tf.matmul(z, self.w_mu) + self.b_mu
    logsig = tf.matmul(z, self.w_logsig) + self.b_logsig
    sigma = tf.sigmoid(logsig)
    sigma = tf.exp(logsig)
    eps = tf.random.normal(
        shape=tf.shape(mu), mean=0, stddev=1, dtype=tf.float32)
    x = mu + sigma * eps
    ent = tf.reduce_sum(-tf.log(sigma + 1e-6), axis=-1)
    return x, mu, logsig, ent


class RNNGenerator(snt.AbstractModule):

  def __init__(self,
               block_size,
               rnn_input_dim=128,
               state_dim=128,
               pc_dim=(2048, 3),
               cell_type='lstm',
               act_hidden=tf.nn.relu,
               gen_depth=3,
               name='rnn_generator'):
    """Args:

      state_dim: dimensionality of hidden states of the RNN cell
      block_size: number of points to generate at once
      pc_dim: a single point cloud's dimension
      cell_type: one of [lstm, gru].
    """
    assert (pc_dim[0] % block_size == 0)
    super(RNNGenerator, self).__init__(name=name)
    self.rnn_input_dim = rnn_input_dim
    self.pc_dim = pc_dim
    self.gauss_dim = block_size * pc_dim[-1]
    self.block_size = block_size
    self.num_blocks = pc_dim[0] // block_size
    self.state_dim = state_dim
    self.cell_type = cell_type

    with self._enter_variable_scope():
      self.input_proj = snt.nets.MLP(
          output_sizes=[rnn_input_dim * 2, rnn_input_dim],
          activation=act_hidden,
          activate_final=True)
      if cell_type == 'lstm':
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(state_dim)
      elif cell_type == 'gru':
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(state_dim)
      else:
        raise ValueError('cell_type {} not implemented'.format(cell_type))
      self.output_lvm = LVMBlock(
          self.gauss_dim, act_hidden=act_hidden, depth=gen_depth)

  def _build(self, z):
    x, mu, logsig, ent = self.output_lvm(z)
    state_input = self.input_proj(tf.concat([x, mu, logsig], axis=-1))
    sample_output = tf.expand_dims(x, 0)
    ent_output = tf.expand_dims(ent, 0)

    if self.cell_type == 'lstm':
      init_state = tf.nn.rnn_cell.LSTMStateTuple(z, z)
    else:
      init_state = z

    def loop_body(prev_state, state_input, sample_output, ent_output):
      state_output, next_state = self.rnn_cell(state_input, prev_state)
      x, mu, logsig, ent = self.output_lvm(state_output)
      sample_output = tf.concat([sample_output, tf.expand_dims(x, 0)], axis=0)
      ent_output = tf.concat([ent_output, tf.expand_dims(ent, 0)], axis=0)

      # prep for next iteration
      state_input = self.input_proj(tf.concat([x, mu, logsig], axis=-1))
      return next_state, state_input, sample_output, ent_output

    def loop_cond(prev_state, state_input, sample_output, ent_output):
      return tf.shape(ent_output)[0] < self.num_blocks

    if self.cell_type == 'lstm':
      shape_invariant = tf.nn.rnn_cell.LSTMStateTuple(
          tf.TensorShape((None, self.state_dim)),
          tf.TensorShape((None, self.state_dim)))
    else:
      shape_invariant = tf.TensorShape((None, self.state_dim))

    _, _, sample_output, ent_output = tf.while_loop(
        loop_cond,
        loop_body, [init_state, state_input, sample_output, ent_output],
        shape_invariants=[
            shape_invariant,
            tf.TensorShape((None, self.rnn_input_dim)),
            tf.TensorShape((None, None, self.gauss_dim)),
            tf.TensorShape((None, None))
        ])

    sample_output = tf.reshape(
        tf.transpose(sample_output, [1, 0, 2]), (-1,) + self.pc_dim)
    ent_output = tf.reduce_sum(ent_output, axis=0)
    return sample_output, ent_output


class GPRNN(snt.AbstractModule):

  def __init__(self,
               block_size,
               act_hidden=tf.nn.relu,
               pc_dim=(2048, 3),
               init_z_dim=128,
               name='rnn_generator'):
    super(GPRNN, self).__init__(name=name)
    self.pc_dim = pc_dim
    gauss_dim = block_size * pc_dim[-1]
    assert (pc_dim[0] % block_size == 0)
    self.num_blocks = pc_dim[0] // block_size - 1

    with self._enter_variable_scope():
      self.first_block = LVMBlock(
          init_z_dim, gauss_dim, act_hidden=self.act_hidden)
      if self.num_blocks > 0:
        self.lvm_block = LVMBlock(
            gauss_dim * pc_dim[-1], gauss_dim, act_hidden=self.act_hidden)

  def _build(self, z):
    list_x = []
    list_ent = []
    x, mu, logsig, ent = self.first_block(z)
    list_x.append(x)
    list_ent.append(ent)
    for _ in range(self.num_blocks):
      x, mu, logsig, ent = self.lvm_block(tf.concat([x, mu, logsig], axis=-1))
      list_x.append(x)
      list_ent.append(ent)
    x = tf.reshape(tf.concat(list_x, axis=-1), (-1,) + self.pc_dim)
    ent = tf.reduce_sum(list_ent, axis=0)
    return x, tf.identity(ent)


class DeterministicEncoder(snt.AbstractModule):
  """The Encoder."""

  def __init__(self, output_sizes):
    super(DeterministicEncoder, self).__init__(name='DeterministicEncoder')
    """CNP encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
    """
    self._output_sizes = output_sizes

  def _build(self, context_x, context_y, num_context_points):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of size bs x observations x m_ch. For this 1D regression
        task this corresponds to the x-values.
      context_y: Tensor of size bs x observations x d_ch. For this 1D regression
        task this corresponds to the y-values.
      num_context_points: A tensor containing a single scalar that indicates the
        number of context_points provided in this iteration.

    Returns:
      representation: The encoded representation averaged over all context
          points.
    """

    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([context_x, context_y], axis=-1)

    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = encoder_input.shape.as_list()
    hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
    hidden.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name='Encoder_layer_{}'.format(i)))

      # Last layer without a ReLu
      hidden = tf.layers.dense(
          hidden, self._output_sizes[-1], name='Encoder_layer_{}'.format(i + 1))

    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

    # Aggregator: take the mean over all points
    representation = tf.reduce_mean(hidden, axis=1)

    return representation


class DeterministicDecoder(snt.AbstractModule):
  """The Decoder."""

  def __init__(self, output_sizes):
    """CNP decoder.

    Args:
      output_sizes: An iterable containing the output sizes of the decoder MLP
        as defined in `basic.Linear`.
    """
    super(DeterministicDecoder, self).__init__(name='DeterministicDecoder')
    self._output_sizes = output_sizes

  def _build(self, representation, target_x, num_total_points):
    """Decodes the individual targets.

    Args:
      representation: The encoded representation of the context
      target_x: The x locations for the target query
      num_total_points: The number of target points.

    Returns:
      dist: A multivariate Gaussian over the target points.
      mu: The mean of the multivariate Gaussian.
      sigma: The standard deviation of the multivariate Gaussian.
    """

    # Concatenate the representation and the target_x
    representation = tf.tile(
        tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
    input = tf.concat([representation, target_x], axis=-1)

    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = input.shape.as_list()
    hidden = tf.reshape(input, (batch_size * num_total_points, -1))
    hidden.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name='Decoder_layer_{}'.format(i)))

      # Last layer without a ReLu
      hidden = tf.layers.dense(
          hidden, self._output_sizes[-1], name='Decoder_layer_{}'.format(i + 1))

    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return dist, mu, sigma


class MLPGen(snt.AbstractModule):

  def __init__(self,
               dim,
               hidden_dim,
               depth,
               output_dim,
               act_hidden=tf.nn.relu,
               sp_iters=0,
               mlp=None,
               name='mlp_gauss'):
    super(MLPGen, self).__init__(name=name)
    self.dim = dim
    with self._enter_variable_scope():
      if mlp is None:
        self.mlp = MLP(self.dim + 31, hidden_dim, depth, 1, act_hidden,
                       sp_iters)
      else:
        self.mlp = mlp

  def _build(self, raw_x):
    z = tf.random.normal(
        shape=[tf.shape(raw_x)[0], tf.shape(raw_x)[1], 32],
        mean=0,
        stddev=1,
        dtype=tf.float32)
    x = tf.concat([raw_x, z], -1)
    y = self.mlp(x)
    y = tf.reshape(y, [-1, tf.shape(raw_x)[1], 1])
    return tf.concat([raw_x, y], -1)


class iCondGen(snt.AbstractModule):

  def __init__(self,
               dim,
               cond_dim,
               num_layers,
               act_hidden='tanh',
               sp_iters=1,
               name='icondres_flow'):
    super(iCondGen, self).__init__(name=name)
    self.dim = dim
    self.cond_dim = cond_dim
    self.i_cond_flow = iCondResFlow(dim, cond_dim, num_layers, act_hidden,
                                    sp_iters)

  def _build(self, raw_x):
    x = tf.reshape(raw_x, [-1, self.dim])
    z = tf.random.normal(
        shape=[tf.shape(x)[0], self.cond_dim],
        mean=0,
        stddev=1,
        dtype=tf.float32)

    y, logp = self.i_cond_flow(z, x, 0)
    y = tf.reshape(y, [-1, tf.shape(raw_x)[1], 1])
    logp = tf.reshape(logp, [-1, 1])
    return tf.concat([raw_x, y], -1), logp


class iDoubleCondGen(snt.AbstractModule):

  def __init__(self,
               dim,
               condx_dim,
               condz_dim,
               num_layers,
               act_hidden='tanh',
               sp_iters=1,
               name='icondres_flow'):
    super(iDoubleCondGen, self).__init__(name=name)
    self.dim = dim
    self.condx_dim = condx_dim
    self.condz_dim = condz_dim
    with self._enter_variable_scope():
      self.i_cond_flow = iCondResFlow(dim, condz_dim + condz_dim, num_layers,
                                      act_hidden, sp_iters)
      self.fc = snt.Linear(condz_dim)
      self.mlp = MLP(condz_dim, condz_dim, 2, condz_dim, tf.nn.relu)

  def _build(self, raw_x, z_cond):
    x = tf.reshape(raw_x, [-1, self.dim])
    z_cond = tf.tile(z_cond, [1, tf.shape(raw_x)[1]])
    z_cond = tf.reshape(z_cond, [-1, self.condz_dim])
    z_cond = self.mlp(z_cond)
    z = tf.random.normal(
        shape=[tf.shape(x)[0], self.condx_dim],
        mean=0,
        stddev=1,
        dtype=tf.float32)
    x = self.fc(x)
    ctx = tf.concat([x, z_cond], axis=-1)
    y, logp = self.i_cond_flow(z, ctx, 0)
    y = tf.reshape(y, [-1, tf.shape(raw_x)[1], 1])
    logp = tf.reshape(logp, [-1, tf.shape(raw_x)[1], 1])
    logp = tf.reduce_sum(logp, axis=1, keepdims=False)
    return tf.concat([raw_x, y], -1), logp


class HyperGen(snt.AbstractModule):

  def __init__(self, dim, condx_dim, condz_dim, num_layers, name='HyperGen'):
    super(HyperGen, self).__init__(name=name)
    self.dim = dim
    self.condx_dim = condx_dim
    self.condz_dim = condz_dim

    with self._enter_variable_scope():
      self.fc = snt.Linear(condz_dim)
      self.norm_flow = NormFlow(self.dim, num_layers, 'planar')
      self.hnet = HyperNet(
          2 * condz_dim, 256, self.norm_flow.num_params, depth=2)

  def _build(self, raw_x, z_cond):
    x = tf.reshape(raw_x, [-1, self.dim])
    z_cond = tf.tile(z_cond, [1, tf.shape(raw_x)[1]])
    z_cond = tf.reshape(z_cond, [-1, self.condz_dim])
    z = tf.random.normal(
        shape=[tf.shape(x)[0], 1, self.dim], mean=0, stddev=1, dtype=tf.float32)
    x = self.fc(x)
    ctx = tf.concat([x, z_cond], axis=-1)

    params = self.hnet(ctx)
    y, logp = self.norm_flow(z, 0, params)
    y = tf.reshape(y, [-1, tf.shape(raw_x)[1], 1])
    logp = tf.reshape(logp, [-1, tf.shape(raw_x)[1], 1])
    logp = tf.reduce_sum(logp, axis=1, keepdims=False)
    return tf.concat([raw_x, y], -1), logp
