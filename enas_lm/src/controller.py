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

"""ENAS controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_float('controller_baseline_dec', 0.999, '')
flags.DEFINE_float('controller_entropy_weight', 1e-5, '')
flags.DEFINE_float('controller_temperature', 5., '')
flags.DEFINE_float('controller_tanh_constant', 2.25, '')
flags.DEFINE_float('controller_learning_rate', 5e-5, '')
flags.DEFINE_integer('controller_num_layers', 9, '')


REWARD_CONSTANT = 80.0


def _build_train_op(loss, tf_vars, learning_rate, train_step, num_aggregate):
  """Build training ops from `loss` tensor."""
  optim = tf.train.AdamOptimizer(learning_rate)
  optim = tf.train.SyncReplicasOptimizer(
      optim, replicas_to_aggregate=num_aggregate, total_num_replicas=1)
  grads = tf.gradients(loss, tf_vars)
  train_op = optim.apply_gradients(zip(grads, tf_vars), global_step=train_step)
  grad_norm = tf.global_norm(grads)
  return train_op, optim, grad_norm


def _lstm(x, prev_c, prev_h, w_lstm):
  """LSTM subgraph."""
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w_lstm)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  return next_c, next_h


def _set_default_params(params):
  """Add controller's default params."""
  params.add_hparam('controller_hidden_size', 64)
  params.add_hparam('controller_num_layers', FLAGS.controller_num_layers)
  params.add_hparam('controller_num_functions', 4)  # tanh, relu, sigmoid, iden

  params.add_hparam('controller_baseline_dec', FLAGS.controller_baseline_dec)
  params.add_hparam('controller_entropy_weight',
                    FLAGS.controller_entropy_weight)
  params.add_hparam('controller_temperature', FLAGS.controller_temperature)
  params.add_hparam('controller_tanh_constant', FLAGS.controller_tanh_constant)
  params.add_hparam('controller_learning_rate', FLAGS.controller_learning_rate)
  params.add_hparam('controller_num_aggregate', 10)
  params.add_hparam('controller_num_train_steps', 25)

  return params


class Controller(object):
  """ENAS controller. Samples architectures and creates training ops."""

  def __init__(self, params, name='controller'):
    print('-' * 80)
    print('Create a controller')
    self.params = _set_default_params(params)
    self.name = name
    self._build_params()
    self._build_sampler()

  def _build_params(self):
    """Create TF parameters."""
    initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
    num_funcs = self.params.controller_num_functions
    hidden_size = self.params.controller_hidden_size
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope('lstm'):
        self.w_lstm = tf.get_variable('w', [2 * hidden_size, 4 * hidden_size])

      with tf.variable_scope('embedding'):
        self.g_emb = tf.get_variable('g', [1, hidden_size])
        self.w_emb = tf.get_variable('w', [num_funcs, hidden_size])

      with tf.variable_scope('attention'):
        self.attn_w_1 = tf.get_variable('w_1', [hidden_size, hidden_size])
        self.attn_w_2 = tf.get_variable('w_2', [hidden_size, hidden_size])
        self.attn_v = tf.get_variable('v', [hidden_size, 1])

    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()
                      if v.name.startswith(self.name)])
    print('Controller has {0} params'.format(num_params))

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""
    hidden_size = self.params.controller_hidden_size
    num_layers = self.params.controller_num_layers

    arc_seq = []
    sample_log_probs = []
    sample_entropy = []
    all_h = [tf.zeros([1, hidden_size], dtype=tf.float32)]
    all_h_w = [tf.zeros([1, hidden_size], dtype=tf.float32)]

    # sampler ops
    inputs = self.g_emb
    prev_c = tf.zeros([1, hidden_size], dtype=tf.float32)
    prev_h = tf.zeros([1, hidden_size], dtype=tf.float32)

    inputs = self.g_emb
    for layer_id in range(1, num_layers+1):
      next_c, next_h = _lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      all_h.append(next_h)
      all_h_w.append(tf.matmul(next_h, self.attn_w_1))

      query = tf.matmul(next_h, self.attn_w_2)
      query = query + tf.concat(all_h_w[:-1], axis=0)
      query = tf.tanh(query)
      logits = tf.matmul(query, self.attn_v)
      logits = tf.reshape(logits, [1, layer_id])

      if self.params.controller_temperature:
        logits /= self.params.controller_temperature
      if self.params.controller_tanh_constant:
        logits = self.params.controller_tanh_constant * tf.tanh(logits)
      diff = tf.to_float(layer_id - tf.range(0, layer_id)) ** 2
      logits -= tf.reshape(diff, [1, layer_id]) / 6.0

      skip_index = tf.multinomial(logits, 1)
      skip_index = tf.to_int32(skip_index)
      skip_index = tf.reshape(skip_index, [1])
      arc_seq.append(skip_index)

      log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=skip_index)
      sample_log_probs.append(log_prob)

      entropy = log_prob * tf.exp(-log_prob)
      sample_entropy.append(tf.stop_gradient(entropy))

      inputs = tf.nn.embedding_lookup(
          tf.concat(all_h[:-1], axis=0), skip_index)
      inputs /= (0.1 + tf.to_float(layer_id - skip_index))

      next_c, next_h = _lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      logits = tf.matmul(next_h, self.w_emb, transpose_b=True)
      if self.params.controller_temperature:
        logits /= self.params.controller_temperature
      if self.params.controller_tanh_constant:
        logits = self.params.controller_tanh_constant * tf.tanh(logits)
      func = tf.multinomial(logits, 1)
      func = tf.to_int32(func)
      func = tf.reshape(func, [1])
      arc_seq.append(func)
      log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=func)
      sample_log_probs.append(log_prob)
      entropy = log_prob * tf.exp(-log_prob)
      sample_entropy.append(tf.stop_gradient(entropy))
      inputs = tf.nn.embedding_lookup(self.w_emb, func)

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = arc_seq

    self.sample_log_probs = tf.concat(sample_log_probs, axis=0)
    self.ppl = tf.exp(tf.reduce_mean(self.sample_log_probs))

    sample_entropy = tf.concat(sample_entropy, axis=0)
    self.sample_entropy = tf.reduce_sum(sample_entropy)

    self.all_h = all_h

  def build_trainer(self, child_model):
    """Build the train ops by connecting Controller with a Child."""
    # actor
    self.valid_loss = tf.to_float(child_model.rl_loss)
    self.valid_loss = tf.stop_gradient(self.valid_loss)
    self.valid_ppl = tf.exp(self.valid_loss)
    self.reward = REWARD_CONSTANT / self.valid_ppl

    if self.params.controller_entropy_weight:
      self.reward += self.params.controller_entropy_weight * self.sample_entropy

    # or baseline
    self.sample_log_probs = tf.reduce_sum(self.sample_log_probs)
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(self.baseline,
                                    ((1 - self.params.controller_baseline_dec) *
                                     (self.baseline - self.reward)))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)
    self.loss = self.sample_log_probs * (self.reward - self.baseline)

    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='train_step')
    tf_vars = [var for var in tf.trainable_variables()
               if var.name.startswith(self.name)]

    self.train_op, self.optimizer, self.grad_norm = _build_train_op(
        loss=self.loss,
        tf_vars=tf_vars,
        learning_rate=self.params.controller_learning_rate,
        train_step=self.train_step,
        num_aggregate=self.params.controller_num_aggregate)

  def train(self, sess, reset_op, log_every=10):
    """Train the controller for `num_steps`."""
    print('-' * 80)
    print('Training controller')
    num_steps = (self.params.controller_num_aggregate *
                 self.params.controller_num_train_steps)
    run_ops = [self.sample_arc,
               self.sample_entropy,
               self.reward,
               self.baseline,
               self.train_op]

    for step in range(num_steps):
      arc, ent, reward, baseline, _ = sess.run(run_ops)
      sess.run(reset_op)
      if step % log_every == 0:
        log_string = 'step={0:<5d}'.format(step)
        log_string += ' ent={0:<7.3f}'.format(ent)
        log_string += ' ppl={0:<7.2f}'.format(REWARD_CONSTANT / reward)
        log_string += ' rw={0:<7.4f}'.format(reward)
        log_string += ' bl={0:<7.4f}'.format(baseline)
        log_string += ' arc=[{0}]'.format(' '.join([str(v) for v in arc]))
        print(log_string)
