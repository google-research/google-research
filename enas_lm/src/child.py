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

"""AWD ENAS fixed model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from enas_lm.src import data_utils
from enas_lm.src import utils


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('child_batch_size', 128, '')
flags.DEFINE_integer('child_bptt_steps', 35, '')


def _gen_mask(shape, drop_prob):
  """Generate a droppout mask."""
  keep_prob = 1. - drop_prob
  mask = tf.random_uniform(shape, dtype=tf.float32)
  mask = tf.floor(mask + keep_prob) / keep_prob
  return mask


def _rnn_fn(sample_arc, x, prev_s, w_prev, w_skip, input_mask, layer_mask,
            params):
  """Multi-layer LSTM.

  Args:
    sample_arc: [num_layers * 2], sequence of tokens representing architecture.
    x: [batch_size, num_steps, hidden_size].
    prev_s: [batch_size, hidden_size].
    w_prev: [2 * hidden_size, 2 * hidden_size].
    w_skip: [None, [hidden_size, 2 * hidden_size] * (num_layers-1)].
    input_mask: `[batch_size, hidden_size]`.
    layer_mask: `[batch_size, hidden_size]`.
    params: hyper-params object.

  Returns:
    next_s: [batch_size, hidden_size].
    all_s: [[batch_size, num_steps, hidden_size] * num_layers].
  """
  batch_size = x.get_shape()[0].value
  num_steps = tf.shape(x)[1]
  num_layers = len(sample_arc) // 2

  all_s = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

  # extract the relevant variables, so that you only do L2-reg on them.
  u_skip = []
  start_idx = 0
  for layer_id in range(num_layers):
    prev_idx = sample_arc[start_idx]
    func_idx = sample_arc[start_idx + 1]
    u_skip.append(w_skip[layer_id][func_idx, prev_idx])
    start_idx += 2
  w_skip = u_skip
  var_s = [w_prev] + w_skip[1:]

  def _select_function(h, function_id):
    h = tf.stack([tf.tanh(h), tf.nn.relu(h), tf.sigmoid(h), h], axis=0)
    h = h[function_id]
    return h

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, prev_s, all_s):
    """Body function."""
    inp = x[:, step, :]

    # important change: first input uses a tanh()
    if layer_mask is not None:
      assert input_mask is not None
      ht = tf.matmul(tf.concat([inp * input_mask, prev_s * layer_mask],
                               axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([inp, prev_s], axis=1), w_prev)
    h, t = tf.split(ht, 2, axis=1)
    h = tf.tanh(h)
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    start_idx = 0
    used = []
    for layer_id in range(num_layers):
      prev_idx = sample_arc[start_idx]
      func_idx = sample_arc[start_idx + 1]
      used.append(tf.one_hot(prev_idx, depth=num_layers, dtype=tf.int32))
      prev_s = tf.stack(layers, axis=0)[prev_idx]
      if layer_mask is not None:
        ht = tf.matmul(prev_s * layer_mask, w_skip[layer_id])
      else:
        ht = tf.matmul(prev_s, w_skip[layer_id])
      h, t = tf.split(ht, 2, axis=1)

      h = _select_function(h, func_idx)
      t = tf.sigmoid(t)
      s = prev_s + t * (h - prev_s)
      s.set_shape([batch_size, params.hidden_size])
      layers.append(s)
      start_idx += 2

    next_s = tf.add_n(layers[1:]) / tf.cast(num_layers, dtype=tf.float32)
    all_s = all_s.write(step, next_s)

    return step + 1, next_s, all_s

  loop_inps = [tf.constant(0, dtype=tf.int32), prev_s, all_s]
  _, next_s, all_s = tf.while_loop(_condition, _body, loop_inps)
  all_s = tf.transpose(all_s.stack(), [1, 0, 2])

  return next_s, all_s, var_s


def _set_default_params(params):
  """Set default hyper-parameters."""
  params.add_hparam('alpha', 0.0)  # activation L2 reg
  params.add_hparam('beta', 1.)  # activation slowness reg
  params.add_hparam('best_valid_ppl_threshold', 5)

  params.add_hparam('batch_size', FLAGS.child_batch_size)
  params.add_hparam('bptt_steps', FLAGS.child_bptt_steps)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_e', 0.10)  # word
  params.add_hparam('drop_i', 0.20)  # embeddings
  params.add_hparam('drop_x', 0.75)  # input to RNN cells
  params.add_hparam('drop_l', 0.25)  # between layers
  params.add_hparam('drop_o', 0.75)  # output
  params.add_hparam('drop_w', 0.00)  # weight

  params.add_hparam('grad_bound', 0.1)
  params.add_hparam('hidden_size', 200)
  params.add_hparam('init_range', 0.04)
  params.add_hparam('learning_rate', 20.)
  params.add_hparam('num_train_epochs', 600)
  params.add_hparam('vocab_size', 10000)

  params.add_hparam('weight_decay', 8e-7)
  return params


class LM(object):
  """Language model."""

  def __init__(self, params, controller, x_train, x_valid, name='child'):
    print('-' * 80)
    print('Building LM')

    self.params = _set_default_params(params)
    self.controller = controller
    self.sample_arc = tf.unstack(controller.sample_arc)
    self.name = name

    # train data
    (self.x_train, self.y_train,
     self.num_train_batches, self.reset_start_idx,
     self.should_reset, self.base_bptt) = data_utils.input_producer(
         x_train, params.batch_size, params.bptt_steps, random_len=True)
    params.add_hparam(
        'num_train_steps', self.num_train_batches * params.num_train_epochs)

    # valid data
    (self.x_valid, self.y_valid,
     self.num_valid_batches) = data_utils.input_producer(
         x_valid, params.batch_size, params.bptt_steps)

    self._build_params()
    self._build_train()
    self._build_valid()

  def _build_params(self):
    """Create model parameters."""

    print('-' * 80)
    print('Building model params')
    initializer = tf.initializers.random_uniform(minval=-self.params.init_range,
                                                 maxval=self.params.init_range)
    num_functions = self.params.controller_num_functions
    num_layers = self.params.controller_num_layers
    hidden_size = self.params.hidden_size
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope('embedding'):
        w_emb = tf.get_variable('w', [self.params.vocab_size, hidden_size])
        dropped_w_emb = tf.layers.dropout(
            w_emb, self.params.drop_e, [self.params.vocab_size, 1],
            training=True)

      with tf.variable_scope('rnn_cell'):
        w_prev = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])
        i_mask = tf.ones([hidden_size, 2 * hidden_size], dtype=tf.float32)
        h_mask = _gen_mask([hidden_size, 2 * hidden_size], self.params.drop_w)
        mask = tf.concat([i_mask, h_mask], axis=0)
        dropped_w_prev = w_prev * mask

        w_skip, dropped_w_skip = [], []
        for layer_id in range(1, num_layers+1):
          with tf.variable_scope('layer_{}'.format(layer_id)):
            w = tf.get_variable(
                'w', [num_functions, layer_id, hidden_size, 2 * hidden_size])
            mask = _gen_mask([1, 1, hidden_size, 2 * hidden_size],
                             self.params.drop_w)
            dropped_w = w * mask
            w_skip.append(w)
            dropped_w_skip.append(dropped_w)

      with tf.variable_scope('init_states'):
        with tf.variable_scope('batch'):
          init_shape = [self.params.batch_size, hidden_size]
          batch_prev_s = tf.get_variable(
              's', init_shape, dtype=tf.float32, trainable=False)
          zeros = np.zeros(init_shape, dtype=np.float32)
          batch_reset = tf.assign(batch_prev_s, zeros)

    self.num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()
                           if v.name.startswith(self.name)]).value
    print('All children have {0} params'.format(self.num_params))

    num_params_per_child = 0
    for v in tf.trainable_variables():
      if v.name.startswith(self.name):
        if 'rnn_cell' in v.name:
          num_params_per_child += v.shape[-2].value * v.shape[-1].value
        else:
          num_params_per_child += np.prod([d.value for d in v.shape])
    print('Each child has {0} params'.format(num_params_per_child))

    self.batch_init_states = {
        's': batch_prev_s,
        'reset': batch_reset,
    }
    self.train_params = {
        'w_emb': dropped_w_emb,
        'w_prev': dropped_w_prev,
        'w_skip': dropped_w_skip,
        'w_soft': w_emb,
    }
    self.eval_params = {
        'w_emb': w_emb,
        'w_prev': w_prev,
        'w_skip': w_skip,
        'w_soft': w_emb,
    }

  def _forward(self, x, y, model_params, init_states, is_training=False):
    """Computes the logits.

    Args:
      x: [batch_size, num_steps], input batch.
      y: [batch_size, num_steps], output batch.
      model_params: a `dict` of params to use.
      init_states: a `dict` of params to use.
      is_training: if `True`, will apply regularizations.

    Returns:
      loss: scalar, cross-entropy loss
    """
    w_emb = model_params['w_emb']
    w_prev = model_params['w_prev']
    w_skip = model_params['w_skip']
    w_soft = model_params['w_soft']
    prev_s = init_states['s']

    emb = tf.nn.embedding_lookup(w_emb, x)
    batch_size = self.params.batch_size
    hidden_size = self.params.hidden_size
    sample_arc = self.sample_arc
    if is_training:
      emb = tf.layers.dropout(
          emb, self.params.drop_i, [batch_size, 1, hidden_size], training=True)

      input_mask = _gen_mask([batch_size, hidden_size], self.params.drop_x)
      layer_mask = _gen_mask([batch_size, hidden_size], self.params.drop_l)
    else:
      input_mask = None
      layer_mask = None

    out_s, all_s, var_s = _rnn_fn(sample_arc, emb, prev_s, w_prev, w_skip,
                                  input_mask, layer_mask, params=self.params)

    top_s = all_s
    if is_training:
      top_s = tf.layers.dropout(
          top_s, self.params.drop_o,
          [self.params.batch_size, 1, self.params.hidden_size], training=True)

    carry_on = [tf.assign(prev_s, out_s)]
    logits = tf.einsum('bnh,vh->bnv', top_s, w_soft)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
    loss = tf.reduce_mean(loss)

    reg_loss = loss  # `loss + regularization_terms` is for training only
    if is_training:
      # L2 weight reg
      self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(w ** 2) for w in var_s])
      reg_loss += self.params.weight_decay * self.l2_reg_loss

      # activation L2 reg
      reg_loss += self.params.alpha * tf.reduce_mean(all_s ** 2)

      # activation slowness reg
      reg_loss += self.params.beta * tf.reduce_mean(
          (all_s[:, 1:, :] - all_s[:, :-1, :]) ** 2)

    with tf.control_dependencies(carry_on):
      loss = tf.identity(loss)
      if is_training:
        reg_loss = tf.identity(reg_loss)

    return reg_loss, loss

  def _build_train(self):
    """Build training ops."""
    print('-' * 80)
    print('Building train graph')
    reg_loss, loss = self._forward(self.x_train, self.y_train,
                                   self.train_params, self.batch_init_states,
                                   is_training=True)

    tf_vars = [v for v in tf.trainable_variables()
               if v.name.startswith(self.name)]
    global_step = tf.train.get_or_create_global_step()
    lr_scale = (tf.cast(tf.shape(self.y_train)[-1], dtype=tf.float32) /
                tf.cast(self.params.bptt_steps, dtype=tf.float32))
    learning_rate = utils.get_lr(global_step, self.params) * lr_scale
    if self.params.grad_bound:
      grads = tf.gradients(reg_loss, tf_vars)
      clipped_grads, grad_norm = tf.clip_by_global_norm(grads,
                                                        self.params.grad_bound)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars),
                                         global_step=global_step)

    self.train_loss = loss
    self.train_op = train_op
    self.grad_norm = grad_norm
    self.learning_rate = learning_rate

  def _build_valid(self):
    print('Building valid graph')
    _, loss = self._forward(self.x_valid, self.y_valid,
                            self.eval_params, self.batch_init_states)
    self.valid_loss = loss
    self.rl_loss = loss

  def eval_valid(self, sess):
    """Eval 1 round on valid set."""
    total_loss = 0
    for _ in range(self.num_valid_batches):
      sess.run(self.batch_init_states['reset'])
      total_loss += sess.run(self.valid_loss)
    valid_ppl = np.exp(total_loss / self.num_valid_batches)
    print('valid_ppl={0:<.2f}'.format(valid_ppl))

    return valid_ppl
