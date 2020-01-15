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

"""AWD LSTM model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from enas_lm.src import data_utils
from enas_lm.src import utils

MOVING_AVERAGE_DECAY = 0.9995


MOVING_AVERAGE_DECAY = 0.9995


def _gen_mask(shape, drop_prob):
  """Generate a droppout mask."""
  keep_prob = 1. - drop_prob
  mask = tf.random_uniform(shape, dtype=tf.float32)
  mask = tf.floor(mask + keep_prob) / keep_prob
  return mask


def _lstm(x, prev_c, prev_h, w_lstm, layer_masks):
  """Multi-layer LSTM.

  Args:
    x: [batch_size, num_steps, hidden_size].
    prev_c: [[batch_size, hidden_size] * num_layers].
    prev_h: [[batch_size, hidden_size] * num_layers].
    w_lstm: [[2 * hidden_size, 4 * hidden_size] * num_layers].
    layer_masks: [([hidden_size, hidden_size] or None)* num_layers].

  Returns:
    next_c: [[batch_size, hidden_size] * num_layers].
    next_h: [[batch_size, hidden_size] * num_layers].
    all_h: [batch_size, num_steps, hidden_size].
  """
  _, num_steps, _ = tf.unstack(tf.shape(x))
  num_layers = len(w_lstm)

  all_h = [tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)
           for _ in range(num_layers)]

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, pprev_c, pprev_h, all_h):
    """Apply LSTM at each step."""
    next_c, next_h = [], []
    for layer_id, (p_c, p_h, w, m) in enumerate(zip(
        pprev_c, pprev_h, w_lstm, layer_masks)):
      inp = x[:, step, :] if layer_id == 0 else next_h[-1]
      if m is not None:
        inp *= m
      ifog = tf.matmul(tf.concat([inp, p_h], axis=1), w)
      i, f, o, g = tf.split(ifog, 4, axis=1)
      i = tf.sigmoid(i)
      f = tf.sigmoid(f)
      o = tf.sigmoid(o)
      g = tf.tanh(g)
      c = i * g + f * p_c
      h = o * tf.tanh(c)
      all_h[layer_id] = all_h[layer_id].write(step, h)
      next_c.append(c)
      next_h.append(h)
    return step + 1, next_c, next_h, all_h

  loop_inps = [tf.constant(0, dtype=tf.int32), prev_c, prev_h, all_h]
  _, next_c, next_h, all_h = tf.while_loop(_condition, _body, loop_inps,
                                           parallel_iterations=1)
  all_h = [tf.transpose(h.stack(), [1, 0, 2])
           for h in all_h]

  return next_c, next_h, all_h


def _set_default_params(params):
  """Set default parameters."""
  params.add_hparam('alpha', 2.)  # activation L2 reg
  params.add_hparam('best_valid_ppl_threshold', 7)
  params.add_hparam('beta', 1.)  # activation slowness reg

  params.add_hparam('batch_size', 12)
  params.add_hparam('bptt_steps', 70)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_e', 0.10)  # word
  params.add_hparam('drop_i', 0.65)  # embeddings
  params.add_hparam('drop_l', 0.30)  # between layers
  params.add_hparam('drop_o', 0.40)  # output
  params.add_hparam('drop_w', 0.50)  # weight

  params.add_hparam('emb_size', 400)
  params.add_hparam('start_decay_epoch', 14)
  params.add_hparam('decay_every_epoch', 1)
  params.add_hparam('decay_rate', 0.98)
  params.add_hparam('grad_bound', 0.25)
  params.add_hparam('hidden_size', 1100)
  params.add_hparam('init_range', 0.1)
  params.add_hparam('learning_rate', 20.)
  params.add_hparam('num_layers', 3)
  params.add_hparam('num_train_epochs', 500)
  params.add_hparam('vocab_size', 10000)

  params.add_hparam('weight_decay', 1.2e-6)
  return params


class LM(object):
  """Language model."""

  def __init__(self, params, x_train, x_valid, x_test, name='language_model'):
    print('-' * 80)
    print('Building LM')

    self.params = _set_default_params(params)
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

    # test data
    (self.x_test, self.y_test,
     self.num_test_batches) = data_utils.input_producer(x_test, 1, 1)

    params.add_hparam('start_decay_step',
                      params.start_decay_epoch * self.num_train_batches)
    params.add_hparam('decay_every_step',
                      params.decay_every_epoch * self.num_train_batches)

    self._build_params()
    self._build_train()
    self._build_valid()
    self._build_test()

  def _build_params(self):
    """Create and count model parameters."""
    print('-' * 80)
    print('Building model params')
    with tf.variable_scope(self.name):
      with tf.variable_scope('embedding'):
        initializer = tf.initializers.random_uniform(
            -self.params.init_range, self.params.init_range)
        w_emb = tf.get_variable(
            'w', [self.params.vocab_size, self.params.emb_size],
            initializer=initializer)
        dropped_w_emb = tf.layers.dropout(
            w_emb, self.params.drop_e, [self.params.vocab_size, 1],
            training=True)

      w_lstm = []
      dropped_w_lstm = []
      with tf.variable_scope('lstm'):
        for i in range(self.params.num_layers):
          inp_size = self.params.emb_size if i == 0 else self.params.hidden_size
          hid_size = (self.params.emb_size if i == self.params.num_layers - 1
                      else self.params.hidden_size)
          init_range = 1.0 / np.sqrt(hid_size)
          initializer = tf.initializers.random_uniform(-init_range, init_range)
          with tf.variable_scope('layer_{0}'.format(i)):
            w = tf.get_variable('w', [inp_size + hid_size, 4 * hid_size],
                                initializer=initializer)
            i_mask = tf.ones([inp_size, 4 * hid_size], dtype=tf.float32)
            h_mask = _gen_mask([hid_size, 4 * hid_size], self.params.drop_w)
            mask = tf.concat([i_mask, h_mask], axis=0)
            dropped_w = w * mask
            w_lstm.append(w)
            dropped_w_lstm.append(dropped_w)

      with tf.variable_scope('init_states'):
        batch_prev_c, batch_prev_h, batch_reset = [], [], []
        test_prev_c, test_prev_h, test_reset = [], [], []
        for i in range(self.params.num_layers):
          inp_size = self.params.emb_size if i == 0 else self.params.hidden_size
          hid_size = (self.params.emb_size if i == self.params.num_layers - 1
                      else self.params.hidden_size)

          with tf.variable_scope('layer_{0}'.format(i)):
            with tf.variable_scope('batch'):
              init_shape = [self.params.batch_size, hid_size]
              batch_prev_c.append(tf.get_variable(
                  'c', init_shape, dtype=tf.float32, trainable=False))
              batch_prev_h.append(tf.get_variable(
                  'h', init_shape, dtype=tf.float32, trainable=False))
              zeros = np.zeros(init_shape, dtype=np.float32)
              batch_reset.append(tf.assign(batch_prev_c[-1], zeros))
              batch_reset.append(tf.assign(batch_prev_h[-1], zeros))
            with tf.variable_scope('test'):
              init_shape = [1, hid_size]
              test_prev_c.append(tf.get_variable(
                  'c', init_shape, dtype=tf.float32, trainable=False))
              test_prev_h.append(tf.get_variable(
                  'h', init_shape, dtype=tf.float32, trainable=False))
              zeros = np.zeros(init_shape, dtype=np.float32)
              test_reset.append(tf.assign(test_prev_c[-1], zeros))
              test_reset.append(tf.assign(test_prev_h[-1], zeros))

    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print('Model has {0} params'.format(num_params))

    self.batch_init_states = {
        'c': batch_prev_c,
        'h': batch_prev_h,
        'reset': batch_reset,
    }
    self.train_params = {
        'w_emb': dropped_w_emb,
        'w_lstm': dropped_w_lstm,
        'w_soft': w_emb,
    }
    self.test_init_states = {
        'c': test_prev_c,
        'h': test_prev_h,
        'reset': test_reset,
    }
    self.eval_params = {
        'w_emb': w_emb,
        'w_lstm': w_lstm,
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
    w_lstm = model_params['w_lstm']
    w_soft = model_params['w_soft']
    prev_c = init_states['c']
    prev_h = init_states['h']

    emb = tf.nn.embedding_lookup(w_emb, x)
    if is_training:
      emb = tf.layers.dropout(
          emb, self.params.drop_i,
          [self.params.batch_size, 1, self.params.emb_size], training=True)

      layer_masks = [None]
      for _ in range(1, self.params.num_layers - 1):
        mask = _gen_mask([self.params.batch_size, self.params.hidden_size],
                         self.params.drop_l)
        layer_masks.append(mask)
      layer_masks.append(None)
    else:
      layer_masks = [None] * self.params.num_layers

    out_c, out_h, all_h = _lstm(emb, prev_c, prev_h, w_lstm, layer_masks)
    top_h = all_h[-1]
    if is_training:
      top_h = tf.layers.dropout(
          top_h, self.params.drop_o,
          [self.params.batch_size, 1, self.params.emb_size], training=True)

    carry_on = []
    for var, val in zip(prev_c + prev_h, out_c + out_h):
      carry_on.append(tf.assign(var, val))

    logits = tf.einsum('bnh,vh->bnv', top_h, w_soft)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
    loss = tf.reduce_mean(loss)  # TODO(hyhieu): watch for num_steps

    reg_loss = loss  # loss + regularization_terms, for training only
    if is_training:
      # L2 weight reg
      reg_loss += self.params.weight_decay * tf.add_n(
          [tf.reduce_sum(w ** 2) for w in tf.trainable_variables()])

      # activation L2 reg
      reg_loss += self.params.alpha * tf.add_n(
          [tf.reduce_mean(h ** 2) for h in all_h[:-1]])

      # activation slowness L2 reg
      reg_loss += self.params.beta * tf.add_n(
          [tf.reduce_mean((h[:, 1:, :] - h[:, :-1, :]) ** 2)
           for h in all_h[:-1]])

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

    tf_vars = tf.trainable_variables()
    global_step = tf.train.get_or_create_global_step()
    lr_scale = (tf.cast(tf.shape(self.y_train)[-1], dtype=tf.float32) /
                tf.cast(self.params.bptt_steps, dtype=tf.float32))
    learning_rate = utils.get_lr(global_step, self.params) * lr_scale
    # learning_rate = tf.Print(
    #     learning_rate,
    #     [learning_rate, lr_scale, self.base_bptt, tf.shape(self.y_train)],
    #     message='lr: ', summarize=3)
    grads = tf.gradients(reg_loss, tf_vars)
    clipped_grads, grad_norm = tf.clip_by_global_norm(grads,
                                                      self.params.grad_bound)

    (self.update_moving_avg_ops, self.use_moving_avg_vars,
     self.restore_normal_vars) = self._create_average_ops()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars),
                                         global_step=global_step)

    self.train_loss = loss
    self.train_op = train_op
    self.grad_norm = grad_norm
    self.learning_rate = learning_rate

  def _create_average_ops(self):
    """Build moving average ops."""
    print('Creating moving average ops')

    with tf.variable_scope('moving_avg_flag'):
      self.moving_avg_started = tf.get_variable(
          'flag', [], tf.int32, initializer=tf.initializers.zeros(),
          trainable=False)
      self.start_moving_avg_op = tf.assign(self.moving_avg_started, 1)

    all_vars = tf.trainable_variables()
    average_pairs = []
    var_cnt = 0
    with tf.variable_scope('average'):
      for v in all_vars:
        avg_v = tf.get_variable(
            str(var_cnt), shape=v.shape, dtype=v.dtype,
            initializer=tf.zeros_initializer, trainable=False)
        var_cnt += 1
        average_pairs.append([v, avg_v])
    backup_pairs = []
    var_cnt = 0
    with tf.variable_scope('backup'):
      for v in all_vars:
        backup_v = tf.get_variable(str(var_cnt), shape=v.shape, dtype=v.dtype,
                                   trainable=False)
        var_cnt += 1
        backup_pairs.append([v, backup_v])

    with tf.variable_scope('avg_step'):
      avg_step = tf.get_variable('step', [], dtype=tf.float32, trainable=False)

    with tf.control_dependencies([tf.assign_add(avg_step, 1.0)]):
      average_op = []
      for v, avg_v in average_pairs:
        mu = 1 / avg_step
        new_avg = mu * v + (1 - mu) * avg_v
        with tf.control_dependencies([new_avg]):
          average_op.append(tf.assign(avg_v, new_avg))

    assert len(average_pairs) == len(all_vars)
    assert len(average_pairs) == len(backup_pairs)
    use_average_op = []
    for i in range(len(average_pairs)):
      v, avg_v = average_pairs[i]
      _, backup_v = backup_pairs[i]
      with tf.control_dependencies([tf.assign(backup_v, v)]):
        use_average_op.append(tf.assign(v, avg_v))
    use_average_op = tf.group(* use_average_op)

    reverse_average_op = []
    for v, backup_v in backup_pairs:
      reverse_average_op.append(tf.assign(v, backup_v))
    reverse_average_op = tf.group(* reverse_average_op)

    return average_op, use_average_op, reverse_average_op

  def _build_valid(self):
    print('Building valid graph')
    _, loss = self._forward(self.x_valid, self.y_valid,
                            self.eval_params, self.batch_init_states)
    self.valid_loss = loss

  def _build_test(self):
    print('Building test graph')
    _, loss = self._forward(self.x_test, self.y_test,
                            self.eval_params, self.test_init_states)
    self.test_loss = loss

  def eval_valid(self, sess, use_moving_avg=False):
    """Eval 1 round on valid set."""
    total_loss = 0
    if use_moving_avg:
      sess.run([self.use_moving_avg_vars, self.batch_init_states['reset']])
    for _ in range(self.num_valid_batches):
      total_loss += sess.run(self.valid_loss)
    valid_ppl = np.exp(total_loss / self.num_valid_batches)
    print('valid_ppl={0:<.2f}'.format(valid_ppl))
    if use_moving_avg:
      sess.run(self.restore_normal_vars)

    return valid_ppl

  def eval_test(self, sess, use_moving_avg=False):
    """Eval 1 round on test set."""
    total_loss = 0
    if use_moving_avg:
      sess.run([self.use_moving_avg_vars, self.test_init_states['reset']])
    for step in range(self.num_test_batches):
      total_loss += sess.run(self.test_loss)
      if (step + 1) % 1000 == 0:
        test_ppl = np.exp(total_loss / (step + 1))
        log_string = 'step={0}'.format(step + 1)
        log_string += ' test_ppl={0:<.2f}'.format(test_ppl)
        print(log_string)
    test_ppl = np.exp(total_loss / self.num_valid_batches)
    log_string = 'step={0}'.format(self.num_test_batches)
    log_string += ' test_ppl={0:<.2f}'.format(test_ppl)
    print(log_string)
    if use_moving_avg:
      sess.run(self.restore_normal_vars)

    return test_ppl
