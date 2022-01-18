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


flags.DEFINE_string('fixed_arc', None, '')
flags.DEFINE_float('child_alpha', 0.7, 'activation L2 reg')
flags.DEFINE_float('child_drop_e', 0.125, 'drop rate words')
flags.DEFINE_float('child_drop_i', 0.175, 'drop rate embeddings')
flags.DEFINE_float('child_drop_l', 0.225, 'drop rate between layers')
flags.DEFINE_float('child_drop_o', 0.75, 'drop rate output')
flags.DEFINE_float('child_drop_w', 0.00, 'drop rate weight')
flags.DEFINE_float('child_drop_x', 0.725, 'drop rate at input of RNN cells')
flags.DEFINE_float('child_init_range', 0.05, '')
flags.DEFINE_float('child_grad_bound', 0.25, '')
flags.DEFINE_float('child_weight_decay', 8e-7, '')
flags.DEFINE_integer('child_num_train_epochs', 3000, '')
flags.DEFINE_integer('child_hidden_size', 800, '')


def _gen_mask(shape, drop_prob):
  """Generate a droppout mask."""
  keep_prob = 1. - drop_prob
  mask = tf.random_uniform(shape, minval=0., maxval=1., dtype=tf.float32)
  mask = tf.floor(mask + keep_prob) / keep_prob
  return mask


def _rnn_fn(x, prev_s, w_prev, w_skip, input_mask, layer_mask, params):
  """Multi-layer LSTM.

  Args:
    x: [batch_size, num_steps, hidden_size].
    prev_s: [batch_size, hidden_size].
    w_prev: [2 * hidden_size, 2 * hidden_size].
    w_skip: [None, [hidden_size, 2 * hidden_size] * (num_layers-1)].
    input_mask: [batch_size, hidden_size].
    layer_mask: [batch_size, hidden_size].
    params: hyper-params object.

  Returns:
    next_s: [batch_size, hidden_size].
    all_s: [[batch_size, num_steps, hidden_size] * num_layers].
  """
  batch_size = x.get_shape()[0].value
  num_steps = tf.shape(x)[1]
  fixed_arc = params.fixed_arc
  num_layers = len(fixed_arc) // 2

  all_s = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, prev_s, all_s):
    """Body fn for `tf.while_loop`."""
    inp = x[:, step, :]
    if layer_mask is not None:
      assert input_mask is not None
      ht = tf.matmul(
          tf.concat([inp * input_mask, prev_s * layer_mask], axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([inp, prev_s], axis=1), w_prev)
    h, t = tf.split(ht, 2, axis=1)
    h = tf.tanh(h)
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    def _select_function(h, function_id):
      if function_id == 0:
        return tf.tanh(h)
      elif function_id == 1:
        return tf.nn.relu(h)
      elif function_id == 2:
        return tf.sigmoid(h)
      elif function_id == 3:
        return h
      raise ValueError('Unknown func_idx {0}'.format(function_id))

    start_idx = 0
    for layer_id in range(num_layers):
      prev_idx = fixed_arc[start_idx]
      func_idx = fixed_arc[start_idx + 1]
      prev_s = layers[prev_idx]
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

  return next_s, all_s


def _set_default_params(params):
  """Set default values for the hparams."""
  params.add_hparam('alpha', FLAGS.child_alpha)  # activation L2 reg
  params.add_hparam('best_valid_ppl_threshold', 10)

  params.add_hparam('batch_size', 64)
  params.add_hparam('bptt_steps', 35)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_e', FLAGS.child_drop_e)  # word
  params.add_hparam('drop_i', FLAGS.child_drop_i)  # embeddings
  params.add_hparam('drop_l', FLAGS.child_drop_l)  # between RNN nodes
  params.add_hparam('drop_o', FLAGS.child_drop_o)  # output
  params.add_hparam('drop_w', FLAGS.child_drop_w)  # weight
  params.add_hparam('drop_x', FLAGS.child_drop_x)  # input to RNN layers

  assert FLAGS.fixed_arc is not None
  params.add_hparam('fixed_arc', [int(d) for d in FLAGS.fixed_arc.split(' ')])

  params.add_hparam('grad_bound', FLAGS.child_grad_bound)
  params.add_hparam('hidden_size', FLAGS.child_hidden_size)
  params.add_hparam('init_range', FLAGS.child_init_range)
  params.add_hparam('learning_rate', 20.)
  params.add_hparam('num_train_epochs', FLAGS.child_num_train_epochs)
  params.add_hparam('num_warmup_epochs', 0.0)
  params.add_hparam('vocab_size', 10000)

  params.add_hparam('weight_decay', FLAGS.child_weight_decay)
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

    params.add_hparam('num_warmup_steps',
                      params.num_warmup_epochs * self.num_train_batches)
    self._build_params()
    self._build_train()
    self._build_valid()
    self._build_test()

  def _build_params(self):
    """Create model parameters."""

    print('-' * 80)
    print('Building model params')
    initializer = tf.initializers.random_uniform(minval=-self.params.init_range,
                                                 maxval=self.params.init_range)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope('embedding'):
        w_emb = tf.get_variable(
            'w', [self.params.vocab_size, self.params.hidden_size],
            initializer=initializer)
        dropped_w_emb = tf.layers.dropout(
            w_emb, self.params.drop_e, [self.params.vocab_size, 1],
            training=True)

      hidden_size = self.params.hidden_size
      fixed_arc = self.params.fixed_arc
      num_layers = len(fixed_arc) // 2
      with tf.variable_scope('rnn_cell'):
        w_prev = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])
        i_mask = tf.ones([hidden_size, 2 * hidden_size], dtype=tf.float32)
        h_mask = _gen_mask([hidden_size, 2 * hidden_size], self.params.drop_w)
        mask = tf.concat([i_mask, h_mask], axis=0)
        dropped_w_prev = w_prev * mask

        w_skip, dropped_w_skip = [], []
        for layer_id in range(num_layers):
          mask = _gen_mask([hidden_size, 2 * hidden_size], self.params.drop_w)
          with tf.variable_scope('layer_{}'.format(layer_id)):
            w = tf.get_variable('w', [hidden_size, 2 * hidden_size])
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
        with tf.variable_scope('test'):
          init_shape = [1, hidden_size]
          test_prev_s = tf.get_variable(
              's', init_shape, dtype=tf.float32, trainable=False)
          zeros = np.zeros(init_shape, dtype=np.float32)
          test_reset = tf.assign(test_prev_s, zeros)

    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print('Model has {0} params'.format(num_params))

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
    self.test_init_states = {
        's': test_prev_s,
        'reset': test_reset,
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
    if is_training:
      emb = tf.layers.dropout(
          emb, self.params.drop_i,
          [self.params.batch_size, 1, hidden_size], training=True)

      input_mask = _gen_mask([batch_size, hidden_size], self.params.drop_x)
      layer_mask = _gen_mask([batch_size, hidden_size], self.params.drop_l)
    else:
      input_mask = None
      layer_mask = None

    out_s, all_s = _rnn_fn(emb, prev_s, w_prev, w_skip, input_mask, layer_mask,
                           self.params)
    top_s = all_s
    if is_training:
      top_s = tf.layers.dropout(top_s, self.params.drop_o,
                                [batch_size, 1, hidden_size], training=True)

    carry_on = [tf.assign(prev_s, out_s)]
    logits = tf.einsum('bnh,vh->bnv', top_s, w_soft)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
    loss = tf.reduce_mean(loss)

    reg_loss = loss  # loss + regularization_terms, for training only
    if is_training:
      # L2 weight reg
      reg_loss += self.params.weight_decay * tf.add_n(
          [tf.reduce_sum(w ** 2) for w in tf.trainable_variables()])

      # activation L2 reg
      reg_loss += self.params.alpha * tf.reduce_mean(all_s ** 2)

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
        log_string = 'step={0:<6d}'.format(step + 1)
        log_string += ' test_ppl={0:<.2f}'.format(test_ppl)
        print(log_string)
    test_ppl = np.exp(total_loss / self.num_test_batches)
    log_string = 'step={0:<6d}'.format(self.num_test_batches)
    log_string += ' test_ppl={0:<.2f}'.format(test_ppl)
    print(log_string)
    if use_moving_avg:
      sess.run(self.restore_normal_vars)

    return test_ppl

