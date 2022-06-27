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

"""AWD ENAS fixed model on TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from enas_lm.src.tpu import utils
from tensorflow.contrib import tpu as contrib_tpu


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('fixed_arc', None, '')
flags.DEFINE_boolean('child_average_loose_ends', True, (
    'if True, average the unused steps in RNNCell, otherwise average all'))
flags.DEFINE_float('child_alpha', 0.0, 'activation L2 reg')
flags.DEFINE_float('child_drop_e', 0.15, 'drop rate words')
flags.DEFINE_float('child_drop_i', 0.20, 'drop rate embeddings')
flags.DEFINE_float('child_drop_l', 0.25, 'drop rate between layers')
flags.DEFINE_float('child_drop_o', 0.75, 'drop rate output')
flags.DEFINE_float('child_drop_w', 0.00, 'drop rate weight')
flags.DEFINE_float('child_drop_x', 0.75, 'drop rate at input of RNN cells')
flags.DEFINE_float('child_init_range', 0.04, '')
flags.DEFINE_float('child_grad_bound', 0.25, '')
flags.DEFINE_float('child_learning_rate', 20.0, '')
flags.DEFINE_float('child_weight_decay', 8e-7, '')
flags.DEFINE_float('child_start_moving_average_step', 2000., '')

flags.DEFINE_integer('child_eval_batch_size', 64, '')
flags.DEFINE_integer('child_num_train_steps', int(2e5), '')
flags.DEFINE_integer('child_num_eval_steps', 10, '')
flags.DEFINE_integer('child_hidden_size', 850, '')

flags.DEFINE_integer('child_bptt_steps', 35, '')


def set_default_params(params):
  """Set default values for the hparams."""
  params.add_hparam('alpha', FLAGS.child_alpha)  # activation L2 reg
  params.add_hparam('average_loose_ends', FLAGS.child_average_loose_ends)
  params.add_hparam('best_valid_ppl_threshold', 10)

  params.add_hparam('train_batch_size', 64)
  params.add_hparam('eval_batch_size', FLAGS.child_eval_batch_size)
  params.add_hparam('bptt_steps', FLAGS.child_bptt_steps)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_e', FLAGS.child_drop_e)  # word
  params.add_hparam('drop_i', FLAGS.child_drop_i)  # embeddings
  params.add_hparam('drop_l', FLAGS.child_drop_l)  # between RNN nodes
  params.add_hparam('drop_o', FLAGS.child_drop_o)  # output
  params.add_hparam('drop_w', FLAGS.child_drop_w)  # weight
  params.add_hparam('drop_x', FLAGS.child_drop_x)  # input to RNN layers

  assert FLAGS.fixed_arc is not None, 'Please specify `--fixed_arc`'
  params.add_hparam('fixed_arc', [int(d) for d in FLAGS.fixed_arc.split(' ')])

  params.add_hparam('grad_bound', FLAGS.child_grad_bound)
  params.add_hparam('hidden_size', FLAGS.child_hidden_size)
  params.add_hparam('init_range', FLAGS.child_init_range)
  params.add_hparam('learning_rate', FLAGS.child_learning_rate)
  params.add_hparam('num_train_steps', FLAGS.child_num_train_steps)
  params.add_hparam('num_eval_steps', FLAGS.child_num_eval_steps)
  params.add_hparam('start_moving_average',
                    FLAGS.child_start_moving_average_step)

  params.add_hparam('vocab_size', 10000)
  params.add_hparam('weight_decay', FLAGS.child_weight_decay)
  return params


def _gen_mask(shape, drop_prob):
  """Generate a droppout mask."""
  assert 0 <= drop_prob < 1.0
  keep_prob = 1. - drop_prob
  mask = tf.random_uniform(shape, minval=0., maxval=1., dtype=tf.float32)
  mask = tf.floor(mask + keep_prob) / keep_prob
  return mask


def _build_params(params):
  """Create model parameters."""

  tf.logging.info('-' * 80)
  tf.logging.info('Building model params')
  initializer = tf.initializers.random_uniform(minval=-params.init_range,
                                               maxval=params.init_range)
  with tf.variable_scope('language_model', initializer=initializer):
    with tf.variable_scope('embedding'):
      w_emb = tf.get_variable(
          'w', [params.vocab_size, params.hidden_size],
          initializer=initializer)
      dropped_w_emb = tf.layers.dropout(
          w_emb, params.drop_e, [params.vocab_size, 1],
          training=True)

    hidden_size = params.hidden_size
    fixed_arc = params.fixed_arc
    num_layers = len(fixed_arc) // 2
    with tf.variable_scope('rnn_cell'):
      w_prev = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])
      i_mask = tf.ones([hidden_size, 2 * hidden_size], dtype=tf.float32)
      h_mask = _gen_mask([hidden_size, 2 * hidden_size], params.drop_w)
      mask = tf.concat([i_mask, h_mask], axis=0)
      dropped_w_prev = w_prev * mask

      w_skip, dropped_w_skip = [], []
      for layer_id in range(num_layers):
        mask = _gen_mask([hidden_size, 2 * hidden_size], params.drop_w)
        with tf.variable_scope('layer_{}'.format(layer_id)):
          w = tf.get_variable('w', [hidden_size, 2 * hidden_size])
          dropped_w = w * mask
          w_skip.append(w)
          dropped_w_skip.append(dropped_w)

    def _create_init_states(batch_size):
      """Create initial states for RNNs and returns the reset ops."""
      init_shape = [batch_size, hidden_size]
      prev_s = tf.get_variable('s', init_shape, dtype=tf.float32,
                               initializer=tf.initializers.zeros(),
                               trainable=False)
      zeros = np.zeros(init_shape, dtype=np.float32)
      reset = tf.assign(prev_s, zeros)
      return prev_s, reset

    with tf.variable_scope('init_states'):
      with tf.variable_scope('batch'):
        batch_prev_s, batch_reset = _create_init_states(params.train_batch_size)
      with tf.variable_scope('test'):
        test_prev_s, test_reset = _create_init_states(1)

  num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('Model has {0} params'.format(num_params))

  batch_init_states = {
      's': batch_prev_s,
      'reset': batch_reset,
  }
  test_init_states = {
      's': test_prev_s,
      'reset': test_reset,
  }
  train_params = {
      'w_emb': dropped_w_emb,
      'w_prev': dropped_w_prev,
      'w_skip': dropped_w_skip,
      'w_soft': w_emb,
  }
  eval_params = {
      'w_emb': w_emb,
      'w_prev': w_prev,
      'w_skip': w_skip,
      'w_soft': w_emb,
  }
  if params.task_mode == 'train':
    return batch_init_states, train_params
  elif params.task_mode == 'valid':
    return batch_init_states, eval_params
  elif params.task_mode == 'test':
    return test_init_states, eval_params
  else:
    raise ValueError('Unknown task_mode {0}'.format(params.task_mode))


class _ENASCell(tf.nn.rnn_cell.RNNCell):
  """Implements ENAS cell interface for TPUs."""

  def __init__(self, params, prev_s, w_prev, w_skip, input_mask, layer_mask):
    self._params = params
    self.prev_s = prev_s
    self.w_prev = w_prev
    self.w_skip = w_skip

    self._input_mask = input_mask
    self._layer_mask = layer_mask

    self.state_size = params.hidden_size
    self.output_size = params.hidden_size

  def input_mask(self):
    return self._input_mask

  def layer_mask(self):
    return self._layer_mask

  def output_size(self):
    return self._output_size

  def state_size(self):
    return self._state_size

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: `2-D Tensor` with shape `[batch_size, self.state_size]`.
      scope: optional cell scope.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: A single `2-D` tensor.
    """
    batch_size, hidden_size = inputs.shape
    fixed_arc = self._params.fixed_arc
    num_layers = len(fixed_arc) // 2
    prev_s = self.prev_s
    w_prev = self.w_prev
    w_skip = self.w_skip
    input_mask = self._input_mask
    layer_mask = self._layer_mask

    if layer_mask is not None:
      assert input_mask is not None
      ht = tf.matmul(tf.concat([inputs * input_mask,
                                state * layer_mask], axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([inputs, state], axis=1), w_prev)
    h, t = tf.split(ht, 2, axis=1)
    h = tf.tanh(h)
    t = tf.sigmoid(t)
    s = state + t * (h - state)
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
    used = np.zeros(num_layers+1, dtype=np.float32)
    for layer_id in range(num_layers):
      prev_idx = fixed_arc[start_idx]
      func_idx = fixed_arc[start_idx + 1]
      prev_s = layers[prev_idx]
      used[prev_idx] = 1
      if layer_mask is not None:
        ht = tf.matmul(prev_s * layer_mask, w_skip[layer_id])
      else:
        ht = tf.matmul(prev_s, w_skip[layer_id])
      h, t = tf.split(ht, 2, axis=1)

      h = _select_function(h, func_idx)
      t = tf.sigmoid(t)
      s = prev_s + t * (h - prev_s)
      s.set_shape([batch_size, hidden_size])
      layers.append(s)
      start_idx += 2

    if self._params.average_loose_ends:
      layers = [l for l, u in zip(layers, used) if u == 0]
      next_s = tf.add_n(layers) / np.sum(1. - used)
    else:
      next_s = tf.add_n(layers[1:]) / tf.cast(num_layers, dtype=tf.float32)
    return next_s, next_s


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

  enas_cell = _ENASCell(params=params,
                        prev_s=prev_s,
                        w_prev=w_prev,
                        w_skip=w_skip,
                        input_mask=input_mask,
                        layer_mask=layer_mask)
  all_s, next_s = tf.nn.dynamic_rnn(
      cell=enas_cell,
      inputs=x,
      initial_state=prev_s,
      dtype=tf.float32)

  return next_s, all_s


def _create_average_ops(params):
  """Build moving average ops."""
  tf.logging.info('Creating moving average ops')

  with tf.variable_scope('moving_average'):
    moving_average_step = tf.get_variable(
        'step', [], dtype=tf.float32, trainable=False)

  all_vars = tf.trainable_variables()
  average_pairs = []
  with tf.variable_scope('average'):
    for v in all_vars:
      v_name = utils.strip_var_name(v.name)
      average_v = tf.get_variable(
          v_name, shape=v.shape, dtype=v.dtype,
          initializer=tf.initializers.zeros(), trainable=False)
      average_pairs.append([v, average_v])

  with tf.control_dependencies([tf.assign_add(moving_average_step, 1.0)]):
    update_average_op = []
    mu = tf.cond(
        tf.greater(moving_average_step, params.start_moving_average),
        lambda: tf.div(1., moving_average_step - params.start_moving_average),
        lambda: 0.)
    for v, average_v in average_pairs:
      new_average = mu * v + (1 - mu) * average_v
      with tf.control_dependencies([new_average]):
        update_average_op.append(tf.assign(average_v, new_average))

  assert len(average_pairs) == len(all_vars)
  use_average_op = []
  for i in range(len(average_pairs)):
    v, average_v = average_pairs[i]
    use_average_op.append(tf.assign(v, average_v))

  return update_average_op, mu, use_average_op


def _forward(params, x, y, model_params, init_states, is_training=False):
  """Computes the logits.

  Args:
    params: hyper-parameters.
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

  emb = tf.einsum('bnv,vh->bnh', tf.one_hot(x, params.vocab_size), w_emb)
  batch_size = x.get_shape()[0].value
  hidden_size = params.hidden_size
  if is_training:
    emb = tf.layers.dropout(
        emb, params.drop_i, [params.batch_size, 1, hidden_size], training=True)

    input_mask = _gen_mask([batch_size, hidden_size], params.drop_x)
    layer_mask = _gen_mask([batch_size, hidden_size], params.drop_l)
  else:
    input_mask = None
    layer_mask = None

  out_s, all_s = _rnn_fn(emb, prev_s, w_prev, w_skip, input_mask, layer_mask,
                         params)
  top_s = all_s
  if is_training:
    top_s = tf.layers.dropout(top_s, params.drop_o,
                              [batch_size, 1, hidden_size], training=True)

  carry_on = [tf.assign(prev_s, out_s)]
  logits = tf.einsum('bnh,vh->bnv', top_s, w_soft)
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=logits)

  if is_training:
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

    # total_loss = mean(cross_entroy_loss) + regularization_terms
    total_loss = cross_entropy_loss

    # L2 weight reg
    total_loss += params.weight_decay * tf.add_n(
        [tf.reduce_sum(w ** 2) for w in tf.trainable_variables()])

    # activation L2 reg
    total_loss += params.alpha * tf.reduce_mean(all_s ** 2)
  else:
    total_loss = tf.reduce_mean(cross_entropy_loss)

  with tf.control_dependencies(carry_on):
    cross_entropy_loss = tf.identity(cross_entropy_loss)
    if is_training:
      total_loss = tf.identity(total_loss)

  return total_loss, cross_entropy_loss


def model_fn(features, labels, mode, params):
  """`model_fn` for training mode for `TPUEstimator`."""
  labels = labels
  is_training = (mode == tf_estimator.ModeKeys.TRAIN)
  x = tf.transpose(features['x'], [1, 0])
  y = tf.transpose(features['y'], [1, 0])
  init_states, model_params = _build_params(params)

  (update_average_ops, moving_average_mu,
   use_moving_average_ops) = _create_average_ops(params)

  if params.moving_average:
    tf.logging.info('swap in moving average')
    with tf.control_dependencies(use_moving_average_ops):
      total_loss, cross_entropy_loss = _forward(
          params, x, y, model_params, init_states, is_training=is_training)
  else:
    if not is_training:
      tf.logging.info('not swap in moving average')
    total_loss, cross_entropy_loss = _forward(
        params, x, y, model_params, init_states, is_training=is_training)

  if is_training:
    tf_vars = tf.trainable_variables()
    global_step = tf.train.get_or_create_global_step()
    lr_scale = (tf.cast(tf.shape(y)[-1], dtype=tf.float32) /
                tf.cast(params.bptt_steps, dtype=tf.float32))
    learning_rate = utils.get_lr(global_step, params) * lr_scale
    grads = tf.gradients(total_loss, tf_vars)
    clipped_grads, grad_norm = tf.clip_by_global_norm(grads,
                                                      params.grad_bound)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if params.use_tpu:
      optimizer = contrib_tpu.CrossShardOptimizer(
          opt=optimizer, reduction=tf.losses.Reduction.MEAN)
    with tf.control_dependencies(update_average_ops):
      train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars),
                                           global_step=global_step)

    names_and_tensors = [
        ('learning_rate', learning_rate),
        ('per_example_cross_entropy', cross_entropy_loss),
        ('train_ppl', tf.exp(cross_entropy_loss)),
        ('grad_norm', grad_norm),
        ('moving_average_mu', moving_average_mu),
    ]
    host_call = utils.build_host_call_fn(params, names_and_tensors)
    return contrib_tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, host_call=host_call)
  else:
    def _metric_fn(cross_entropy_loss):
      """Computes metrics for EstimatorSpec."""
      metrics = {
          'log_ppl/{0}'.format(params.task_mode): tf.metrics.mean(
              values=cross_entropy_loss),
      }
      return metrics

    return contrib_tpu.TPUEstimatorSpec(
        mode=tf_estimator.ModeKeys.EVAL,
        loss=total_loss,
        eval_metrics=(_metric_fn, [cross_entropy_loss]))
