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

# pylint: disable=logging-format-interpolation
# pylint: disable=g-direct-tensorflow-import

r"""Common utils."""

import os
import re
import threading

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


def get_worker_name(worker_id):
  """Returns `/job:tpu_worker/task:{worker_id}`."""
  return f'/job:tpu_worker/task:{worker_id}'


def get_device_name(worker_id, core_id):
  """Returns `/job:tpu_worker/task:{worker_id}/device:tpu:{core_id}`."""
  return f'/job:tpu_worker/task:{worker_id}/device:TPU:{core_id}'


def count_params():
  """Count model params."""
  num_params = sum([np.prod([d.value for d in w.shape])
                    for w in tf.trainable_variables()
                    if 'teacher' not in w.name.lower()])
  return num_params


def strip_var_name(var_name):
  """Strips variable name of sub-strings blocking variable name matching.

  Removes sub-strings that should be ignored when matching checkpointed variable
  names to variable names in the training graph, namely:
  - trailing colon + number, e.g. "W:0" --> "W"
  - partitioning info., e.g. "/a/part_12/b" --> "a/b".
  (Note that checkpointed variables do not have partitioning info in their name,
  while model variables do).

  Args:
    var_name: str, variable name.

  Returns:
    stripped variable name.
  """
  # Strip trailing number, e.g. convert "lstm/W_0:0" to "lstm/W_0".
  var_name = re.sub(r':\d+$', '', var_name)
  # Strip partitioning info, e.g. convert "W_0/part_3/Adagrad" to "W_0/Adagrad".
  var_name = re.sub(r'/part_\d+', '', var_name)
  return var_name


def get_saver(max_to_keep=1, restore_ema=False):
  """Constructs a `Saver`."""
  var_list = {}
  if restore_ema:
    logging.info('Restore EMA values')
    for v in tf.global_variables():
      if v.name.startswith('ema'):
        logging.fatal(f'wrong ema var name `{v.name}`')
      if 'global_step' in v.name:
        var_list['global_step'] = v
      else:
        var_list['ema/' + strip_var_name(v.name)] = v
  else:
    for v in tf.global_variables():
      var_list[strip_var_name(v.name)] = v
  saver = tf.train.Saver(var_list,
                         max_to_keep=max_to_keep,
                         save_relative_paths=True)
  return saver


class AsyncCheckpoint(object):
  """Saves checkpoint using a separated thread."""

  def __init__(self, saver, ckpt_dir, max_to_keep=None):
    self._saver = saver
    self._ckpt_dir = ckpt_dir
    self._max_to_keep = max_to_keep
    self._thread = None
    self.latest_checkpoint = None

  def join(self):
    if self._thread is not None:
      self._thread.join()

  def save(self, sess, step):
    """Docs."""

    def _save_fn():
      """Run the saver process."""
      raw_sess = sess if isinstance(sess, tf.Session) else sess.raw_session()
      ckpt_path = self._saver.save(
          raw_sess,
          save_path=os.path.join(self._ckpt_dir, 'ckpt'),
          global_step=step,
          write_meta_graph=False,
          write_state=False)
      self.latest_checkpoint = ckpt_path[len(self._ckpt_dir) + 1:]
      logging.info(f'Saved checkpoint `{ckpt_path}`')

      all_checkpoints = get_all_checkpoints(self._ckpt_dir)
      assert all_checkpoints is not None
      new_ckpt_content = [f'model_checkpoint_path: "{all_checkpoints[-1]}"']
      if (self._max_to_keep is not None and
          self._max_to_keep < len(all_checkpoints)):
        pattern = all_checkpoints[0] + '*'
        tf.io.gfile.BulkDelete(tf.io.gfile.Glob(pattern))
        # pylint: disable=invalid-unary-operand-type
        all_checkpoints = all_checkpoints[-self._max_to_keep:]
        # pylint: enable=invalid-unary-operand-type
      for ckpt_name in all_checkpoints:
        new_ckpt_content.append(f'all_model_checkpoint_paths: "{ckpt_name}"')
      checkpoint_file = os.path.join(self._ckpt_dir, 'checkpoint')
      with tf.io.gfile.GFile(checkpoint_file, 'w') as fout:
        fout.write('\n'.join(new_ckpt_content))

    if self._thread is not None:
      self._thread.join(timeout=0.1)
      if self._thread.is_alive():
        logging.info('Saver thread still in progress, skipping checkpoint.')
        return

    self._thread = threading.Thread(target=_save_fn)
    self._thread.start()


def should_log(params):
  """Returns a Boolean `tf.Tensor` dictating whether we should log values."""
  global_step = tf.train.get_or_create_global_step()
  first_run = tf.equal(global_step, 1)
  log_every = tf.equal(tf.floormod(global_step, params.log_every), 0)
  return tf.logical_or(first_run, log_every)


def get_all_checkpoints(ckpt_dir):
  """Returns a list of all checkpoints, eg `['ckpt-100', 'ckpt-500']`."""
  if not tf.io.gfile.IsDirectory(ckpt_dir):
    return []
  pattern = ckpt_dir + '/ckpt-*'
  s = len(ckpt_dir) + len('/ckpt-')
  checkpoints = [int(f.split('.')[0][s:]) for f in tf.io.gfile.Glob(pattern)]
  checkpoints = [os.path.join(ckpt_dir, 'ckpt-{0}'.format(v))
                 for v in sorted(set(checkpoints))]
  return checkpoints


def get_latest_checkpoint(ckpt_dir):
  """Returns a list of all checkpoints, eg `['ckpt-100', 'ckpt-500']`."""
  all_checkpoints = get_all_checkpoints(ckpt_dir)
  all_checkpoints = [ckpt for ckpt in all_checkpoints if 'temp' not in ckpt]
  if all_checkpoints:
    return all_checkpoints[-1]
  else:
    return None


def get_outfeed_ops(params, signature):
  """Create TPU outfeed ops."""
  outfeed_dtypes, outfeed_shapes = [], []
  for dtype, shape in signature.values():
    outfeed_dtypes.append(dtype)
    outfeed_shapes.append(shape)
  outfeed_ops = []
  outfeed_graph = tf.Graph()

  dev_assign = params.device_assignment
  host_to_tpus = {}
  for replica_id in range(params.num_replicas):
    host_device = dev_assign.host_device(replica=replica_id, logical_core=0)
    tpu_ordinal = dev_assign.tpu_ordinal(replica=replica_id, logical_core=0)
    if host_device not in host_to_tpus:
      host_to_tpus[host_device] = [tpu_ordinal]
    else:
      assert tpu_ordinal not in host_to_tpus[host_device]
      host_to_tpus[host_device].append(tpu_ordinal)

  with outfeed_graph.as_default():
    for host, tpus in host_to_tpus.items():
      with tf.device(host):
        for device_ordinal in tpus:
          device_outfeed = tf.raw_ops.OutfeedDequeueTuple(
              dtypes=outfeed_dtypes,
              shapes=outfeed_shapes,
              device_ordinal=device_ordinal)
          outfeed_ops.append(device_outfeed)

  return outfeed_ops, outfeed_graph


class InfeedThread(object):
  """InfeedTread wrapper."""

  def __init__(self, params, infeed_ops, infeed_graphs, name='infeed_thread'):
    if infeed_graphs is not None:
      assert isinstance(infeed_graphs, list)
      assert len(infeed_graphs) == len(infeed_ops)

    self.infeed_ops = infeed_ops
    self.infeed_graphs = infeed_graphs

    self.sessions = []
    for g in infeed_graphs:
      with g.as_default():
        sess = tf.Session(target=params.master, graph=g)
        self.sessions.append(sess)

    self.name = name
    self._threads = []

  def stop(self):
    self.join()
    for sess in self.sessions:
      sess.close()

  def join(self):
    for thread in self._threads:
      if thread is not None:
        thread.join(timeout=0.1)
        del thread

  def start(self, verbose=False):
    """Docs."""
    if verbose:
      logging.info(f'Start thread for `{self.name}`')

    def _infeed_fn(sess, infeed_op, infeed_graph):
      """Run the infeed process."""
      with infeed_graph.as_default():
        sess.run(infeed_op)

    for sess, op, g in zip(self.sessions, self.infeed_ops, self.infeed_graphs):
      thread = threading.Thread(target=_infeed_fn, args=(sess, op, g))
      thread.daemon = True
      thread.start()
      self._threads.append(thread)


class OutfeedThread(object):
  """OutfeedThread wrapper."""

  def __init__(self, params, outfeed_ops, outfeed_graph, outfeed_signature,
               name='outfeed_thread'):
    self.params = params
    self.outfeed_ops = outfeed_ops
    self.outfeed_graph = outfeed_graph
    self.outfeed_signature = outfeed_signature

    with outfeed_graph.as_default():
      self.session = tf.Session(target=params.master, graph=outfeed_graph)

    self.name = name
    self._thread = None

  def join(self):
    if self._thread is not None:
      self._thread.join(timeout=0.1)
      self._thread = None
    self.session.close()

  def start(self, verbose=False):
    """Docs."""
    if verbose:
      logging.info(f'Start thread for `{self.name}`')
    if self._thread is not None:
      return

    params = self.params
    outfeed_signature = self.outfeed_signature

    def _outfeed_fn():
      """Read from `outfeed_dequeue` and write `Summary`."""
      train_logdir = os.path.join(params.output_dir, 'logs', 'train')
      summary_writer = tf.summary.FileWriter(train_logdir)
      summary_tags = list(outfeed_signature.keys())
      while True:
        outfeeds = self.session.run(self.outfeed_ops)
        outfeeds = np.array(outfeeds).reshape([params.num_replicas, -1])
        outfeeds = np.sum(outfeeds, axis=0).tolist()
        summary_values = []
        for tag, value in zip(summary_tags, outfeeds):
          if tag == 'global_step':
            value /= params.num_replicas
            step = value
          else:
            summary_values.append(tf.Summary.Value(tag=tag, simple_value=value))
        summary_writer.add_summary(tf.Summary(value=summary_values), step)
        summary_writer.flush()
        if step >= params.num_train_steps:
          summary_writer.close()
          break

    self._thread = threading.Thread(target=_outfeed_fn)
    self._thread.daemon = True
    self._thread.start()


def setup_ema(params, name_scope=None):
  """Create exponential moving average for all variables under `name_scope`."""
  logging.info(f'ema_decay with rate {params.ema_decay}')
  all_vars = tf.global_variables()
  ema_ops = []
  step = tf.cast(tf.train.get_or_create_global_step() - params.ema_start,
                 tf.float32)
  decay = 1. - tf.minimum(params.ema_decay, (step+1.) / (step+10.))
  decay = tf.cond(tf.train.get_or_create_global_step() < params.ema_start,
                  lambda: tf.constant(1, tf.float32), lambda: decay)

  def should_skip(v):
    key_words = ['momentum', 'rms', 'global_step', 'debug', 'adam', 'lars']
    conditions = [k in v.name.lower() for k in key_words]
    if name_scope is not None:
      conditions += [not v.name.lower().startswith(name_scope)]
    return any(conditions)

  def get_init(v_name):
    key_words = ['variance', 'beta']
    if any([k in v_name for k in key_words]):
      return tf.initializers.ones()
    return tf.initializers.zeros()

  with tf.variable_scope('ema'):
    for v in all_vars:
      if not should_skip(v):
        v_name = strip_var_name(v.name)
        with tf.device(v.device):
          ema_var = tf.get_variable(
              name=v_name,
              shape=v.shape.as_list(),
              initializer=get_init(v_name),
              trainable=False)
          ema_op = tf.assign_sub(ema_var, decay * (ema_var-v), use_locking=True)
        ema_ops.append(ema_op)
  ema_op = tf.group(*ema_ops)
  return ema_op


def get_session(params, isolate_session_state=True):
  """Builds and returns a `tf.Session`."""
  config = tf.ConfigProto(
      isolate_session_state=isolate_session_state,
      allow_soft_placement=True,
      graph_options=tf.GraphOptions(
          optimizer_options=tf.OptimizerOptions(
              opt_level=tf.OptimizerOptions.L0,
              do_common_subexpression_elimination=False,
              do_function_inlining=False,
              do_constant_folding=False)))
  return tf.Session(target=params.master, config=config)


def get_learning_rate(params, initial_lr=None, num_warmup_steps=None,
                      num_wait_steps=None):
  """Build learning rate."""
  global_step = tf.train.get_or_create_global_step()

  if initial_lr is None:
    initial_lr = params.lr
  initial_lr = initial_lr * params.train_batch_size / 256.

  if num_warmup_steps is None:
    num_warmup_steps = params.num_warmup_steps

  if num_wait_steps is not None:
    global_step = global_step - num_wait_steps

  if params.lr_decay_type == 'constant':
    lr = tf.constant(initial_lr, dtype=tf.float32)
  elif params.lr_decay_type == 'exponential':
    lr = tf.train.exponential_decay(
        learning_rate=initial_lr,
        global_step=global_step-num_warmup_steps,
        decay_steps=params.num_decay_steps,
        decay_rate=params.lr_decay_rate,
        staircase=True)
  elif params.lr_decay_type == 'cosine':
    if num_wait_steps is None:
      lr = tf.train.cosine_decay(
          learning_rate=initial_lr,
          global_step=global_step-num_warmup_steps,
          decay_steps=params.num_train_steps-num_warmup_steps,
          alpha=0.0)
    else:
      lr = tf.train.cosine_decay(
          learning_rate=initial_lr,
          global_step=global_step-num_warmup_steps,
          decay_steps=params.num_train_steps-num_warmup_steps-num_wait_steps,
          alpha=0.0)
  else:
    raise ValueError(f'Unknown lr_decay_type `{params.lr_decay_type}`')

  r = (tf.cast(global_step+1, tf.float32) /
       tf.cast(num_warmup_steps, tf.float32))
  warmup_lr = initial_lr * r
  lr = tf.cond(global_step < num_warmup_steps, lambda: warmup_lr, lambda: lr)

  if num_wait_steps is not None:
    lr = tf.cond(global_step < 0,
                 lambda: tf.constant(0., tf.float32), lambda: lr)

  return lr


def get_optimizer(params, learning_rate=None):
  """Build optimizer."""
  if learning_rate is None:
    learning_rate = get_learning_rate(params)

  if params.optim_type.lower() == 'sgd':
    logging.info('Use SGD')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                                  use_locking=True)
  elif params.optim_type.lower() == 'momentum':
    logging.info('Use Momentum')
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9,
                                           use_nesterov=True,
                                           use_locking=True)
  elif params.optim_type.lower() == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=params.rmsprop_rho,
                                          momentum=params.rmsprop_momentum,
                                          epsilon=params.rmsprop_epsilon,
                                          use_locking=True)
  elif params.optim_type.lower() == 'lars':
    class LARSOptimizer(tf.train.Optimizer):
      """Layer-wise Adaptive Rate Scaling for large batch training.

      Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
      I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

      Implements the LARS learning rate scheme presented in the paper above.
      This optimizer is useful when scaling the batch size to up to 32K without
      significant performance degradation. It is recommended to use the
      optimizer in conjunction with:
          - Gradual learning rate warm-up
          - Linear learning rate scaling
          - Poly rule learning rate decay

      Note, LARS scaling is currently only enabled for dense tensors. Sparse
      tensors use the default momentum optimizer.
      """

      def __init__(
          self,
          learning_rate,
          momentum=0.9,
          weight_decay=0.0001,
          # The LARS coefficient is a hyperparameter
          eeta=0.001,
          epsilon=0.0,
          name='LARSOptimizer',
          # Enable skipping variables from LARS scaling.
          # TODO(sameerkm): Enable a direct mechanism to pass a
          # subset of variables to the optimizer.
          skip_list=None,
          use_nesterov=False):
        """Construct a new LARS Optimizer.

        Args:
          learning_rate: A `Tensor` or floating point value.
          momentum: A floating point value. Momentum hyperparameter.
          weight_decay: A floating point value. Weight decay hyperparameter.
          eeta: LARS coefficient as used in the paper. Dfault set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the
            highest scaling factor in LARS.
          epsilon: Optional epsilon parameter to be set in models that have very
            small gradients. Default set to 0.0.
          name: Optional name prefix for variables and ops created.
          skip_list: List of strings to enable skipping variables from scaling.
            If any of the strings in skip_list is a subset of var.name, variable
            'var' is skipped from LARS scaling. For a typical classification
            model with batch normalization, the skip_list is
            ['batch_normalization', 'bias']
          use_nesterov: when set to True, nesterov momentum will be enabled

        Raises:
          ValueError: If a hyperparameter is set to a non-sensical value.
        """
        if momentum < 0.0:
          raise ValueError(f'momentum should be positive: {momentum}')
        if weight_decay < 0.0:
          raise ValueError(f'weight_decay should be positive: {weight_decay}')
        super(LARSOptimizer, self).__init__(use_locking=False, name=name)

        self._learning_rate = learning_rate
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._eeta = eeta
        self._epsilon = epsilon
        self._name = name
        self._skip_list = skip_list
        self._use_nesterov = use_nesterov

      def _create_slots(self, var_list):
        for v in var_list:
          self._zeros_slot(v, 'momentum', self._name)

      def compute_lr(self, grad, var):
        scaled_lr = self._learning_rate
        if self._skip_list is None or not any(v in var.name
                                              for v in self._skip_list):
          w_norm = tf.norm(var, ord=2)
          g_norm = tf.norm(grad, ord=2)
          trust_ratio = tf.where(
              tf.math.greater(w_norm, 0),
              tf.where(
                  tf.math.greater(g_norm, 0),
                  (self._eeta * w_norm / (
                      g_norm + self._weight_decay * w_norm + self._epsilon)),
                  1.0),
              1.0)
          scaled_lr = self._learning_rate * trust_ratio
          # Add the weight regularization gradient
          grad = grad + self._weight_decay * var
        return scaled_lr, grad

      def _apply_dense(self, grad, var):
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, 'momentum')
        return tf.raw_ops.ApplyMomentum(
            var,
            mom,
            tf.cast(1.0, var.dtype.base_dtype),
            grad * scaled_lr,
            self._momentum,
            use_locking=False,
            use_nesterov=self._use_nesterov)

      def _resource_apply_dense(self, grad, var):
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, 'momentum')
        return tf.raw_ops.ResourceApplyMomentum(
            var=var.handle,
            accum=mom.handle,
            lr=tf.cast(1.0, var.dtype.base_dtype),
            grad=grad * scaled_lr,
            momentum=self._momentum,
            use_locking=False,
            use_nesterov=self._use_nesterov)

      # Fallback to momentum optimizer for sparse tensors
      def _apply_sparse(self, grad, var):
        mom = self.get_slot(var, 'momentum')
        return tf.raw_ops.SparseApplyMomentum(
            var,
            mom,
            tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            tf.cast(self._momentum_tensor, var.dtype.base_dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov).op

      def _resource_apply_sparse(self, grad, var, indices):
        mom = self.get_slot(var, 'momentum')
        return tf.raw_ops.ResourceSparseApplyMomentum(
            var.handle,
            mom.handle,
            tf.cast(self._learning_rate_tensor, grad.dtype),
            grad,
            indices,
            tf.cast(self._momentum_tensor, grad.dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov)

      def _prepare(self):
        learning_rate = self._learning_rate
        if callable(learning_rate):
          learning_rate = learning_rate()
        self._learning_rate_tensor = tf.convert_to_tensor(
            learning_rate, name='learning_rate')
        momentum = self._momentum
        if callable(momentum):
          momentum = momentum()
        self._momentum_tensor = tf.convert_to_tensor(momentum, name='momentum')

    optimizer = LARSOptimizer(
        learning_rate=learning_rate,
        weight_decay=params.weight_decay,
        skip_list=['batch_norm', 'batchnorm', 'gamma', 'beta', 'bias'],
        use_nesterov=True)
  else:
    raise ValueError(f'Unknown optim_type `{params.optim_type}`')
  return learning_rate, optimizer


def get_l2_loss(excluded_keywords=None):
  """Traverse `tf.trainable_variables` compute L2 reg. Ignore `batch_norm`."""
  def _is_excluded(v):
    """Guess whether a variable belongs to `batch_norm`."""
    keywords = ['batchnorm', 'batch_norm', 'bn',
                'layernorm', 'layer_norm']
    if excluded_keywords is not None:
      keywords += excluded_keywords
    return any([k in v.name.lower() for k in keywords])

  l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
               if not _is_excluded(v)]
  return tf.add_n(l2_losses)
