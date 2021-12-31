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

"""Custom training loop for all sequence alignments experiments."""

import enum
import itertools
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from dedal import multi_task
from dedal.train import checkpoint
from dedal.train import logger

Example = Mapping[str, tf.Tensor]


def get_strategy(use_tpu = False,
                 tpu_job_name = 'tpu_worker',
                 master = 'local'):
  """Builds the proper strategy based on the parameters.

  Args:
    use_tpu: whether the job must be trained on tpu or not.
    tpu_job_name: the name of the tpu_job if applies.
    master: the master job name.

  Returns:
    A tf.distribute.Strategy.
  """
  use_remote_eager = master and master != 'local'
  if use_tpu:
    logging.info('Use TPU at %s with job name "%s".', master, tpu_job_name)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=master, job_name=tpu_job_name)
    if use_remote_eager:
      tf.config.experimental_connect_to_cluster(resolver)
      logging.warning('Remote eager configured. Remote eager can be slow.')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    if use_remote_eager:
      tf.config.experimental_connect_to_host(master, job_name='gpu_worker')
      logging.warning('Remote eager configured. Remote eager can be slow.')
    gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
    if gpus:
      logging.info('Found GPUs: %s', gpus)
      strategy = tf.distribute.MirroredStrategy()
    else:
      strategy = tf.distribute.OneDeviceStrategy('CPU')

  return strategy


@gin.constants_from_enum
class Task(enum.Enum):
  TRAIN = 0
  EVAL = 1
  DOWNSTREAM = 2


@gin.configurable
class TrainingLoop:
  """Hand made training loop."""

  def __init__(self,
               workdir,
               strategy,
               dataset_builder=None,
               logger_cls=logger.Logger,
               model_cls = None,
               loss_fn = None,
               optimizer_cls = None,
               batch_size = 128,
               num_steps = 10000,
               num_eval_steps = None,
               num_steps_per_train_iteration = 10,
               graph_mode = True,
               separate_eval = True,
               reference_workdir = None,
               num_reference_steps = None):
    self._workdir = workdir
    self.strategy = strategy

    self._dataset_builder = dataset_builder
    self._logger_cls = logger_cls
    self._loss_fn = loss_fn
    self._model_cls = model_cls
    self._optimizer_cls = optimizer_cls

    self._batch_size = batch_size
    self._num_steps = num_steps
    self._num_eval_steps = num_eval_steps
    self._num_steps_per_train_iteration = num_steps_per_train_iteration
    self._graph_mode = graph_mode
    self._separate_eval = separate_eval

    with self.strategy.scope():
      self.model = self._model_cls()
      self._step = tf.Variable(0, dtype=tf.int64, trainable=False, name='step')
      self._optimizer = optimizer_cls() if optimizer_cls is not None else None

    self._checkpointer = checkpoint.Checkpointer(
        self._workdir, self.strategy, self.model, self._step, self._optimizer)

    # For eval / downstream jobs, the reference checkpointing.
    if reference_workdir is not None:
      with self.strategy.scope():
        self._reference_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name='ref_step')
        self._num_reference_steps = num_reference_steps
      self._reference_ckpt = checkpoint.Checkpointer(
          reference_workdir, self.strategy, self.model, self._reference_step)
    else:
      self._reference_ckpt = None
      self._reference_step = None
      self._num_reference_steps = None

  def run(self, task = Task.TRAIN):
    """Run the training loop for the given task."""
    tasks = {
        Task.TRAIN: self.train,
        Task.EVAL: self.evaluate,
        Task.DOWNSTREAM: self.downstream
    }
    task = task if isinstance(task, Task) else Task[task.upper()]
    task_fn = tasks.get(task, None)
    if task_fn is None:
      raise ValueError(
          f'Unknown task {task}. Possible values are '
          f'{[t.name for t in tasks.keys()]}')
    task_fn()

  @property
  def _learning_rate(self):
    lr = self._optimizer.lr
    return lr if not callable(lr) else lr(self._step)

  @gin.configurable(module='TrainingLoop')
  def train_step(self,
                 inputs,
                 log,
                 training = True):
    """Runs one training step."""

    def step_fn(features, y_true, weights, metadata):
      """step_fn is replicated when running with TPUStrategy."""
      with tf.GradientTape() as tape:
        y_pred = self.model(features, training=training)
        local_loss, individual_losses = self._loss_fn(y_true, y_pred, weights)
        local_loss += sum(self.model.losses)

      replica_ctx = tf.distribute.get_replica_context()
      grads = tape.gradient(
          local_loss, self.model.trainable_variables,
          unconnected_gradients=tf.UnconnectedGradients.ZERO)
      grads = replica_ctx.all_reduce('sum', grads)
      self._optimizer.apply_gradients(
          grads_and_vars=zip(grads, self.model.trainable_variables),
          experimental_aggregate_gradients=False)

      loss = replica_ctx.all_reduce('sum', local_loss)
      individual_losses = {k: replica_ctx.all_reduce('sum', v)
                           for k, v in individual_losses.items()}
      grad_norm = tf.linalg.global_norm(grads)
      log.update_mean('loss', loss)
      for k, v in individual_losses.items():
        log.update_mean(k, v)
      log.update_mean('gradient_norm', grad_norm)
      # TODO(fllinares, oliviert): do not average LR??
      log.update_mean('learning_rate', self._learning_rate)
      for m in self.model.metrics:
        log.update_mean(m.name, m.result())
      log.update(y_true, y_pred, weights, metadata)
      self._step.assign_add(1)

    for _ in tf.range(self._num_steps_per_train_iteration):
      x, y_true, weights, metadata = next(inputs)
      self.strategy.run(step_fn, args=[x, y_true, weights, metadata])

  def parse_eval_splits(self, verbose = True):
    """Returns preconfigured list of `split` args for `dataset_builder.make`."""
    # Uses eval splits configured in the dataset builder. If none are present,
    # defaults to `tfds.Split.TEST`.
    split = self._dataset_builder.split
    split = tfds.Split.TEST if split is None else split
    # In single-input mode, i.e. when dataset_builder is a `DatasetBuilder`
    # instance, `split` can be
    #  + a `str` (one eval split),
    #  + a `Sequence[str]` (multiple eval splits within the same job).
    # In multi-input mode, i.e. when dataset_builder is a `MultiDatasetBuilder`,
    # instance, `split` can be
    #  + a `str` (one eval split, all subdatasets share the same split name),
    #  + a `Sequence[str]` (multiple eval splits within the same job, all
    #    subdatasets share the same split name),
    #  + a `Sequence[Sequence[str]]` (multiple eval splits within the same job,
    #    each subdataset configured with a different split name)
    splits = (split,) if isinstance(split, str) else tuple(split)
    if verbose:
      for i, split in enumerate(splits):
        logging.info(
            'Eval splits (%d / %d): %s.', i + 1, len(splits), ', '.join(split))
    return splits

  def make_ds(self, split=None):
    return self._dataset_builder.make(split, self._batch_size, self.strategy)

  def make_logger(self, split, task, dummy = False):
    if dummy:
      return logger.DummyLogger()
    split = split if isinstance(split, str) else ','.join(split)
    return self._logger_cls(self._workdir, self.strategy, split, task)

  def train(self, freeze = False, silent = False):
    """Trains the network."""
    train_step_fn = (tf.function(self.train_step) if self._graph_mode
                     else self.train_step)

    logging.info('Starting training.')
    train_split = tfds.Split.TRAIN
    train_examples = iter(self.make_ds(train_split))
    logging.info('train: train dataset ready.')
    eval_ds = None
    if not self._separate_eval:
      eval_splits = self.parse_eval_splits()
      eval_ds = [self.make_ds(split) for split in eval_splits]

    log = self.make_logger(train_split, 'train', dummy=silent)

    first_inputs, _, _, _ = next(train_examples)
    self.may_transfer(first_inputs, freeze=freeze)
    self._checkpointer.restore()

    while self._step.numpy() < self._num_steps:
      train_step_fn(train_examples, log=log)
      step = self._step.numpy()
      logged = log.log_and_reset(step, step >= self._num_steps)
      self._checkpointer.may_save(step >= self._num_steps)
      if logged and eval_ds is not None:
        # TODO(fllinares): should we use one logger per ds instead?
        for ds in eval_ds:
          self.evaluate_once(ds, log)

      # Just for debug.
      if step < 10:
        logging.info('Train step %i completed', step)

  @gin.configurable(module='TrainingLoop')
  def eval_step(self, inputs, y_true, weights, metadata, log, training=False):
    """Run a single eval step, in a distributed strategy."""

    def step_fn(x, y_true, weights, metadata):
      y_pred = self.model(x, training=training)
      local_loss, individual_losses = self._loss_fn(y_true, y_pred, weights)
      local_loss += sum(self.model.losses)

      replica_ctx = tf.distribute.get_replica_context()
      loss = replica_ctx.all_reduce('sum', local_loss)
      individual_losses = {k: replica_ctx.all_reduce('sum', v)
                           for k, v in individual_losses.items()}
      log.update_mean('loss', loss)
      for k, v in individual_losses.items():
        log.update_mean(k, v)
      log.update(y_true, y_pred, weights, metadata)

    self.strategy.run(step_fn, args=[inputs, y_true, weights, metadata])

  def evaluate_once(self, ds, log):
    """Evaluate by passing once through the dataset."""
    # TODO(oliviert, fllinares): try jit_compile=True.
    eval_step_fn = (tf.function(self.eval_step) if self._graph_mode
                    else self.eval_step)
    for x, y_true, weights, metadata in itertools.islice(
        ds, 0, self._num_eval_steps):
      eval_step_fn(x, y_true, weights, metadata, log)
    log.log_and_reset(self._step.numpy(), force=True)

  @gin.configurable(module='TrainingLoop')
  def evaluate(self, finetune_fn=None):
    """Evaluates the trained network by reading the train checkpoints.

    Args:
      finetune_fn: A (typically, gin-configured) callable that takes a
        TrainingLoop object as its first argument. Its main purpose is to allow
        arbitrary postprocessing of the model prior to eval. Note, however, that
        these changes will *not* be persistent *nor* saved as a checkpoint.
    """
    logging.info('Starting evaluation.')
    splits = self.parse_eval_splits()
    eval_ds = [self.make_ds(split) for split in splits]
    logging.info('evaluate: eval dataset(s) ready.')
    eval_logs = [self.make_logger(split, 'evaluate') for split in splits]
    ckpt = (self._checkpointer if self._reference_ckpt is None
            else self._reference_ckpt)
    step = self._step if self._reference_step is None else self._reference_step
    while step.numpy() < self._num_steps - 1:
      ckpt.restore_after(step.numpy())
      if finetune_fn is not None:
        logging.info('evaluate: executing finetune_fn...')
        finetune_fn(loop=self)
        logging.info('evaluate: finetune_fn completed.')
      self._step.assign(step)
      for ds, log in zip(eval_ds, eval_logs):
        # TODO(fllinares): double-check this doesn't mess with the clocks.
        log.restart_clock()
        self.evaluate_once(ds, log)

  def downstream(self):
    """Runs a downstream task (train and test) based on upstream checkpoints."""
    logging.info('Starting downstream.')
    if self._reference_ckpt is None:
      logging.info('No reference pass for the upstream task')

    # Dataset and logger for train and test.
    eval_splits = self.parse_eval_splits()
    ds_logs = [(self.make_ds(split), self.make_logger(split, 'downstream'))
               for split in (tfds.Split.TRAIN,) + eval_splits]

    while self._reference_step.numpy() < self._num_reference_steps - 1:
      # Re-initializes model, preventing weights in head being trained from
      # being re-used from the last train-eval cycle.
      with self.strategy.scope():
        self.model = self._model_cls()
      self._checkpointer.set_model(self.model)
      self._reference_ckpt.set_model(self.model)
      # TODO(oliviert): Consider instead having one logging folder per upstream
      # step to save the whole downstream learning curve.
      self.train(freeze=True, silent=True)
      # Logs at the proper upstream step and the end of training.
      self._step.assign(self._reference_step)
      for ds, log in ds_logs:
        self.evaluate_once(ds, log)
      # Enables re-training from scratch.
      self._checkpointer.delete()
      self._step.assign(0)

  @gin.configurable(module='TrainingLoop')
  def may_transfer(
      self,
      inputs,
      freeze = False,
      reset_head = False):
    """Tries to restore the weights from the reference model, if exists."""
    logging.info('Trying to transfer weights.')
    if self._reference_ckpt is None:
      return

    logging.info('Transferring weights from %s',
                 self._reference_ckpt._workdir)  # pylint: disable=protected-access

    if isinstance(reset_head, bool):
      reset_head = self.model.heads.constant_copy(reset_head)

    # Initializes the weights to be able to restore them.
    @tf.function
    def predict(inputs):
      self.model(inputs)
    self.strategy.run(predict, args=[inputs])

    # Backs up random init vals of params of output heads that are not to be
    # restored from the reference checkpoint.
    heads_init_vars = reset_head.constant_copy([])
    for head, head_init_vars, reset_head_flag in zip(
        self.model.heads, heads_init_vars, reset_head):
      if reset_head_flag:
        head_init_vars.extend([v.value() for v in head.variables])

    self._reference_ckpt.restore_after(self._reference_step.numpy())
    # TODO(oliviert): make this more configurable.
    if freeze:
      self.model.encoder.trainable = False
      if self.model.aligner is not None:
        self.model.aligner.trainable = False

    logging.info('Weights transferred complete.')

    # Optionally, reinitializes (a subset of) the output heads.
    for head, head_init_vars, reset_head_flag in zip(
        self.model.heads, heads_init_vars, reset_head):  # pylint: disable=protected-access
      if reset_head_flag:
        for var, init_var in zip(head.variables, head_init_vars):
          var.assign(init_var)
        logging.info('Output head %s was reset.', head.name)
