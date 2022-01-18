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

"""Basic Trainer classe."""

import functools
import time

from absl import logging
from flax import jax_utils
from flax.deprecated import nn
from flax.training import common_utils
import jax
from jax.experimental.optimizers import clip_grads
import jax.numpy as jnp
import numpy as np

from gift.pipelines import pipeline_utils
from gift.tasks import metrics
from gift.train_lib import lr_schedulers
from gift.train_lib import optimizers


class Trainer(object):
  """Basic Training pipeline."""

  def __init__(self, model_cls, task, hparams, experiment_dir,
               tb_summary_writer, rng):
    self.includes_self_supervision = False
    self.task = task
    self.hparams = hparams
    self.experiment_dir = experiment_dir
    self.tb_summary_writer = tb_summary_writer

    # calculate the total number of training steps
    (self.total_steps,
     self.steps_per_epoch) = pipeline_utils.get_num_training_steps(
         self.hparams, self.task.dataset.meta_data)
    self.setup_steps_per_eval()
    self.eval_frequency = hparams.get('eval_frequency') or self.steps_per_epoch
    if not self.eval_frequency:
      raise ValueError("'eval_frequency' should be specified in the config.")
    self.checkpoint_frequency = hparams.get(
        'checkpoint_frequency') or self.eval_frequency

    self.set_train_state(model_cls, rng)

    self.learning_rate_fn = self.get_learning_rate_fn()

    self.setup_pmapped_tain_and_eval_steps()

  def setup_steps_per_eval(self):
    # TODO(samiraabnar): Fix data pipeline: set drop remainder to False for eval
    self.total_eval_steps = self.get_total_eval_steps()
    self.steps_per_eval = self.hparams.get(
        'steps_per_eval') or self.total_eval_steps

  def get_total_eval_steps(self):
    return (int(
        np.ceil(self.task.dataset.meta_data['num_eval_examples'] /
                self.hparams.eval_batch_size)))

  def metrics_fn(self, logits, batch):
    return self.task.metrics_fn(logits=logits, batch=batch)

  def training_loss_fn(self, flax_module, train_state, batch, dropout_rng):
    """Runs forward pass and computes loss.

    Args:
      flax_module: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    raise NotImplementedError

  def setup_pmapped_tain_and_eval_steps(self):
    """Define pmapped methods."""
    self.pmapped_train_step = jax.pmap(
        self.train_step,
        axis_name='batch',
        in_axes=(0, 0),
        # Donate the batch argument.
        donate_argnums=(1,),
    )
    self.pmapped_eval_step = jax.pmap(
        self.eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        # Donate the batch argument.
        donate_argnums=(1,),
    )

    self.pmapped_forward_pass = jax.pmap(
        self.forward_pass,
        axis_name='batch',
        in_axes=(0, 0, 0, 0),
        static_broadcasted_argnums=(4, 5))

  def forward_pass(self,
                   flax_model,
                   train_state,
                   batch,
                   rng,
                   input_layer_key='input',
                   train=True):
    """Forward pass.

    Args:
      flax_model: flax.nn.Model; Flax model.
      train_state: TrainState object.
      batch: dict; A batch of examples.
      rng: float; Jax random number generator key.
      input_layer_key: str; Which layer the input should be plugged in.
      train: bool; Train flag.

    Returns:
      logits, hidden activations, activations of key layer, and new model state.
    """
    # bind the rng to the host/device we are on.
    rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    inputs = batch['inputs']

    with nn.stochastic(rng):
      (logits, all_reps, selected_reps,
       new_model_state) = pipeline_utils.forward_pass_with_reps(
           inputs, flax_model, train_state.model_state, input_layer_key, train)

    selected_reps = selected_reps.reshape(
        (selected_reps.shape[0], selected_reps.shape[1], -1))

    return logits, all_reps, selected_reps, new_model_state

  def train_step(self, train_state, batch):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.

    Returns:
      Updated state of training and calculated metrics.

    """
    max_grad_norm = self.hparams.get('max_grad_norm', None)
    new_rng, rng = jax.random.split(train_state.rng)

    # bind the rng to the host/device we are on.
    dropout_rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    train_loss_fn = functools.partial(
        self.training_loss_fn,
        train_state=train_state,
        batch=batch,
        dropout_rng=dropout_rng)
    new_train_state, metric_dict = self.compute_grads_and_update(
        batch, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metric_dict

  def get_learning_rate(self, step):
    lr = self.learning_rate_fn(step)

    return lr

  # TODO(samiraabnar): Try to avoid code duplication when overriding this fn.
  def compute_grads_and_update(self, batch, max_grad_norm, new_rng,
                               train_loss_fn, train_state):
    """Compute grads and updates parameters.

    Args:
      batch: dict; Batch of examples.
      max_grad_norm: float; Max value for grad norm (used for grad clipping).
      new_rng: Jax RNG key.
      train_loss_fn: fn(params)--> loss; Loss function (for which grad is
        computed).
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.

    Returns:
      Updated state of training and calculated metrics.
    """

    # Compute learning rate:
    lr = self.get_learning_rate(train_state.global_step)

    compute_gradient_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
    (_, (new_model_state,
         logits)), grad = compute_gradient_fn(train_state.optimizer.target)
    # re-use same axis_name as in the call to `pmap(...train_step...)` below
    grad = jax.lax.pmean(grad, axis_name='batch')

    if max_grad_norm is not None:
      grad = clip_grads(grad, max_grad_norm)

    new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_train_state = train_state.replace(
        global_step=train_state.global_step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state,
        rng=new_rng)

    metric_dict = self.collect_metrics(batch, logits, lr)

    return new_train_state, metric_dict

  def collect_metrics(self, batch, logits, lr):
    # Collect metrics:
    metric_dict = self.metrics_fn(logits, batch)
    metric_dict['learning_rate'] = lr

    return metric_dict

  def eval_step(self, train_state, batch):
    """Evaluate the model on the whole evaluation set.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.

    Returns:
      Calculated metrics.
    """

    metric_fn = functools.partial(self.metrics_fn)
    return pipeline_utils.eval_step(train_state, batch, metric_fn)

  def get_learning_rate_fn(self):
    """Get learning rate scheduler."""
    return lr_schedulers.get_learning_rate_fn(self.hparams)

  def set_train_state(self, model_cls, rng):
    """Set up train state.

    Args:
      model_cls: Type of the flax module.
      rng: Jax PRNG.
    """
    # Build flax_model.
    self.hparams.output_dim = self.task.task_params.output_dim
    flax_module, self.hparams = model_cls.build_flax_module(self.hparams)

    # Initialize flax module.
    rng, dropout_rng = jax.random.split(rng)
    (flax_module, model_state,
     self.num_trainable_params) = pipeline_utils.create_flax_module(
         flax_module, self.task.dataset.meta_data['input_shape'],
         self.hparams, dropout_rng,
         self.task.dataset.meta_data.get('input_dtype', jnp.float32))

    if self.hparams.get('pretrained', None):
      pretrained_config = self.hparams.pretrained.get('config')
      pretrained_checkpoint_path = self.hparams.pretrained.get(
          'checkpoint_path')
      pretrained_checkpoint_step = self.hparams.pretrained.get(
          'checkpoint_step', None)

      rng, new_rng = jax.random.split(rng)
      # Create and loads the model from the pretrained path.
      if pretrained_checkpoint_step is not None:
        logging.info('load pretrained model at step %d',
                     pretrained_checkpoint_step)
      pretrained_train_state = pipeline_utils.load_model(
          rng=new_rng,
          model_config=pretrained_config,
          model_ckpt=pretrained_checkpoint_path,
          task=self.task,
          load_full_train_state=self.hparams.pretrained.get(
              'full_trainstate_ckpt', True),
          checkpoint_step=pretrained_checkpoint_step)

      if self.hparams.pretrained.get('full_trainstate_ckpt', True):
        pretrained_model = pretrained_train_state.optimizer.target
        pretrained_model_state = pretrained_train_state.model_state
      else:
        (pretrained_model, pretrained_model_state) = pretrained_train_state

      if self.hparams.pretrained.get('only_backbone_pretrained', False):
        # Update params with pretrained params
        for m_key, m_params in pretrained_model.params.items():
          logging.info(m_key)
          if m_key not in ['head'] and ('disc' not in m_key):
            flax_module.params[m_key] = m_params
          else:
            logging.info('Not updated!')
        # Update model_state with pretrained model_state
        new_state_dict = {}
        for state_key, state_val in pretrained_model_state.as_dict().items():
          logging.info(state_key)
          if 'head' not in state_key and ('disc' not in state_key):
            new_state_dict[state_key] = pretrained_model_state[state_key]
          else:
            logging.info('Not updated!')
            new_state_dict[state_key] = state_val
        model_state = nn.Collection(new_state_dict)
      else:
        flax_module = pretrained_model
        model_state = pretrained_model_state

    # Create optimizer.
    optimizer = optimizers.get_optimizer(self.hparams).create(flax_module)

    # Create train state.
    rng, train_rng = jax.random.split(rng)
    train_state = pipeline_utils.TrainState(
        global_step=0,
        optimizer=optimizer,
        model_state=model_state,
        rng=train_rng)
    self.start_step = train_state.global_step

    # Reset gift regularizer's init point.
    if self.hparams.get('gift_factor', None):
      self.task.regularisers = [
          functools.partial(
              metrics.parameter_distance,
              base_params=train_state.optimizer.target.params,
              norm_factor=self.hparams.get('gift_factor'),
              mode='l2')
      ]

    if self.hparams.checkpoint:
      train_state, self.start_step = pipeline_utils.restore_checkpoint(
          self.experiment_dir, train_state)
      logging.info('Loading checkpoint at step %d', self.start_step)

    # Replicate the optimzier, state, and rng.
    self.train_state = jax_utils.replicate(train_state)
    del flax_module  # do not keep a copy of the initial model

    # Save the initial state.
    if self.start_step == 0 and self.hparams.checkpoint:
      self.checkpoint(self.train_state, self.start_step)

  def get_next_batch(self, data_iter):
    return next(data_iter)

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0
    train_metrics = []
    train_summary, eval_summary = None, None

    tick = time.time()

    # Main train loop.
    for step in range(self.start_step + 1, self.total_steps + 1):
      train_batch = self.get_next_batch(self.task.dataset.data_iters.train)
      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batch)
      train_metrics.append(t_metrics)

      eval_summary, train_metrics, train_summary, tick = self.maybe_eval_and_log(
          eval_summary, master, step, tick, train_metrics, train_summary)

      # sync and save
      self.train_state = self.checkpoint(self.train_state, step)

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def maybe_eval_and_log(self, eval_summary, master, step, tick, train_metrics,
                         train_summary):
    """Maybe evaluate and log based on the current step value."""
    if (step % self.eval_frequency == 0) or (step == self.total_steps):
      del eval_summary
      del train_summary

      train_metrics = common_utils.get_metrics(train_metrics)
      train_summary = pipeline_utils.compute_global_mean_metrics(train_metrics)

      tock = time.time()
      steps_per_sec = self.eval_frequency / (tock - tick)
      tick = tock

      # log train summary
      if master:
        self.write_train_summary(
            step=step,
            metric_dict=train_metrics,
            summary=train_summary,
            steps_per_sec=steps_per_sec)
      # reset metric accumulation for next evaluation cycle
      del train_metrics
      train_metrics = []

      # sync model state across replicas
      self.train_state = pipeline_utils.sync_model_state_across_replicas(
          self.train_state)

      # evaluate and log the results
      eval_summary, _ = self.eval(step, self.train_state)
    return eval_summary, train_metrics, train_summary, tick

  def eval(self, step, train_state, split_name='validation'):
    """Evaluation loop.

    Args:
      step: int; Training step.
      train_state: TrainState; Object containing training state.
      split_name: str; Name of the dataset split to evaluate on.

    Returns:
      eval_summart, train_state
    """
    train_state = pipeline_utils.sync_model_state_across_replicas(train_state)
    eval_summary, eval_metrics = self.eval_split(
        train_state=train_state, split_name=split_name)

    # log eval summary
    master = jax.host_id() == 0

    if master:
      self.write_eval_summary(
          step=step, metric_dict=eval_metrics, summary=eval_summary)
    return eval_summary, train_state

  def eval_split(self, train_state, split_name):
    """Evaluation loop on the specified split.

    Args:
      train_state: TrainState; Object containing training state.
      split_name: str; Name of the data split we want to evaluate the model on.

    Returns:
      eval_summary, train_state
    """
    data_iters = self.task.dataset.data_iters[split_name]
    eval_metrics = []
    for _ in range(self.steps_per_eval):
      env_eval_batches = self.get_next_batch(data_iters)
      e_metrics = self.pmapped_eval_step(train_state, env_eval_batches)
      eval_metrics.append(e_metrics)

    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_summary = pipeline_utils.compute_global_mean_metrics(eval_metrics)

    return eval_summary, eval_metrics

  def checkpoint(self, train_state, step):
    """Saves checkpoint.

    Syncs the model state across replicas if needed.

    Args:
      train_state: TrainSate; A flax struct that keeps model state and optimizer
        state.
      step: int; Number of steps passes so far during training.

    Returns:
      train_state
    """
    checkpoint_flag = False
    if self.hparams.get('ckpnt_steps', None) and self.hparams.checkpoint:
      if step in self.hparams.get('ckpnt_steps'):
        checkpoint_flag = True
    elif ((step % self.checkpoint_frequency == 0) or
          (step == self.total_steps)) and self.hparams.checkpoint:
      checkpoint_flag = True

    if checkpoint_flag:
      # Sync model state across replicas.
      train_state = pipeline_utils.sync_model_state_across_replicas(train_state)
      if jax.host_id() == 0:
        pipeline_utils.save_checkpoint(
            self.experiment_dir, train_state, keep=self.hparams.keep_ckpts)

    return train_state

  def write_train_summary(self, step, metric_dict, summary, steps_per_sec=None):
    """Logging and summarizing for the train phase.

    Args:
      step: int; Training step.
      metric_dict: dict; Dictionary of metric_name --> list of metric value for
        batches.
      summary: dict; Dictionary of metric_name --> average metric value (over
        all batches).
      steps_per_sec: float; Average steps per second.
    """
    mode = 'train'
    self.log(step, summary, mode, steps_per_sec)

    if steps_per_sec is not None:
      if self.hparams.write_summary:
        self.tb_summary_writer.scalar(f'{mode}_steps_per_sec', steps_per_sec,
                                      step)
    if self.hparams.write_summary:
      self.tb_summary_writer.scalar('num_trainable_params',
                                    self.num_trainable_params, step)
    if self.hparams.write_summary:
      for key, vals in metric_dict.items():
        if isinstance(vals, tuple):
          # If val is tuple of (value, normalizer), for most of the metrics
          # this is the case.
          for i, val in enumerate(zip(vals[0], vals[1])):
            report_step = step - len(vals[0]) + i + 1
            self.tb_summary_writer.scalar(
                f'{mode}_{key}', val[0] / val[1] if val[1] > 0 else 0.0,
                report_step)
        else:
          # If it is not a tuple, for example learning rate does not have
          # a normalizer.
          for i, val in enumerate(vals):
            report_step = step - len(vals) + i + 1
            self.tb_summary_writer.scalar(f'{mode}_{key}', val, report_step)

  def write_eval_summary(self, step, metric_dict, summary):
    """Logging and summarizing for the eval phase.

    Args:
      step: int; Training step.
      metric_dict: dict; Dictionary of metric_name --> list of metric value for
        batches.
      summary: dict; Dictionary of metric_name --> average metric value (over
        all batches).
    """
    mode = 'valid'

    self.log(step, summary, mode)
    if step is not None:
      if self.hparams.write_summary:
        for key, vals in metric_dict.items():
          if isinstance(vals, tuple):
            # If val is tuple of (value, normalizer), for most of the metrics
            # this is the case.
            self.tb_summary_writer.scalar(f'{mode}_{key}',
                                          vals[0].sum() / vals[1].sum(), step)
          else:
            # If it is not a tuple, for example learning rate does not have
            # a normalizer.
            self.tb_summary_writer.scalar(f'{mode}_{key}', vals.mean(), step)

  def log(self, step, summary, mode, steps_per_sec=None):
    """Log metrics.

    Args:
      step: int; Training step.
      summary: dict; Dictionary of metric_name --> average metric value (over
        all batches).
      mode: str; Eval or train.
      steps_per_sec: float; Average steps per second.

    Returns:

    """
    message = ''
    for key, val in summary.items():
      message += f'{key}: {val} | '
      if np.isnan(val):
        logging.info('step: %d -- {%s} -- {%s}', step or 0, mode, message)
        raise pipeline_utils.TrainingDivergedError(
            'NaN detected in {}'.format(f'train_{key}'))
    logging.info('step: %d -- {%s} -- {%s}', step or 0, mode, message)
    if steps_per_sec is not None:
      logging.info('train_steps_per_sec: %f, %d', steps_per_sec, step)
