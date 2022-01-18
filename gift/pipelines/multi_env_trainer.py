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

"""Multi environment trainer."""

import functools
import time

from flax.deprecated import nn
from flax.training import common_utils
import jax
from jax.experimental.optimizers import clip_grads
import jax.numpy as jnp
import numpy as np

from gift.pipelines import pipeline_utils
from gift.pipelines import trainer


class MultiEnvTrainer(trainer.Trainer):
  """Base class for multi environment trainers."""

  def get_total_eval_steps(self):
    total_eval_steps = {}

    def get_num_steps(split, env):
      return np.ceil(self.task.dataset.splits[split][env].num_examples /
                     self.hparams.eval_batch_size)

    for split in self.task.dataset.splits:

      total_eval_steps[split] = {
          env: int(get_num_steps(split, env))
          for env in self.task.dataset.splits[split]
      }

    return total_eval_steps

  def metrics_fn(self, env_logits, env_batch, env_ids, model_params):
    return self.task.metrics_fn(
        env_logits=env_logits,
        env_batches=env_batch,
        env_ids=env_ids,
        params=model_params)

  def training_loss_fn(self, flax_model, train_state, batch, dropout_rng,
                       env_ids):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      env_ids: list(int); List if environment ids.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    raise NotImplementedError

  def setup_pmapped_tain_and_eval_steps(self):
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, _ = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    train_env_ids = list(map(int, train_env_ids))

    self.p_train_step = functools.partial(
        self.train_step, env_ids=train_env_ids)
    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)

    self.pmapped_train_step = jax.pmap(
        self.p_train_step,
        axis_name='batch',
        in_axes=(0, 0, 0),
        donate_argnums=(1, 2))

    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2))

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
    # bind the rng to the host/device we are on.
    rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    inputs = pipeline_utils.get_multi_env_inputs(batch, 'inputs')

    with nn.stochastic(rng):
      (env_logits, all_env_reps, selected_env_reps,
       new_model_state) = pipeline_utils.vmapped_flax_module_with_reps(
           inputs, flax_model, train_state.model_state, input_layer_key, train)

    selected_env_reps = selected_env_reps.reshape(
        (selected_env_reps.shape[0], selected_env_reps.shape[1], -1))

    return env_logits, all_env_reps, selected_env_reps, new_model_state

  def get_next_batch(self, data_iter):
    """Return the next batch for multi environment datasets.

    Args:
      data_iter: list(map) List of iterators on the different domains of the
        dataset split (train/test/valid).

    Returns:
      List of batches.
    """
    return jax.tree_map(next, data_iter)

  def train_step(self, train_state, batch, env_ids):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.
      env_ids: list(int): List of training environments codes.

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
        dropout_rng=dropout_rng,
        env_ids=env_ids)

    new_train_state, metrics = self.compute_grads_and_update(
        batch, env_ids, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metrics

  # TODO(samiraabnar): Try to avoid code duplication when overriding this fn.
  def compute_grads_and_update(self, batch, env_ids, max_grad_norm, new_rng,
                               train_loss_fn, train_state):

    # Compute learning rate:
    lr = self.get_learning_rate(train_state.global_step)

    # Compute gradients:
    compute_gradient_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
    (_, (new_model_state, logits,
         logs)), grad = compute_gradient_fn(train_state.optimizer.target)

    # Update parameters:
    grad = jax.lax.pmean(grad, axis_name='batch')
    # Clip gradients:
    if max_grad_norm is not None:
      grad = clip_grads(grad, max_grad_norm)

    new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

    # Get the new (updated) train_state:
    new_train_state = pipeline_utils.TrainState(
        global_step=train_state.global_step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state,
        rng=new_rng)

    metrics = self.collect_metrics(batch, env_ids, logits, logs, lr,
                                   train_state.optimizer.target)

    return new_train_state, metrics

  def collect_metrics(self, batch, env_ids, logits, logs, lr, model_params):
    """Collect metrics."""

    metrics_dict = self.metrics_fn(logits, batch, env_ids, model_params)
    metrics_dict['learning_rate'] = lr
    if isinstance(logs, dict):
      for key in logs:
        if jnp.isscalar(logs[key]):
          metrics_dict[key] = logs[key]
        else:
          metrics_dict[f'mean_{key}'] = jnp.mean(logs[key])

    return metrics_dict

  def eval_step(self, train_state, batch, env_id, all_env_ids):
    """Runs a single step of evaluation.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data. a metrics function, that given logits and
        batch of data, calculates the metrics as well as the loss.
      env_id: int: Eval environments code.
      all_env_ids: List of eval all environment ids.

    Returns:
      Calculated metrics.
    """
    flax_model = train_state.optimizer.target
    inputs = pipeline_utils.get_multi_env_inputs(batch, 'inputs')

    with nn.stateful(train_state.model_state, mutable=False):
      env_logits = pipeline_utils.vmapped_flax_module_eval(flax_model, inputs)

    if env_id >= 0:
      metrics = self.metrics_fn(env_logits, batch, [env_id], flax_model)
    else:
      metrics = self.metrics_fn(env_logits, batch, all_env_ids, flax_model)

    return metrics

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0

    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, train_iters = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    train_env_ids = list(map(int, train_env_ids))

    for step in range(self.start_step + 1, self.total_steps + 1):
      train_batches = self.get_next_batch(train_iters)
      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batches)

      t_metrics = jax.tree_map(lambda x: x[0], t_metrics)
      train_metrics.append(t_metrics)

      eval_summary, train_metrics, train_summary, tick = self.maybe_eval_and_log(
          eval_env_ids, eval_summary, master, step, tick, train_metrics,
          train_summary)

      # Sync and save
      self.train_state = self.checkpoint(self.train_state, step)

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def maybe_eval_and_log(self, eval_env_ids, eval_summary, master, step, tick,
                         train_metrics, train_summary):
    if (step % self.eval_frequency == 0) or (step == self.total_steps):
      train_metrics = jax.device_get(train_metrics)
      train_metrics = common_utils.stack_forest(train_metrics)
      train_summary = pipeline_utils.compute_global_mean_metrics(train_metrics)
      tock = time.time()
      steps_per_sec = self.eval_frequency / (tock - tick)
      tick = tock

      # Log train summary:
      if master:
        self.write_train_summary(
            step=step,
            metric_dict=train_metrics,
            summary=train_summary,
            steps_per_sec=steps_per_sec)

      # Reset metric accumulation for next evaluation cycle:
      train_metrics = []

      # Sync model state across replicas:
      self.train_state = pipeline_utils.sync_model_state_across_replicas(
          self.train_state)

      # Evaluate and log the results:
      eval_summary, self.train_state = self.eval(step, self.train_state,
                                                 eval_env_ids)
    return eval_summary, train_metrics, train_summary, tick

  def eval(self, step, train_state, eval_env_ids=None):
    """Evaluation loop.

    Args:
      step: int; Training step.
      train_state: TrainState; Object containing training state.
      eval_env_ids: list(int); Eval environments ids.

    Returns:
      eval_summart, train_state
    """
    eval_summary, eval_metrics = self.eval_split(
        train_state=train_state,
        eval_env_ids=eval_env_ids,
        split_name='validation')
    # log eval summary
    master = jax.host_id() == 0
    if master:
      self.write_eval_summary(
          step=step, metric_dict=eval_metrics, summary=eval_summary)
    return eval_summary, train_state

  def eval_split(self, train_state, split_name, eval_env_ids=None):
    """Evaluation loop on the specified split.

    Args:
      train_state: TrainState; Object containing training state.
      split_name: str; Name of the data split we want to evaluate the model on.
      eval_env_ids: list(int); Eval environments ids.

    Returns:
      eval_summary, train_state
    """
    data_iters = self.task.dataset.data_iters[split_name]
    if eval_env_ids is None:
      eval_env_ids = list(map(int, data_iters.keys()))

    eval_metrics = {}
    if isinstance(self.steps_per_eval, dict):
      for env_id in eval_env_ids:
        env_id_str = str(env_id)
        env_eval_metrics = []
        for _ in range(self.steps_per_eval[split_name][env_id_str]):
          env_eval_batches = self.get_next_batch([data_iters[env_id_str]])
          e_metrics = self.pmapped_eval_step(train_state, env_eval_batches,
                                             env_id)
          env_eval_metrics.append(e_metrics)

        env_eval_metrics = common_utils.get_metrics(env_eval_metrics)
        eval_metrics.update(env_eval_metrics)

      eval_summary = pipeline_utils.compute_global_mean_metrics(eval_metrics)
    else:
      _, data_iters = list(zip(*dict(data_iters).items()))
      eval_metrics = []
      for _ in range(self.steps_per_eval):
        env_eval_batches = self.get_next_batch(data_iters)
        e_metrics = self.pmapped_eval_step(train_state, env_eval_batches, -1)
        eval_metrics.append(e_metrics)

      eval_metrics = common_utils.get_metrics(eval_metrics)
      eval_summary = pipeline_utils.compute_global_mean_metrics(eval_metrics)

    return eval_summary, eval_metrics
