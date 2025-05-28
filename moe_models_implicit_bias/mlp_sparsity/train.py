# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Contains the training code for the activation sparsity experiment."""

import functools
import time
from typing import Any
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
from moe_models_implicit_bias.mlp_sparsity import input_pipeline
from moe_models_implicit_bias.mlp_sparsity import models

NUM_CLASSES = 100


def create_model(*, model_cls, config):
  return model_cls(config)


def initialized(key, config, model):
  input_shape = (1, config.dim)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones(input_shape, jnp.float32))
  return variables['params'], variables['batch_stats']


def projected_mse_loss(logits, labels):
  mse_loss = jnp.mean((logits - labels)**2)
  return mse_loss


def compute_metrics(logits, z_all, labels):
  """Compute loss and other activation sparsity related metrics from a batch.

  Args:
    logits: the logits produced by the model.
    z_all: the hidden activations at the intermediate layers.
    labels: the ground truth labels.

  Returns:
    metrics: a dict containing the computed metrics.
  """
  loss = jnp.mean((logits - labels)**2)
  metrics = {
      'loss': loss,
  }
  ii = 1
  for z in z_all:
    z /= jnp.linalg.norm(z, axis=-1, keepdims=True)
    z = jnp.abs(z)

    metrics[f'sparsity_{ii}'] = jnp.mean((z > 0.0))
    metrics[f'dead_sparsity_{ii}'] = jnp.mean((jnp.mean(z, axis=0) > 0.0))
    ii += 1
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float, steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate, transition_steps=10)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def train_step(state, batch, learning_rate_fn, config):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    (logits, z), new_model_state = state.apply_fn(
        {
            'params': params,
            'batch_stats': state.batch_stats
        },
        batch[0],
        mutable=['batch_stats'])
    loss = projected_mse_loss(logits, batch[1])
    weight_penalty_params = jax.tree.leaves(params)
    weight_decay = config.weight_decay
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits, z)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)
  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
  new_model_state, logits, z = aux[1]
  metrics = compute_metrics(logits, z, batch[1])
  metrics['learning_rate'] = lr
  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree.map(
            functools.partial(jnp.where, is_fin), new_state.opt_state,
            state.opt_state),
        params=jax.tree.map(
            functools.partial(jnp.where, is_fin), new_state.params,
            state.params),
        dynamic_scale=dynamic_scale)
    metrics['scale'] = dynamic_scale.scale
  return new_state, metrics


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree.map(_prepare, xs)


def create_input_iter(config, batch_size):
  ds = input_pipeline.create_split(config, batch_size)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree.map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(rng,
                       config: ml_collections.ConfigDict,
                       model,
                       learning_rate_fn=None):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None
  params, batch_stats = initialized(rng, config, model)
  if config.optim == 'adam':
    tx = optax.adam(learning_rate=learning_rate_fn)
  elif config.optim == 'sgd':
    tx = optax.sgd(learning_rate=learning_rate_fn, momentum=.9)
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale)
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)
  rng = random.PRNGKey(config.seed)
  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()
  platform = jax.local_devices()[0].platform
  print('Platform:', platform, flush=True)
  train_iter = create_input_iter(config, local_batch_size)
  steps_per_epoch = (config.num_samples // config.batch_size)
  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  steps_per_checkpoint = steps_per_epoch * 10
  base_learning_rate = config.learning_rate * config.batch_size / 256.
  model_cls = getattr(models, config.model)
  model = create_model(model_cls=model_cls, config=config)
  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                             steps_per_epoch)
  state = create_train_state(
      rng, config, model, learning_rate_fn=learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)
  # making each entry be a unit normal
  p_train_step = jax.pmap(
      functools.partial(
          train_step, learning_rate_fn=learning_rate_fn, config=config),
      axis_name='batch')
  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')
    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'{k}': v
            for k, v in jax.tree.map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return state
