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

r"""Train.

Example command:

python -m aux_tasks.grid.train \
  --base_dir=/tmp/pw \
  --reverb_address=localhost:1234 \
  --config=aux_tasks/grid/config.py:implicit

"""
import functools
import pathlib
import signal
from typing import Callable, NamedTuple, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
import chex
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from ml_collections import config_flags
import optax
import tqdm

from aux_tasks.grid import dataset
from aux_tasks.grid import loss_utils
from aux_tasks.grid import utils

_BASE_DIR = flags.DEFINE_string('base_dir', None, 'Base directory')
_CONFIG = config_flags.DEFINE_config_file('config', lock_config=True)


@struct.dataclass
class TrainMetrics(clu_metrics.Collection):
  loss: clu_metrics.Average.from_output('loss')
  rank: clu_metrics.Average.from_output('rank')


# @struct.dataclass
# class EvalMetrics(clu_metrics.Collection):
#   grassman_distance: clu_metrics.LastValue.from_output('grassman_distance')
#   dot_product: clu_metrics.LastValue.from_output('dot_product')
#   top_singular_value: clu_metrics.LastValue.from_output('top_singular_value')


@struct.dataclass
class EvalMetrics(clu_metrics.Collection):
  eval_loss: clu_metrics.Average.from_output('loss')


class SpectralDense(nn.Module):
  """Spectral Dense."""
  features: int
  kernel_init: Callable[[chex.PRNGKey, tuple[int, Ellipsis], jnp.dtype], chex.Array]

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features))
    # TODO(jfarebro): use power iteration
    _, s, _ = jnp.linalg.svd(kernel, full_matrices=False)
    kernel /= s[0]

    return jnp.einsum('...x,xf->...f', inputs, kernel)


class MLPEncoder(nn.Module):
  """MLP Encoder."""
  num_layers: int
  num_units: int
  embedding_dim: int

  @nn.compact
  def __call__(self, inputs):
    kernel_init = nn.initializers.xavier_uniform()

    x = inputs
    for _ in range(self.num_layers - 1):
      x = nn.Dense(self.num_units, kernel_init=kernel_init)(x)
      x = nn.PReLU()(x)
    x = nn.Dense(self.embedding_dim, kernel_init=kernel_init)(x)
    x = nn.PReLU()(x)
    return x


class ModuleOutputs(NamedTuple):
  phi: chex.Array
  predictions: Optional[chex.Array]


class ImplicitModule(nn.Module):
  encoder: nn.Module

  @nn.compact
  def __call__(self, inputs):
    return ModuleOutputs(self.encoder(inputs), None)


class ExplicitModule(nn.Module):
  """Explicit Module."""
  encoder: nn.Module
  num_tasks: int

  @nn.compact
  def __call__(self, inputs):
    kernel_init = nn.initializers.xavier_uniform()

    phi = self.encoder(inputs)
    x = nn.Dense(self.num_tasks, kernel_init=kernel_init)(phi)
    return ModuleOutputs(phi, x)


class LinearModule(nn.Module):
  """Linear Module."""
  num_tasks: int

  @nn.compact
  def __call__(self, phi):
    kernel_init = nn.initializers.xavier_uniform()
    x = nn.Dense(self.num_tasks, kernel_init=kernel_init)(phi)
    return x


@functools.partial(
    jax.jit, donate_argnums=(0, 1), static_argnames=('stop_grad', 'l2_coeff'))
def train_step_naive(state,
                     metrics,
                     inputs,
                     targets,
                     *,
                     stop_grad = True,
                     rcond = 1e-5):
  """Train naive CG."""

  def loss_fn(params):
    outputs = state.apply_fn(params, inputs)
    phis = outputs.phi

    # ws = jax.scipy.sparse.linalg.cg(
    #     phis.T @ phis, phis.T @ targets, tol=1e-12)[0]
    ws, _, _, _ = jnp.linalg.lstsq(phis, targets, rcond=rcond)
    if stop_grad:
      ws = jax.lax.stop_gradient(ws)

    task_outputs = phis @ ws
    loss = jnp.mean(optax.l2_loss(task_outputs, targets))

    rank = jnp.linalg.matrix_rank(phis.T @ phis)
    metrics_update = metrics.single_from_model_output(loss=loss, rank=rank)

    return loss, metrics_update

  grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
  grads, metrics_update = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = metrics.merge(metrics_update)
  return state, metrics


@functools.partial(
    jax.jit, static_argnames=('batch_sizes', 'alpha'), donate_argnums=(0, 1))
def train_step_implicit(state, metrics,
                        inputs, targets, *,
                        batch_sizes,
                        alpha):
  """Train step implicit."""

  def loss_fn(params):
    outputs = state.apply_fn(params, inputs)
    phis = outputs.phi

    rank = jnp.linalg.matrix_rank(phis.T @ phis)

    # Split out phis for implicit least squares grad computation
    phis = loss_utils.split_in_chunks(phis, [
        batch_sizes.main,
        batch_sizes.weight,
        batch_sizes.weight,
        batch_sizes.cov,
        batch_sizes.cov,
    ])  # pyformat: disable
    # Split out psis for implicit least squares grad computation
    psis = loss_utils.split_in_chunks(targets, [
        batch_sizes.main,
        batch_sizes.weight,
        batch_sizes.weight
    ])  # pyformat: disable
    loss = loss_utils.implicit_least_squares(*phis, *psis, alpha=alpha)

    metrics_update = metrics.single_from_model_output(loss=loss, rank=rank)
    return loss, metrics_update

  grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
  grads, metrics_update = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = metrics.merge(metrics_update)
  return state, metrics


@functools.partial(
    jax.jit, static_argnames=('batch_sizes', 'alpha'), donate_argnums=(0, 1))
def train_step_naive_implicit(state, metrics,
                              inputs, targets, *,
                              alpha):
  """Train naive implicit."""

  def loss_fn(params):
    outputs = state.apply_fn(params, inputs)
    phis = outputs.phi
    # Split out phis for implicit least squares grad computation
    loss = loss_utils.naive_implicit_least_squares(phis, targets, alpha=alpha)
    rank = jnp.linalg.matrix_rank(phis.T @ phis)
    metrics_update = metrics.single_from_model_output(loss=loss, rank=rank)

    return loss, metrics_update

  grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
  grads, metrics_update = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = metrics.merge(metrics_update)
  return state, metrics


@functools.partial(
    jax.jit, static_argnames=('batch_sizes', 'alpha'), donate_argnums=(0, 1))
def train_step_explicit(state,
                        metrics,
                        inputs,
                        targets):
  """Train naive implicit."""

  def loss_fn(params):
    outputs = state.apply_fn(params, inputs)
    predictions = outputs.predictions
    phis = outputs.phi
    # Split out phis for implicit least squares grad computation
    loss = jnp.mean(optax.l2_loss(predictions, targets))
    rank = jnp.linalg.matrix_rank(phis.T @ phis)
    metrics_update = metrics.single_from_model_output(loss=loss, rank=rank)

    return loss, metrics_update

  grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
  grads, metrics_update = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = metrics.merge(metrics_update)
  return state, metrics


# @functools.partial(jax.jit, static_argnames=('config'))
# def evaluate_mdp(state: TrainState, aux_task_matrix: chex.Array,
#                  config: config_dict.ConfigDict) -> clu_metrics.Collection:
#   """Evaluate."""
#   u = loss_utils.top_d_singular_vectors(aux_task_matrix, config.embedding_dim)

#   num_states = u.shape[0]
#   states = jax.nn.one_hot(jnp.arange(num_states), num_states)
#   phis = state.apply_fn(state.params, states)

#   top_singular_value = jnp.linalg.norm(phis.T @ phis, ord=2)
#   grassman_distance = loss_utils.grassman_distance(phis, u)

#   if phis.shape[-1] == 1:
#     dot_product = phis.T @ u / (jnp.linalg.norm(phis) * jnp.linalg.norm(u))
#     dot_product = dot_product.flatten()
#   else:
#     dot_product = None

#   return EvalMetrics.single_from_model_output(
#       grassman_distance=grassman_distance,
#       dot_product=dot_product,
#       top_singular_value=top_singular_value)


def create_default_writer():
  return metric_writers.create_default_writer()  # pylint: disable=unreachable


@functools.partial(jax.jit, donate_argnums=(1, 2))
def evaluate_step(
    train_state,
    eval_state,
    metrics,
    inputs,
    targets):
  """Eval train step."""

  outputs = train_state.apply_fn(train_state.params, inputs)
  phis = outputs.phi

  def loss_fn(params):
    predictions = jax.vmap(eval_state.apply_fn, in_axes=(None, 0))(params, phis)
    loss = jnp.mean(optax.l2_loss(predictions, targets))
    metrics_update = EvalMetrics.single_from_model_output(loss=loss)
    return loss, metrics_update

  grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
  grads, metrics_update = grad_fn(eval_state.params)
  eval_state = eval_state.apply_gradients(grads=grads)
  metrics = metrics.merge(metrics_update)
  return eval_state, metrics


def evaluate(base_dir, config, *,
             train_state):
  """Eval function."""
  chkpt_manager = checkpoint.Checkpoint(str(base_dir / 'eval'))

  writer = create_default_writer()

  key = jax.random.PRNGKey(config.eval.seed)
  model_init_key, ds_key = jax.random.split(key)

  linear_module = LinearModule(config.eval.num_tasks)
  params = linear_module.init(model_init_key,
                              jnp.zeros((config.encoder.embedding_dim,)))
  lr = optax.cosine_decay_schedule(config.eval.learning_rate,
                                   config.num_eval_steps)
  optim = optax.adam(lr)

  ds = dataset.get_dataset(config, ds_key, num_tasks=config.eval.num_tasks)
  ds_iter = iter(ds)

  state = TrainState.create(
      apply_fn=linear_module.apply, params=params, tx=optim)
  state = chkpt_manager.restore_or_initialize(state)

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_eval_steps, writer=writer)
  hooks = [
      report_progress,
      periodic_actions.Profile(num_profile_steps=5, logdir=str(base_dir))
  ]

  def handle_preemption(signal_number, _):
    logging.info('Received signal %d, saving checkpoint.', signal_number)
    with report_progress.timed('checkpointing'):
      chkpt_manager.save(state)
    logging.info('Finished saving checkpoint.')

  signal.signal(signal.SIGTERM, handle_preemption)

  metrics = EvalMetrics.empty()
  with metric_writers.ensure_flushes(writer):
    for step in tqdm.tqdm(range(state.step, config.num_eval_steps)):
      with jax.profiler.StepTraceAnnotation('eval', step_num=step):
        states, targets = next(ds_iter)
        state, metrics = evaluate_step(
            train_state, state, metrics, states, targets)

      if step % config.log_metrics_every == 0:
        writer.write_scalars(step, metrics.compute())
        metrics = EvalMetrics.empty()

      for hook in hooks:
        hook(step)

    # Finally, evaluate on the true(ish) test aux task matrix.
    states, targets = dataset.EvalDataset(config, ds_key).get_batch()

    @jax.jit
    def loss_fn():
      outputs = train_state.apply_fn(train_state.params, states)
      phis = outputs.phi
      predictions = jax.vmap(
          state.apply_fn, in_axes=(None, 0))(state.params, phis)
      return jnp.mean(optax.l2_loss(predictions, targets))

    test_loss = loss_fn()
    writer.write_scalars(config.num_eval_steps + 1, {'test_loss': test_loss})


def train(base_dir, config):
  """Train function."""
  print(config)
  chkpt_manager = checkpoint.Checkpoint(str(base_dir / 'train'))

  writer = create_default_writer()

  # Initialize dataset
  key = jax.random.PRNGKey(config.seed)
  key, subkey = jax.random.split(key)
  ds = dataset.get_dataset(config, subkey, num_tasks=config.num_tasks)
  ds_iter = iter(ds)

  key, subkey = jax.random.split(key)
  encoder = MLPEncoder(**config.encoder)

  train_config = config.train.to_dict()
  train_method = train_config.pop('method')

  module_config = train_config.pop('module')
  module_class = module_config.pop('name')

  module = globals().get(module_class)(encoder, **module_config)
  train_step = globals().get(f'train_step_{train_method}')
  train_step = functools.partial(train_step, **train_config)

  params = module.init(subkey, next(ds_iter)[0])
  lr = optax.cosine_decay_schedule(config.learning_rate, config.num_train_steps)
  optim = optax.chain(optax.adam(lr),
                      # optax.adaptive_grad_clip(0.15)
                     )

  state = TrainState.create(apply_fn=module.apply, params=params, tx=optim)
  state = chkpt_manager.restore_or_initialize(state)

  # Hooks
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  hooks = [
      report_progress,
      periodic_actions.Profile(num_profile_steps=5, logdir=str(base_dir))
  ]

  def handle_preemption(signal_number, _):
    logging.info('Received signal %d, saving checkpoint.', signal_number)
    with report_progress.timed('checkpointing'):
      chkpt_manager.save(state)
    logging.info('Finished saving checkpoint.')

  signal.signal(signal.SIGTERM, handle_preemption)

  metrics = TrainMetrics.empty()
  with metric_writers.ensure_flushes(writer):
    for step in tqdm.tqdm(range(state.step, config.num_train_steps)):
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        states, targets = next(ds_iter)
        state, metrics = train_step(state, metrics, states, targets)

      logging.log_first_n(logging.INFO, 'Finished training step %d', 5, step)

      if step % config.log_metrics_every == 0:
        writer.write_scalars(step, metrics.compute())
        metrics = TrainMetrics.empty()

      # if step % config.log_eval_metrics_every == 0 and isinstance(
      #     ds, dataset.MDPDataset):
      #   eval_metrics = evaluate_mdp(state, ds.aux_task_matrix, config)
      #   writer.write_scalars(step, eval_metrics.compute())

      for hook in hooks:
        hook(step)

  chkpt_manager.save(state)
  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  base_dir = pathlib.Path(_BASE_DIR.value)
  config = _CONFIG.value

  train_state = train(base_dir, config)
  evaluate(base_dir, config, train_state=train_state)


if __name__ == '__main__':
  app.run(main)
