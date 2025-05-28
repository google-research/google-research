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

"""Offline Training."""

import contextlib
import functools
import signal
import sys
from typing import Callable, Dict, Mapping, Optional, Tuple, cast

from absl import flags
from absl import logging
import chex
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from etils import epath
import flax
from flax import linen as nn
import jax
from jax import profiler
from jax.experimental import pjit
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as onp
import optax
from orbax import checkpoint
import tensorflow as tf
import tqdm
from pvn import indicator_functions
from pvn import networks
from pvn import state
from pvn.datasets import atari
from pvn.utils import config_utils
from pvn.utils import mesh_utils
from pvn.utils import tree_utils

FLAGS = flags.FLAGS

_DATASETS = {'atari': atari.create_dataset}

ElementSpec = Mapping[str, jax.ShapeDtypeStruct]
Mesh = jax.sharding.Mesh
TargetParamsUpdateFunction = Callable[
    [chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree
]


@flax.struct.dataclass
class TrainMetrics(clu_metrics.Collection):
  """Train metrics."""

  tde_loss: clu_metrics.Average.from_output('tde_loss')
  tde_loss_std: clu_metrics.Std.from_output('tde_loss')

  average_prediction: clu_metrics.Average.from_output('dsm_action_preds')

  gradient_norm: clu_metrics.Average.from_output('gradient_norm')


@flax.struct.dataclass
class IndicatorMetrics(clu_metrics.Collection):
  average_reward: clu_metrics.Average.from_output('dsm_rewards')


@flax.struct.dataclass
class RNDIndicatorMetrics(clu_metrics.Collection):
  average_reward: clu_metrics.Average.from_output('dsm_rewards')

  proportion_loss: clu_metrics.Average.from_output('proportion_loss')
  proportion_loss_std: clu_metrics.Std.from_output('proportion_loss')


def construct_hard_target_params_update_fn(
    update_period,
):
  """Constructs a periodic hard target parameter copy closure.

  Args:
    update_period: int => The copy rate in terms of gradient steps.

  Returns:
    TargetParamsUpdateFunction => Function with the signature
      Callable[[chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree]
      that takes new_params, old_params, step. When step % update_period == 0
      then new_params == old_params.
  """
  return functools.partial(optax.periodic_update, update_period=update_period)


def construct_soft_target_params_update_fn(
    tau,
):
  """Constructs a closure that takes a moving average of online parameters.

  Args:
    tau: float => Moving average parameter, i.e., (1 - tau) * new + tau * old is
      the full soft update.

  Returns:
    TargetParamsUpdateFunction => Function with the signature
      Callable[[chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree]
      that takes new_params, old_params, step and returns
      (1 - tau) * new_params + tau * old_params.
      Note: This function doesn't use the step parameter.
  """

  def wrapper(
      new_params, old_params, unused_step
  ):
    # The current step is unused as we just EMA the params.
    ema = lambda new, old: (1.0 - tau) * new + tau * old
    return jax.tree.map(ema, new_params, old_params)

  return wrapper


@chex.assert_max_traces(n=1)
@profiler.annotate_function
def train_step(
    train_state,
    indicator_state,
    batch,
    train_metrics,
    indicator_metrics,
    config,
):
  """Train DSM."""

  # Double transpose trick for observation data
  def _transpose_observations(name, x):
    if name == 'observation' or name == 'next_observation':
      x = jnp.transpose(x, axes=[3, 0, 1, 2])
    return mesh_utils.with_sharding_constraint(
        x, mesh_utils.map_leading_axis_to_pspec(x, 'data')  # pytype: disable=wrong-arg-types  # jnp-type
    )

  with jax.profiler.TraceAnnotation('transpose-observation'):
    batch = tree_utils.tree_map_with_names(_transpose_observations, batch)

  # Actions should be one-hot for easier TPU computation
  # Terminals have an extra dimension added so we don't need to do so in the
  # train step.
  chex.assert_rank(
      [
          batch['observation'],
          batch['action'],
          batch['terminal'],
          batch['next_observation'],
      ],
      [4, 2, 2, 4],
  )

  def loss_fn(
      dsm_params, indicator_params
  ):
    # DSM indicators
    with jax.profiler.TraceAnnotation('indicator-rewards'):
      indicators = jax.vmap(
          indicator_state.apply_fn,
          in_axes=(None, 0),
      )(indicator_params, batch['observation'])

    if (
        config.offline.indicator.type == 'rnd'
        and config.offline.indicator.train_on_unthresholded_rewards
    ):
      rewards = mesh_utils.with_sharding_constraint(
          indicators.pre_threshold,
          mesh_utils.create_partition_spec('data', 'model'),
      )
    else:
      rewards = mesh_utils.with_sharding_constraint(
          indicators.rewards, mesh_utils.create_partition_spec('data', 'model')
      )
    rewards = jax.lax.stop_gradient(rewards)

    # Perform forward passes
    with jax.profiler.TraceAnnotation('online-action-values'):
      action_values = jax.vmap(
          train_state.apply_fn,
          in_axes=(None, 0),
      )(dsm_params, batch['observation'])
    with jax.profiler.TraceAnnotation('target-action-values'):
      target_action_values = jax.vmap(
          train_state.apply_fn,
          in_axes=(None, 0),
      )(train_state.target_params, batch['next_observation'])

    # === DSM Loss ===
    # Actions are one-hot for easier TPU computation, slicing is very
    # slow on TPUs.
    chosen_action_preds = jnp.einsum(
        'bta,ba->bt',
        action_values,
        batch['action'],
    )
    chosen_action_preds = mesh_utils.with_sharding_constraint(
        chosen_action_preds, mesh_utils.create_partition_spec('data', 'model')
    )

    # Compute TD targets
    targets = jnp.mean(target_action_values, axis=2)
    targets = mesh_utils.with_sharding_constraint(
        targets, mesh_utils.create_partition_spec('data', 'model')
    )
    chex.assert_rank([chosen_action_preds, targets], [2, 2])

    td_targets = (
        rewards + (1.0 - batch['terminal']) * config.offline.discount * targets
    )
    td_targets = jax.lax.stop_gradient(td_targets)
    td_targets = mesh_utils.with_sharding_constraint(
        td_targets, mesh_utils.create_partition_spec('data', 'model')
    )
    # TD errors
    td_errors = td_targets - chosen_action_preds
    td_errors = mesh_utils.with_sharding_constraint(
        td_errors, mesh_utils.create_partition_spec('data', 'model')
    )
    chex.assert_rank(td_errors, 2)
    tde_loss = jnp.mean(optax.l2_loss(td_errors))

    loss = tde_loss

    train_infos = {
        'tde_loss': tde_loss.astype(jnp.float32),
        'dsm_action_preds': jnp.mean(action_values).astype(jnp.float32),
    }
    indicator_infos = {
        'dsm_rewards': jnp.mean(rewards).astype(jnp.float32),
    }

    # === Quantile Regression Loss ===
    if config.offline.indicator.type == 'rnd':
      target_reward_proportion = (
          config.offline.indicator.target_reward_proportion
      )
      dsm_pre_rewards = mesh_utils.with_sharding_constraint(
          indicators.pre_threshold,
          mesh_utils.create_partition_spec('data', 'model'),
      )
      proportion_loss = dsm_pre_rewards * (
          (1.0 - target_reward_proportion) - (dsm_pre_rewards < 0.0)
      )
      chex.assert_rank(proportion_loss, 2)
      proportion_loss = jnp.mean(proportion_loss)
      # Mask out the TDE loss if we haven't taken enough warmup steps
      loss *= (
          indicator_state.step > config.offline.indicator.num_qr_warmup_steps
      )
      # Add the proportion loss term
      indicator_infos |= {
          'proportion_loss': proportion_loss.astype(jnp.float32)
      }
      loss += (
          indicator_state.step < config.offline.indicator.num_qr_steps
      ) * proportion_loss

    return loss, (train_infos, indicator_infos)  # pytype: disable=bad-return-type  # jnp-type

  grad_fn = jax.grad(loss_fn, argnums=(0, 1), has_aux=True, allow_int=True)
  (train_grads, indicator_grads), (train_infos, indicator_infos) = grad_fn(
      train_state.params, indicator_state.params
  )

  # Hash params are int so we get back float0 grads. Just use zero grad
  if config.offline.indicator.type == 'hash':
    indicator_grads = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(leaf, dtype=jnp.float32), indicator_grads
    )

  # This works for our two encoder networks - NatureDQN and Impala.
  train_infos |= {
      'gradient_norm': jnp.linalg.norm(
          train_grads['params']['encoder']['Dense_0']['kernel']
      )
  }

  train_state = train_state.apply_gradients(grads=train_grads)
  indicator_state = indicator_state.apply_gradients(grads=indicator_grads)

  train_metrics_update = train_metrics.single_from_model_output(**train_infos)
  train_metrics = train_metrics.merge(train_metrics_update)

  indicator_metrics_update = indicator_metrics.single_from_model_output(
      **indicator_infos
  )
  indicator_metrics = indicator_metrics.merge(indicator_metrics_update)

  return train_state, indicator_state, train_metrics, indicator_metrics


def create_train_state_with_optional_mesh(
    element_spec,
    *,
    mesh,
    config,
    rng,
):
  """Create train & indicator state with optional mesh."""

  # pyformat: disable
  if ((config.offline.target_params_update_every is not None) ==
      (config.offline.target_params_soft_update_tau is not None)):
    raise ValueError('Must provide exactly one of '
                     '`target_params_soft_update_tau` or '
                     '`target_params_update_every`')
  # pyformat: enable

  # Create target network update function
  if config.offline.target_params_update_every is not None:
    target_params_update_fn = construct_hard_target_params_update_fn(
        config.offline.target_params_update_every
    )
  elif config.offline.target_params_soft_update_tau is not None:
    target_params_update_fn = construct_soft_target_params_update_fn(
        config.offline.target_params_soft_update_tau
    )
  else:
    raise ValueError(
        'Must provide exactly one of target_params_update_every '
        'or target_params_soft_update_tau'
    )

  encoder: nn.Module = config_utils.get_configurable(
      networks, config.encoder, name='encoder'
  )
  model: nn.Module = config_utils.get_configurable(
      networks,
      config.offline.model,
      num_actions=atari.get_num_actions(config.game),
      encoder=encoder,
  )
  optim: optax.GradientTransformation = config_utils.get_configurable(
      optax, config.offline.optim
  )

  if mesh:
    # If we have a mesh we'll construct the partition specs and configure
    # our parallel jit function.
    train_state_pspec = state.create_train_state_partition_spec(
        element_spec,
        model=model,
        optim=optim,
        target_params_update_fn=target_params_update_fn,
    )

    with mesh:
      create_train_state = pjit.pjit(
          functools.partial(
              state.create_train_state,
              element_spec,
              model=model,
              optim=optim,
              target_params_update_fn=target_params_update_fn,
              rng=rng,
          ),
          in_shardings=None,
          out_shardings=train_state_pspec,
      )
      train_state = create_train_state()
  else:
    # If we don't have a mesh just use regular jit
    create_train_state = state.create_train_state
    train_state = state.create_train_state(
        element_spec,
        model=model,
        optim=optim,
        target_params_update_fn=target_params_update_fn,
        rng=rng,
    )

  return train_state


def create_indicator_state_with_optional_mesh(
    element_spec,
    *,
    mesh,
    config,
    rng,
):
  """Create train & indicator state with optional mesh."""
  model: nn.Module = config_utils.get_configurable(
      indicator_functions, config.offline.indicator.module
  )
  optim: optax.GradientTransformation = config_utils.get_configurable(
      optax, config.offline.optim
  )

  # Mask out the encoder parameters as these are never optimized.
  if config.offline.indicator.type == 'hash':
    optim_mask = {'params': False}
  elif config.offline.indicator.type == 'rnd':
    optim_mask = {'params': {'encoder': False, 'reward_bias': True}}
  else:
    raise ValueError
  optim = optax.masked(optim, optim_mask)

  if mesh:
    # If we have a mesh we'll construct the partition specs and configure
    # our parallel jit function.
    indicator_state_pspec = state.create_indicator_state_partition_spec(
        element_spec, model=model, optim=optim
    )

    with mesh:
      create_indicator_state = pjit.pjit(
          functools.partial(
              state.create_indicator_state,
              element_spec,
              model=model,
              optim=optim,
              rng=rng,
          ),
          in_shardings=None,
          out_shardings=indicator_state_pspec,
      )

      indicator_state = create_indicator_state()
  else:
    # If we don't have a mesh just use regular jit
    create_indicator_state = state.create_indicator_state
    indicator_state = state.create_indicator_state(
        element_spec, model=model, optim=optim, rng=rng
    )

  return indicator_state


def jit_train_step_with_optional_mesh(
    element_spec,
    train_state,
    indicator_state,
    *,
    mesh,
):
  """Create jitted train function with optional mesh."""
  if mesh:
    batch_pspec = tree_utils.tree_map_with_regex(
        functools.partial(
            mesh_utils.map_trailing_axis_to_pspec, mesh_axis_name='data'
        ),
        element_spec,
        [('observation',), ('next_observation',)],
        functools.partial(
            mesh_utils.map_leading_axis_to_pspec, mesh_axis_name='data'
        ),
    )

    train_state_shape = jax.eval_shape(lambda x: x, train_state)
    train_state_pspec = state.create_train_state_partition_spec_from_shape(
        train_state_shape
    )

    indicator_state_shape = jax.eval_shape(lambda x: x, indicator_state)
    indicator_state_pspec = (
        state.create_indicator_state_partition_spec_from_shape(
            indicator_state_shape
        )
    )

    return pjit.pjit(
        train_step,
        donate_argnums=(0, 1, 3, 4),
        static_argnums=(5,),
        in_shardings=(
            train_state_pspec,
            indicator_state_pspec,
            batch_pspec,
            None,
            None,
        ),
        out_shardings=(
            train_state_pspec,
            indicator_state_pspec,
            None,
            None,
        ),
    )
  else:
    return jax.jit(
        train_step,
        donate_argnums=(0, 1, 3, 4),
        static_argnums=(5,),
    )


def train(
    workdir,
    *,
    config,
):
  """Train function."""

  dataset = config_utils.get_configurable(_DATASETS, config.offline.dataset)
  dataset: tf.data.Dataset = cast(tf.data.Dataset, dataset)
  # Create dataset iterator
  dataset_iter = dataset.as_numpy_iterator()

  # Metric Writer
  writer = metric_writers.create_default_writer(
      just_logging=jax.process_index() > 0,
  )
  checkpointers = {
      'train': checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler()),
      'indicator': checkpoint.Checkpointer(
          checkpoint.PyTreeCheckpointHandler()
      ),
  }
  ckpt_manager_options = checkpoint.CheckpointManagerOptions(max_to_keep=3)
  ckpt_manager = checkpoint.CheckpointManager(
      workdir, checkpointers=checkpointers, options=ckpt_manager_options
  )

  # Hooks
  running_in_colab = 'google.colab' in sys.modules

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.offline.num_grad_updates, writer=writer
  )

  # Create partition specs & global mesh
  # Right now we'll greedily shard over the data dimension based on how
  # many local devices we have
  global_mesh_spec = [('data', len(jax.local_devices())), ('model', 1)]
  mesh = mesh_utils.create_global_mesh(global_mesh_spec)

  # Convert the TensorFlow dataset elementspec to Jax so we can use it
  # with jax.eval_shape.
  def convert_tf_tensor_spec(spec):
    return jax.ShapeDtypeStruct(spec.shape, spec.dtype.as_numpy_dtype())

  element_spec = jax.tree_util.tree_map(
      convert_tf_tensor_spec, dataset.element_spec
  )

  # Create train and indicator states. Both will require the mesh.
  seed = config.seed
  if seed is None:
    seed = onp.random.SeedSequence().generate_state(1).item()
  rng = jax.random.PRNGKey(seed)
  train_rng, indicator_rng = jax.random.split(rng)

  train_state = create_train_state_with_optional_mesh(
      element_spec, mesh=mesh, config=config, rng=train_rng
  )
  indicator_state = create_indicator_state_with_optional_mesh(
      element_spec, mesh=mesh, config=config, rng=indicator_rng
  )
  with report_progress.timed('checkpoint-restore'):
    (
        train_state,
        indicator_state,
    ) = state.maybe_restore_train_and_indicator_state(
        train_state,
        indicator_state,
        ckpt_manager=ckpt_manager,
        mesh=mesh,
    )

  jitted_train_step = jit_train_step_with_optional_mesh(
      element_spec,
      train_state,
      indicator_state,
      mesh=mesh,
  )

  def save_checkpoint(*, wait_until_finished):
    logging.info('Saving checkpoint at step %d.', train_state.step)
    with report_progress.timed('checkpoint-save'):
      state.save_train_and_indicator_state(
          train_state.step,
          train_state,
          indicator_state,
          ckpt_manager=ckpt_manager,
          wait_until_finished=wait_until_finished,
      )
    logging.info('Finished saving checkpoint.')

  hooks = [report_progress]

  # Variables to be modified by the signal so that we can checkpoint
  # if we're about to be evicted.
  about_to_be_evicted = False
  exit_signal = 0

  def signal_handler(signal_number, _):
    nonlocal about_to_be_evicted, exit_signal
    logging.info('Received signal %d', signal_number)
    about_to_be_evicted = True
    exit_signal = signal_number

  signal.signal(signal.SIGTERM, signal_handler)
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGABRT, signal_handler)

  with contextlib.ExitStack() as stack:
    stack.enter_context(metric_writers.ensure_flushes(writer))

    initial_step = int(train_state.step) + 1
    final_step = config.offline.num_grad_updates
    logging.info('Starting training from step %d', initial_step)

    train_metrics = TrainMetrics.empty()
    indicator_metrics = IndicatorMetrics.empty()
    if config.offline.indicator.type == 'rnd':
      indicator_metrics = RNDIndicatorMetrics.empty()
    for step in tqdm.trange(
        initial_step,
        final_step + 1,
        unit='grad updates',
        disable=not running_in_colab,
    ):
      with contextlib.ExitStack() as step_stack:
        step_stack.enter_context(
            profiler.StepTraceAnnotation('train', step_num=step)
        )
        if mesh:
          step_stack.enter_context(mesh)
        # We must sample transitions from the dataset inside the
        # StepTraceAnnotation to get some additional xprof features.
        batch = next(dataset_iter)

        # pjit doesn't support kwargs so everything must be passed
        # as positional args.
        (
            train_state,
            indicator_state,
            train_metrics,
            indicator_metrics,
        ) = jitted_train_step(
            train_state,
            indicator_state,
            batch,
            train_metrics,
            indicator_metrics,
            config,
        )

      # Quick indication training is happening
      logging.log_first_n(logging.INFO, 'Finished training step %d', 5, step)

      # Write scalars, this is async and won't block
      if step % config.offline.log_metrics_every == 0 or step == final_step:
        computed_train_metrics = {
            f'offline/train/{key}': value
            for key, value in train_metrics.compute().items()
        }
        computed_indicator_metrics = {
            f'offline/indicator/{key}': value
            for key, value in indicator_metrics.compute().items()
        }
        writer.write_scalars(
            step, computed_train_metrics | computed_indicator_metrics
        )
        train_metrics = TrainMetrics.empty()
        indicator_metrics = IndicatorMetrics.empty()
        if config.offline.indicator.type == 'rnd':
          indicator_metrics = RNDIndicatorMetrics.empty()

      if about_to_be_evicted:
        save_checkpoint(wait_until_finished=True)
        logging.info('Successfully saved checkpoint on eviction.')
        sys.exit(exit_signal)
      # Save checkpoint periodically so we don't rely only on the exit signal
      if step % config.offline.checkpoint_every == 0:
        save_checkpoint(wait_until_finished=False)

      # Call hooks, most of the hooks are async and won't block
      for hook in hooks:
        hook(step)

  save_checkpoint(wait_until_finished=True)
  logging.info(
      'Finished offline after %d gradient updates.',
      config.offline.num_grad_updates,
  )
