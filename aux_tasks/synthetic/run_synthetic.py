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

r"""Implicit aux tasks training.

Example command:

python -m aux_tasks.synthetic.run_synthetic

"""
# pylint: disable=invalid-name
import functools
from typing import Callable, Optional, Union

from absl import app
from absl import flags
from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
from etils import epath
from etils import etqdm
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
import optax

from aux_tasks.synthetic import estimates
from aux_tasks.synthetic import utils

_config = config_dict.ConfigDict()

_config.method: str = 'explicit'
_config.optimizer: str = 'sgd'
_config.num_epochs: int = 2_000_000
_config.rescale_psi = ''
_config.use_mnist = False
_config.sample_with_replacement = True

_config.S: int = 10  # Number of states
_config.T: int = 10  # Number of aux. tasks
_config.d: int = 1  # feature dimension

# The theoretical maximum for kappa is 2, and 1.9 works well.
_config.kappa: float = 1.9  # Lissa kappa

_config.covariance_batch_size: int = 32
_config.main_batch_size: int = 32
_config.weight_batch_size: int = 32

_config.seed: int = 4753849
_config.lr: float = 0.01

_config.suite = 'synthetic'  # synthetic or puddle_world

# SYNTHETIC SUITE CONFIG
# If the SVD has precomputed, supply the path here to avoid recomputing it.
_config.svd_path: str = ''
_config.use_tabular_gradient = True

# PUDDLE WORLD SUITE CONFIG
_config.puddle_world_path = ''
_config.puddle_world_arena = 'sutton_10'  # or sutton_20, sutton_100
_config.phi_hidden_layers = 1
_config.phi_hidden_layer_width = 100
_config.use_center_states_only = False

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Base directory to store stats.', required=True
)
_CONFIG = config_flags.DEFINE_config_dict('config', _config, lock_config=True)

Parameters = dict[str, Union[flax.core.FrozenDict, jnp.ndarray]]


def compute_grassman_distance(Y1, Y2):
  """Grassman distance between subspaces spanned by Y1 and Y2."""
  Q1, _ = jnp.linalg.qr(Y1)
  Q2, _ = jnp.linalg.qr(Y2)

  _, sigma, _ = jnp.linalg.svd(Q1.T @ Q2)
  sigma = jnp.round(sigma, decimals=6)
  return jnp.linalg.norm(jnp.arccos(sigma))


def compute_cosine_similarity(Y1, Y2):
  try:
    projection_weights = jnp.linalg.solve(Y1.T @ Y1, Y1.T @ Y2)
    projection = Y1 @ projection_weights

    return jnp.linalg.norm(projection)
  except np.linalg.LinAlgError:
    pass
  return jnp.nan  # pytype: disable=bad-return-type  # jnp-type


def compute_normalized_dot_product(
    Y1, Y2
):
  return jnp.abs(
      jnp.squeeze(Y1.T @ Y2 / (jnp.linalg.norm(Y1) * jnp.linalg.norm(Y2)))
  )


def eigengame_subspace_distance(
    Phi, optimal_subspace
):
  """Compute subspace distance as per the eigengame paper."""
  try:
    d = Phi.shape[1]
    U_star = optimal_subspace @ optimal_subspace.T

    U_phi, _, _ = jnp.linalg.svd(Phi)
    U_phi = U_phi[:, :d]
    P_star = U_phi @ U_phi.T

    return 1 - 1 / d * jnp.trace(U_star @ P_star)
  except np.linalg.LinAlgError:
    return jnp.nan  # pytype: disable=bad-return-type  # jnp-type


def compute_metrics(
    Phi, optimal_subspace
):
  """Computes a variety of learning curve-type metrics for the given run.

  Args:
    Phi: Feature matrix.
    optimal_subspace: The optimal subspace.

  Returns:
    dict with keys:
      cosine_similarity: a jnp.array of size num_update_steps with cosine
        similarity between Phi and the d-principal subspace of Psi.
      feature_norm: the mean norm of the state feature vectors
        (averaged across states) over time.
      eigengame_subspace_distance: the subspace distance from the
        eigengame paper.
  """
  feature_norm = jnp.linalg.norm(Phi) / Phi.shape[0]
  cosine_similarity = compute_cosine_similarity(Phi, optimal_subspace)

  metrics = {
      'cosine_similarity': cosine_similarity,
      'feature_norm': feature_norm,
      'eigengame_subspace_distance': eigengame_subspace_distance(
          Phi, optimal_subspace
      ),
  }

  _, d = Phi.shape
  if d > 1:
    grassman_distance = compute_grassman_distance(Phi, optimal_subspace)
    metrics |= {'grassman_distance': grassman_distance}
  elif d == 1:
    dot_product = compute_normalized_dot_product(Phi, optimal_subspace)
    metrics |= {'dot_product': dot_product}

  return metrics


@functools.partial(
    jax.jit,
    static_argnames=(
        'compute_phi',
        'compute_psi',
        'method',
        'main_batch_size',
        'covariance_batch_size',
        'weight_batch_size',
        'd',
        'compute_feature_norm_on_oracle_states',
        'sample_states',
        'use_tabular_gradient',
    ),
)
def compute_gradient(
    *,
    source_states,
    task,
    compute_phi,
    compute_psi,
    params,
    key,
    method,
    oracle_states,
    lissa_kappa,
    main_batch_size,
    covariance_batch_size,
    weight_batch_size,
    d,
    compute_feature_norm_on_oracle_states,
    sample_states,
    use_tabular_gradient = True,
):
  """Computes the gradient under a specific method."""
  # The argument passed to vjp should be a function of parameters
  # to Phi.
  phi_params = params['phi_params']
  source_phi, phi_vjp = jax.vjp(
      lambda params: compute_phi(params, source_states), phi_params
  )

  # We needed compute_phi to take params as an argument to work out
  # gradients w.r.t the parameters in the vjp above. However,
  # for future use we can just wrap the parameters up in the function
  # for other usage.
  compute_phi_no_params = functools.partial(compute_phi, phi_params)

  if method == 'lissa' and compute_feature_norm_on_oracle_states:
    oracle_phis = compute_phi(phi_params, oracle_states)  # pytype: disable=wrong-arg-types  # jax-ndarray
    feature_norm = utils.compute_max_feature_norm(oracle_phis)
  else:
    feature_norm = None

  # This determines the weight vectors to be used to perform the gradient step.
  if method == 'explicit':
    # With the explicit method we maintain a running weight vector.
    explicit_weight_matrix = params['explicit_weight_matrix']
    weight_1 = jnp.squeeze(explicit_weight_matrix[:, task], axis=1)
    weight_2 = jnp.squeeze(explicit_weight_matrix[:, task], axis=1)
  else:  # Implicit methods.
    if method == 'oracle':
      # This exactly determines the covariance in the tabular case,
      # i.e. when oracle_states = S.
      Phi = compute_phi_no_params(oracle_states)
      num_states = oracle_states.shape[0]

      covariance_1 = jnp.linalg.pinv(Phi.T @ Phi) * num_states
      covariance_2 = covariance_1

      # Use all states for weight vector.
      weight_states_1 = oracle_states
      weight_states_2 = weight_states_1
    if method == 'naive':
      # The naive method uses one covariance matrix for both weight vectors.
      covariance_1, key = estimates.naive_inverse_covariance_matrix(
          compute_phi_no_params, sample_states, key, d, covariance_batch_size
      )
      covariance_2 = covariance_1

      weight_states_1, key = sample_states(key, weight_batch_size)
      weight_states_2 = weight_states_1
    elif method == 'naive++':
      # The naive method uses one covariance matrix for both weight vectors.
      covariance_1, key = estimates.naive_inverse_covariance_matrix(
          compute_phi_no_params, sample_states, key, d, covariance_batch_size
      )
      covariance_2, key = estimates.naive_inverse_covariance_matrix(
          compute_phi_no_params, sample_states, key, d, covariance_batch_size
      )

      weight_states_1, key = sample_states(key, weight_batch_size)
      weight_states_2, key = sample_states(key, weight_batch_size)
    elif method == 'lissa':
      # Compute two independent estimates of the inverse covariance matrix.
      covariance_1, key = estimates.lissa_inverse_covariance_matrix(
          compute_phi_no_params,
          sample_states,
          key,
          d,
          covariance_batch_size,
          lissa_kappa,
          feature_norm=feature_norm,
      )
      covariance_2, key = estimates.lissa_inverse_covariance_matrix(
          compute_phi_no_params,
          sample_states,
          key,
          d,
          covariance_batch_size,
          lissa_kappa,
          feature_norm=feature_norm,
      )

      # Draw two separate sets of states for the weight vectors (important!)
      weight_states_1, key = sample_states(key, weight_batch_size)
      weight_states_2, key = sample_states(key, weight_batch_size)

    # Compute the weight estimates by combining the inverse covariance
    # estimate and the sampled Phi & Psi's.
    weight_1 = (
        covariance_1
        @ compute_phi(phi_params, weight_states_1).T  # pytype: disable=wrong-arg-types  # jax-ndarray
        @ compute_psi(weight_states_1, task)
    ) / len(weight_states_1)
    weight_2 = (
        covariance_2
        @ compute_phi(phi_params, weight_states_2).T  # pytype: disable=wrong-arg-types  # jax-ndarray
        @ compute_psi(weight_states_2, task)
    ) / len(weight_states_2)

  prediction = jnp.dot(source_phi, weight_1)
  estimated_error = prediction - compute_psi(source_states, task)

  if use_tabular_gradient:
    # We use the same weight vector to move all elements of our batch, but
    # they have different errors.
    partial_gradient = jnp.reshape(
        jnp.tile(weight_2, main_batch_size), (main_batch_size, d)
    )

    # Line up the shapes of error and weight vectors so we can construct the
    # gradient.
    expanded_estimated_error = jnp.expand_dims(estimated_error, axis=1)
    partial_gradient = partial_gradient * expanded_estimated_error

    # Note: this doesn't work for duplicate indices. However, it shouldn't
    # add any bias to the algorithm, and is faster than checking for
    # duplicate indices. Most of the time we care about the case where our
    # batch size is much smaller than the number of states, so duplicate
    # indices should be rare.
    phi_gradient = jnp.zeros_like(phi_params)
    phi_gradient = phi_gradient.at[source_states, :].set(partial_gradient)
  else:
    # Calculate implicit gradient (Phi @ w_1 - Psi) @ w_2.T
    implicit_gradient = jnp.outer(estimated_error, weight_2)
    # Pullback implicit gradient to get the full Phi gradient.
    (phi_gradient,) = phi_vjp(implicit_gradient)
  gradient = {'phi_params': phi_gradient}

  if method == 'explicit':
    weight_gradient = source_phi.T @ estimated_error
    expanded_gradient = jnp.expand_dims(weight_gradient, axis=1)

    explicit_weight_matrix = params['explicit_weight_matrix']
    explicit_weight_gradient = jnp.zeros_like(explicit_weight_matrix)
    explicit_weight_gradient = explicit_weight_gradient.at[:, task].set(
        expanded_gradient
    )
    gradient['explicit_weight_matrix'] = explicit_weight_gradient

  return gradient, key


@functools.partial(
    jax.jit,
    static_argnames=(
        'compute_phi',
        'compute_psi',
        'optimizer',
        'method',
        'covariance_batch_size',
        'main_batch_size',
        'covariance_batch_size',
        'weight_batch_size',
        'd',
        'num_tasks',
        'compute_feature_norm_on_oracle_states',
        'sample_states',
        'use_tabular_gradient',
    ),
)
def _train_step(
    *,
    compute_phi,
    compute_psi,
    params,
    optimizer,
    optimizer_state,
    key,
    method,
    oracle_states,
    lissa_kappa,
    main_batch_size,
    covariance_batch_size,
    weight_batch_size,
    d,
    num_tasks,
    compute_feature_norm_on_oracle_states,
    sample_states,
    use_tabular_gradient = True,
):
  """Computes one training step.

  Args:
    compute_phi: A function that takes params and states and returns a matrix of
      phis.
    compute_psi: A function implementing a mapping from (state, task) pairs to
      real values. In the finite case, this can be implemented as a function
      that indexes into a matrix. Note: the code does not currently support an
      infinite number of tasks.
    params: Parameters used as the first argument for compute_phi.
    optimizer: An optax optimizer to use.
    optimizer_state: The current state of the optimizer.
    key: The jax prng key.
    method: 'naive', 'lissa', or 'oracle'.
    oracle_states: The set of states to use for the oracle method.
    lissa_kappa: The parameter of the lissa method, if used.
    main_batch_size: How many states to update at once.
    covariance_batch_size: the 'J' parameter. For the naive method, this is how
      many states we sample to construct the inverse. For the lissa method,
      ditto -- these are also "iterations".
    weight_batch_size: How many states to construct the weight vector.
    d: The dimension of the representation.
    num_tasks: The total number of tasks.
    compute_feature_norm_on_oracle_states: If True, computes the feature norm
      using the oracle states (all the states in synthetic experiments).
      Otherwise, computes the norm using the sampled batch. Only applies to
      LISSA.
    sample_states: A function that takes an rng key and a number of states to
      sample, and returns a tuple containing (a vector of sampled states, an
      updated rng key).
    use_tabular_gradient: If true, the train step will calculate the gradient
      using the tabular calculation. Otherwise, it will use a jax.vjp to
      backpropagate the gradient.

  Returns:
    A dict containing updated values for Phi, key,
      and optimizer_state, as well as the the computed gradient.
  """
  source_states_key, task_key, key = jax.random.split(key, num=3)
  source_states, key = sample_states(source_states_key, main_batch_size)
  task = jax.random.choice(task_key, num_tasks, (1,))

  gradient, key = compute_gradient(
      source_states=source_states,
      task=task,
      compute_phi=compute_phi,
      compute_psi=compute_psi,
      params=params,
      key=key,
      method=method,
      oracle_states=oracle_states,
      lissa_kappa=lissa_kappa,
      main_batch_size=main_batch_size,
      covariance_batch_size=covariance_batch_size,
      weight_batch_size=weight_batch_size,
      d=d,
      compute_feature_norm_on_oracle_states=compute_feature_norm_on_oracle_states,
      sample_states=sample_states,
      use_tabular_gradient=use_tabular_gradient,
  )

  updates, optimizer_state = optimizer.update(gradient, optimizer_state)
  params = optax.apply_updates(params, updates)

  return {  # pytype: disable=bad-return-type  # numpy-scalars
      'params': params,
      'key': key,
      'optimizer_state': optimizer_state,
  }


def train(
    *,
    workdir,
    compute_phi,
    compute_psi,
    params,
    optimal_subspace,
    num_epochs,
    learning_rate,
    key,
    method,
    lissa_kappa,
    optimizer,
    covariance_batch_size,
    main_batch_size,
    weight_batch_size,
    d,
    num_tasks,
    compute_feature_norm_on_oracle_states,
    sample_states,
    eval_states,
    use_tabular_gradient = True,
):
  """Training function.

  For lissa, the total number of samples is
  2 x covariance_batch_size + main_batch_size + 2 x weight_batch_size.

  Args:
    workdir: Work directory, where we'll save logs.
    compute_phi: A function that takes params and states and returns a matrix of
      phis.
    compute_psi: A function that takes an array of states and an array of tasks
      and returns Psi[states, tasks].
    params: Parameters used as the first argument for compute_phi.
    optimal_subspace: Top-d left singular vectors of Psi.
    num_epochs: How many gradient steps to perform. (Not really epochs)
    learning_rate: The step size parameter for sgd.
    key: The jax prng key.
    method: 'naive', 'lissa', or 'oracle'.
    lissa_kappa: The parameter of the lissa method, if used.
    optimizer: Which optimizer to use. Only 'sgd' is supported.
    covariance_batch_size: the 'J' parameter. For the naive method, this is how
      many states we sample to construct the inverse. For the lissa method,
      ditto -- these are also "iterations".
    main_batch_size: How many states to update at once.
    weight_batch_size: How many states to construct the weight vector.
    d: The dimension of the representation.
    num_tasks: The total number of tasks.
    compute_feature_norm_on_oracle_states: If True, computes the feature norm
      using the oracle states (all the states in synthetic experiments).
      Otherwise, computes the norm using the sampled batch. Only applies to
      LISSA.
    sample_states: A function that takes an rng key and a number of states to
      sample, and returns a tuple containing (a vector of sampled states, an
      updated rng key).
    eval_states: An array of states to use to compute metrics on. This will be
      used to compute Phi = compute_phi(params, eval_states).
    use_tabular_gradient: If true, the train step will calculate the gradient
      using the tabular calculation. Otherwise, it will use a jax.vjp to
      backpropagate the gradient.
  """
  # Create an explicit weight vector (needed for explicit method only).
  if method == 'explicit':
    key, weight_key = jax.random.split(key)
    explicit_weight_matrix = jax.random.normal(
        weight_key, (d, num_tasks), dtype=jnp.float32
    )
    params['explicit_weight_matrix'] = explicit_weight_matrix

  if optimizer == 'sgd':
    optimizer = optax.sgd(learning_rate)
  elif optimizer == 'adam':
    optimizer = optax.adam(learning_rate)
  else:
    raise ValueError(f'Unknown optimizer {optimizer}.')
  optimizer_state = optimizer.init(params)

  chkpt_manager = checkpoint.Checkpoint(base_directory=_WORKDIR.value)
  initial_step, params, optimizer_state = chkpt_manager.restore_or_initialize(
      (0, params, optimizer_state)
  )

  writer = metric_writers.create_default_writer(
      logdir=str(workdir),
  )

  # Checkpointing and logging too much can use a lot of disk space.
  # Therefore, we don't want to checkpoint more than 10 times an experiment,
  # or keep more than 1k Phis per experiment.
  checkpoint_period = max(num_epochs // 10, 100_000)
  log_period = max(1_000, num_epochs // 1_000)

  def _checkpoint_callback(step, t, params, optimizer_state):
    del t  # Unused.
    chkpt_manager.save((step, params, optimizer_state))

  hooks = [
      periodic_actions.PeriodicCallback(
          every_steps=checkpoint_period, callback_fn=_checkpoint_callback
      )
  ]

  fixed_train_kwargs = {
      'compute_phi': compute_phi,
      'compute_psi': compute_psi,
      'optimizer': optimizer,
      'method': method,
      # In the tabular case, the eval_states are all the states.
      'oracle_states': eval_states,
      'lissa_kappa': lissa_kappa,
      'main_batch_size': main_batch_size,
      'covariance_batch_size': covariance_batch_size,
      'weight_batch_size': weight_batch_size,
      'd': d,
      'num_tasks': num_tasks,
      'compute_feature_norm_on_oracle_states': (
          compute_feature_norm_on_oracle_states
      ),
      'sample_states': sample_states,
      'use_tabular_gradient': use_tabular_gradient,
  }
  variable_kwargs = {
      'params': params,
      'optimizer_state': optimizer_state,
      'key': key,
  }

  @jax.jit
  def _eval_step(phi_params):
    eval_phi = compute_phi(phi_params, eval_states)
    eval_psi = compute_psi(eval_states)  # pytype: disable=wrong-arg-count

    metrics = compute_metrics(eval_phi, optimal_subspace)
    metrics |= {'frob_norm': utils.outer_objective_mc(eval_phi, eval_psi)}
    return metrics

  # Perform num_epochs gradient steps.
  with metric_writers.ensure_flushes(writer):
    for step in etqdm.tqdm(
        range(initial_step + 1, num_epochs + 1),
        initial=initial_step,
        total=num_epochs,
    ):
      variable_kwargs = _train_step(**fixed_train_kwargs, **variable_kwargs)

      if step % log_period == 0:
        metrics = _eval_step(variable_kwargs['params']['phi_params'])
        writer.write_scalars(step, metrics)

      for hook in hooks:
        hook(
            step,
            params=variable_kwargs['params'],
            optimizer_state=variable_kwargs['optimizer_state'],
        )

  writer.flush()


def main(_):
  config: config_dict.ConfigDict = _CONFIG.value
  logging.info(config)

  if config.suite == 'synthetic':
    experiment = utils.create_synthetic_experiment(config)
    compute_feature_norm_on_oracle_states = True
  elif config.suite == 'puddle_world':
    experiment = utils.create_puddle_world_experiment(config)
    compute_feature_norm_on_oracle_states = False
  else:
    raise ValueError(f'Unknown experiment suite {config.suite}.')

  workdir = epath.Path(_WORKDIR.value)
  workdir.mkdir(exist_ok=True)

  train(  # pytype: disable=wrong-arg-types  # jax-ndarray
      workdir=workdir,
      compute_phi=experiment.compute_phi,
      compute_psi=experiment.compute_psi,
      params={'phi_params': experiment.params},
      optimal_subspace=experiment.optimal_subspace,
      num_epochs=config.num_epochs,
      learning_rate=config.lr,
      key=experiment.key,
      method=config.method,
      lissa_kappa=config.kappa,
      optimizer=config.optimizer,
      covariance_batch_size=config.covariance_batch_size,
      main_batch_size=config.main_batch_size,
      weight_batch_size=config.weight_batch_size,
      d=config.d,
      num_tasks=config.T,
      compute_feature_norm_on_oracle_states=(
          compute_feature_norm_on_oracle_states
      ),
      sample_states=experiment.sample_states,
      eval_states=experiment.eval_states,
      use_tabular_gradient=config.use_tabular_gradient,
  )


if __name__ == '__main__':
  app.run(main)
