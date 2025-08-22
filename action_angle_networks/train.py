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

"""Training code."""

import functools
import logging
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import chex
from clu import metric_writers
from clu import parameter_overview
import distrax
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax

from action_angle_networks import chm_simulation
from action_angle_networks import models
from action_angle_networks import orbit_simulation
from action_angle_networks import scalers
from action_angle_networks import shm_simulation


def get_generate_canonical_coordinates_fn(
    config
):
  """Returns a function that creates trajectories of shape [num_samples, num_trajectories]."""
  if config.simulation == 'shm':
    return jax.jit(
        jax.vmap(
            jax.vmap(
                shm_simulation.generate_canonical_coordinates,
                in_axes=(0, None)),
            in_axes=(None, 0),
            out_axes=1))

  if config.simulation == 'chm':
    return jax.jit(
        jax.vmap(
            chm_simulation.generate_canonical_coordinates,
            in_axes=(0, None),
            out_axes=0))

  if config.simulation == 'orbit':
    return jax.jit(
        jax.vmap(
            orbit_simulation.generate_canonical_coordinates,
            in_axes=(0, None),
            out_axes=0))

  raise ValueError('Unsupported simulation: {config.simulation}.')


def get_compute_hamiltonian_fn(
    config
):
  """Returns a function that computes the Hamiltonian over trajectories of shape [num_samples, num_trajectories]."""
  if config.simulation == 'shm':
    return jax.jit(
        jax.vmap(shm_simulation.compute_hamiltonian, in_axes=(0, 0, None)))

  if config.simulation == 'chm':
    return jax.jit(
        jax.vmap(chm_simulation.compute_hamiltonian, in_axes=(0, 0, None)))

  if config.simulation == 'orbit':
    return jax.jit(
        jax.vmap(orbit_simulation.compute_hamiltonian, in_axes=(0, 0, None)))

  raise ValueError('Unsupported simulation: {config.simulation}.')


def create_masked_coupling_flow(
    config,
    init_shape):
  """Builds the normalizing flow model."""
  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.
  def bijector_fn(params):
    return distrax.RationalQuadraticSpline(
        params,
        range_min=config.spline_range_min,
        range_max=config.spline_range_max)

  num_bijector_params = 3 * config.spline_bins + 1

  # Construct activation.
  activation = getattr(jax.nn, config.activation, None)
  conditioners = []
  for _ in range(config.num_flow_layers):
    conditioner = models.MaskedCouplingFlowConditioner(
        event_shape=init_shape[-1],
        latent_sizes=config.latent_sizes,
        activation=activation,
        num_bijector_params=num_bijector_params)
    conditioners.append(conditioner)

  return models.MaskedCouplingNormalizingFlow(
      event_shape=init_shape[-1],
      bijector_fn=bijector_fn,
      conditioners=conditioners)


def create_pointwise_flow(config,
                          init_shape):
  """Builds the pointwise normalizing flow model."""
  activation = models.create_activation_bijector(config.activation)
  base_flow_input_dims = init_shape[-1] // 2
  base_flow = models.OneDimensionalNormalizingFlow(
      num_layers=config.num_flow_layers, activation=activation)
  return models.PointwiseNormalizingFlow(
      base_flow=base_flow, base_flow_input_dims=base_flow_input_dims)


def create_shear_flow(config,
                      init_shape):
  """Builds the shear normalizing flow model."""
  linear_input_dims = conditioner_input_dims = init_shape[-1] // 2
  activation = getattr(jax.nn, config.activation, None)
  switch = False
  flows = [models.SymplecticLinearFlow(linear_input_dims)]
  for _ in range(config.num_flow_layers):
    conditioner = models.GradientBasedConditioner(
        projection_dims=config.latent_size, activation=activation)
    shear_flow = models.ShearNormalizingFlow(
        conditioner=conditioner,
        conditioner_input_dims=conditioner_input_dims,
        switch=switch)
    flows.append(shear_flow)
    switch = not switch
  return models.SequentialFlow(flows)


def create_flow(config,
                init_shape):
  """Creates the flow model."""
  if config.flow_type == 'shear':
    return create_shear_flow(config, init_shape)
  if config.flow_type == 'pointwise':
    return create_pointwise_flow(config, init_shape)
  if config.flow_type == 'masked-coupling':
    return create_masked_coupling_flow(config, init_shape)
  raise ValueError(f'Unsupported flow type: {config.flow_type}.')


def create_scaler(config):
  """Constructs the scaler."""
  if config.scaler == 'standard':
    return scalers.StandardScaler()
  if config.scaler == 'identity':
    return scalers.IdentityScaler()
  raise ValueError(f'Unsupported scaler: {config.scaler}.')


def create_model(config):
  """Creates the model."""
  activation = getattr(jax.nn, config.activation, None)
  latent_size = config.latent_size

  if config.encoder_decoder_type == 'mlp':
    encoder = models.MLPEncoder(
        position_encoder=models.MLP([latent_size],
                                    activation,
                                    skip_connections=False),
        momentum_encoder=models.MLP([latent_size],
                                    activation,
                                    skip_connections=False),
        transform_fn=models.MLP([latent_size],
                                activation,
                                skip_connections=True),
        latent_position_decoder=models.MLP([latent_size, 1],
                                           activation,
                                           skip_connections=True),
        latent_momentum_decoder=models.MLP([latent_size, 1],
                                           activation,
                                           skip_connections=True),
        name='encoder')
    decoder = models.MLPDecoder(
        latent_position_encoder=models.MLP([latent_size],
                                           activation,
                                           skip_connections=False),
        latent_momentum_encoder=models.MLP([latent_size],
                                           activation,
                                           skip_connections=False),
        transform_fn=models.MLP([latent_size],
                                activation,
                                skip_connections=True),
        position_decoder=models.MLP([latent_size, 1],
                                    activation,
                                    skip_connections=True),
        momentum_decoder=models.MLP([latent_size, 1],
                                    activation,
                                    skip_connections=True),
        name='decoder')
  if config.encoder_decoder_type == 'flow':
    flow = create_flow(  # pytype: disable=wrong-arg-types  # numpy-scalars
        config, init_shape=(config.batch_size, 2 * config.num_trajectories))
    encoder = models.FlowEncoder(flow)
    decoder = models.FlowDecoder(flow)

  # Action-Angle Neural Network.
  if config.model == 'action-angle-network':
    return models.ActionAngleNetwork(
        encoder=encoder,
        angular_velocity_net=models.MLP([latent_size, config.num_trajectories],
                                        activation,
                                        skip_connections=True,
                                        name='angular_velocity_net'),
        decoder=decoder,
        polar_action_angles=config.polar_action_angles,
        single_step_predictions=config.single_step_predictions)

  # Baseline Euler Update Network.
  if config.model == 'euler-update-network':
    return models.EulerUpdateNetwork(
        encoder=encoder,
        derivative_net=models.MLP([latent_size, latent_size, latent_size, 2],
                                  activation,
                                  skip_connections=True,
                                  name='derivative'),
        decoder=decoder)

  raise ValueError('Unsupported model.')


def create_train_state(
    config, rng,
    init_samples):
  """Creates the training state."""
  model = create_model(config)
  params = model.init(rng, *init_samples)
  tx = optax.adam(learning_rate=config.learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def compute_predictions(
    state, positions, momentums,
    time_deltas):
  """Computes model predictions."""
  return state.apply_fn(state.params, positions, momentums, time_deltas)


@jax.jit
def compute_loss(predicted_positions,
                 predicted_momentums,
                 target_positions,
                 target_momentums,
                 auxiliary_predictions = None,
                 regularizations = None):
  """Computes the loss for the given predictions."""
  assert predicted_positions.shape == target_positions.shape, f'Got predicted_positions: {predicted_positions.shape}, target_positions: {target_positions.shape}'
  assert predicted_momentums.shape == target_momentums.shape, f'Got predicted_momentums: {predicted_momentums.shape}, target_momentums: {target_momentums.shape}'

  loss = optax.l2_loss(
      predictions=predicted_positions, targets=target_positions)
  loss += optax.l2_loss(
      predictions=predicted_momentums, targets=target_momentums)
  loss = jnp.mean(loss)

  if auxiliary_predictions is not None:
    angular_velocities = auxiliary_predictions['angular_velocities']
    angular_velocities_variances = jnp.var(angular_velocities, axis=0).sum()
    loss += regularizations['angular_velocities'] * angular_velocities_variances

    actions = auxiliary_predictions['actions']
    actions_variances = jnp.var(actions, axis=0).sum()
    loss += regularizations['actions'] * actions_variances
  return loss  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def compute_updates(
    state,
    positions,
    momentums,
    time_deltas,
    target_positions,
    target_momentums,
    regularizations,
    compute_encoded_decoded_loss = False):
  """Computes gradients for a single batch."""

  def loss_fn(params):
    curr_state = state.replace(params=params)
    predicted_positions, predicted_momentums, auxiliary_predictions = compute_predictions(
        curr_state, positions, momentums, time_deltas)
    regularized_prediction_loss = compute_loss(
        predicted_positions, predicted_momentums, target_positions,
        target_momentums, auxiliary_predictions, regularizations)
    total_loss = regularized_prediction_loss

    if compute_encoded_decoded_loss:
      encoded_decoded_positions, encoded_decoded_momentums, _ = compute_predictions(
          curr_state, positions, momentums, 0.)
      encoded_decoded_loss = compute_loss(encoded_decoded_positions,
                                          encoded_decoded_momentums, positions,
                                          momentums)
      total_loss += regularizations[
          'encoded_decoded_differences'] * encoded_decoded_loss

    return total_loss

  return jax.grad(loss_fn)(state.params)


@functools.partial(jax.jit, static_argnames='compute_hamiltonian_fn')
def compute_mean_change_in_hamiltonians(
    curr_positions, curr_momentums,
    predicted_positions, predicted_momentums,
    compute_hamiltonian_fn, simulation_parameters):
  """Computes the mean change in Hamiltonian for current coordinates versus predicted coordinates."""
  curr_hamiltonians = compute_hamiltonian_fn(curr_positions, curr_momentums,
                                             simulation_parameters)
  predicted_hamiltonians = compute_hamiltonian_fn(predicted_positions,
                                                  predicted_momentums,
                                                  simulation_parameters)
  return jnp.mean(jnp.abs(curr_hamiltonians - predicted_hamiltonians))  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def compute_metrics(
    predicted_positions,
    predicted_momentums,
    target_positions,
    target_momentums,
    current_positions,
    current_momentums,
    encoded_decoded_positions,
    encoded_decoded_momentums,
    auxiliary_predictions,
    regularizations,
    compute_encoded_decoded_loss = False):
  """Computes loss and other metrics."""
  prediction_loss = compute_loss(predicted_positions, predicted_momentums,
                                 target_positions, target_momentums)
  regularized_prediction_loss = compute_loss(predicted_positions,
                                             predicted_momentums,
                                             target_positions, target_momentums,
                                             auxiliary_predictions,
                                             regularizations)
  total_loss = regularized_prediction_loss
  if compute_encoded_decoded_loss:
    encoded_decoded_loss = compute_loss(encoded_decoded_positions,
                                        encoded_decoded_momentums,
                                        current_positions, current_momentums)
    total_loss += regularizations[
        'encoded_decoded_differences'] * encoded_decoded_loss
  return {
      'prediction_loss': prediction_loss,
      'regularized_prediction_loss': regularized_prediction_loss,
      'total_loss': total_loss
  }


@functools.partial(jax.jit, static_argnames=['jump', 'compute_hamiltonian_fn'])
def compute_metrics_helper(
    state, positions, momentums,
    jump, time_delta, scaler,
    compute_hamiltonian_fn,
    simulation_parameters,
    regularizations):
  """Helper for compute_metrics that evaluates the current training state."""
  curr_positions, curr_momentums, target_positions, target_momentums = get_coordinates_for_time_jump(
      positions, momentums, jump)
  predicted_positions, predicted_momentums, auxiliary_predictions = compute_predictions(
      state, curr_positions, curr_momentums, jump * time_delta)
  encoded_decoded_positions, encoded_decoded_momentums, _ = compute_predictions(
      state, curr_positions, curr_momentums, 0)
  metrics = compute_metrics(predicted_positions, predicted_momentums,
                            target_positions, target_momentums, curr_positions,
                            curr_momentums, encoded_decoded_positions,
                            encoded_decoded_momentums, auxiliary_predictions,
                            regularizations)

  # Rescale to original units before computing Hamiltonian.
  curr_positions, curr_momentums = inverse_transform_with_scaler(
      curr_positions, curr_momentums, scaler)
  predicted_positions, predicted_momentums = inverse_transform_with_scaler(
      predicted_positions, predicted_momentums, scaler)
  metrics['mean_change_in_hamiltonians'] = compute_mean_change_in_hamiltonians(
      curr_positions, curr_momentums, predicted_positions, predicted_momentums,
      compute_hamiltonian_fn, simulation_parameters)

  return metrics


def log_metrics(step,
                metrics,
                summary_writer,
                prefix = ''):
  """Logs all metrics."""
  # Formatting for accuracy.
  for metric in metrics:
    if 'accuracy' in metric:
      metrics[metric] *= 100

  # Add prefix to all metric names.
  metrics = {
      prefix + metric: metric_val for metric, metric_val in metrics.items()
  }

  for metric, metric_val in metrics.items():
    logging.info('num_steps: % 3d, %s: %.4f', step, metric, metric_val)

  summary_writer.write_scalars(step, metrics)
  summary_writer.flush()


@jax.jit
def fit_scaler(positions, momentums,
               scaler):
  """Fits the scaler."""
  if positions.ndim == 1:
    positions = positions[jnp.newaxis, Ellipsis]
    momentums = momentums[jnp.newaxis, Ellipsis]

  assert positions.ndim == 2, f'Got positions of shape {positions.shape}.'
  assert momentums.ndim == 2, f'Got momentums of shape {momentums.shape}.'

  coords = jnp.concatenate((positions, momentums), axis=1)
  return scaler.fit(coords)


@jax.jit
def transform_with_scaler(
    positions, momentums,
    scaler):
  """Performs the rescaling to normalized coordinates."""
  if positions.ndim == 1:
    positions = positions[jnp.newaxis, Ellipsis]
    momentums = momentums[jnp.newaxis, Ellipsis]

  assert positions.ndim == 2, f'Got positions of shape {positions.shape}.'
  assert momentums.ndim == 2, f'Got momentums of shape {momentums.shape}.'

  coords = jnp.concatenate((positions, momentums), axis=1)
  coords = scaler.transform(coords)
  num_positions = coords.shape[1] // 2
  positions = coords[:, :num_positions]
  momentums = coords[:, num_positions:]
  return positions, momentums


@jax.jit
def inverse_transform_with_scaler(
    positions, momentums,
    scaler):
  """Performs the inverse rescaling to unnormalized coordinates."""
  if positions.ndim == 1:
    positions = positions[jnp.newaxis, Ellipsis]
    momentums = momentums[jnp.newaxis, Ellipsis]

  assert positions.ndim == 2, f'Got positions of shape {positions.shape}.'
  assert momentums.ndim == 2, f'Got momentums of shape {momentums.shape}.'

  coords = jnp.concatenate((positions, momentums), axis=1)
  coords = scaler.inverse_transform(coords)
  num_positions = coords.shape[1] // 2
  positions = coords[:, :num_positions]
  momentums = coords[:, num_positions:]
  return positions, momentums


@functools.partial(jax.jit, static_argnames='num_trajectories')
def sample_simulation_parameters(
    simulation_parameter_ranges,
    num_trajectories,
    rng):
  """Samples simulation parameters."""

  is_tuple = lambda val: isinstance(val, tuple)
  ranges_flat, ranges_treedef = jax.tree.flatten(
      simulation_parameter_ranges, is_leaf=is_tuple)
  rng, shuffle_rng, *rngs = jax.random.split(rng, len(ranges_flat) + 2)
  shuffle_indices = jax.random.permutation(shuffle_rng, num_trajectories)
  rng_tree = jax.tree.unflatten(ranges_treedef, rngs)

  def sample_simulation_parameter(simulation_parameter_range,
                                  parameter_rng):
    """Sample a single simulation parameter."""
    del parameter_rng
    minval, maxval = simulation_parameter_range
    samples = jnp.linspace(minval, maxval, num=num_trajectories)
    return jnp.sort(samples)[shuffle_indices]

  return jax.tree.map(
      sample_simulation_parameter,
      simulation_parameter_ranges,
      rng_tree,
      is_leaf=is_tuple)


@functools.partial(jax.jit, static_argnames='jump')
def get_coordinates_for_time_jump(
    positions, momentums,
    jump):
  """Returns the current and target coordinates for the given jump."""
  # Input arrays are of shape [num_samples, num_trajectories].
  assert positions.ndim == 2, f'Got positions of shape {positions.shape}.'
  assert momentums.ndim == 2, f'Got momentums of shape {momentums.shape}.'

  curr_positions, target_positions = positions[:-jump], positions[jump:]
  curr_momentums, target_momentums = momentums[:-jump], momentums[jump:]
  return curr_positions, curr_momentums, target_positions, target_momentums


@functools.partial(jax.jit, static_argnames='max_jump')
def get_coordinates_for_time_jumps(
    positions, momentums,
    max_jump):
  """Returns the current and target coordinates for jumps upto max_jump."""
  # Input arrays are of shape [num_samples, num_trajectories].
  assert positions.ndim == 2, f'Got positions of shape {positions.shape}.'
  assert momentums.ndim == 2, f'Got momentums of shape {momentums.shape}.'

  def map_to_target_indices(index):
    return index + jnp.arange(1, max_jump + 1)

  num_samples = positions.shape[1]
  curr_indices = jnp.arange(0, num_samples - max_jump)
  target_indices = jax.vmap(map_to_target_indices)(curr_indices)  # pytype: disable=wrong-arg-types  # jnp-type
  curr_positions = positions[curr_indices]
  target_positions = positions[target_indices]
  curr_momentums = momentums[curr_indices]
  target_momentums = momentums[target_indices]
  return curr_positions, curr_momentums, target_positions, target_momentums


def get_coordinates_fn(
    config
):
  """Returns a function that sets up coordinates."""
  # Setup for one-step predictions.
  if config.single_step_predictions:
    return get_coordinates_for_time_jump

  # Setup for multi-step predictions.
  return get_coordinates_for_time_jumps


def sample_constant_time_jump(step, constant_jump,
                              rng):
  """Returns a constant jump size."""
  del step, rng
  return constant_jump


def sample_time_jump_with_linear_increase(step, num_train_steps,
                                          min_jump, max_jump,
                                          rng):
  """Returns a stochastic jump size, with linearly increasing mean."""
  max_time_jump_for_step = min_jump + (step / (num_train_steps - 1)) * (
      max_jump - min_jump)
  max_time_jump_for_step = jnp.round(max_time_jump_for_step)
  jump = jax.random.randint(rng, (), min_jump, max_time_jump_for_step + 1)
  jump = int(jump)
  return jump


def get_time_deltas_fn(
    config):
  """Returns a function that sets up time deltas."""
  # Setup for one-step predictions.
  if config.single_step_predictions:
    return lambda jump: jump * config.time_delta

  # Setup for multi-step predictions.
  return lambda jump: jnp.arange(1, jump + 1) * config.time_delta


@functools.partial(jax.jit, static_argnames='index')
def get_trajectory_with_parameters(
    index, positions, momentums,
    simulation_parameters
):
  """Gets the trajectory and simulation parameters for a particular index."""
  return jax.tree.map(
      lambda arr: arr[index], (positions, momentums, simulation_parameters))


def train_and_evaluate(
    config, workdir
):
  """Performs training and evaluation with the given configuration."""
  # Set up logging.
  summary_writer = metric_writers.create_default_writer(workdir)
  summary_writer.write_hparams(config.to_dict())

  time_delta = config.time_delta
  train_time_jump_range = config.train_time_jump_range
  test_time_jumps = config.test_time_jumps
  num_trajectories = config.num_trajectories
  num_samples = config.num_samples
  train_split_proportion = config.train_split_proportion
  num_train_steps = config.num_train_steps
  regularizations = config.regularizations.to_dict()
  eval_cadence = config.eval_cadence

  # Get simulation functions.
  generate_canonical_coordinates_fn = get_generate_canonical_coordinates_fn(
      config)
  compute_hamiltonian_fn = get_compute_hamiltonian_fn(config)

  # Generate data.
  rng = jax.random.PRNGKey(config.rng_seed)
  rng, simulation_parameters_rng = jax.random.split(rng)
  simulation_parameters = sample_simulation_parameters(
      config.simulation_parameter_ranges.to_dict(), num_trajectories,
      simulation_parameters_rng)
  times = jnp.arange(num_samples) * time_delta
  all_positions, all_momentums = generate_canonical_coordinates_fn(
      times, simulation_parameters)

  # Train-test split.
  if config.split_on == 'times':
    num_train_samples = int(num_samples * train_split_proportion)
    train_positions = all_positions[:num_train_samples]
    test_positions = all_positions[num_train_samples:]
    train_momentums = all_momentums[:num_train_samples]
    test_momentums = all_momentums[num_train_samples:]

    train_simulation_parameters = simulation_parameters
    test_simulation_parameters = simulation_parameters
  else:
    raise ValueError(f'Unsupported feature for split: {config.split_on}.')

  # Rescale.
  scaler = create_scaler(config)
  scaler = fit_scaler(train_positions, train_momentums, scaler)
  train_positions, train_momentums = transform_with_scaler(
      train_positions, train_momentums, scaler)
  test_positions, test_momentums = transform_with_scaler(
      test_positions, test_momentums, scaler)

  # Initialize model.
  state = create_train_state(
      config, rng, (train_positions[:1], train_momentums[:1], time_delta))
  best_state = state
  parameter_overview.log_parameter_overview(state.params)

  # Setup for coordinates and time deltas.
  coordinates_fn = get_coordinates_fn(config)
  time_deltas_fn = get_time_deltas_fn(config)

  # Setup sampling for time jumps.
  sample_time_jump_fn = functools.partial(
      sample_time_jump_with_linear_increase,
      min_jump=train_time_jump_range[0],
      max_jump=train_time_jump_range[1],
      num_train_steps=num_train_steps)

  min_train_loss = jnp.inf
  all_train_metrics = {}
  all_test_metrics = {}

  for step in range(num_train_steps):
    step_rng = jax.random.fold_in(rng, step)

    # Sample time jump.
    step_rng, jump_rng = jax.random.split(step_rng)
    jump = sample_time_jump_fn(step, rng=jump_rng)

    # Setup inputs and targets on all trajectories.
    train_curr_positions, train_curr_momentums, train_target_positions, train_target_momentums = coordinates_fn(
        train_positions, train_momentums, jump)
    time_deltas = time_deltas_fn(jump)

    # Sample indices.
    num_samples_on_trajectory = train_curr_positions.shape[0]
    sample_indices = jax.random.choice(step_rng, num_samples_on_trajectory,
                                       (config.batch_size,))
    batch_curr_positions = train_curr_positions[sample_indices]
    batch_curr_momentums = train_curr_momentums[sample_indices]
    batch_target_positions = train_target_positions[sample_indices]
    batch_target_momentums = train_target_momentums[sample_indices]

    # Update parameters.
    grads = compute_updates(state, batch_curr_positions, batch_curr_momentums,
                            time_deltas, batch_target_positions,
                            batch_target_momentums, regularizations)
    state = state.apply_gradients(grads=grads)

    # Evaluate, if required.
    is_last_step = (step == num_train_steps - 1)
    if step % eval_cadence == (eval_cadence - 1) or is_last_step:
      train_metrics = compute_metrics_helper(
          state, train_positions, train_momentums, jump,
          time_delta, scaler, compute_hamiltonian_fn,
          train_simulation_parameters, regularizations)
      log_metrics(step, train_metrics, summary_writer, prefix='train_')
      all_train_metrics[step] = train_metrics

      test_metrics = {}
      for test_jump in test_time_jumps:
        test_metrics[test_jump] = compute_metrics_helper(
            state, test_positions, test_momentums, test_jump,
            time_delta, scaler, compute_hamiltonian_fn,
            test_simulation_parameters, regularizations)
        log_metrics(
            step,
            test_metrics[test_jump],
            summary_writer,
            prefix=f'test_jump_{test_jump}_')
      all_test_metrics[step] = test_metrics

      # Save best state seen so far.
      if train_metrics['total_loss'] < min_train_loss:
        min_train_loss = train_metrics['total_loss']
        best_state = state

  auxiliary_data = {
      'train': {
          'positions': train_positions,
          'momentums': train_momentums,
          'simulation_parameters': train_simulation_parameters,
          'metrics': all_train_metrics,
      },
      'test': {
          'positions': test_positions,
          'momentums': test_momentums,
          'simulation_parameters': test_simulation_parameters,
          'metrics': all_test_metrics,
      },
  }
  return scaler, best_state, auxiliary_data  # pytype: disable=bad-return-type  # numpy-scalars
