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

"""Definition of models."""

import abc
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


@jax.jit
def inverse_leaky_relu(y):
  """Inverse of the default jax.nn.leaky_relu."""
  alpha = jnp.where(y > 0, 1, 0.01)
  return y / alpha


@jax.jit
def inverse_softplus(y):
  """Inverse of jax.nn.softplus, adapted from TensorFlow Probability."""
  threshold = jnp.log(jnp.finfo(jnp.float32).eps) + 2.
  is_too_small = y < jnp.exp(threshold)
  is_too_large = y > -threshold
  too_small_value = jnp.log(y)
  too_large_value = y
  y = jnp.where(is_too_small | is_too_large, 1., y)
  x = y + jnp.log(-jnp.expm1(-y))
  return jnp.where(is_too_small, too_small_value,
                   jnp.where(is_too_large, too_large_value, x))


_CUSTOM_ACTIVATIONS = {
    'leaky_relu': jax.nn.leaky_relu,
    'softplus': jax.nn.softplus,
}


_INVERSE_CUSTOM_ACTIVATIONS = {
    'leaky_relu': inverse_leaky_relu,
    'softplus': inverse_softplus,
}


def create_activation_bijector(activation):
  """Creates a bijector for the given activation function."""
  if activation in _CUSTOM_ACTIVATIONS:
    return distrax.Lambda(
        forward=_CUSTOM_ACTIVATIONS[activation],
        inverse=_INVERSE_CUSTOM_ACTIVATIONS[activation])

  activation_fn = getattr(jax.nn, activation, None)
  return distrax.as_bijector(activation_fn)


def cartesian_to_polar(x,
                       y):
  """Converts cartesian (x, y) coordinates to polar (r, theta) coordinates."""
  r = jnp.sqrt(x**2 + y**2)
  theta = jnp.arctan2(y, x)
  return r, theta


def polar_to_cartesian(r,
                       theta):
  """Converts polar (r, theta) coordinates to cartesian (x, y) coordinates."""
  x = r * jnp.cos(theta)
  y = r * jnp.sin(theta)
  return x, y


class MLP(nn.Module):
  """A multi-layer perceptron (MLP)."""

  latent_sizes: Sequence[int]
  activation: Optional[Callable[[chex.Array], chex.Array]]
  skip_connections: bool = True
  activate_final: bool = False

  @nn.compact
  def __call__(self, inputs):
    for index, dim in enumerate(self.latent_sizes):
      next_inputs = nn.Dense(dim)(inputs)

      if index != len(self.latent_sizes) - 1 or self.activate_final:
        if self.activation is not None:
          next_inputs = self.activation(next_inputs)

      if self.skip_connections and next_inputs.shape == inputs.shape:
        next_inputs = next_inputs + inputs

      inputs = next_inputs
    return inputs


class NormalizingFlow(abc.ABC, nn.Module):
  """Base class for normalizing flows."""

  @abc.abstractmethod
  def forward(self, inputs):
    """Computes the forward map."""

  @abc.abstractmethod
  def inverse(self, inputs):
    """Computes the inverse map."""

  def __call__(self, inputs):
    return self.forward(inputs)


class MaskedCouplingFlowConditioner(nn.Module):
  """Conditioner for the masked coupling normalizing flow."""

  event_shape: Sequence[int]
  latent_sizes: Sequence[int]
  activation: Callable[[chex.Array], chex.Array]
  num_bijector_params: int

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
    inputs = MLP(
        self.latent_sizes, self.activation, activate_final=True)(
            inputs)
    inputs = nn.Dense(np.prod(self.event_shape) * self.num_bijector_params)(
        inputs)
    inputs = jnp.reshape(
        inputs, inputs.shape[:-1] + tuple(self.event_shape) +
        (self.num_bijector_params,))
    return inputs


class MaskedCouplingNormalizingFlow(NormalizingFlow):
  """Implements a masked coupling normalizing flow."""

  event_shape: Sequence[int]
  bijector_fn: Callable[[optax.Params], distrax.Bijector]
  conditioners: Sequence[MaskedCouplingFlowConditioner]

  def setup(self):
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(self.event_shape)) % 2
    mask = jnp.reshape(mask, self.event_shape)
    mask = mask.astype(bool)

    layers = []
    for conditioner in self.conditioners:
      layer = distrax.MaskedCoupling(
          mask=mask, bijector=self.bijector_fn, conditioner=conditioner)
      layers.append(layer)

      # Flip the mask after each layer.
      mask = jnp.logical_not(mask)

    # Chain layers to create the flow.
    self.flow = distrax.Chain(layers)

  def forward(self, inputs):
    """Encodes inputs as latent vectors."""
    return self.flow.forward(inputs)

  def inverse(self, inputs):
    """Applies the inverse flow to the latents."""
    return self.flow.inverse(inputs)


class OneDimensionalNormalizingFlow(NormalizingFlow):
  """Implements a one-dimensional normalizing flow."""

  num_layers: int
  activation: distrax.Bijector

  def setup(self):
    layers = []
    for index in range(self.num_layers):
      scale = self.param(f'scale_{index}', nn.initializers.lecun_normal(),
                         (1, 1))
      shift = self.param(f'shift_{index}', nn.initializers.lecun_normal(),
                         (1, 1))
      layer = distrax.ScalarAffine(scale=scale, shift=shift)
      layer = distrax.Chain([self.activation, layer])
      layers.append(layer)

    self.flow = distrax.Chain(layers)

  def inverse(self, inputs):
    """Applies the inverse flow to the latents."""
    return self.flow.inverse(inputs)

  def forward(self, inputs):
    """Encodes inputs as latent vectors."""
    return self.flow.forward(inputs)


class PointwiseNormalizingFlow(NormalizingFlow):
  """Implements a pointwise symplectic normalizing flow."""

  base_flow: NormalizingFlow
  base_flow_input_dims: int
  switch: bool = False

  def forward(self, inputs):
    num_dims = self.base_flow_input_dims
    assert inputs.shape[
        -1] == 2 * num_dims, f'Got inputs of shape {inputs.shape} for num_dims = {num_dims}.'

    first_coords = inputs[Ellipsis, :num_dims]
    second_coords = inputs[Ellipsis, num_dims:]

    if self.switch:
      first_coords, second_coords = second_coords, first_coords

    def dot_product_for_forward(coords_transformed):
      coords = self.base_flow.inverse(coords_transformed)
      return jnp.dot(coords.squeeze(axis=-1), second_coords.squeeze(axis=-1))  # pytype: disable=bad-return-type  # jnp-type

    first_coords_transformed = self.base_flow.forward(first_coords)
    second_coords_transformed = jax.grad(dot_product_for_forward)(
        first_coords_transformed)

    return jnp.concatenate(
        (first_coords_transformed, second_coords_transformed), axis=-1)

  def inverse(self, inputs):
    num_dims = self.base_flow_input_dims
    assert inputs.shape[
        -1] == 2 * num_dims, f'Got inputs of shape {inputs.shape} for num_dims = {num_dims}.'

    first_coords = inputs[Ellipsis, :num_dims]
    second_coords = inputs[Ellipsis, num_dims:]

    if self.switch:
      first_coords, second_coords = second_coords, first_coords

    def dot_product_for_inverse(coords_inverted):
      coords = self.base_flow.forward(coords_inverted)
      return jnp.dot(coords.squeeze(axis=-1), second_coords.squeeze(axis=-1))  # pytype: disable=bad-return-type  # jnp-type

    first_coords_inverted = self.base_flow.inverse(first_coords)
    second_coords_inverted = jax.grad(dot_product_for_inverse)(
        first_coords_inverted)

    return jnp.concatenate((first_coords_inverted, second_coords_inverted),
                           axis=-1)


class LinearBasedConditioner(nn.Module):
  """Linear module from SympNets."""

  @nn.compact
  def __call__(self, inputs):
    num_dims = inputs.shape[-1]
    w = self.param('w', nn.initializers.normal(0.01), (num_dims, num_dims))
    return inputs @ (w + w.T)


class ActivationBasedConditioner(nn.Module):
  """Activation module from SympNets."""

  activation: Callable[[chex.Array], chex.Array]

  @nn.compact
  def __call__(self, inputs):
    num_dims = inputs.shape[-1]
    a = self.param('a', nn.initializers.normal(0.01), (num_dims,))
    return self.activation(inputs) * a


class GradientBasedConditioner(nn.Module):
  """Gradient module from SympNets."""

  activation: Callable[[chex.Array], chex.Array]
  projection_dims: int
  skip_connections: bool = True

  @nn.compact
  def __call__(self, inputs):
    num_dims = inputs.shape[-1]
    w = self.param('w', nn.initializers.normal(0.01),
                   (num_dims, self.projection_dims))
    b = self.param('b', nn.initializers.normal(0.01), (self.projection_dims,))
    a = self.param('a', nn.initializers.zeros, (self.projection_dims,))
    gate = self.param('gate', nn.initializers.zeros, (num_dims,))

    outputs = self.activation(inputs @ w + b)
    outputs = outputs * a
    outputs = outputs @ w.T
    if self.skip_connections:
      outputs += inputs
    outputs *= gate
    return outputs


class ShearNormalizingFlow(NormalizingFlow):
  """Implements a shearing normalizing flow."""

  conditioner: nn.Module
  conditioner_input_dims: int
  switch: bool = False

  def forward(self, inputs):
    num_dims = self.conditioner_input_dims
    assert inputs.shape[-1] == 2 * num_dims, (
        f'Got inputs of shape {inputs.shape} for num_dims = {num_dims}.')

    first_coords = inputs[Ellipsis, :num_dims]
    second_coords = inputs[Ellipsis, num_dims:]

    if self.switch:
      first_coords, second_coords = second_coords, first_coords

    first_coords_transformed = first_coords + self.conditioner(second_coords)

    if self.switch:
      first_coords_transformed, second_coords = second_coords, first_coords_transformed

    return jnp.concatenate((first_coords_transformed, second_coords), axis=-1)

  def inverse(self, inputs):
    num_dims = self.conditioner_input_dims
    assert inputs.shape[-1] == 2 * num_dims, (
        f'Got inputs of shape {inputs.shape} for num_dims = {num_dims}.')

    first_coords = inputs[Ellipsis, :num_dims]
    second_coords = inputs[Ellipsis, num_dims:]

    if self.switch:
      first_coords, second_coords = second_coords, first_coords

    first_coords_inverted = first_coords - self.conditioner(second_coords)

    if self.switch:
      first_coords_inverted, second_coords = second_coords, first_coords_inverted

    return jnp.concatenate((first_coords_inverted, second_coords), axis=-1)


class SymplecticLinearFlow(NormalizingFlow):
  """Implements a SymplecticLinear layer from 'Learning Symmetries of Classical Integrable Systems', consisting of a shift, scale and rotation."""

  operation_input_dims: int

  def setup(self):
    num_dims = self.operation_input_dims
    self.shift_val = self.param('shift_val', nn.initializers.zeros, (num_dims,))
    self.scale_val = self.param('scale_val', nn.initializers.ones, (num_dims,))
    self.rotate_val = self.param('rotate_val', nn.initializers.zeros,
                                 (num_dims,))

  def extract_coords(self, inputs):
    """Returns the two sets of coordinates."""
    num_dims = self.operation_input_dims
    assert inputs.shape[
        -1] == 2 * num_dims, f'Got inputs of shape {inputs.shape} for num_dims = {num_dims}.'

    first_coords = inputs[Ellipsis, :num_dims]
    second_coords = inputs[Ellipsis, num_dims:]
    return first_coords, second_coords

  def forward(self, inputs):
    """Runs the forward pass."""
    first_coords, second_coords = self.extract_coords(inputs)
    first_coords, second_coords = self.shift(
        first_coords, second_coords, forward=True)
    first_coords, second_coords = self.scale(
        first_coords, second_coords, forward=True)
    first_coords, second_coords = self.rotate(
        first_coords, second_coords, forward=True)
    return jnp.concatenate((first_coords, second_coords), axis=-1)

  def inverse(self, inputs):
    """Computes the inverse of self.forward()."""
    first_coords, second_coords = self.extract_coords(inputs)
    first_coords, second_coords = self.rotate(
        first_coords, second_coords, forward=False)
    first_coords, second_coords = self.scale(
        first_coords, second_coords, forward=False)
    first_coords, second_coords = self.shift(
        first_coords, second_coords, forward=False)
    return jnp.concatenate((first_coords, second_coords), axis=-1)

  def shift(self, first_coords, second_coords,
            forward):
    """Performs the shift transformation on one of the coordinates."""
    shift = self.shift_val
    if not forward:
      shift = -shift
    first_coords_shifted = first_coords + second_coords * shift
    second_coords_shifted = second_coords
    return first_coords_shifted, second_coords_shifted

  def scale(self, first_coords, second_coords,
            forward):
    """Scales the coordinates in a symplectic manner."""
    scale = self.scale_val
    if not forward:
      scale = 1 / scale
    first_coords_scaled = first_coords * scale
    second_coords_scaled = second_coords / scale
    return first_coords_scaled, second_coords_scaled

  def rotate(self, first_coords, second_coords,
             forward):
    """Rotates all of the coordinates."""
    theta = self.rotate_val
    if not forward:
      theta = -theta
    first_coords_rotated = first_coords * jnp.cos(
        theta) - second_coords * jnp.sin(theta)
    second_coords_rotated = first_coords * jnp.sin(
        theta) + second_coords * jnp.cos(theta)
    return first_coords_rotated, second_coords_rotated


class SequentialFlow(NormalizingFlow):
  """Adaptation of nn.Sequential() for flows."""

  flows: Sequence[NormalizingFlow]

  def forward(self, inputs):
    for flow in self.flows:
      inputs = flow.forward(inputs)
    return inputs

  def inverse(self, inputs):
    for flow in reversed(self.flows):
      inputs = flow.inverse(inputs)
    return inputs

  def __call__(self, inputs):
    return self.forward(inputs)


class CoordinateEncoder(abc.ABC, nn.Module):
  """Base class for encoders."""

  @abc.abstractmethod
  def __call__(self, positions,
               momentums):
    """Returns corresponding angles and momentums by encoding inputs."""


class CoordinateDecoder(abc.ABC, nn.Module):
  """Base class for decoders."""

  @abc.abstractmethod
  def __call__(self, actions,
               angles):
    """Returns corresponding positions and momentums by decoding inputs."""


class MLPEncoder(CoordinateEncoder):
  """MLP-based encoder."""

  position_encoder: nn.Module
  momentum_encoder: nn.Module
  transform_fn: nn.Module
  latent_position_decoder: nn.Module
  latent_momentum_decoder: nn.Module

  def __call__(self, positions,
               momentums):
    # Encode input coordinates.
    positions = self.position_encoder(positions)
    momentums = self.momentum_encoder(momentums)

    # Transform to new coordinates.
    coords = jnp.concatenate([positions, momentums], axis=-1)
    coords = self.transform_fn(coords)

    # Decode to final coordinates.
    latent_positions = self.latent_position_decoder(coords)
    latent_momentums = self.latent_momentum_decoder(coords)
    return latent_positions, latent_momentums


class MLPDecoder(CoordinateDecoder):
  """MLP-based decoder."""

  latent_position_encoder: nn.Module
  latent_momentum_encoder: nn.Module
  transform_fn: nn.Module
  position_decoder: nn.Module
  momentum_decoder: nn.Module

  def __call__(self, latent_positions,
               latent_momentums):
    # Encode input coordinates.
    latent_positions = self.latent_position_encoder(latent_positions)
    latent_momentums = self.latent_momentum_encoder(latent_momentums)

    # Transform to new coordinates.
    coords = jnp.concatenate([latent_positions, latent_momentums], axis=-1)
    coords = self.transform_fn(coords)

    # Decode to final coordinates.
    positions = self.position_decoder(coords)
    momentums = self.momentum_decoder(coords)
    return positions, momentums


class FlowEncoder(CoordinateEncoder):
  """Flow-based encoder for the Action-Angle Neural Network."""

  flow: NormalizingFlow

  def __call__(self, positions,
               momentums):
    # Pass through forward flow to obtain latent positions and momentums.
    coords = jnp.concatenate([positions, momentums], axis=-1)
    coords = self.flow.forward(coords)

    assert len(coords.shape) == 2, coords.shape
    assert coords.shape[-1] % 2 == 0, coords.shape

    num_positions = coords.shape[-1] // 2
    latent_positions = coords[Ellipsis, :num_positions]
    latent_momentums = coords[Ellipsis, num_positions:]
    return latent_positions, latent_momentums


class FlowDecoder(CoordinateDecoder):
  """Flow-based decoder for the Action-Angle Neural Network."""

  flow: NormalizingFlow

  def __call__(self, latent_positions,
               latent_momentums):
    # Pass through inverse flow to obtain positions and momentums.
    coords = jnp.concatenate([latent_positions, latent_momentums], axis=-1)
    coords = self.flow.inverse(coords)

    assert len(coords.shape) == 2, coords.shape
    assert coords.shape[-1] % 2 == 0, coords.shape

    num_positions = coords.shape[-1] // 2
    positions = coords[Ellipsis, :num_positions]
    momentums = coords[Ellipsis, num_positions:]
    return positions, momentums


class ActionAngleNetwork(nn.Module):
  """Implementation of an Action-Angle Neural Network."""

  encoder: CoordinateEncoder
  angular_velocity_net: nn.Module
  decoder: CoordinateDecoder
  polar_action_angles: bool
  single_step_predictions: bool = True

  def predict_single_step(
      self, positions, momentums,
      time_deltas
  ):
    """Predicts future coordinates with one-step prediction."""
    time_deltas = jnp.squeeze(time_deltas)
    time_deltas = jnp.expand_dims(time_deltas, axis=range(time_deltas.ndim, 2))
    assert time_deltas.ndim == 2

    # Encode.
    current_latent_positions, current_latent_momentums = self.encoder(
        positions, momentums)
    if self.polar_action_angles:
      actions, current_angles = jax.vmap(cartesian_to_polar)(
          current_latent_positions, current_latent_momentums)
    else:
      actions, current_angles = current_latent_positions, current_latent_momentums

    # Compute angular velocities.
    angular_velocities = self.angular_velocity_net(actions)
    assert angular_velocities.shape[-1] == current_angles.shape[-1]

    # Fast-forward.
    future_angles = current_angles + angular_velocities * time_deltas
    if self.polar_action_angles:
      future_angles = (future_angles + jnp.pi) % (2 * jnp.pi) - (jnp.pi)

    # Decode.
    if self.polar_action_angles:
      future_latent_positions, future_latent_momentums = jax.vmap(
          polar_to_cartesian)(actions, future_angles)
    else:
      future_latent_positions, future_latent_momentums = actions, future_angles
    predicted_positions, predicted_momentums = self.decoder(
        future_latent_positions, future_latent_momentums)

    return predicted_positions, predicted_momentums, dict(
        current_latent_positions=current_latent_positions,
        current_latent_momentums=current_latent_momentums,
        actions=actions,
        current_angles=current_angles,
        angular_velocities=angular_velocities,
        future_angles=future_angles,
        future_latent_positions=future_latent_positions,
        future_latent_momentums=future_latent_momentums)

  def predict_multi_step(
      self, init_positions, init_momentums,
      time_deltas
  ):
    """Predicts future coordinates with multi-step prediction."""
    time_deltas = jnp.expand_dims(time_deltas, axis=range(time_deltas.ndim, 2))
    assert time_deltas.ndim == 2

    # init_positions and init_positions have shape [1 x num_trajectories].
    assert len(init_positions.shape) == 2
    assert len(init_momentums.shape) == 2
    assert init_positions.shape[0] == 1
    assert init_momentums.shape[0] == 1

    # Encode.
    current_latent_positions, current_latent_momentums = self.encoder(
        init_positions, init_momentums)
    if self.polar_action_angles:
      actions, current_angles = jax.vmap(cartesian_to_polar)(
          current_latent_positions, current_latent_momentums)
    else:
      actions, current_angles = current_latent_positions, current_latent_momentums

    # Compute angular velocities.
    angular_velocities = self.angular_velocity_net(actions)
    assert angular_velocities.shape[-1] == current_angles.shape[-1]

    # Fast-forward.
    future_angles = current_angles + angular_velocities * time_deltas

    # actions has shape [1 x num_trajectories].
    # future_angles has shape [T x num_trajectories].
    if self.polar_action_angles:
      future_angles = (future_angles + jnp.pi) % (2 * jnp.pi) - (jnp.pi)
      future_latent_positions, future_latent_momentums = jax.vmap(
          polar_to_cartesian, in_axes=(None, 0))(actions[0], future_angles)
    else:
      future_latent_positions, future_latent_momentums = jax.vmap(
          lambda x, y: (x, y), in_axes=(None, 0))(actions[0], future_angles)

    # predicted_positions has shape [T x num_trajectories].
    # predicted_momentums has shape [T x num_trajectories].
    predicted_positions, predicted_momentums = self.decoder(
        future_latent_positions, future_latent_momentums)

    return predicted_positions, predicted_momentums, dict(
        current_latent_positions=current_latent_positions,
        current_latent_momentums=current_latent_momentums,
        actions=actions,
        current_angles=current_angles,
        angular_velocities=angular_velocities,
        future_angles=future_angles,
        future_latent_positions=future_latent_positions,
        future_latent_momentums=future_latent_momentums)

  def encode_decode(self, positions,
                    momentums):
    """Encodes and decodes the given coordinates."""
    actions, current_angles = self.encoder(positions, momentums)
    return self.decoder(actions, current_angles)

  def __call__(
      self, positions, momentums,
      time_deltas
  ):
    time_deltas = jnp.asarray(time_deltas)
    if self.single_step_predictions or time_deltas.ndim == 0:
      return self.predict_single_step(positions, momentums, time_deltas)
    return self.predict_multi_step(positions, momentums, time_deltas)


class EulerUpdateNetwork(nn.Module):
  """A neural network that performs Euler updates with predicted position and momentum derivatives."""

  encoder: CoordinateEncoder
  derivative_net: nn.Module
  decoder: CoordinateDecoder

  @nn.compact
  def __call__(self, positions, momentums,
               time_delta):
    # Encode.
    positions, momentums = self.encoder(positions, momentums)

    # Predict derivatives for each coordinate.
    coords = jnp.concatenate([positions, momentums], axis=-1)
    derivatives = self.derivative_net(coords)

    # Unpack.
    num_positions = derivatives.shape[-1] // 2
    position_derivative = derivatives[Ellipsis, :num_positions]
    momentum_derivative = derivatives[Ellipsis, num_positions:]

    # Perform Euler update.
    predicted_positions = positions + position_derivative * time_delta
    predicted_momentums = momentums + momentum_derivative * time_delta

    # Decode.
    predicted_positions, predicted_momentums = self.decoder(
        predicted_positions, predicted_momentums)
    return predicted_positions, predicted_momentums, None
