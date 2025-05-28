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

"""Conditional VAE models."""
import functools as ft
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from wildfire_perc_sim import utils

Tensor = jnp.ndarray


class Block(nn.Module):
  """A (non-bottleneck) block for the ResNet module."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  activation: Callable[[Tensor], Tensor] = nn.relu
  normalizer: Callable[Ellipsis, nn.Module] = nn.BatchNorm
  convolution: Callable[Ellipsis, nn.Module] = ft.partial(nn.Conv, use_bias=False)

  @nn.compact
  def __call__(self, x):
    res = x
    x = self.convolution(self.filters, (3, 3), self.strides)(x)
    x = self.normalizer()(x)
    x = self.activation(x)
    x = self.convolution(self.filters, (3, 3))(x)
    x = self.normalizer(scale_init=nn.initializers.zeros)(x)
    if x.shape != res.shape:
      res = self.convolution(self.filters, (1, 1), self.strides)(res)
      res = self.normalizer()(res)
    return self.activation(res + x)


class ResNetEncoder(nn.Module):
  """Resnet Encoder Block."""
  stage_sizes: Sequence[int]
  latent_dim: Optional[int] = None
  num_filters: int = 64
  block_cls: Any = Block
  dtype: Any = jnp.float32
  act: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  conv: Any = nn.Conv
  embedding_dimension: Optional[Tuple[int, int, int]] = None

  @nn.compact
  def __call__(self, x):
    conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = ft.partial(nn.GroupNorm, epsilon=1e-5, dtype=self.dtype)

    x = conv(
        self.num_filters, (7, 7), (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init')(
            x)
    x = norm(num_groups=min(16, x.shape[3]), name='gn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        num_filters = self.num_filters * 2**i
        x = self.block_cls(
            num_filters,
            strides=strides,
            normalizer=ft.partial(norm, num_groups=min(16, num_filters)),
            activation=self.act,
            convolution=conv)(
                x)
    if self.embedding_dimension is not None:
      x = conv(self.embedding_dimension[2], (1, 1))(x)
      x = jax.image.resize(x, (x.shape[0], *self.embedding_dimension),
                           'nearest')
      return x
    elif self.latent_dim is None:
      raise ValueError('If `embedding_dimension` = None, `latent_dim` must be '
                       'definied.')
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.latent_dim, dtype=self.dtype)(x)
    return jnp.asarray(x, self.dtype)


class ResNetDecoder(nn.Module):
  """Resnet Decoder Block."""
  stage_sizes: Any
  image_size: Tuple[int, int, int]
  num_filters: int = 64
  block_cls: Any = Block
  dtype: Any = jnp.float32
  act: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  conv: Any = nn.ConvTranspose

  @nn.compact
  def __call__(self, x):
    conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = ft.partial(nn.GroupNorm, epsilon=1e-5, dtype=self.dtype)

    num_upsamples = len(self.stage_sizes) - 1
    start_image_size = ((self.image_size[0] //
                         (2**num_upsamples), self.image_size[1] //
                         (2**num_upsamples)) + (self.num_filters,))

    x = nn.Dense(np.prod(start_image_size), dtype=self.dtype)(x)
    x = jnp.reshape(x, (x.shape[0],) + start_image_size)
    x = norm(num_groups=min(16, x.shape[3]))(x)
    x = nn.relu(x)

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        num_filters = max(16, self.num_filters //
                          2**i) if (i < len(self.stage_sizes) - 1 or
                                    j < block_size - 1) else self.image_size[2]
        x = self.block_cls(
            num_filters,
            strides=strides,
            normalizer=ft.partial(norm, num_groups=min(16, num_filters)),
            activation=self.act,
            convolution=conv)(
                x)
    return x


class EncoderConvLSTM(nn.Module):
  """Encodes a Sequence of Observations into a Vector using an ConvLSTM."""
  features: int
  kernel_shape: Tuple[int, int]
  memory_shape: Tuple[int, int, int]
  encoding_dim: int

  def setup(self):
    self.convlstm = nn.ConvLSTMCell(self.features, self.kernel_shape)
    self.encoding_layer = nn.Dense(self.encoding_dim)

  def __call__(self, inputs):
    # Assume that batch_size is constant for all the inputs
    batch_size = inputs[0].shape[0]
    memory_shape = (batch_size,) + self.memory_shape
    carry = (jnp.zeros(memory_shape), jnp.zeros(memory_shape))
    for i in range(len(inputs)):
      carry, encoding = self.convlstm(carry, inputs[i])
    encoding = jnp.reshape(encoding, (batch_size, -1))
    encoding = self.encoding_layer(encoding)
    return encoding


class ConditionalEncoderConv(nn.Module):
  """Creates a conditional encoding of `x` with condition `c`."""
  latent_dim: int
  encoder_block: Any

  @nn.compact
  def __call__(self, x, c):
    base_encoding = self.encoder_block()(x)
    y = jnp.concatenate([base_encoding, c], axis=1)
    return nn.Dense(self.latent_dim)(nn.relu(y))


class ConditionalDecoderConv(nn.Module):
  """Conditionally decodes an image `y` from `x` given condition `c`."""
  decoder_block: Any

  @nn.compact
  def __call__(self, x, c):
    x = jnp.concatenate([x, c], axis=1)
    x = nn.Dense(x.shape[1])(x)
    x = nn.relu(x)
    return self.decoder_block()(x)


class ConditionalVAEConv(nn.Module):
  """Conditional Convolutional VAE Model."""
  latent_dim: int
  conditional_encoder_block: Any
  conditional_decoder_block: Any
  conditional_block: Any

  def setup(self):
    self.condition_net = self.conditional_block()
    self.encoder = self.conditional_encoder_block()
    self.decoder = self.conditional_decoder_block()

  def __call__(
      self, x, c, prng
  ):
    cond = self.condition_net(c)
    encoding = self.encoder(x, cond)

    split_dim = self.latent_dim // 2
    mean, logvar = encoding[:, :split_dim], encoding[:, split_dim:]

    prng, key = random.split(prng)
    latent_vector = utils.reparameterize(key, mean, logvar)

    x_reconstructed = self.decoder(latent_vector, cond)

    return x_reconstructed, mean, logvar

  def sample(self, nsamples, c,
             prng):
    prng, key = random.split(prng)
    latent_vector = random.normal(
        key, (nsamples * c[0].shape[0], self.latent_dim // 2))
    cond = jnp.concatenate([self.condition_net(c)] * nsamples, axis=0)
    res = self.decoder(latent_vector, cond)
    return res.reshape((nsamples, -1, *res.shape[1:]))


class ConvolutionOperatorPredictor(nn.Module):
  """Predict a convolution operator for percolation models.

  Attributes:
    num_kernels: Number of Convolution Kernels that need to be Generated.
    window_size: Size of the Convolution Kernel.
    num_channels: Number of Channels of the output after applying the generated
      convolution kernel.
    predictor: Partial Function which takes as input `embedding_dimension`. The
      constructed layer must take a 4D array and return a 4D array of size
      `embedding_dimension`.
  """
  num_kernels: int
  window_size: Tuple[int, int]
  num_channels: int
  predictor: Any

  @nn.compact
  def __call__(self, x):
    kernel = self.predictor(
        embedding_dimension=(*self.window_size, x.shape[3] * self.num_channels *
                             self.num_kernels))(
                                 x)
    kernel = kernel.reshape((x.shape[0], *self.window_size, x.shape[3],
                             self.num_channels, self.num_kernels))
    kernel = jnp.transpose(kernel, (5, 0, 1, 2, 3, 4))
    return kernel


class PercolationPropagator(nn.Module):
  """Use a ConvolutionPredictorOperator to propagate `x` for `unroll_steps`.

  Attributes:
    convolution_operator_predictor: Should be a partial constructor for
      `ConvolutionOperatorPredictor` which takes zero inputs.
    static_kernel: If set to `True`, then the convolution operator is generated
      using `x` and it is reused for every forward propagation. If `False`, the
      operator is regenerated at every step, using the current state `x_t`.
  """
  convolution_operator_predictor: Any
  static_kernel: bool = True

  @nn.compact
  def __call__(
      self, x, unroll_steps
  ):
    assert x.ndim == 4
    op_predictor = self.convolution_operator_predictor()
    kernel = op_predictor(x)
    states = []
    if not self.static_kernel:
      kernels = [kernel]

    for i in range(unroll_steps):
      x = utils.apply_percolation_convolution(kernel, x)
      states.append(x)
      if not self.static_kernel and i != unroll_steps - 1:
        kernel = op_predictor(x)
        kernels.append(kernel)

    if self.static_kernel:
      return states, kernel
    return states, kernels


class StandardPropagator(nn.Module):
  """Propagates a hidden state to generate a sequence of hidden states and observations."""
  observation_predictor: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]]
  hidden_state_predictor: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]]

  @nn.compact
  def __call__(
      self, x,
      unroll_steps):
    assert x.ndim == 4

    observation_predictor = self.observation_predictor()
    hidden_state_predictor = self.hidden_state_predictor()

    hstates, observations = [x], [observation_predictor(x)]

    for _ in range(unroll_steps - 1):
      hidden_state_new = hidden_state_predictor(hstates[-1])
      hstates.append(hidden_state_new)
      observations.append(observation_predictor(hidden_state_new))

    return tuple(hstates), tuple(observations)


class DeterministicPredictorPropagator(nn.Module):
  """Given a sequence of observations predict a sequence of hidden states and observations."""
  observation_channels: int = 2
  latent_dim: int = 128
  field_shape: Tuple[int, int] = (64, 64)
  hidden_state_channels: int = 9
  stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2)
  decoder_num_starting_filters: int = 512
  observation_fn: Optional[Callable[[], nn.Module]] = None
  single_step_fn: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    encoder = EncoderConvLSTM(
        features=self.observation_channels,
        encoding_dim=self.latent_dim,
        memory_shape=(*self.field_shape, self.observation_channels),
        kernel_shape=(3, 3))

    decoder = ResNetDecoder(
        stage_sizes=self.stage_sizes,
        image_size=(*self.field_shape, self.hidden_state_channels),
        num_filters=self.decoder_num_starting_filters,
    )

    self.predictor = nn.Sequential([encoder, decoder])

    if self.observation_fn is None:

      def observation_fn():
        return lambda x: x[Ellipsis, -self.observation_channels:]
    else:
      observation_fn = self.observation_fn

    if self.single_step_fn is None:

      def single_step_fn():
        return nn.Sequential([
            nn.Conv(self.hidden_state_channels * 4, (3, 3), padding='SAME'),
            nn.GroupNorm(num_groups=4),
            nn.relu,
            nn.Conv(self.hidden_state_channels, (3, 3), padding='SAME'),
        ])
    else:
      single_step_fn = self.single_step_fn

    self.propagator = StandardPropagator(
        hidden_state_predictor=single_step_fn,
        observation_predictor=observation_fn,
    )

  def __call__(
      self,
      observation_sequence,
      unroll_steps = None,
      prediction_mode = False
  ):
    if unroll_steps is None:
      unroll_steps = len(observation_sequence)
    predicted_hidden_state = self.predictor(observation_sequence[::-1])
    predicted_hidden_states, predicted_observations = self.propagator(
        predicted_hidden_state, unroll_steps)

    # Observations are binary
    if prediction_mode:
      predicted_observations = [(nn.sigmoid(obs) >= 0.5).astype(jnp.float32)
                                for obs in predicted_observations]

    return predicted_hidden_states, predicted_observations


class VariationalPredictorPropagator(nn.Module):
  """Generative models for predicting hidden state and propagating it."""
  observation_channels: int = 2
  latent_dim: int = 128
  field_shape: Tuple[int, int] = (64, 64)
  hidden_state_channels: int = 9
  stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2)
  decoder_num_starting_filters: int = 512
  observation_fn: Optional[Callable[[], nn.Module]] = None
  single_step_fn: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    conditional_block = ft.partial(
        EncoderConvLSTM,
        features=self.observation_channels,
        encoding_dim=self.latent_dim,
        memory_shape=(*self.field_shape, self.observation_channels),
        kernel_shape=(3, 3))

    encoder_block = ft.partial(
        ResNetEncoder, stage_sizes=self.stage_sizes, latent_dim=self.latent_dim)

    conditional_encoder_block = ft.partial(
        ConditionalEncoderConv,
        latent_dim=self.latent_dim,
        encoder_block=encoder_block)

    decoder_block = ft.partial(
        ResNetDecoder,
        stage_sizes=self.stage_sizes,
        image_size=(*self.field_shape, self.hidden_state_channels),
        num_filters=self.decoder_num_starting_filters,
    )

    conditional_decoder_block = ft.partial(
        ConditionalDecoderConv, decoder_block=decoder_block)

    self.encoder = conditional_encoder_block()
    self.decoder = conditional_decoder_block()
    self.condition_net = conditional_block()

    if self.observation_fn is None:

      def observation_fn():
        return lambda x: x[Ellipsis, -self.observation_channels:]
    else:
      observation_fn = self.observation_fn

    if self.single_step_fn is None:

      def single_step_fn():
        return nn.Sequential([
            nn.Conv(self.hidden_state_channels * 4, (3, 3), padding='SAME'),
            nn.GroupNorm(num_groups=4),
            nn.relu,
            nn.Conv(self.hidden_state_channels, (3, 3), padding='SAME'),
        ])
    else:
      single_step_fn = self.single_step_fn

    self.propagator = StandardPropagator(
        hidden_state_predictor=single_step_fn,
        observation_predictor=observation_fn,
    )

  def __call__(
      self,
      hidden_state,
      observation_sequence,
      key,
      unroll_steps = None
  ):
    if unroll_steps is None:
      unroll_steps = len(observation_sequence)
    cond = self.condition_net(observation_sequence[::-1])
    encoding = self.encoder(hidden_state, cond)

    split_dim = self.latent_dim // 2
    mean, logvar = encoding[:, :split_dim], encoding[:, split_dim:]

    latent_vector = utils.reparameterize(key, mean, logvar)

    predicted_hidden_state = self.decoder(latent_vector, cond)
    # Stopping gradient is the mathematically right thing to do here but
    # it leads to higher observation loss
    predicted_hidden_states, predicted_observations = self.propagator(
        predicted_hidden_state, unroll_steps)

    # predicted_hidden_state == predicted_hidden_states[0] but we need to return
    # it since we stop gradients for the VAE in the propagator call.
    return (predicted_hidden_state, predicted_hidden_states,
            predicted_observations, mean, logvar)

  def sample(
      self, unroll_length, observation_sequence,
      key
  ):
    latent_vector = random.normal(
        key, (observation_sequence[0].shape[0], self.latent_dim // 2))
    cond = self.condition_net(observation_sequence[::-1])
    hstate = self.decoder(latent_vector, cond)

    predicted_hidden_states, predicted_observations = self.propagator(
        hstate, unroll_length)

    # Observations are binary
    predicted_observations = [(nn.sigmoid(obs) >= 0.5).astype(jnp.float32)
                              for obs in predicted_observations]

    return predicted_hidden_states, predicted_observations
