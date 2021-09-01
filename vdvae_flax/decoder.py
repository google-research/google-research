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

"""Implementation of the decoder of a VDVAE."""

import functools
import operator

from typing import Mapping, Optional, Sequence, Tuple

import chex
import distrax
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from vdvae_flax import blocks


@flax.struct.dataclass
class DecoderBlockOutput:
  outputs: chex.Array
  kl: chex.Array


class DecoderBlock(nn.Module):
  """Building block of the decoder."""
  num_channels: int
  bottlenecked_num_channels: int
  latent_dim: int
  weights_scale: float
  upsampling_rate: int = 1
  precision: Optional[jax.lax.Precision] = None

  def _compute_posterior(
      self,
      inputs,
      encoder_outputs,
      context_vectors,
  ):
    """Computes the posterior branch of the DecoderBlock."""
    chex.assert_rank(inputs, 4)
    resolution = inputs.shape[1]
    try:
      encoded_image = encoder_outputs[resolution]
    except KeyError:
      raise KeyError('encoder_outputs does not contain the required '  # pylint: disable=g-doc-exception
                     f'resolution ({resolution}). encoder_outputs resolutions '
                     f'are {list(encoder_outputs.keys())}.')
    posterior_block = blocks.ResBlock(
        self.bottlenecked_num_channels,
        self.latent_dim * 2,
        use_residual_connection=False,
        precision=self.precision,
        name='posterior_block')
    concatenated_inputs = jnp.concatenate([inputs, encoded_image], axis=3)
    posterior_output = posterior_block(concatenated_inputs, context_vectors)
    posterior_mean, posterior_log_std = jnp.split(posterior_output, 2, axis=3)
    return posterior_mean, posterior_log_std

  def _compute_prior_and_features(
      self,
      inputs,
      context_vectors,
  ):
    """Computes the prior branch of the DecoderBlock."""
    chex.assert_rank(inputs, 4)
    prior_block = blocks.ResBlock(
        self.bottlenecked_num_channels,
        self.latent_dim * 2 + self.num_channels,
        use_residual_connection=False,
        last_weights_scale=0.0,
        precision=self.precision,
        name='prior_block')
    prior_output = prior_block(inputs, context_vectors)
    prior_mean = prior_output[Ellipsis, :self.latent_dim]
    prior_log_std = prior_output[Ellipsis, self.latent_dim:self.latent_dim * 2]
    features = prior_output[Ellipsis, self.latent_dim * 2:]

    return prior_mean, prior_log_std, features

  def _compute_outputs(self, inputs, features,
                       latent):
    """Computes the outputs of the DecoderBlock."""
    latent_projection = blocks.get_vdvae_convolution(
        self.num_channels, (1, 1),
        self.weights_scale,
        name='latent_projection',
        precision=self.precision)
    output = inputs + features + latent_projection(latent)
    final_res_block = blocks.ResBlock(
        self.bottlenecked_num_channels,
        self.num_channels,
        use_residual_connection=True,
        last_weights_scale=self.weights_scale,
        precision=self.precision,
        name='final_residual_block')
    return final_res_block(output)

  @nn.compact
  def __call__(
      self,
      inputs,
      sample_rng,
      context_vectors = None,
      encoder_outputs = None,
      temperature = 1.,
  ):
    """Evaluates the DecoderBlock.

    Args:
      inputs: a batch of input images of shape [B, H, W, C], where H=W is the
        resolution, and C matches the number of channels of the DecoderBlock.
      sample_rng: random key for sampling.
      context_vectors: optional batch of shape [B, D]. These are typically used
        to condition the VDVAE.
      encoder_outputs: a mapping from resolution to encoded images corresponding
        to the output of an Encoder. This mapping should contain the resolution
        of `inputs`. For each resolution R in encoder_outputs, the corresponding
        value has shape [B, R, R, C].
      temperature: when encoder outputs are not provided, the decoder block
        samples a latent unconditionnally using the mean of the prior
        distribution, and its log_std + log(temperature).

    Returns:
      A DecoderBlockOutput object holding the outputs of the decoder block,
      which have the same shape as the in
      puts, as well as the KL divergence
      between the prior and posterior.

    Raises:
      ValueError: if the inputs are not square images, or they have a number
      of channels incompatible with the settings of the DecoderBlock.
    """
    chex.assert_rank(inputs, 4)
    if inputs.shape[1] != inputs.shape[2]:
      raise ValueError('VDVAE only works with square images, but got '
                       f'rectangular images of shape {inputs.shape[1:3]}.')
    if inputs.shape[3] != self.num_channels:
      raise ValueError('inputs have incompatible number of channels: '
                       f'got {inputs.shape[3]} channels but expeced '
                       f'{self.num_channels}.')

    if self.upsampling_rate > 1:
      current_res = inputs.shape[1]
      target_res = current_res * self.upsampling_rate
      target_shape = (inputs.shape[0], target_res, target_res, inputs.shape[3])
      inputs = jax.image.resize(inputs, shape=target_shape, method='nearest')

    prior_mean, prior_log_std, features = self._compute_prior_and_features(
        inputs, context_vectors)
    if encoder_outputs is not None:
      posterior_mean, posterior_log_std = self._compute_posterior(
          inputs, encoder_outputs, context_vectors)
    else:
      posterior_mean = prior_mean
      posterior_log_std = prior_log_std + jnp.log(temperature)

    posterior_distribution = distrax.Independent(
        distrax.Normal(posterior_mean, jnp.exp(posterior_log_std)),
        reinterpreted_batch_ndims=3)
    prior_distribution = distrax.Independent(
        distrax.Normal(prior_mean, jnp.exp(prior_log_std)),
        reinterpreted_batch_ndims=3)
    latent = posterior_distribution.sample(seed=sample_rng)
    kl = posterior_distribution.kl_divergence(prior_distribution)

    outputs = self._compute_outputs(inputs, features, latent)
    return DecoderBlockOutput(outputs=outputs, kl=kl)


class Decoder(nn.Module):
  """Decoder of a VDVAE.

  Builds a VDVAE decoder.
  """
  num_blocks: int
  num_channels: int
  bottlenecked_num_channels: int
  latent_dim: int
  upsampling_rates: Sequence[Tuple[int, int]]
  output_image_resolution: int
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    prod = lambda seq: functools.reduce(operator.mul, seq, 1)
    self.total_upsampling_rate = prod(
        [rate for _, rate in self.upsampling_rates])
    if self.output_image_resolution % self.total_upsampling_rate != 0:
      raise ValueError('Total upsampling should divide requested output '
                       'image resolution, but got total upsampling of '
                       f'{self.total_upsampling_rate} for output resolution of '
                       f'{self.output_image_resolution}.')
    self.input_resolution = self.output_image_resolution // self.total_upsampling_rate

    sampling_rates = sorted(self.upsampling_rates)
    num_blocks = self.num_blocks
    current_sequence_start = 0
    blocks_list = []
    for block_idx, rate in sampling_rates:
      if rate == 1:
        continue
      sequence_length = block_idx - current_sequence_start
      if sequence_length > 0:
        # Add sequence of non-downsampling blocks as a single layer stack.
        for i in range(current_sequence_start, block_idx):
          blocks_list.append(
              DecoderBlock(
                  self.num_channels,
                  self.bottlenecked_num_channels,
                  self.latent_dim,
                  np.sqrt(1.0 / self.num_blocks),
                  upsampling_rate=1,
                  precision=self.precision,
                  name=f'res_block_{i}'))

      # Add upsampling block
      blocks_list.append(
          DecoderBlock(
              self.num_channels,
              self.bottlenecked_num_channels,
              self.latent_dim,
              np.sqrt(1.0 / self.num_blocks),
              upsampling_rate=rate,
              precision=self.precision,
              name=f'res_block_{block_idx}'))
      # Update running parameters
      current_sequence_start = block_idx + 1
    # Add remaining blocks after last upsampling block
    sequence_length = num_blocks - current_sequence_start
    if sequence_length > 0:
      # Add sequence of non-downsampling blocks as a single layer stack.
      for i in range(current_sequence_start, num_blocks):
        blocks_list.append(
            DecoderBlock(
                self.num_channels,
                self.bottlenecked_num_channels,
                self.latent_dim,
                np.sqrt(1.0 / self.num_blocks),
                upsampling_rate=1,
                precision=self.precision,
                name=f'res_block_{i}'))

    self._blocks = blocks_list

  def _validate_encoder_outputs(
      self,
      encoder_outputs,
  ):
    """Validates the encoder_outputs structure and returns batch_size."""
    batch_size = None
    if encoder_outputs is not None:
      if not encoder_outputs:
        raise ValueError('encoder_outputs should be either None or a non-empty '
                         'mapping.')
      for encoder_out in encoder_outputs.values():
        encoder_out_shape = encoder_out.shape
        if len(encoder_out_shape) != 4:
          raise ValueError('encoder_outputs should be of rank 4 but '
                           f'encountered shape {encoder_out_shape}.')
        if batch_size is not None:
          if encoder_out_shape[0] != batch_size:
            raise ValueError('encoder_outputs values should have the same '
                             'leading dimension (batch size) but found '
                             f'{batch_size} and {encoder_out_shape[0]}')
        batch_size, height, width, num_channels = encoder_out_shape
        if height != width:
          raise ValueError('encoder_outputs should be square arrays, but got '
                           f'height of {height} and width of {width}.')
        if num_channels != self.num_channels:
          raise ValueError('encoder_outputs number of channels should equal '
                           'the number of channels of the Decoder, but got '
                           f'{num_channels} vs {self.num_channels}.')
    return batch_size

  def __call__(
      self,
      sample_rng,
      num_samples_to_generate,
      context_vectors = None,
      encoder_outputs = None,
      temperature = 1.,
  ):
    """Generates an image.

    Args:
      sample_rng: random key for sampling.
      num_samples_to_generate: number of images to generate from the prior
        distributions, conditioned only on potential context vectors. This
        argument should be provided only when encoder_outputs is not provided.
        If provided, it should be positive.
      context_vectors: optional batch of shape [B, D]. These are typically used
        to condition the VDVAE.
      encoder_outputs: a mapping from resolution to encoded images corresponding
        to the output of an Encoder. This mapping should contain all the
        intermediate resolutions of the Decoder. For each resolution R_i in
        encoder_outputs, the corresponding value has shape [B, R_i, R_i, C],
        where B is a common batch size, and C should equal the number of
        channels of the Decoder.
      temperature: when encoder outputs are not provided, each decoder block
        samples a latent unconditionnally using the mean of the prior
        distribution, and its log_std + log(temperature).

    Returns:
      A tuple (generated_image, kl) where generated image is of shape
      (B, R, R, C) with B the batch size (or num_samples_to_generate if it is
      provided), R the output image resolution, and C the number of channels.
      kl is a 2-D array of shape (num_blocks, batch_size) containing the KL
      divergence between prior and posterior for each block.
    """

    if num_samples_to_generate is not None and num_samples_to_generate < 1:
      raise ValueError(
          'If provided, num_samples_to_generate should be > 1 but got '
          f'{num_samples_to_generate}.')
    batch_size = num_samples_to_generate
    if context_vectors is not None:
      if len(context_vectors.shape) != 2:
        raise ValueError('Context vectors should have rank 2, but got '
                         f'shape {context_vectors.shape}.')

    inputs = jnp.zeros((batch_size,) +
                       (self.input_resolution, self.input_resolution,
                        self.num_channels))

    kl = jnp.zeros((self.num_blocks, batch_size))

    for block_index, block in enumerate(self._blocks):
      sample_rng, sample_rng_now = jax.random.split(sample_rng)
      block_output = block(
          inputs,
          sample_rng_now,
          context_vectors=context_vectors,
          encoder_outputs=encoder_outputs,
          temperature=temperature)

      kl = jax.ops.index_update(kl, block_index, block_output.kl)

      inputs = block_output.outputs

    return inputs, kl
