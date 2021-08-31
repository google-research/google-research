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

"""Implementation of Very Deep VAEs (https://arxiv.org/abs/2011.10650)."""

from typing import Optional, Any

import chex
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from vdvae_flax import decoder
from vdvae_flax import encoder
from vdvae_flax import vdvae_utils


@flax.struct.dataclass
class VdvaeOutput:
  samples: chex.Array  # [B, H, W, C]
  elbo: Optional[chex.Array]  # [B]
  reconstruction_loss: Optional[chex.Array]  # [B]
  kl_per_decoder_block: Optional[chex.Array]  # [num_decoder_blocks, B]


class Vdvae(nn.Module):
  """Very Deep VAE."""

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(
      self,
      sample_rng,
      num_samples_to_generate,
      inputs = None,
      context_vectors = None,
      temperature = 1.,
  ):
    """Evaluates a VDVAE.

    Args:
      sample_rng: random key for sampling.
      num_samples_to_generate: number of images to generate from the prior
        distribution, conditioned only on optional context vectors. This
        argument should be provided only when inputs is not provided. If
        provided, it should be positive.
      inputs: an optional batch of input RGB images of shape [B, H, W, C], where
        H=W. These should be provided when training the VDVAE. These inputs
        should be of type uint8.
      context_vectors: an optional batch of input context vectors of shape [B,
        D] that the VDVAE is conditioned on. These can be omitted, in which case
        the samples will be conditioned on the inputs only if they are provided,
        or not conditioned on anything otherwise.
      temperature: when inputs are not provided, each decoder block samples a
        latent unconditionally using the mean of the prior distribution, and its
        log_std + log(temperature).

    Returns:
      A VdvaeOutput object containing sampled images. If inputs were provided,
      the sampled images are sampled using the posterior distribution,
      conditioned on the inputs (and optional context vectors). In this case,
      the output also contains the elbo, reconstruction loss and KL divergences
      between prior and posterior for each block of the decoder. If inputs are
      not provided, samples are produced using the prior distribution,
      conditioned only on the optional context vectors.
    """
    encoder_model = encoder.Encoder(**self.config.encoder)
    decoder_model = decoder.Decoder(**self.config.decoder)
    sampler_model = vdvae_utils.QuantizedLogisticMixtureNetwork(
        **self.config.sampler)

    if inputs is None:
      encoder_outputs = None
    else:
      if inputs.dtype != jnp.uint8:
        raise ValueError("Expected inputs to be of type uint8 but got "
                         f"{inputs.dtype}")
      preprocessed_inputs = vdvae_utils.cast_to_float_center_and_normalize(
          inputs)
      encoder_outputs = encoder_model(
          preprocessed_inputs, context_vectors=context_vectors)

    sample_rng, sample_rng_now = jax.random.split(sample_rng)
    decoder_outputs, kl = decoder_model(
        sample_rng=sample_rng_now,
        num_samples_to_generate=num_samples_to_generate,
        context_vectors=context_vectors,
        encoder_outputs=encoder_outputs,
        temperature=temperature)

    sample_rng, sample_rng_now = jax.random.split(sample_rng)
    sampler_output = sampler_model(sample_rng_now, decoder_outputs, inputs)
    if inputs is not None:
      reconstruction_loss = sampler_output.negative_log_likelihood
      total_kl = jnp.sum(kl, axis=0)
      elbo = reconstruction_loss + total_kl
    else:
      reconstruction_loss = None
      elbo = None

    return VdvaeOutput(
        samples=sampler_output.samples,
        elbo=elbo,
        kl_per_decoder_block=kl,
        reconstruction_loss=reconstruction_loss)
