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

"""Models for program synthesis with discrete latent variables."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from latent_programmer.models import base_models as models
from latent_programmer.models import vqvae


@struct.dataclass
class LatentTransformerConfig:
  """Global hyperparameters for latent transformer."""
  base_cfg: models.TransformerConfig
  latent_vocab_size: int
  c: int = 2
  train_vq: bool = True
  commitment_cost_vq: float = 0.25


class Autoencoder(nn.Module):
  """Encodes target sequence into shorter embedding sequence."""

  config: models.TransformerConfig
  c: int  # Sequence length is reduced by 2^c to make latent length.

  @nn.compact
  def __call__(self,
               targets,
               targets_mask=None):
    """Autoencodes program task.

    Args:
      targets: target data `[batch_size, length]`
      targets_mask: padding mask for targets.

    Returns:
      embedding sequence.
    """
    cfg = self.config
    assert targets.ndim == 2  # (batch, len)

    if targets_mask is None:
      targets_mask = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
    encoder_mask = nn.make_attention_mask(
        targets_mask, targets_mask, dtype=cfg.dtype)

    output_embed = nn.Embed(
        num_embeddings=cfg.output_vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='embed_output')

    # Add num_io dimension to latents and latents_mask.
    x = targets.astype('int32')
    x = output_embed(x)
    x = models.AddPositionEmbs(config=cfg, cache=cfg.decode, name='posembed')(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)

    for lyr in range(cfg.num_layers):
      x, _ = models.EncoderBlock(   # Attend to inputs.
          config=cfg, name=f'encoderblock_{lyr}')(x, encoder_mask)

    y = x * targets_mask[Ellipsis, None]
    for i in range(self.c):  # Strided convolutions to decrease length.
      y = nn.Conv(
          features=cfg.emb_dim,
          kernel_size=(2,),
          strides=(2,),
          name=f'conv_{i}')(y)

    return y


class LatentProgramTransformer(nn.Module):
  """Transformer model for program synthesis with discrete latent variables."""

  config: LatentTransformerConfig

  def setup(self):
    cfg = self.config

    self.encoder = models.TransformerIOEncoder(config=cfg.base_cfg,
                                               name='encoder')
    self.decoder = models.TransformerDecoder(config=cfg.base_cfg,
                                             name='decoder')

    self.ae = Autoencoder(config=cfg.base_cfg, c=cfg.c, name='ae')
    self.vq = vqvae.VectorQuantizerEMA(
        config=cfg.base_cfg,
        num_embeddings=cfg.latent_vocab_size,
        commitment_cost=cfg.commitment_cost_vq,
        name='vq')
    self.latent_pos_emb = models.AddPositionEmbs(
        config=cfg.base_cfg, cache=False, name='posembed_latent')

  def encode(self,
             inputs,
             outputs):
    """Applies encoder on input specification."""
    # i/o shape = (batch_size, num_io, length)
    assert inputs.ndim == 3, ('Number of i/o dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    assert outputs.ndim == inputs.ndim

    return self.encoder(inputs, outputs)

  def quantize(self, latent_indices):
    """Gets latent embeddings from tokens."""
    if latent_indices.ndim == 2:
      latents = self.vq.quantize(latent_indices)
    else:
      # Passed in distribution instead.
      assert latent_indices.ndim == 3
      latents = self.vq.quantize(latent_indices, soft_quantize=True)
    return latents

  def decode(self,
             programs,
             latents,
             encoded,
             latents_padding_mask,
             encoded_padding_mask):
    """Applies decoder on programs and encoded specification."""
    cfg = self.config

    assert programs.ndim == 2, ('Number of program dimensions should be 2,'
                                ' but it is: %d' % programs.ndim)
    assert latents.ndim == 3, ('Number of latents dimensions should be 3,'
                               ' but it is: %d' % latents.ndim)
    assert encoded.ndim == 4, ('Number of encoded dimensions should be 4,'
                               ' but it is: %d' % encoded.ndim)

    # Collapse num_io dimension
    flat_encoded = models.flatten_num_io_dim(encoded)
    flat_encoded_padding_mask = models.flatten_num_io_dim(encoded_padding_mask)

    latents = self.latent_pos_emb(latents)
    # Concatenate the i/o encoding and latents together.
    flat_encoded = jnp.concatenate([flat_encoded, latents], axis=1)

    # Make attention masks.
    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      decoder_mask = None
      latent_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), latents_padding_mask, dtype=cfg.dtype)
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), flat_encoded_padding_mask, dtype=cfg.dtype)
      encoder_decoder_mask = jnp.concatenate(
          [encoder_decoder_mask, latent_decoder_mask], axis=-1)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(programs > 0, programs > 0, dtype=cfg.dtype),
          nn.make_causal_mask(programs, dtype=cfg.dtype))
      latent_decoder_mask = nn.make_attention_mask(
          programs > 0, latents_padding_mask, dtype=cfg.dtype)
      encoder_decoder_mask = nn.make_attention_mask(
          programs > 0, flat_encoded_padding_mask, dtype=cfg.dtype)
      encoder_decoder_mask = jnp.concatenate(
          [encoder_decoder_mask, latent_decoder_mask], axis=-1)

    return self.decoder(
        programs, flat_encoded, decoder_mask, encoder_decoder_mask)

  def __call__(self,
               inputs,
               outputs,
               programs,
               emb_mask=None,
               pretrain=False):
    """Applies Transformer autoencoder on the inputs."""
    cfg = self.config

    encoded = self.encode(inputs, outputs)
    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)

    autoencoded = self.ae(programs)
    latents_padding_mask = jnp.where(  # Latent Masks
        programs > 0, 1, 0).astype(jnp.float32)[:, ::2**cfg.c]
    vq = self.vq(autoencoded,
                 train=cfg.train_vq if not pretrain else False,
                 emb_mask=emb_mask,
                 padding_mask=latents_padding_mask)
    latents = vq['latents']

    # To avoid bypassing, only use latents for decoding during pretraining.
    if pretrain:
      encoded_padding_mask *= 0

    output = self.decode(programs,
                         latents,
                         encoded,
                         latents_padding_mask,
                         encoded_padding_mask)
    return output, vq
