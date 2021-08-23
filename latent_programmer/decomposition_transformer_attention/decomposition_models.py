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

"""Models for decomposition experiment."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from typing import Any
from flax import linen as nn
import jax.numpy as jnp

from latent_programmer.models import base_models


def make_partial_program_mask(programs,
                              bos_token = 1,
                              dtype = jnp.float32):
  """Make mask that segments program into partial programs."""
  num_partials = jnp.cumsum(jnp.where(programs == bos_token, 1, 0), axis=-1)

  mask = jnp.equal(jnp.expand_dims(num_partials, axis=-1),
                   jnp.expand_dims(num_partials, axis=-2))
  mask = jnp.expand_dims(mask, axis=-3)
  return mask.astype(dtype)


class DecomposeAttentionTransformer(nn.Module):
  """Transformer model for program synthesis with i/o examples."""

  config: base_models.TransformerConfig

  def setup(self):
    cfg = self.config

    self.encoder = base_models.TransformerIOEncoder(config=cfg, name='encoder')
    # Shifting is done before call to decoder in order to compute masks.
    self.decoder = base_models.TransformerDecoder(
        config=cfg.replace(shift=False), name='decoder')

  def encode(self,
             inputs,
             outputs):
    """Applies encoder on input specification."""
    # i/o shape = (batch_size, num_io, length)
    assert inputs.ndim == 3, ('Number of i/o dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    assert outputs.ndim == inputs.ndim

    return self.encoder(inputs, outputs)

  def decode(self,
             programs,
             encoded,
             encoded_padding_mask):
    """Applies decoder on programs and encoded specification."""
    cfg = self.config

    assert programs.ndim == 2, ('Number of program dimensions should be 2,'
                                ' but it is: %d' % programs.ndim)
    assert encoded.ndim == 4, ('Number of encoded dimensions should be 4,'
                               ' but it is: %d' % encoded.ndim)

    # Collapse num_io dimension
    flat_encoded = base_models.flatten_num_io_dim(encoded)
    flat_encoded_padding_mask = base_models.flatten_num_io_dim(
        encoded_padding_mask)

    preshift_programs = programs  # Save pre-shifted programs for padding mask.
    if cfg.shift:
      programs = base_models.shift_right(programs, cfg.bos_token)

    # Make attention masks.
    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      # TODO(jxihong): Fast decoding currently does not work with new attention.
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), flat_encoded_padding_mask, dtype=cfg.dtype)
    else:
      # BOS tokens attend to all previous BOS tokens.
      decoder_bos_mask = nn.combine_masks(
          nn.make_attention_mask(
              programs == cfg.bos_token,
              programs == cfg.bos_token,
              dtype=cfg.dtype),
          nn.make_causal_mask(programs, dtype=cfg.dtype))
      # Program tokens attend to all previous tokens in partial program.
      decoder_partial_mask = nn.combine_masks(
          make_partial_program_mask(
              programs, bos_token=cfg.bos_token, dtype=cfg.dtype),
          nn.make_causal_mask(programs, dtype=cfg.dtype))
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(
              preshift_programs > 0, preshift_programs > 0, dtype=cfg.dtype),
          jnp.logical_or(decoder_bos_mask, decoder_partial_mask))
      encoder_decoder_mask = nn.make_attention_mask(
          programs > 0, flat_encoded_padding_mask, dtype=cfg.dtype)

    return self.decoder(
        programs, flat_encoded, decoder_mask, encoder_decoder_mask)

  def __call__(self,
               inputs,
               outputs,
               programs):
    """Applies Transformer model on the inputs."""
    encoded = self.encode(inputs, outputs)
    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)

    return self.decode(programs, encoded, encoded_padding_mask)
