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
from flax import struct
import jax.numpy as jnp

from latent_programmer.models import base_models


@struct.dataclass
class DecomposeAttentionTransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  base_config: base_models.TransformerConfig
  # Options: baseline, bos_to_bos, bos_to_last, bos_to_bos_and_last,
  # bos_full_attention
  attention_mask_type: str
  # Whether to use special relative attention computation for BOS tokens
  bos_special_attention: bool
  # Whether to clip the relative positions
  clip_relative_attention: bool  


def shift_left(x):
  """Shift the input to the left."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[-1] = (0, 1)  # Padding on axis=-1
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[Ellipsis, 1:]


def make_partial_program_mask(programs,
                              bos_token = 1,
                              dtype = jnp.float32):
  """Make mask that segments program into partial programs."""
  num_partials = jnp.cumsum(jnp.where(programs == bos_token, 1, 0), axis=-1)

  mask = jnp.equal(jnp.expand_dims(num_partials, axis=-1),
                   jnp.expand_dims(num_partials, axis=-2))
  mask = jnp.expand_dims(mask, axis=-3)
  return mask.astype(dtype)


def make_relative_position(programs,
                           dtype=jnp.int32):
  program_position = jnp.arange(programs.shape[-1], dtype=jnp.int32)

  relative_position = program_position[None, :] - program_position[:, None]
  relative_position = jnp.broadcast_to(
      relative_position, programs.shape[:-1] + relative_position.shape)
  return relative_position.astype(dtype)


def make_partial_program_relative_position(programs,
                                           bos_token=1,
                                           dtype=jnp.int32):
  """Make relative positions for bos tokens of partial programs."""
  program_partial_position = jnp.cumsum(
      jnp.where(programs == bos_token, 1, 0), axis=-1)

  bos_relative_position = (program_partial_position[Ellipsis, None, :] -
                           program_partial_position[Ellipsis, None])
  return bos_relative_position.astype(dtype)


class DecomposeAttentionTransformer(nn.Module):
  """Transformer model for program synthesis with i/o examples."""

  config: DecomposeAttentionTransformerConfig

  def setup(self):
    base_config = self.config.base_config

    is_initialized = (self.has_variable('relative_attention', 'max_input_distance') and
                      self.has_variable('relative_attention', 'max_output_distance') and
                      self.has_variable('relative_attention', 'max_io_distance') and
                      self.has_variable('relative_attention', 'max_program_distance')
                      )
    if is_initialized and self.config.clip_relative_attention:
      max_input_distance = self.max_input_distance.item()
      max_output_distance = self.max_output_distance.item()
      max_io_distance = self.max_io_distance.item()
      max_program_distance = self.max_program_distance.item()

      base_encoder_config = base_config.replace(
          max_distance=max_input_distance,
          num_relative_position_buckets=2 * max_input_distance,
          max_output_distance=max_output_distance,
          num_output_relative_position_buckets=2 * max_output_distance)
      base_decoder_config=base_config.replace(
          max_distance=max_io_distance,
          num_relative_position_buckets=2 * max_io_distance,
          max_output_distance=max_program_distance,
          num_output_relative_position_buckets=2 * max_program_distance,
          shift=False),  # Shifting is done separately in decoder.
    else:
      base_encoder_config = base_config
      base_decoder_config = base_config.replace(shift=False)

    self.encoder = base_models.TransformerIOEncoder(
        config=base_encoder_config, name='encoder')
    self.decoder = base_models.TransformerDecoder(
        config=base_decoder_config, name='decoder')

    self.max_input_distance = self.variable(
        'relative_attention', 'max_input_distance', lambda: jnp.zeros((), jnp.int32))
    self.max_output_distance = self.variable(
        'relative_attention', 'max_output_distance', lambda: jnp.zeros((), jnp.int32))
    self.max_io_distance = self.variable(
        'relative_attention', 'max_io_distance', lambda: jnp.zeros((), jnp.int32))
    self.max_program_distance = self.variable(
        'relative_attention', 'max_program_distance', lambda: jnp.zeros((), jnp.int32))

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
    cfg = self.config.base_config

    assert programs.ndim == 2, ('Number of program dimensions should be 2,'
                                ' but it is: %d' % programs.ndim)
    assert encoded.ndim == 4, ('Number of encoded dimensions should be 4,'
                               ' but it is: %d' % encoded.ndim)

    # Collapse num_io dimension
    flat_encoded = base_models.flatten_num_io_dim(encoded)
    flat_encoded_padding_mask = base_models.flatten_num_io_dim(
        encoded_padding_mask)

    if cfg.shift:
      programs = base_models.shift_right(programs, cfg.bos_token)

    # Make attention masks.
    decoder_mask = None
    decoder_relative_position = None  # Relative positions.
    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      # TODO(jxihong): Fast decoding currently does not work with new attention.
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), flat_encoded_padding_mask, dtype=cfg.dtype)
    else:
      attention_mask_type = self.config.attention_mask_type
      if attention_mask_type == 'baseline':
        decoder_mask = nn.combine_masks(
            nn.make_attention_mask(programs > 0, programs > 0, dtype=cfg.dtype),
            nn.make_causal_mask(programs, dtype=cfg.dtype))
      else:
        if attention_mask_type == 'bos_to_bos':
          # BOS tokens attend to all previous BOS tokens.
          decoder_bos_mask = nn.combine_masks(
              nn.make_attention_mask(
                  programs == cfg.bos_token,
                  programs == cfg.bos_token,
                  dtype=cfg.dtype),
              nn.make_causal_mask(programs, dtype=cfg.dtype))
        elif attention_mask_type == 'bos_to_last':
          # BOS tokens attend to all last partial program tokens.
          bos_mask = nn.combine_masks(
              nn.make_attention_mask(
                  programs == cfg.bos_token,
                  programs == cfg.bos_token,
                  dtype=cfg.dtype),
              nn.make_causal_mask(programs, dtype=cfg.dtype))
          # Shift bos mask to left to get all previous last partial program
          # tokens.
          decoder_bos_mask = shift_left(bos_mask)
        elif attention_mask_type == 'bos_to_bos_and_last':
          # BOS tokens attend to all previous BOS + last partial program tokens.
          bos_mask = nn.combine_masks(
              nn.make_attention_mask(
                  programs == cfg.bos_token,
                  programs == cfg.bos_token,
                  dtype=cfg.dtype),
              nn.make_causal_mask(programs, dtype=cfg.dtype))
          # Shift bos mask to left to get all previous last partial program
          # tokens.
          decoder_bos_mask = jnp.logical_or(bos_mask, shift_left(bos_mask))
        elif attention_mask_type == 'bos_full_attention':
          # BOS tokens attend to all previous tokens, including program tokens.
          decoder_bos_mask = nn.combine_masks(
              nn.make_attention_mask(
                  programs == cfg.bos_token,
                  programs > 0,
                  dtype=cfg.dtype),
              nn.make_causal_mask(programs, dtype=cfg.dtype))
        else:
          raise ValueError('Unhandled attention_mask_type: {}'.format(
              attention_mask_type))
        # Program tokens attend to all previous tokens in partial program.
        decoder_partial_mask = nn.combine_masks(
            make_partial_program_mask(
                programs, bos_token=cfg.bos_token, dtype=cfg.dtype),
            nn.make_causal_mask(programs, dtype=cfg.dtype))
        decoder_mask = nn.combine_masks(
            nn.make_attention_mask(
                programs > 0, programs > 0, dtype=cfg.dtype),
            jnp.logical_or(decoder_bos_mask, decoder_partial_mask))

        if self.config.bos_special_attention:
          # Make custom relative positions where BOS are separately indexed.
          decoder_relative_position = make_relative_position(programs)
          decoder_partial_relative_position = (
              make_partial_program_relative_position(programs,
                                                     bos_token=cfg.bos_token))
          decoder_relative_position = jnp.where(
              (programs == cfg.bos_token)[Ellipsis, None],
              decoder_partial_relative_position,
              decoder_relative_position)
        else:
          decoder_relative_position = None

      encoder_decoder_mask = nn.make_attention_mask(
          programs > 0, flat_encoded_padding_mask, dtype=cfg.dtype)

    return self.decoder(
        programs, flat_encoded, decoder_mask, encoder_decoder_mask,
        decoder_relative_position)

  def __call__(self,
               inputs,
               outputs,
               programs):
    """Applies Transformer model on the inputs."""
    encoded = self.encode(inputs, outputs)
    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)

    if self.has_variable('relative_attention', 'max_input_distance'):
      max_input_distance = jnp.max(jnp.count_nonzero(inputs == 0, axis=0))
      self.max_input_distance.value = jnp.maximum(
        self.max_input_distance.value, max_input_distance)
    if self.has_variable('relative_attention', 'max_input_distance'):
      max_output_distance = jnp.max(jnp.count_nonzero(outputs == 0, axis=0))
      self.max_output_distance.value = jnp.maximum(
        self.max_output_distance.value, max_output_distance)
    if self.has_variable('relative_attention', 'max_io_distance'):
      max_io_distance = jnp.max(
          jnp.count_nonzero(jnp.concatenate((inputs, outputs), axis=0) == 0, axis=0))
      self.max_io_distance.value = jnp.maximum(
        self.max_io_distance.value, max_io_distance)
    if self.has_variable('relative_attention', 'max_program_distance'):
      max_program_distance = jnp.max(jnp.count_nonzero(programs == 0, axis=0))
      self.max_program_distance.value = jnp.maximum(
        self.max_program_distance.value, max_program_distance)

    return self.decode(programs, encoded, encoded_padding_mask)
