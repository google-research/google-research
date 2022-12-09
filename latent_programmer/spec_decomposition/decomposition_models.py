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

"""Models for decomposition experiment."""

from typing import Any
from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from latent_programmer.models import base_models
from latent_programmer.models import relative_attention


@struct.dataclass
class DecomposeAttentionTransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  base_config: base_models.TransformerConfig
  # Options: baseline, bos_to_bos, bos_to_last, bos_to_bos_and_last,
  # bos_full_attention
  attention_mask_type: str = 'baseline'
  # Whether to use special relative attention computation for BOS tokens
  bos_special_attention: bool = False
  # The kind of dataset: 'robust_fill' or 'scan'.
  dataset_type: str = 'robust_fill'
  # Whether to do relative self-attention on the flat encoding of the I/O
  # examples, where the positions are taken modulo the length of 1 example.
  flat_encoded_self_attention: bool = False
  # Whether to align relative dot-product attention position between the target
  # (next spec part for each I/O example) and the encoded I/O examples, using
  # separator tokens in the target.
  aligned_relative_attention: bool = False
  separator_token_id: int = -1


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
  """Makes mask that segments program into partial programs."""
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
  """Makes relative positions for bos tokens of partial programs."""
  program_partial_position = jnp.cumsum(
      jnp.where(programs == bos_token, 1, 0), axis=-1)

  bos_relative_position = (program_partial_position[Ellipsis, None, :] -
                           program_partial_position[Ellipsis, None])
  return bos_relative_position.astype(dtype)


def make_aligned_relative_position(targets,
                                   flat_encoded,
                                   max_input_length,
                                   separator_token,
                                   dtype=jnp.int32):
  """Makes relative positions, aligning examples based on separator tokens."""
  # Running example: targets is `a b | c d e | f g`, max_input_length is 100.

  # targets_position: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  targets_position = jnp.arange(targets.shape[-1], dtype=jnp.int32)

  # Reset the target positions after every separator token. That's why we roll
  # by 1, to mark the positions immediately after separators.
  # separator_locs: [[0, 0, 0, 1, 0, 0, 0, 1, 0], ...]
  # shape: [batch_size, targets_length]
  separator_locs = jnp.roll(jnp.where(targets == separator_token, 1, 0),
                            shift=1, axis=-1)
  # shift: [[0, 0, 0, 3, 3, 3, 3, 7, 7], ...]
  # shape: [batch_size, targets_length]
  shift = jax.lax.cummax(
      jnp.where(separator_locs == 0,
                jnp.zeros_like(targets_position), targets_position),
      axis=1)
  # Track the number of separator tokens that have been seen.
  # separator_counts: [[0, 0, 0, 1, 1, 1, 1, 2, 2], ...]
  # shape: [batch_size, targets_length]
  separator_counts = jnp.cumsum(jnp.where(separator_locs == 1, 1, 0), axis=-1)
  # aligned_targets_position: [[0, 1, 2, 100, 101, 102, 103, 200, 201], ...]
  # shape: [batch_size, targets_length]
  aligned_targets_position = (max_input_length * separator_counts
                              + targets_position - shift)
  # flat_encoded_position: [0, 1, 2, ...]
  # shape: [num_io_examples * max_input_length]
  flat_encoded_position = jnp.arange(flat_encoded.shape[-2], dtype=jnp.int32)
  # shape: [batch_size, targets_length, num_io_examples * max_input_length]
  relative_position = (flat_encoded_position[None, None, :]
                       - aligned_targets_position[:, :, None]).astype(dtype)
  return relative_position


class DecomposeAttentionTransformer(nn.Module):
  """Transformer model for program synthesis with i/o examples."""

  config: DecomposeAttentionTransformerConfig

  def setup(self):
    base_config = self.config.base_config

    if self.config.dataset_type in ['robust_fill', 'deepcoder']:
      self.encoder = base_models.TransformerIOEncoder(config=base_config,
                                                      name='encoder')
    elif self.config.dataset_type in ['robust_fill_base', 'scan']:
      self.encoder = base_models.TransformerEncoder(config=base_config,
                                                    name='encoder')
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(
          self.config.dataset_type))
    # Shifting is done separately in decoder.
    self.decoder = base_models.TransformerDecoder(
        config=base_config.replace(shift=False), name='decoder')

  def encode(self,
             inputs,
             outputs):
    """Applies encoder on input specification."""
    # i/o shape = (batch_size, num_io, length)
    assert inputs.ndim == 3, ('Number of i/o dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    assert outputs.ndim == inputs.ndim

    return self.encoder(inputs, outputs)

  @nn.compact
  def do_flat_encoded_self_attention(self, flat_encoded, mod_position):
    """Does self-attention for the flat encoding."""
    cfg = self.config.base_config
    x = nn.LayerNorm(dtype=cfg.dtype)(flat_encoded)
    x = relative_attention.RelativeSelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        bidirectional=True,
        num_relative_position_buckets=(
            cfg.num_flat_encoding_relative_position_buckets),
        max_distance=cfg.max_flat_encoding_distance,
        mod_position=mod_position)(
            x, None, None)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + flat_encoded
    return x

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

    if self.config.flat_encoded_self_attention:
      per_example_encoding_len = encoded.shape[2]
      flat_encoded = self.do_flat_encoded_self_attention(
          flat_encoded, mod_position=per_example_encoding_len)

    if cfg.shift:
      programs = base_models.shift_right(programs, cfg.bos_token)

    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      # TODO(jxihong): Fast decoding currently does not work with new attention.
      raise NotImplementedError()

    # Make attention masks.
    decoder_mask = None
    decoder_relative_position = None  # Custom relative positions.
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

    # Compute relative attention positions.
    if self.config.aligned_relative_attention:
      encoder_decoder_relative_position = make_aligned_relative_position(
          programs,
          flat_encoded,
          encoded.shape[2],  # shape: (batch_size, num_io, io_length, embed_dim)
          self.config.separator_token_id)
    else:
      encoder_decoder_relative_position = None

    return self.decoder(
        programs, flat_encoded, decoder_mask, encoder_decoder_mask,
        decoder_relative_position, encoder_decoder_relative_position)

  def __call__(self,
               inputs,
               outputs,
               programs):
    """Applies Transformer model on the inputs."""
    encoded = self.encode(inputs, outputs)
    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)

    return self.decode(programs, encoded, encoded_padding_mask)
