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

"""Models for decomposition experiment."""

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from latent_programmer.models import base_models


@struct.dataclass
class DecomposeAttentionTransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  base_config: base_models.TransformerConfig
  # The kind of dataset: 'robustfill' or 'deepcoder'.
  dataset_type: str = 'robustfill'
  # Whether to align relative dot-product attention position between the target
  # (next spec part for each I/O example) and the encoded I/O examples, using
  # separator tokens in the target.
  aligned_relative_attention: bool = False
  separator_token_id: int = -1


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

    if self.config.dataset_type in ['robustfill', 'deepcoder']:
      encoder_fn = base_models.TransformerIOEncoder
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(
          self.config.dataset_type))

    self.encoder = encoder_fn(config=base_config, name='encoder')

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

    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      # TODO(jxihong): Fast decoding currently does not work with new attention.
      raise NotImplementedError()

    # Make attention masks.
    decoder_mask = nn.combine_masks(
        nn.make_attention_mask(programs > 0, programs > 0, dtype=cfg.dtype),
        nn.make_causal_mask(programs, dtype=cfg.dtype))
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
