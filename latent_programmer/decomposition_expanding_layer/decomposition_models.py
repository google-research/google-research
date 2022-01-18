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

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from flax import linen as nn
import jax.numpy as jnp

from latent_programmer.models import base_models


# Utility functions to handle multiple partial programs.


def add_and_tile_dim(x, num_repeats, axis=1):
  """Creates new dimension in non-scalar array and tiles into it."""
  if x.ndim == 0:
    return x
  x = jnp.expand_dims(x, axis=axis)
  tile_dims = [1] * x.ndim
  tile_dims[axis] = num_repeats
  return jnp.tile(x, tile_dims)


def split_embedding_dim(x, num_splits):
  if x.ndim == 0:
    return x
  emb_dim_per_split = x.shape[-1] // num_splits
  total_usable_size = emb_dim_per_split * num_splits
  return x[Ellipsis, :total_usable_size].reshape(
      x.shape[:-1] + (num_splits, emb_dim_per_split))


class DecomposeExpandingLayerTransformer(nn.Module):
  """Transformer model for program synthesis with i/o examples."""

  config: base_models.TransformerConfig
  num_partial_programs: int = 1
  use_expanding_layer: bool = True

  def setup(self):
    cfg = self.config

    self.encoder = base_models.TransformerIOEncoder(config=cfg, name='encoder')
    self.decoder = base_models.TransformerDecoder(config=cfg, name='decoder')
    if self.use_expanding_layer:
      self.expand = nn.Dense(
          self.num_partial_programs * cfg.emb_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          name='expandembed')

  def encode(self,
             inputs,
             outputs):
    """Applies encoder on input specification."""
    # i/o shape = (batch_size, (num_partial), num_io, length)
    assert inputs.ndim in [3, 4], ('Number of i/o dimensions should be 3 or 4,'
                                   ' but it is: %d' % inputs.ndim)
    assert outputs.ndim == inputs.ndim

    return self.encoder(inputs, outputs)

  def decompose(self, encoded):
    """Splits the embedding layer into num_partial_programs pieces."""
    # encoded shape == [batch_size, num_io, length, dim]
    assert encoded.ndim == 4, ('Number of encoded dimensions should be 4,'
                               ' but it is: %d' % encoded.ndim)

    if self.use_expanding_layer:
      encoded = self.expand(encoded)
    encoded = jnp.transpose(
        split_embedding_dim(encoded, self.num_partial_programs),
        (0, 3, 1, 2, 4))

    return encoded

  def decode(self,
             programs,
             encoded,
             encoded_padding_mask):
    """Applies decoder on programs and encoded specification."""
    cfg = self.config

    # Allow for decoding without num_partial dimension for beam search.
    # programs shape == [batch_size, (num_partial), length]
    assert programs.ndim in [2, 3], ('Number of program dimensions should be'
                                     '2 or 3, but it is: %d' % programs.ndim)
    assert encoded.ndim == programs.ndim + 2

    # Collapse num_io dimension.
    num_io_axis = 1 if programs.ndim == 2 else 2
    flat_encoded = base_models.flatten_num_io_dim(encoded, axis=num_io_axis)
    flat_encoded_padding_mask = base_models.flatten_num_io_dim(
        encoded_padding_mask, axis=num_io_axis)

    # Make attention masks.
    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), flat_encoded_padding_mask, dtype=cfg.dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(programs > 0, programs > 0, dtype=cfg.dtype),
          nn.make_causal_mask(programs, dtype=cfg.dtype))
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

    # encoded shape == [batch_size, (num_partial_programs), num_io, length, dim]
    assert encoded.ndim in [4, 5], ('Number of encoded dimensions should be'
                                    ' 4 or 5, but it is: %d' % encoded.ndim)

    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
    # If programs are decomposed but i/o encodings are not, then split encodings
    # using the model.
    if encoded.ndim == 4 and programs.ndim == 3:
      encoded = self.decompose(encoded)
      encoded_padding_mask = add_and_tile_dim(
          encoded_padding_mask, self.num_partial_programs, axis=1)

    return self.decode(programs, encoded, encoded_padding_mask)
