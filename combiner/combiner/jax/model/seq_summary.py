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

# pylint: skip-file
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import lax
from functools import partial

from combiner.jax.model.transformer_base import TransformerConfig, EncoderDecoder1DBlock, MultiDimEncoderDecoder1DBlock


def do_pooling(pool_spec, input_embed, keepdims):
  if pool_spec == 'last':
    summary = input_embed[Ellipsis, -1, :]
    if keepdims:
      summary = jnp.expand_dims(summary, axis=-2)
    return summary
  else:
    pool_func = getattr(jnp, pool_spec, None)
    if pool_func is None:
      raise ValueError('unknown pooling method %s' % pool_spec)
    return pool_func(input_embed, axis=-2, keepdims=keepdims)


class JustPooling(nn.Module):
  config: TransformerConfig

  @nn.compact
  def __call__(self, input_embed, keepdims=True):
    pool_spec = self.config.seq_summary.split('-')[1]
    return do_pooling(pool_spec, input_embed, keepdims)


class SelfAttPooling(nn.Module):
  config: TransformerConfig
  num_repeat: int

  def setup(self):
    if self.num_repeat == -1:
      self.self_att = EncoderDecoder1DBlock(config=self.config, is_self_att=True)
    else:
      self.self_att = MultiDimEncoderDecoder1DBlock(config=self.config, num_repeat=self.num_repeat, is_self_att=True)

  def __call__(self, input_embed, keepdims=True):
    """
    Args:
      input_embed: embedding of size
            `[batch_sizes..., length, input_embed_dim]`.
    Returns:
      summary: tensor of shape `[batch_sizes..., 1, input_embed_dim]`.
    """
    all_att = self.self_att(input_embed)
    pool_spec = self.config.seq_summary.split('-')[1]
    return do_pooling(pool_spec, all_att, keepdims)


class CrossAttSummary(nn.Module):
  config: TransformerConfig
  num_repeat: int

  def setup(self):
    if self.num_repeat == -1:
      self.cross_att = EncoderDecoder1DBlock(config=self.config, is_self_att=False)
    else:
      self.cross_att = MultiDimEncoderDecoder1DBlock(config=self.config, num_repeat=self.num_repeat, is_self_att=False)

  @nn.compact
  def __call__(self, input_embed, keepdims=True):
    """
    Args:
      input_embed: embedding of size
            `[batch_sizes..., length, input_embed_dim]`.
    Returns:
      summary: tensor of shape `[batch_sizes..., 1, input_embed_dim]`.
    """
    if self.config.seq_summary == 'cross-cls':  # use cls embedding for query
      cls_embedding = self.param('cls_embed', self.config.kernel_init, (1, input_embed.shape[-1]))
      tile_times = []
      for i in range(len(input_embed.shape) - 2):
        cls_embedding = jnp.expand_dims(cls_embedding, axis=0)
        tile_times.append(input_embed.shape[i])
      tile_times += [1, 1]
      query = jnp.tile(cls_embedding, tile_times)
    else:
      assert self.config.seq_summary == 'cross-last'  # use last embedding for query
      query = jnp.expand_dims(input_embed[Ellipsis, -1, :], axis=-2)

    summary = self.cross_att(inputs=query, inputs_kv=input_embed)
    if not keepdims:
      summary = jnp.squeeze(summary, axis=-2)
    return summary


def get_seq_summary_module(config, num_repeat=-1):
  if config.seq_summary.startswith('pool-'):
    return partial(SelfAttPooling, config, num_repeat)
  elif config.seq_summary.startswith('just-'):
    return partial(JustPooling, config, num_repeat)
  elif config.seq_summary.startswith('cross-'):
    return partial(CrossAttSummary, config, num_repeat)
  else:
    raise NotImplementedError
