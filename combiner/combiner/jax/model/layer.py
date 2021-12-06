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
from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import lax
from flax.linen.module import merge_param
from flax.linen.linear import DenseGeneral
from flax.linen.attention import dot_product_attention

from combiner.jax.model.util import log_2_ceil, make_causal_mask, shift_right, shift_left
from combiner.jax.model.transformer_base import TransformerConfig, MlpBlock
from functools import partial
import sys


class _SelfAttLogn(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  def get_dropout_png(self, cfg):
    dropout_rng = None
    if not cfg.deterministic and cfg.attention_dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
    return dropout_rng

  @nn.compact
  def __call__(self, input_qkv):
    cfg = self.config
    log_len = log_2_ceil(cfg.max_len - 1)
    bsize = input_qkv.shape[0]
    features = self.out_features or input_qkv.shape[-1]
    query, key, value, head_dim = get_qkv(cfg, input_qkv)

    joint_logits = []
    list_vals = []
    for l in range(log_len):
      ctx_len = 2 ** l
      last_pos = cfg.max_len - cfg.max_len % ctx_len
      num_ctx = cfg.max_len // ctx_len

      if l == 0:
        span_key = jnp.reshape(key, [-1, 1, cfg.num_heads, head_dim])
        span_val = value.reshape(span_key.shape)
        self_logits = jnp.expand_dims(jnp.sum(query * key, axis=-1), -1)
        joint_logits.append(self_logits)
      else:
        left_query = query[:, :last_pos, :, :].reshape([-1, ctx_len, cfg.num_heads, head_dim])
        span_query = jnp.max(left_query, axis=1, keepdims=True)
        left_key = key[:, :last_pos, :, :].reshape(left_query.shape)
        left_val = value[:, :last_pos, :, :].reshape(left_query.shape)
        span_val = dot_product_attention(span_query * jnp.sqrt(head_dim),
                                  left_key,
                                  left_val,
                                  dropout_rng=self.get_dropout_png(cfg),
                                  dropout_rate=cfg.attention_dropout_rate,
                                  broadcast_dropout=False,
                                  deterministic=cfg.deterministic,
                                  dtype=cfg.dtype)
        span_key = jnp.max(left_key, axis=1, keepdims=True)
      rolled_q = jnp.roll(query, -ctx_len, axis=1)[:, :last_pos, :, :].reshape([-1, ctx_len, cfg.num_heads, head_dim])

      rolled_mask = jnp.concatenate([(jnp.arange(cfg.max_len - ctx_len) // ctx_len) % 2,
                                     jnp.ones(last_pos + ctx_len - cfg.max_len, dtype=jnp.int32)], axis=0)
      rolled_mask = jnp.reshape(rolled_mask, [1, -1, 1, 1])
      rolled_logits = jnp.einsum('...qhd,...khd->...qhk', rolled_q, span_key)
      # bsize, last_pos, h, 1
      rolled_logits = jnp.reshape(rolled_logits, [bsize, -1, cfg.num_heads, 1]) + rolled_mask.astype(rolled_q.dtype) * -1e9
      orig_logits = jnp.pad(rolled_logits,
                            [(0, 0), (0, cfg.max_len - last_pos), (0, 0), (0, 0)],
                            constant_values=-1e9)
      orig_logits = jnp.roll(orig_logits, ctx_len, axis=1)
      joint_logits.append(orig_logits)
      list_vals.append(span_val)
    joint_logits = jnp.concatenate(joint_logits, axis=-1)
    attn_weights = jax.nn.softmax(joint_logits).astype(cfg.dtype)
    local_weights = jnp.split(attn_weights, log_len + 1, axis=-1)
    local_weighted_sums = []
    joint_merged = local_weights[0] * value
    for l in range(log_len):
      ctx_len = 2 ** l
      last_pos = cfg.max_len - cfg.max_len % ctx_len
      num_ctx = cfg.max_len // ctx_len

      rolled_w = jnp.roll(local_weights[l + 1], -ctx_len, axis=1)[:, :last_pos, :, :].reshape(bsize * num_ctx, ctx_len, cfg.num_heads, 1)
      rolled_v = jnp.reshape(rolled_w * list_vals[l], [bsize, -1, cfg.num_heads, head_dim])
      rolled_v = jnp.pad(rolled_v, [(0, 0), (0, cfg.max_len - last_pos), (0, 0), (0, 0)])
      orig_v = jnp.roll(rolled_v, ctx_len, axis=1)
      joint_merged = joint_merged + orig_v
    x = DenseGeneral(features=features,
                  axis=(-2, -1),
                  kernel_init=cfg.kernel_init,
                  bias_init=cfg.bias_init,
                  use_bias=False,
                  dtype=cfg.dtype)(joint_merged)
    return x


class SelfAttLognLayer(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs):
    cfg = self.config
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = _SelfAttLogn(config=cfg,
                         out_features=self.out_features)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs
    z = nn.LayerNorm(dtype=cfg.dtype)(x)
    z = MlpBlock(config=cfg)(z)
    return x + z


class _AxialMixtureAtt(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self, input_qkv):
    cfg = self.config
    cfg.max_len % cfg.max_seg_len == 0

    assert input_qkv.ndim == 3
    bsize = input_qkv.shape[0]
    features = self.out_features or input_qkv.shape[-1]
    qkv_features = cfg.qkv_dim or input_qkv.shape[-1]
    assert qkv_features % cfg.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // cfg.num_heads

    dense = partial(DenseGeneral,
                    axis=-1,
                    features=(cfg.num_heads, head_dim),
                    kernel_init=cfg.kernel_init,
                    bias_init=cfg.bias_init,
                    use_bias=False)
    query, key, value = (dense(dtype=cfg.dtype, name='query')(input_qkv) / jnp.sqrt(head_dim),
                         dense(dtype=cfg.dtype, name='key')(input_qkv),
                         dense(dtype=cfg.dtype, name='value')(input_qkv))
    num_seg = cfg.max_len // cfg.max_seg_len

    ##################
    cur_query = query.reshape([bsize, num_seg, cfg.max_seg_len, cfg.num_heads, head_dim])
    cur_key = key.reshape([bsize, num_seg, cfg.max_seg_len, cfg.num_heads, head_dim])
    cur_value = value.reshape([bsize, num_seg, cfg.max_seg_len, cfg.num_heads, head_dim])

    num_attn_dims = 2
    col_logit_expr = 'BSUNK,BTUNK->BUNST'
    col_attn_expr = 'BUNST,BTUNK->BSUNK'
    col_strict_mask = make_causal_mask(cur_query, length_axis=1, strict=True) # strict lower triangular matrix so that the token won't repeatedly attend to itself
    col_strict_mask = jnp.expand_dims(col_strict_mask, axis=1)
    # (bsize, 1, 1, num_seg, num_seg)
    col_strict_bias = lax.select(
          col_strict_mask > 0,
          jnp.full(col_strict_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(col_strict_mask.shape, -1e10).astype(cfg.dtype))

    row_logit_expr = 'BUSNK,BUTNK->BUNST'
    row_attn_expr = 'BUNST,BUTNK->BUSNK'
    row_mask = make_causal_mask(cur_query, length_axis=2)[:, 0:1, :, :, :]
    # (bsize, 1, 1, max_seg_len, max_seg_len)
    row_bias = lax.select(
          row_mask > 0,
          jnp.full(row_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(row_mask.shape, -1e10).astype(cfg.dtype))

    col_logits = jnp.einsum(col_logit_expr, cur_query, cur_key) + col_strict_bias
    # (bsize, max_seg_len, num_head, num_seg, num_seg)
    row_logits = jnp.einsum(row_logit_expr, cur_query, cur_key) + row_bias
    # (bsize, num_seg, num_head, max_seg_len, max_seg_len)
    ###############################

    col_up2down_query = jax.lax.cummax(cur_query, axis=1)
    col_up2down_key = shift_right(jax.lax.cummax(cur_key, axis=1), axis=1) # shift down in some sense
    col_mask = make_causal_mask(cur_query, length_axis=1)
    col_mask = jnp.expand_dims(col_mask, axis=1)
    col_bias = lax.select(
          col_mask > 0,
          jnp.full(col_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(col_mask.shape, -1e10).astype(cfg.dtype))
    col_up2down_logits = jnp.einsum(col_logit_expr, col_up2down_query, cur_key) + col_bias
    col_up2down_attn_weights = jax.nn.softmax(col_up2down_logits).astype(cfg.dtype)
    col_up2down_summary = jnp.einsum(col_attn_expr, col_up2down_attn_weights, cur_value)
    col_up2down_summary = shift_right(col_up2down_summary, axis=1) # shift down in some sense

    row_only_myself_mask = jnp.expand_dims(jnp.eye(cur_query.shape[2]), (0,1,2))
    row_without_myself_bias = lax.select(
          row_only_myself_mask == 0,
          jnp.full(row_only_myself_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(row_only_myself_mask.shape, -1e10).astype(cfg.dtype)) # attend to all tokens in the previous row except for the token right up to the token in the previous token because this is already taken care of in the local col attention
    all_maskout = jnp.full(row_only_myself_mask.shape, -1e10).astype(cfg.dtype)
    row_without_myself_bias = jnp.concatenate([all_maskout] + [row_without_myself_bias]*(cur_query.shape[1]-1), axis=1) # the first row also has no previous row to attend, so just mask out all logits calculated here
    previous_row_logits = jnp.einsum(row_logit_expr, cur_query, col_up2down_key) + row_without_myself_bias


    row_left2right_query = jax.lax.cummax(cur_query, axis=2)
    row_left2right_key = shift_right(jax.lax.cummax(cur_key, axis=2), axis=2)
    row_left2right_logits = jnp.einsum(row_logit_expr, row_left2right_query, cur_key) + row_bias
    row_left2right_attn_weights = jax.nn.softmax(row_left2right_logits).astype(cfg.dtype)
    row_left2right_summary = jnp.einsum(row_attn_expr, row_left2right_attn_weights, cur_value)
    row_left2right_summary = shift_right(row_left2right_summary, axis=2)

    all_maskout = jnp.full(col_strict_bias.shape, -1e10).astype(cfg.dtype)
    col_strict_without_first_bias = jnp.concatenate([all_maskout] + [col_strict_bias]*(cur_query.shape[2]-1), axis=1)
    top_left_col_logits = jnp.einsum(col_logit_expr, cur_query, row_left2right_key) + col_strict_without_first_bias
    ##################################
    row_right2left_query = jax.lax.cummax(cur_query, axis=2, reverse=True)
    row_right2left_key = shift_left(jax.lax.cummax(cur_key, axis=2, reverse=True), axis=2)
    row_strict_mask = make_causal_mask(cur_query, length_axis=2, strict=True)[:, 0:1, :, :, :]
    # (bsize, 1, 1, max_seg_len, max_seg_len)
    row_upper_bias = lax.select(
          row_strict_mask == 0,
          jnp.full(row_strict_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(row_strict_mask.shape, -1e10).astype(cfg.dtype)) # an upper triangular matrix since we attend all tokens on the right
    row_right2left_logits = jnp.einsum(row_logit_expr, row_right2left_query, cur_key) + row_upper_bias
    row_right2left_attn_weights = jax.nn.softmax(row_right2left_logits).astype(cfg.dtype)
    row_right2left_summary = jnp.einsum(row_attn_expr, row_right2left_attn_weights, cur_value)
    row_right2left_summary = shift_left(row_right2left_summary, axis=2)

    col_strict_without_last_bias = jnp.concatenate([col_strict_bias]*(cur_query.shape[2]-1) + [all_maskout], axis=1)
    top_right_col_logits = jnp.einsum(col_logit_expr, cur_query, row_right2left_key) + col_strict_without_last_bias
    ####


    joint_logits = jnp.concatenate((col_logits.transpose([0, 3, 2, 1, 4]), row_logits, previous_row_logits, top_left_col_logits.transpose([0, 3, 2, 1, 4]), top_right_col_logits.transpose([0, 3, 2, 1, 4])), axis=-1) # follow row, row first, the shape should be (bsize, num_seg, num_head, max_seg_len, num_seg+max_seg_len+max_seg_len+num_seg+num_seg)
    attn_weights = jax.nn.softmax(joint_logits).astype(cfg.dtype)

    col_att, row_att, previous_row_att, top_left_col_att, top_right_col_att = jnp.split(attn_weights, [num_seg, num_seg+cfg.max_seg_len, num_seg+cfg.max_seg_len*2, num_seg*2+cfg.max_seg_len*2], axis=-1)
    col_att = col_att.transpose([0, 3, 2, 1, 4])
    top_left_col_att = top_left_col_att.transpose([0, 3, 2, 1, 4])
    top_right_col_att = top_right_col_att.transpose([0, 3, 2, 1, 4])
    col_merged = jnp.einsum(col_attn_expr, col_att, cur_value)
    row_merged = jnp.einsum(row_attn_expr, row_att, cur_value)
    previous_row_merged = jnp.einsum(row_attn_expr, previous_row_att, col_up2down_summary)
    top_left_merged = jnp.einsum(col_attn_expr, top_left_col_att, row_left2right_summary)
    top_right_merged = jnp.einsum(col_attn_expr, top_right_col_att, row_right2left_summary)

    joint_merged = (col_merged + row_merged + previous_row_merged + top_left_merged + top_right_merged).reshape([bsize, num_seg*cfg.max_seg_len, cfg.num_heads, head_dim])
    x = DenseGeneral(features=features,
                  axis=(-2, -1),
                  kernel_init=cfg.kernel_init,
                  bias_init=cfg.bias_init,
                  use_bias=False,
                  dtype=cfg.dtype)(joint_merged)

    return x


class AxialMixtureSelfAttLayer(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs,
               layer_id=None):
    cfg = self.config
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = _AxialMixtureAtt(config=cfg,
                       out_features=self.out_features)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs
    z = nn.LayerNorm(dtype=cfg.dtype)(x)
    z = MlpBlock(config=cfg)(z)
    return x + z


class _SelfAttAxialRowmajor(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self, input_qkv):
    cfg = self.config
    cfg.max_len % cfg.max_seg_len == 0
    bsize = input_qkv.shape[0]
    features = self.out_features or input_qkv.shape[-1]
    num_seg = cfg.max_len // cfg.max_seg_len
    x_sqr = input_qkv.reshape([bsize, num_seg, cfg.max_seg_len, input_qkv.shape[-1]])
    q_row_local, key_row_local, value_row_local, head_dim = get_qkv(cfg, x_sqr)
    local_logits = jnp.einsum('...qhd,...khd->...qhk', q_row_local, key_row_local)
    row_probs = jax.nn.softmax(local_logits)
    if not cfg.deterministic and cfg.attention_dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
      row_probs = dropatt(row_probs, dropout_rng, 1 - cfg.attention_dropout_rate)
    row_attn_out = jnp.einsum('...qhk,...khd->...qhd', row_probs, value_row_local)

    key_row = DenseGeneral(features=input_qkv.shape[-1],
                           axis=(-2, -1),
                           kernel_init=cfg.kernel_init,
                           bias_init=cfg.bias_init,
                           use_bias=False,
                           dtype=cfg.dtype)(row_attn_out)
    key_row = nn.Dropout(rate=cfg.dropout_rate)(key_row, deterministic=cfg.deterministic)
    key_row = key_row + x_sqr
    key_row = nn.LayerNorm(dtype=cfg.dtype)(key_row)
    key_row = DenseGeneral(axis=-1,
                           features=(cfg.num_heads, head_dim),
                           kernel_init=cfg.kernel_init,
                           bias_init=cfg.bias_init,
                           use_bias=False,
                           dtype=cfg.dtype)(key_row)
    idx_cols = jnp.arange(cfg.max_seg_len)
    local_mask = nn.make_attention_mask(idx_cols, idx_cols, jnp.less, extra_batch_dims=1)
    local_mask = jnp.expand_dims(local_mask, axis=-2) * -1e10
    local_logits = local_logits + local_mask

    global_logits = jnp.einsum('bqlhd,bklhd->bqlhk', q_row_local, key_row)
    idx_rows = jnp.arange(num_seg)
    global_mask = nn.make_attention_mask(idx_rows, idx_rows, jnp.less_equal)
    global_mask = global_mask[:, :, jnp.newaxis, jnp.newaxis, :] * -1e10
    global_logits = global_logits + global_mask

    joint_logits = jnp.concatenate((local_logits, global_logits), axis=-1)
    attn_probs = jax.nn.softmax(joint_logits, axis=-1)
    local_att, global_att = jnp.split(attn_probs, [cfg.max_seg_len], axis=-1)
    if not cfg.deterministic and cfg.attention_dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
      local_att = dropatt(local_att, dropout_rng, 1 - cfg.attention_dropout_rate)
    local_merged = jnp.einsum('bsqhk,bskhd->bsqhd', local_att, value_row_local)
    global_merged = jnp.einsum('bqlhv,bvlhd->bqlhd', global_att, row_attn_out)
    joint_merged = jnp.reshape(local_merged + global_merged, [bsize, cfg.max_len, cfg.num_heads, head_dim])
    x = DenseGeneral(features=features,
                  axis=(-2, -1),
                  kernel_init=cfg.kernel_init,
                  bias_init=cfg.bias_init,
                  use_bias=False,
                  dtype=cfg.dtype)(joint_merged)
    return x


class SelfAttAxialRowmajorLayer(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs):
    cfg = self.config
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = _SelfAttAxialRowmajor(config=cfg,
                              out_features=self.out_features)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs
    z = nn.LayerNorm(dtype=cfg.dtype)(x)
    z = MlpBlock(config=cfg)(z)
    return x + z


class _SelfAttSqrtn(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self, input_qkv):
    cfg = self.config
    cfg.max_len % cfg.max_seg_len == 0
    bsize = input_qkv.shape[0]
    features = self.out_features or input_qkv.shape[-1]
    query, key, value, head_dim = get_qkv(cfg, input_qkv)

    num_seg = cfg.max_len // cfg.max_seg_len
    cur_query = query.reshape([-1, cfg.max_seg_len, query.shape[-2], query.shape[-1]])
    merged_query = jnp.max(cur_query, axis=1, keepdims=True) * jnp.sqrt(head_dim)
    cur_key = key.reshape([-1, cfg.max_seg_len, key.shape[-2], key.shape[-1]])
    cur_value = value.reshape([-1, cfg.max_seg_len, value.shape[-2], value.shape[-1]])
    dropout_rng = None
    if not cfg.deterministic and cfg.attention_dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
    s = dot_product_attention(merged_query,
                              cur_key,
                              cur_value,
                              dropout_rng=dropout_rng,
                              dropout_rate=cfg.attention_dropout_rate,
                              broadcast_dropout=False,
                              deterministic=cfg.deterministic,
                              dtype=cfg.dtype)
    span_val = jnp.reshape(s, [bsize, -1, s.shape[-2], s.shape[-1]])
    span_key = jnp.max(cur_key, axis=1, keepdims=True)
    # (bsize, n_seg, n_head, dim_per_head)
    span_key = jnp.reshape(span_key, [bsize, -1, span_key.shape[-2], span_key.shape[-1]])

    local_mask = make_causal_mask(cur_query, length_axis=1).transpose([0, 2, 1, 3])
    local_bias = lax.select(
          local_mask > 0,
          jnp.full(local_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(local_mask.shape, -1e10).astype(cfg.dtype))
    # (bsize * n_seg, seg_len, n_head, seg_len)
    local_logits = jnp.einsum('...qhd,...khd->...qhk', cur_query, cur_key) + local_bias
    local_logits = jnp.reshape(local_logits, [bsize, -1, cfg.num_heads, cfg.max_seg_len])
    idx = jnp.broadcast_to(jnp.arange(span_key.shape[1], dtype=jnp.int32),
                           span_key.shape[:2])
    prev_mask = nn.make_attention_mask(idx, idx, jnp.greater, extra_batch_dims=0, dtype=jnp.float32).transpose([0, 2, 1, 3])
    prev_mask = jnp.repeat(prev_mask, cfg.max_seg_len, axis=-3)
    prev_bias = lax.select(
          prev_mask > 0,
          jnp.full(prev_mask.shape, 0.).astype(cfg.dtype),
          jnp.full(prev_mask.shape, -1e10).astype(cfg.dtype))
    # (bsize, max_len, n_head, num_segs)
    prev_logits = jnp.einsum('...qhd,...khd->...qhk', query, span_key) + prev_bias
    joint_logits = jnp.concatenate((local_logits, prev_logits), axis=-1)
    # (bsize x max_len,  n_head, seg_len + num_segs)
    attn_weights = jax.nn.softmax(joint_logits).astype(cfg.dtype)
    local_att, prev_att = jnp.split(attn_weights, [cfg.max_seg_len], axis=-1)
    local_att = local_att.reshape([bsize * num_seg, cfg.max_seg_len, cfg.num_heads, cfg.max_seg_len])
    local_merged = jnp.einsum('...qhk,...khd->...qhd', local_att, cur_value)
    prev_merged = jnp.einsum('...qhk,...khd->...qhd', prev_att, span_val)
    joint_merged = jnp.reshape(local_merged, prev_merged.shape) + prev_merged
    x = DenseGeneral(features=features,
                  axis=(-2, -1),
                  kernel_init=cfg.kernel_init,
                  bias_init=cfg.bias_init,
                  use_bias=False,
                  dtype=cfg.dtype)(joint_merged)
    return x


class SelfAttSqrtnLayer(nn.Module):
  config: TransformerConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs):
    cfg = self.config
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = _SelfAttSqrtn(config=cfg,
                         out_features=self.out_features)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs
    z = nn.LayerNorm(dtype=cfg.dtype)(x)
    z = MlpBlock(config=cfg)(z)
    return x + z


def get_qkv(cfg, input_qkv):
  qkv_features = cfg.qkv_dim or input_qkv.shape[-1]
  assert qkv_features % cfg.num_heads == 0, (
          'Memory dimension must be divisible by number of heads.')
  head_dim = qkv_features // cfg.num_heads

  dense = partial(DenseGeneral,
                  axis=-1,
                  features=(cfg.num_heads, head_dim),
                  kernel_init=cfg.kernel_init,
                  bias_init=cfg.bias_init,
                  use_bias=False)
  query, key, value = (dense(dtype=cfg.dtype, name='query')(input_qkv) / jnp.sqrt(head_dim),
                        dense(dtype=cfg.dtype, name='key')(input_qkv),
                        dense(dtype=cfg.dtype, name='value')(input_qkv))
  return query, key, value, head_dim


def dropatt(x, rng, keep_prob):
  keep = jax.random.bernoulli(rng, keep_prob, x.shape)
  multiplier = (keep.astype(x.dtype) / jnp.asarray(keep_prob, dtype=x.dtype))
  return x * multiplier


if __name__ == '__main__':
  key = jax.random.PRNGKey(1)
  max_len = 12

  train_cfg = TransformerConfig(
    vocab_size=1,
    output_vocab_size=1,
    num_heads=2,
    max_len=max_len,
    max_seg_len=3,
    dropout_rate=0,
    attention_dropout_rate=0.5,
    deterministic=False,
    seq_summary='cross-cls',
    # seq_summary='pool-max',
    share_param=False
  )
  eval_cfg = train_cfg.replace(deterministic=True)

  layer_fn = SelfAttLognLayer
  model = layer_fn(eval_cfg)
  model_train = layer_fn(train_cfg)

  key, model_key = jax.random.split(key)
  a = jax.random.uniform(key, (2, max_len, 3))
  params = model.init(model_key, a)
  dropout_key, permute_key = jax.random.split(key)
  out = model_train.apply(params, a, rngs={"permute": permute_key,
                                           "dropout": dropout_key})
  print(out.shape)
