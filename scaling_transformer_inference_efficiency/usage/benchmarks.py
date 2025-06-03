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

"""Benchmarking of prefill and generate."""

from enum import Enum  # pylint:disable = g-importing-member
import functools
import json
import time
from typing import Any, Tuple

import jax
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import pjit
from jax.experimental.maps import xmap
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np


 import humanize
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import sampling
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.attention import KVCache
from scaling_transformer_inference_efficiency.checkpoint import HParams
from scaling_transformer_inference_efficiency.chunk import ChunkResult
from scaling_transformer_inference_efficiency.incremental import InferenceModel
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import layers_serial
from scaling_transformer_inference_efficiency.layers import one_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers.two_d_parallel_xmap import AttnAllToAll


jax.config.update('jax_enable_custom_prng', True)


def div_up(x, y):
  return (x + y - 1) // y


def div_exact(x, y):
  assert x % y == 0
  return x // y


class Layout(Enum):
  WEIGHT_STATIONARY_2D = 'WEIGHT_STATIONARY_2D'
  WEIGHT_STATIONARY_1D = 'WEIGHT_STATIONARY_1D'


# pylint:disable = redefined-outer-name
# pylint:disable = invalid-name
# pylint:disable = broad-except
# pylint:disable = unused-argument
def run_generate_sweep():
  """Sweeps generate hparams."""
  AttnAllToAll = partitioning.AttnAllToAll

  multihead = False
  quantized = False

  num_devices = len(jax.devices())
  for batch in [64, 128, 256, 512, 1024]:
    for quantized in [False, True]:
      for latency_collectives in [False, True]:
        if num_devices == 256 or (
            num_devices == 128 and not latency_collectives
        ):
          model = checkpoint.HParams.PALM_540B_256HEADS
        if num_devices == 128 or (
            num_devices == 64 and not latency_collectives
        ):
          model = checkpoint.HParams.PALM_540B_128HEADS
        else:
          model = checkpoint.HParams.PALM_540B_64HEADS

        for attn_all_to_all in [
            AttnAllToAll.AXIS_Z,
            AttnAllToAll.AXES_YZ,
            AttnAllToAll.AXES_YZX,
        ]:
          cached_seqlen = 2048
          gen_seqlen = 1
          layout = Layout.WEIGHT_STATIONARY_2D
          try:
            run_weight_stationary_layer(
                '  result',
                model.replace(layers=8),
                batch,
                cached_seqlen=cached_seqlen,
                gen_seqlen=gen_seqlen,
                quantized=quantized,
                attn_all_to_all=attn_all_to_all,
                multihead=multihead,
                layout=layout,
                latency_collectives=latency_collectives,
            )
          except BaseException as err:
            print('  failed: ', err)


def run_smaller_generate_sweep(model):
  """Runs generate over small sizes."""
  AttnAllToAll = partitioning.AttnAllToAll

  multihead = False
  quantized = False

  for batch in [32, 64, 128, 256, 512, 1024]:
    for quantized in [False, True]:
      for latency_collectives in [False, True]:
        for attn_all_to_all in [
            AttnAllToAll.AXIS_Z,
            AttnAllToAll.AXES_YZ,
            AttnAllToAll.AXES_YZX,
        ]:
          cached_seqlen = 2048
          gen_seqlen = 1
          layout = Layout.WEIGHT_STATIONARY_2D
          try:
            run_weight_stationary_layer(
                '  result',
                model.replace(layers=8),
                batch,
                cached_seqlen=cached_seqlen,
                gen_seqlen=gen_seqlen,
                quantized=quantized,
                attn_all_to_all=attn_all_to_all,
                multihead=multihead,
                layout=layout,
                latency_collectives=latency_collectives,
            )
          except BaseException as err:
            print('  failed: ', err)


def run_attention_ablation():
  """Runs all attention variants for both multihead and not."""
  AttnAllToAll = partitioning.AttnAllToAll

  quantized = False
  latency_collectives = True
  model = checkpoint.HParams.PALM_540B_64HEADS
  batch = 256
  layout = Layout.WEIGHT_STATIONARY_2D

  num_devices = len(jax.devices())
  for multihead, attn_layouts in [
      (False, [AttnAllToAll.NONE, AttnAllToAll.AXES_YZX]),
      (True, [AttnAllToAll.NONE]),
  ]:
    for attn_all_to_all in attn_layouts:
      for cached_seqlen in [128, 512, 2048, 8192, 16384, 32768, 65536, 131072]:
        gen_seqlen = 1
        try:
          run_weight_stationary_layer(
              '  result',
              model.replace(layers=8),
              batch,
              cached_seqlen=cached_seqlen,
              gen_seqlen=gen_seqlen,
              quantized=quantized,
              attn_all_to_all=attn_all_to_all,
              multihead=multihead,
              layout=layout,
              latency_collectives=latency_collectives,
          )
        except BaseException as err:
          print('  failed: ', err)

        k_and_v = 2
        bytes_per_number = 2
        kv_cache_per_chip = (
            batch
            * cached_seqlen
            * model.layers
            * model.qkv
            * k_and_v
            * bytes_per_number
        ) // num_devices
        if multihead:
          # n heads, but each head is half the dimension
          kv_cache_per_chip *= model.heads // 2
        elif attn_all_to_all == AttnAllToAll.NONE:
          kv_cache_per_chip *= num_devices

        if kv_cache_per_chip > 10 * 1024 * 1024 * 1024:
          print('  OOM')
  print('Done')


def run_layout_ablation():
  """Runs ablation over 1D and 2D layouts for different chip counts."""
  AttnAllToAll = partitioning.AttnAllToAll

  multihead = False
  quantized = False

  num_devices = len(jax.devices())
  for latency_collectives in [False, True]:
    if num_devices == 128:
      model_1d = checkpoint.HParams.PALM_540B_128HEADS
      model_2d = checkpoint.HParams.PALM_540B_128HEADS
    elif num_devices == 256:
      # 1D partitioning needs heads==chips. 2D does not, but it can sometimes
      # improve compute/communication overlap to do so.
      model_1d = checkpoint.HParams.PALM_540B_256HEADS
      model_2d = checkpoint.HParams.PALM_540B_128HEADS
    elif num_devices <= 64:
      model_1d = checkpoint.HParams.PALM_540B_64HEADS
      model_2d = checkpoint.HParams.PALM_540B_64HEADS

    for attn_all_to_all in [
        AttnAllToAll.AXIS_Z,
        AttnAllToAll.AXES_YZ,
        AttnAllToAll.AXES_YZX,
    ]:
      for batch in [64, 128, 256, 512]:
        cached_seqlen = 2048
        gen_seqlen = 1
        for model, layout in [
            (model_2d, Layout.WEIGHT_STATIONARY_2D),
            (model_1d, Layout.WEIGHT_STATIONARY_1D),
        ]:
          try:
            run_weight_stationary_layer(
                '  result',
                model.replace(layers=8),
                batch,
                cached_seqlen=cached_seqlen,
                gen_seqlen=gen_seqlen,
                quantized=quantized,
                attn_all_to_all=attn_all_to_all,
                multihead=multihead,
                layout=layout,
                latency_collectives=latency_collectives,
            )
          except BaseException as err:
            print('  failed: ', err)


def run_weight_stationary_layer(
    name,
    h,
    batch,
    cached_seqlen,
    gen_seqlen,
    quantized,
    attn_all_to_all,
    multihead,
    layout,
    latency_collectives,
    shard_seqlen_vs_batch = False,
    use_xprof=False,
):
  """Runs xmap layer as a micro benchmark."""
  if multihead:
    h = h.replace(qkv=div_exact(h.qkv, 2))
  print(
      f'batch: {batch}, quantized: {quantized}, attn: {attn_all_to_all},'
      f' multihead: {multihead}, seqlen: {cached_seqlen}, gen_seqlen:'
      f' {gen_seqlen}, layers: {h.layers}, embed: {h.embed}, heads: {h.heads},'
      f' q_wi_per_head: {h.q_wi_per_head}, o_wo_per_head: {h.o_wo_per_head},'
      f' layout: {layout}, latency_collectives: {latency_collectives}'
  )

  mesh = get_3d_mesh()
  x_axis, y_axis, z_axis = mesh.devices.shape

  weights_dtype = jnp.bfloat16
  if quantized:
    weights_dtype = jnp.int8

  if layout == Layout.WEIGHT_STATIONARY_1D:
    weight_head_sharding = x_axis * y_axis * z_axis
    weight_embed_sharding = 1
    residual_embed_sharding = x_axis
    residual_batch_sharding = 1
  elif layout == Layout.WEIGHT_STATIONARY_2D:
    weight_head_sharding = y_axis * z_axis
    weight_embed_sharding = x_axis
    residual_embed_sharding = x_axis * y_axis
    residual_batch_sharding = z_axis

  @functools.partial(
      xmap,
      in_axes=(['x', Ellipsis], ['y', Ellipsis], ['z', Ellipsis]),
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def make_inputs(x_index, y_index, z_index):  # pylint: disable = unused-argument
    q_wi = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            div_exact(h.embed, weight_embed_sharding),
            h.q_wi_per_head,
        ),
        weights_dtype,
    )
    q_wi_scale = jnp.zeros(
        (
            h.layers,
            div_up(h.heads, y_axis * z_axis * x_axis),
            1,
            h.q_wi_per_head,
        ),
        jnp.float32,
    )
    kv = jnp.zeros(
        (h.layers, div_exact(h.embed, x_axis), 1, 2 * h.qkv), weights_dtype
    )
    kv_scale = jnp.zeros((h.layers, 1, 1, 2 * h.qkv), jnp.float32)
    o_wo = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            h.o_wo_per_head,
            div_exact(h.embed, weight_embed_sharding),
        ),
        weights_dtype,
    )
    o_wo_scale = jnp.zeros(
        (h.layers, 1, 1, div_up(h.embed, residual_embed_sharding)), jnp.float32
    )
    layernorm_scale = jnp.zeros(
        (h.layers, div_exact(h.embed, x_axis)), jnp.bfloat16
    )

    heads_yz = h.heads // (y_axis * z_axis)
    if heads_yz >= x_axis:
      B = 1
      X = x_axis
    else:
      B = x_axis // heads_yz
      X = heads_yz

    if attn_all_to_all == AttnAllToAll.NONE:
      attn_batch_sharding = 1
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      attn_batch_sharding = z_axis
    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      attn_batch_sharding = y_axis * z_axis
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      attn_batch_sharding = y_axis * z_axis * X

    if batch >= attn_batch_sharding * B:
      attn_batch_sharding *= B

    sin = jnp.zeros(
        (div_up(batch, attn_batch_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )
    cos = jnp.zeros(
        (div_up(batch, attn_batch_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )

    lengths = jnp.zeros((div_up(batch, attn_batch_sharding),), jnp.int32)
    if multihead:
      attn_head_sharding = div_exact(
          x_axis * y_axis * z_axis, attn_batch_sharding
      )
      k = jnp.zeros(
          (
              cached_seqlen,
              h.layers,
              div_up(batch, attn_batch_sharding),
              div_exact(h.heads, attn_head_sharding),
              h.qkv,
          ),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (
              cached_seqlen,
              h.layers,
              div_up(batch, attn_batch_sharding),
              div_exact(h.heads, attn_head_sharding),
              h.qkv,
          ),
          jnp.bfloat16,
      )
    else:
      k = jnp.zeros(
          (cached_seqlen, h.layers, div_up(batch, attn_batch_sharding), h.qkv),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (cached_seqlen, h.layers, div_up(batch, attn_batch_sharding), h.qkv),
          jnp.bfloat16,
      )
    if shard_seqlen_vs_batch:
      x = jnp.zeros(
          (
              batch,
              div_exact(gen_seqlen, residual_batch_sharding),
              div_exact(h.embed, residual_embed_sharding),
          ),
          jnp.bfloat16,
      )
    else:
      x = jnp.zeros(
          (
              div_exact(batch, residual_batch_sharding),
              gen_seqlen,
              div_exact(h.embed, residual_embed_sharding),
          ),
          jnp.bfloat16,
      )

    return (
        q_wi,
        q_wi_scale,
        kv,
        kv_scale,
        o_wo,
        o_wo_scale,
        layernorm_scale,
        sin,
        cos,
        lengths,
        k,
        v,
        x,
    )

  @functools.partial(
      xmap,
      in_axes=['x', 'y', 'z', Ellipsis],
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def the_benchmark(params, sin, cos, kv_caches, x0):
    def loop_body(layer, carry):
      x, k, v = carry

      if layout == Layout.WEIGHT_STATIONARY_2D:
        impl = two_d_parallel_xmap.transformer_layer_weight_stationary
      else:
        impl = one_d_parallel_xmap.weight_stationary

      if cached_seqlen == 0:
        x, layer_k, layer_v = impl(
            h,
            layer,
            params,
            sin,
            cos,
            [],
            x,
            x_axis,
            y_axis,
            z_axis,
            attn_all_to_all,
            latency_collectives,
        )
      else:
        x, layer_k, layer_v = impl(
            h,
            layer,
            params,
            sin,
            cos,
            [kv_caches],
            x,
            x_axis,
            y_axis,
            z_axis,
            attn_all_to_all,
            latency_collectives,
        )
      k = lax.dynamic_update_index_in_dim(
          k, jnp.swapaxes(layer_k, 0, 1), layer, 0
      )
      v = lax.dynamic_update_index_in_dim(
          v, jnp.swapaxes(layer_v, 0, 1), layer, 0
      )
      return x, k, v

    k = jnp.zeros(
        (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
        jnp.bfloat16,
    )
    v = jnp.zeros(
        (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
        jnp.bfloat16,
    )
    x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x0, k, v))
    return x, k, v

  with mesh:
    (
        q_wi,
        q_wi_scale,
        kv,
        kv_scale,
        o_wo,
        o_wo_scale,
        layernorm_scale,
        sin,
        cos,
        lengths,
        k,
        v,
        x,
    ) = make_inputs(jnp.arange(x_axis), jnp.arange(y_axis), jnp.arange(z_axis))
    if quantized:
      params = weights.QuantizedLayer(
          q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale, layernorm_scale
      )
    else:
      params = weights.Layer(q_wi, kv, o_wo)
    kv_caches = KVCache(lengths, k, v, jnp.zeros([0], jnp.int32))

    def run():
      compiled_fn = the_benchmark.lower(
          params, sin, cos, kv_caches, x
      ).compile()
      compiled_fn(params, sin, cos, kv_caches, x)[0].block_until_ready()

    return benchmark_one(
        run, name, 'while', 1.0 / h.layers / gen_seqlen, use_xprof
    )


def run_weight_gathered_xmap_layer(
    name,
    h,
    batch,
    cached_seqlen,
    gen_seqlen,
    quantized,
    use_xprof = False,
):
  """Runs prefill layer as a micro benchmark."""
  print(
      f'batch: {batch}, quantized: {quantized}, seqlen: {cached_seqlen},'
      f' gen_seqlen: {gen_seqlen}, layers: {h.layers}, embed: {h.embed}, heads:'
      f' {h.heads}, q_wi_per_head: {h.q_wi_per_head}, o_wo_per_head:'
      f' {h.o_wo_per_head}'
  )

  mesh = get_3d_mesh()
  x_axis, y_axis, z_axis = mesh.devices.shape

  weights_dtype = jnp.bfloat16
  if quantized:
    weights_dtype = jnp.int8

  @functools.partial(
      xmap,
      in_axes=(['x', Ellipsis], ['y', Ellipsis], ['z', Ellipsis]),
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def make_inputs(x_index, y_index, z_index):
    q_wi = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, y_axis * z_axis),
            div_exact(h.embed, x_axis),
            h.q_wi_per_head,
        ),
        weights_dtype,
    )
    q_wi_scale = jnp.zeros((h.layers, h.heads, 1, h.q_wi_per_head), jnp.float32)
    kv = jnp.zeros((h.layers, h.embed, 1, 2 * h.qkv), weights_dtype)
    kv_scale = jnp.zeros((h.layers, 1, 1, 2 * h.qkv), jnp.float32)
    o_wo = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, y_axis * z_axis),
            h.o_wo_per_head,
            div_exact(h.embed, x_axis),
        ),
        weights_dtype,
    )
    o_wo_scale = jnp.zeros((h.layers, 1, 1, h.embed), jnp.float32)
    layernorm_scale = jnp.zeros((h.layers, h.embed), jnp.bfloat16)

    attn_sharding = x_axis * y_axis * z_axis
    sin = jnp.zeros(
        (div_up(batch, attn_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )
    cos = jnp.zeros(
        (div_up(batch, attn_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )
    lengths = jnp.zeros((div_up(batch, attn_sharding),), jnp.int32)
    k = jnp.zeros(
        (cached_seqlen, h.layers, div_up(batch, attn_sharding), h.qkv),
        jnp.bfloat16,
    )
    v = jnp.zeros(
        (cached_seqlen, h.layers, div_up(batch, attn_sharding), h.qkv),
        jnp.bfloat16,
    )

    x = jnp.zeros(
        (div_exact(batch, x_axis * y_axis * z_axis), gen_seqlen, h.embed),
        jnp.bfloat16,
    )

    return (
        q_wi,
        q_wi_scale,
        kv,
        kv_scale,
        o_wo,
        o_wo_scale,
        layernorm_scale,
        sin,
        cos,
        lengths,
        k,
        v,
        x,
    )

  @functools.partial(
      xmap,
      in_axes=['x', 'y', 'z', Ellipsis],
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def the_benchmark(params, sin, cos, kv_caches, x0):
    def loop_body(layer, carry):
      x, k, v = carry
      if cached_seqlen == 0:
        x, layer_k, layer_v = (
            two_d_parallel_xmap.transformer_layer_weight_gathered(
                h, layer, params, sin, cos, [], x, x_axis, y_axis, z_axis
            )
        )
      else:
        x, layer_k, layer_v = (
            two_d_parallel_xmap.transformer_layer_weight_gathered(
                h,
                layer,
                params,
                sin,
                cos,
                [kv_caches],
                x,
                x_axis,
                y_axis,
                z_axis,
            )
        )

      k = lax.dynamic_update_index_in_dim(
          k, jnp.swapaxes(layer_k, 0, 1), layer, 0
      )
      v = lax.dynamic_update_index_in_dim(
          v, jnp.swapaxes(layer_v, 0, 1), layer, 0
      )
      return x, k, v

    k = jnp.zeros(
        (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
        jnp.bfloat16,
    )
    v = jnp.zeros(
        (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
        jnp.bfloat16,
    )
    x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x0, k, v))
    return x, k, v

  with mesh:
    (
        q_wi,
        q_wi_scale,
        kv,
        kv_scale,
        o_wo,
        o_wo_scale,
        layernorm_scale,
        sin,
        cos,
        lengths,
        k,
        v,
        x,
    ) = make_inputs(jnp.arange(x_axis), jnp.arange(y_axis), jnp.arange(z_axis))
    if quantized:
      params = weights.QuantizedLayer(
          q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale, layernorm_scale
      )
    else:
      params = weights.Layer(q_wi, kv, o_wo)
    kv_caches = KVCache(lengths, k, v, jnp.zeros([0], jnp.int32))

    def run():
      the_benchmark(params, sin, cos, kv_caches, x)[0].block_until_ready()

    return benchmark_one(
        run, name, 'while', 1.0 / h.layers / gen_seqlen, use_xprof
    )


# pylint: disable = unused-argument
def embed_unembed_topp(
    h,
    x,
    embed,
    sample,
    rng,
    x_axis,
    y_axis,
    z_axis,
):
  """Runs non-layer stack components."""
  # x: int32[batch, maxlen]
  # embed: bfloat16[dmodel.X, vocab.YZ]
  _, vocab_yz = embed.shape
  yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
  vocab_start = yz_index * vocab_yz

  # Initial embedding lookup:
  with jax.named_scope('embed'):

    def embed_one(one_x):
      one_x -= vocab_start
      result = lax.dynamic_index_in_dim(embed, one_x, axis=1, keepdims=False)
      return jnp.where((one_x >= 0) & (one_x < vocab_yz), result, 0)

    x = jax.vmap(jax.vmap(embed_one))(x)
    x = lax.psum(x, axis_name=('y', 'z'))

  # x: bfloat16[batch, maxlen, dmodel.X]

  ## Transformer stack would go here ##

  # x: bfloat16[batch, maxlen, dmodel.X]

  # Final layernorm after transformer stack
  with jax.named_scope('layernorm'):
    epsilon = 1e-6
    mean2 = lax.pmean(
        jnp.mean(lax.square(jnp.float32(x)), axis=-1, keepdims=True),
        axis_name='x',
    )
    x = jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))

  # x: bfloat16[batch, maxlen, dmodel.X]

  with jax.named_scope('unembed'):
    logits_unreduced = jnp.einsum(
        'bte,ev->btv', jnp.float32(x), jnp.float32(embed)
    )
    logits = lax.psum_scatter(
        logits_unreduced, 'x', scatter_dimension=0, tiled=True
    )
    # logits: float32[batch.X, maxlen, vocab.YZ]

  if not sample:
    return logits

  with jax.named_scope('sample'):
    # logits:
    # float32[batch.X, maxlen, vocab.YZ]
    #   -> float32[batch.XYZ, maxlen, vocab]
    batch_x, _, vocab_yz = logits.shape
    padded_batch_x = max(batch_x, y_axis * z_axis)
    if padded_batch_x > batch_x:
      logits = jnp.pad(
          logits,
          pad_width=((0, padded_batch_x - batch_x), (0, 0), (0, 0)),
          mode='constant',
      )
    logits = lax.all_to_all(
        logits, ('y', 'z'), split_axis=0, concat_axis=2, tiled=True
    )
    # logits = binary_search.topp_mask(logits, 0.9, -1e10)
    # TODO(reinerp): Do we still need t5x binary search?
    sample = jax.random.categorical(rng, logits).astype(jnp.int32)
    # sample: int32[batch.XYZ, maxlen]
    sample = lax.all_gather(sample, ('x', 'y', 'z'), axis=0, tiled=True)
    return sample


def run_embed_umembed_sweep():
  """Benchmarks embed and unembed layer timing."""
  h = HParams.PALM_540B_64HEADS
  csv = ['batch,seqlen,time,xprof']
  # Prefill
  for maxlen in [20, 60, 128]:
    for batch in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
      dur, url = run_embed_unembed_topp(h, batch, maxlen, sample=False)
      csv.append(f'{batch},{maxlen},{dur},{url}')
  # Generate
  for batch in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    dur, url = run_embed_unembed_topp(h, batch, 1, sample=True)
    csv.append(f'{batch},1,{dur},{url}')
  print('CSV:')
  print('\n'.join(csv))


def run_embed_unembed_topp(
    h, batch, maxlen, sample, use_xprof = False
):
  """Runs all the non-Transformer-layer parts of the network."""
  print(f'batch: {batch}, maxlen: {maxlen}, sample: {sample}')
  mesh = get_3d_mesh()
  x_axis, y_axis, z_axis = mesh.devices.shape

  @functools.partial(
      xmap,
      in_axes=(['x', Ellipsis], ['y', Ellipsis], ['z', Ellipsis]),
      out_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis], [None, Ellipsis]),
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def make_inputs(x_index, y_index, z_index):
    x = jnp.zeros((batch, maxlen), jnp.int32)
    embed = jnp.zeros(
        (div_up(h.embed, x_axis), div_up(h.vocab, y_axis * z_axis)),
        jnp.bfloat16,
    )
    rng = jax.random.PRNGKey(0, impl='rbg')
    return x, embed, rng

  @functools.partial(
      xmap,
      in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis], [None, Ellipsis]),
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def the_benchmark(x, embed, rng):
    return embed_unembed_topp(h, x, embed, sample, rng, x_axis, y_axis, z_axis)

  with mesh:
    x, embed, rng = make_inputs(
        jnp.arange(x_axis), jnp.arange(y_axis), jnp.arange(z_axis)
    )

    def run():
      the_benchmark(x, embed, rng).block_until_ready()

    return benchmark_one(run, '  result', 'xmap(the_benchmark)', 1.0, use_xprof)


def get_3d_mesh():
  """Creates a device mesh for use with xmap over x/y/z axes."""
  devices = jax.devices()
  if len(devices) == 8:
    # retur
    x, y, z = 2, 2, 2
  elif len(devices) == 16:
    # 2,4,2 or 4,2,2 is good
    x, y, z = 2, 4, 2
  elif len(devices) == 32:
    x, y, z = 4, 2, 4
  elif len(devices) == 64:
    x, y, z = 4, 4, 4
  elif len(devices) == 128:
    x, y, z = 8, 4, 4
    # x, y, z = 4, 4, 8
  elif len(devices) == 256:
    # x, y, z = 8, 4, 8
    x, y, z = 4, 8, 8
  elif len(devices) == 512:
    x, y, z = 8, 8, 8
  else:
    raise NotImplementedError
  return Mesh(mesh_utils.create_device_mesh((x, y, z)), ('x', 'y', 'z'))


def benchmark_generate(
    name, hparams, batch, seqlen, num_samples
):
  """Benchmarks a few steps of `generate`."""
  model, params = init_model(hparams)
  benchmark_generate_with_model(
      name, hparams, batch, seqlen, num_samples, model, params
  )


def init_model(hparams):
  """Zero-initializes a model with the specified HParams.

  Args:
    hparams: Model shape.

  Returns:
    A model, plus HBM-resident weights.
  """
  eos_id = 1
  mesh = partitioning.make_mesh()
  model = InferenceModel(
      hparams,
      eos_id,
      functools.partial(
          inference.infer, hparams, layers_pjit.pjit_transformer_layer
      ),
      sampling.sample,
      mesh,
      rules=partitioning.make_rules_two_d(0),
  )
  # TODO(sholto): Make benchmark rules configurable (and the benchmarks to
  # reflect e2e model availability).
  with model.mesh:

    def init_weights():
      return jax.tree.map(
          lambda array: jnp.zeros(array.shape, array.dtype),
          weights.Weights.make_shaped_arrays(hparams),
      )

    params = pjit.pjit(
        init_weights,
        in_shardings=(),
        out_shardings=weights.Weights.physical_axes(),
    )()
  return model, params


def sweep_context_length(hparams):
  """Runs a sweep over context length on the specified model shape."""
  print(
      f'hparams: q_wi_per_head: {hparams.q_wi_per_head}, '
      f'o_wo_per_head: {hparams.o_wo_per_head}, num_layers: {hparams.layers}'
  )
  model, params = init_model(hparams)
  batch = 256
  num_samples = 1
  # for seqlen in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
  for seqlen in [4096]:
    name = f'sweep_context_length[batch={batch}, len={seqlen}]'
    print(f'starting {name}...')
    benchmark_generate_with_model(
        name, hparams, batch, seqlen, num_samples, model, params
    )


def benchmark_generate_with_model(
    name,
    hparams,
    batch,
    seqlen,
    num_samples,
    model,
    params,
):
  """Benchmarks a few steps of `generate`."""
  steps = 4  # All steps are the same. Run just enough to get a few samples
  generate_fn = model.instantiate_generating_fn(steps)
  with model.mesh:
    context = pjit.pjit(
        ChunkResult.zeros,
        in_shardings=(),
        out_shardings=jax.tree.map(
            partitioning.logical_to_physical, ChunkResult.logical_axes()
        ),
        static_argnums=(0, 1, 2),
    )(hparams, batch, seqlen)

  def run():
    model.generate(
        params,
        generate_fn,
        [context],
        np.arange(batch * num_samples),
        sampling.SamplingHyperParams(temperature=0.7),
    )[0].tokens.block_until_ready()

  return benchmark_one(
      run, name, 'pjit__generate_impl', 1.0 / steps, use_xprof=False
  )


def run_serial_layer(
    name,
    h,
    batch,
    cached_seqlen,
    gen_seqlen,
    quantized,
    attn_all_to_all,
    multihead,
    layout,
    latency_collectives,
    swiglu=True,
):
  """Runs xmap layer as a micro benchmark."""
  # if multihead:
  #   h = h.replace(qkv=div_exact(h.qkv, 2))
  print(
      f'batch: {batch}, quantized: {quantized}, attn: {attn_all_to_all},'
      f' multihead: {multihead}, seqlen: {cached_seqlen}, gen_seqlen:'
      f' {gen_seqlen}, layers: {h.layers}, embed: {h.embed}, heads: {h.heads},'
      f' q_wi_per_head: {h.q_wi_per_head}, o_wo_per_head: {h.o_wo_per_head},'
      f' layout: {layout}, latency_collectives: {latency_collectives}'
  )

  mesh = get_3d_mesh()
  x_axis, y_axis, z_axis = mesh.devices.shape

  weights_dtype = jnp.bfloat16
  if quantized:
    weights_dtype = jnp.int8

  if layout == Layout.WEIGHT_STATIONARY_1D:
    weight_head_sharding = x_axis * y_axis * z_axis
    weight_embed_sharding = 1
    residual_embed_sharding = x_axis
    residual_batch_sharding = 1
  elif layout == Layout.WEIGHT_STATIONARY_2D:
    weight_head_sharding = y_axis * z_axis
    weight_embed_sharding = x_axis
    residual_embed_sharding = x_axis * y_axis
    residual_batch_sharding = z_axis

  @functools.partial(
      xmap,
      in_axes=(['x', Ellipsis], ['y', Ellipsis], ['z', Ellipsis]),
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def make_inputs(x_index, y_index, z_index):  # pylint: disable=unused-argument
    if swiglu:
      q_wi_per_head = h.q_wi_per_head
    else:
      q_wi_per_head = h.o_wo_per_head  # as this does not have the swiglu 2x

    q = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            div_exact(h.embed, weight_embed_sharding),
            h.qkv,
        ),
        weights_dtype,
    )
    wi = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            div_exact(h.embed, weight_embed_sharding),
            q_wi_per_head - h.qkv,
        ),
        weights_dtype,
    )
    # q_wi_scale = jnp.zeros((h.layers, div_up(
    #     h.heads, y_axis * z_axis * x_axis), 1, q_wi_per_head), jnp.float32)
    if multihead:
      kv = jnp.zeros(
          (
              h.layers,
              div_exact(h.embed, x_axis),
              div_exact(h.heads, weight_head_sharding),
              2 * h.qkv,
          ),
          weights_dtype,
      )
    else:
      kv = jnp.zeros(
          (h.layers, div_exact(h.embed, x_axis), 1, 2 * h.qkv), weights_dtype
      )

    kv_scale = jnp.zeros((h.layers, 1, 1, 2 * h.qkv), jnp.float32)
    o = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            h.qkv,
            div_exact(h.embed, weight_embed_sharding),
        ),
        weights_dtype,
    )
    wo = jnp.zeros(
        (
            h.layers,
            div_exact(h.heads, weight_head_sharding),
            h.o_wo_per_head - h.qkv,
            div_exact(h.embed, weight_embed_sharding),
        ),
        weights_dtype,
    )
    # o_wo_scale = jnp.zeros(
    # (h.layers, 1, 1, div_up(h.embed, residual_embed_sharding)), jnp.float32)
    layernorm_scale = jnp.zeros(
        (h.layers, div_exact(h.embed, x_axis)), jnp.bfloat16
    )

    heads_yz = h.heads // (y_axis * z_axis)
    if heads_yz >= x_axis:
      B = 1
      X = x_axis
    else:
      B = x_axis // heads_yz
      X = heads_yz

    if attn_all_to_all == AttnAllToAll.NONE:
      attn_batch_sharding = 1
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      attn_batch_sharding = z_axis
    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      attn_batch_sharding = y_axis * z_axis
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      attn_batch_sharding = y_axis * z_axis * X

    if batch >= attn_batch_sharding * B:
      attn_batch_sharding *= B

    sin = jnp.zeros(
        (div_up(batch, attn_batch_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )
    cos = jnp.zeros(
        (div_up(batch, attn_batch_sharding), gen_seqlen, div_exact(h.qkv, 2)),
        jnp.float32,
    )
    lengths = jnp.zeros((div_up(batch, attn_batch_sharding),), jnp.int32)
    if multihead:
      attn_head_sharding = div_exact(
          x_axis * y_axis * z_axis, attn_batch_sharding
      )
      k = jnp.zeros(
          (
              div_exact(h.heads, attn_head_sharding),
              h.layers,
              div_up(batch, attn_batch_sharding),
              cached_seqlen,
              h.qkv,
          ),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (
              div_exact(h.heads, attn_head_sharding),
              h.layers,
              div_up(batch, attn_batch_sharding),
              cached_seqlen,
              h.qkv,
          ),
          jnp.bfloat16,
      )
      # print(f"k_cache, {k.shape}, v_cache, {v.shape}")
    else:
      k = jnp.zeros(
          (cached_seqlen, h.layers, div_up(batch, attn_batch_sharding), h.qkv),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (cached_seqlen, h.layers, div_up(batch, attn_batch_sharding), h.qkv),
          jnp.bfloat16,
      )

    x = jnp.zeros(
        (
            div_exact(batch, residual_batch_sharding),
            gen_seqlen,
            div_exact(h.embed, residual_embed_sharding),
        ),
        jnp.bfloat16,
    )
    return (
        q,
        wi,
        kv,
        kv_scale,
        o,
        wo,
        layernorm_scale,
        sin,
        cos,
        lengths,
        k,
        v,
        x,
    )

  @functools.partial(
      xmap,
      in_axes=['x', 'y', 'z', Ellipsis],
      out_axes=['x', 'y', 'z', Ellipsis],
      axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
  )
  def the_benchmark(params, sin, cos, kv_caches, x0):
    def loop_body(layer, carry):
      x, k, v = carry

      impl = layers_serial.transformer_layer_weight_stationary_serial
      if impl == Layout.WEIGHT_STATIONARY_1D:
        raise NotImplementedError

      if cached_seqlen == 0:
        x, layer_k, layer_v = impl(
            h,
            layer,
            params,
            sin,
            cos,
            [],
            x,
            x_axis,
            y_axis,
            z_axis,
            attn_all_to_all,
            latency_collectives,
            swiglu=swiglu,
        )
      else:
        x, layer_k, layer_v = impl(
            h,
            layer,
            params,
            sin,
            cos,
            [kv_caches],
            x,
            x_axis,
            y_axis,
            z_axis,
            attn_all_to_all,
            latency_collectives,
            swiglu=swiglu,
        )
      k = lax.dynamic_update_index_in_dim(
          k, jnp.swapaxes(layer_k, 0, 1), layer, 0
      )
      v = lax.dynamic_update_index_in_dim(
          v, jnp.swapaxes(layer_v, 0, 1), layer, 0
      )
      return x, k, v

    if multihead:
      k = jnp.zeros(
          (
              h.layers,
              gen_seqlen,
              batch,
              h.heads // (y_axis * z_axis * x_axis),
              h.qkv,
          ),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (
              h.layers,
              gen_seqlen,
              batch,
              h.heads // (y_axis * z_axis * x_axis),
              h.qkv,
          ),
          jnp.bfloat16,
      )
    else:
      k = jnp.zeros(
          (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
          jnp.bfloat16,
      )
      v = jnp.zeros(
          (h.layers, gen_seqlen, batch // (y_axis * z_axis * x_axis), h.qkv),
          jnp.bfloat16,
      )
    # print(f'output k: {k.shape} v: {v.shape}')
    x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x0, k, v))
    return x, k, v

  with mesh:
    q, wi, kv, _, o, wo, _, sin, cos, lengths, k, v, x = make_inputs(
        jnp.arange(x_axis), jnp.arange(y_axis), jnp.arange(z_axis)
    )
    params = layers_serial.SerialLayer(q, wi, kv, o, wo)
    # print(jax.tree.map(jnp.shape, params))
    kv_caches = KVCache(lengths, k, v, jnp.zeros([0], jnp.int32))

    def run():
      compiled_fn = the_benchmark.lower(
          params, sin, cos, kv_caches, x
      ).compile()
      compiled_fn(params, sin, cos, kv_caches, x)[0].block_until_ready()

    return benchmark_one(
        run, name, 'while', 1.0 / h.layers / gen_seqlen, use_xprof=False
    )


def benchmark_one(
    run,
    name,
    search_name,
    scale_duration,
    use_xprof,
):
  """Benchmarks steps, handles xprof url."""
  run()  # Warmup, call once
  if use_xprof:  # internally xprof allows for more accurate timing
    # Aiming to implement in external xprof next week
    if jax.config.read('jax_xla_backend') == 'pathways':
      with pathways.xprof_trace(
          trace_options=xprof_service_pb2.XprofRequest.TraceOptions(
              enable_python_tracer=True,
          ),
          block_until_start=True,
          devices=jax.devices()[:1],
      ) as url:
        run()
      session_id = url[0].split('/trace_viewer/')[-1]
      # url currently goes to google internal link, removed

    else:
      with xprof.session() as url:
        run()
      session_id = url[0].split('session_id=')[-1]

    # print(url)
    xprof_client = xprof_analysis_client.XprofAnalysisClient()
    _, trace_str = xprof_client.get_profile_data(
        'trace_viewer.json', session_id
    )  # pytype: disable=attribute-error
    trace = json.loads(trace_str)

    # We will see the event in each attached TensorCore, even if Megacore is
    # enabled. Track the shortest duration, and ensure that we see it only once
    # per core, i.e. only once per 'pid' of the event.
    dur = None
    seen_pids = set()
    names = set()
    for event in trace['traceEvents']:
      event_name = event.get('name')
      names.add(event_name)
      if isinstance(event_name, str) and event_name.startswith(search_name):
        pid = event.get('pid')
        assert (
            pid not in seen_pids
        ), 'Benchmark called multiple times per TensorCore'
        event_dur = event.get('dur')
        if dur is None or (event_dur > dur):
          dur = event_dur

    if dur is None:
      text = (
          f'Function {search_name} not found. Available names: {names}, xprof:'
          f' {url}'
      )
      print(text)
      raise ValueError(text)
    dur = dur / 1e6
    # while external xprof support is added, use a simple timing loop
  else:
    print('Warning: timing without xprof may be less accurate.')
    t0 = time.time()
    # loop a few times
    num_trials = 5
    for _ in range(0, num_trials):
      run()
    dur = (time.time() - t0) / num_trials
    url = 'No url returned without xprof'
  # adjust for factors such as num layers
  dur = dur * scale_duration
  dur_str = humanize.DecimalPrefix(dur, 's', precision=3, min_scale=-3)
  print(f'{name}: {dur_str}/layer/step, {url}')

  return dur_str, url


def prefill_sweep():
  """Prefill sweep over model sizes."""
  num_devices = len(jax.local_devices())
  print(jax.local_devices())
  start = num_devices
  stop = num_devices * 16
  for quantized in [False, True]:
    for batch in [64, 128, 256, 512, 1024, 2048]:
      if batch >= start and batch <= stop:
        try:
          run_weight_gathered_xmap_layer(
              '  result',
              checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
              batch,
              cached_seqlen=0,
              gen_seqlen=2048,
              quantized=quantized,
          )
        except Exception as e:
          print(e)
          print(f'Batch size {batch} too large')
        print('----------------------------------------')

  for quantized in [False, True]:
    for batch in [64, 128, 256, 512, 1024, 2048]:
      if batch >= start and batch <= stop:
        try:
          run_weight_gathered_xmap_layer(
              '  result',
              checkpoint.HParams.PALM_62B,
              batch,
              cached_seqlen=0,
              gen_seqlen=2048,
              quantized=quantized,
          )
        except Exception as e:
          print(e)
          print(f'Batch size {batch} too large')
        print('----------------------------------------')

  for quantized in [False, True]:
    for batch in [64, 128, 256, 512, 1024, 2048]:
      if batch >= start and batch <= stop:
        try:
          run_weight_gathered_xmap_layer(
              '  result',
              checkpoint.HParams.PALM_8B,
              batch,
              cached_seqlen=0,
              gen_seqlen=2048,
              quantized=quantized,
          )
        except Exception as e:
          print(e)
          print(f'Batch size {batch} too large')
        print('----------------------------------------')


def run_prefill_small_size(model):
  """Runs prefill at small sizes with weight stationary strategy."""
  for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
    for gen_seqlen in [2048]:
      # for batch in [1]:
      #   for gen_seqlen in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
      for quantized in [False, True]:
        for latency_collectives in [False, True]:
          try:
            run_weight_stationary_layer(
                '  result',
                model.replace(layers=8),
                batch,
                cached_seqlen=0,
                gen_seqlen=gen_seqlen,
                quantized=quantized,
                attn_all_to_all=AttnAllToAll.NONE,
                multihead=False,
                layout=Layout.WEIGHT_STATIONARY_2D,
                latency_collectives=latency_collectives,
                shard_seqlen_vs_batch=True,
            )
          except BaseException as err:
            print('  failed: ', err)


def PaLM_benchmark():
  """Runs PaLM architecture on FasterTransformer benchmark lengths."""
  for latency_collectives in [True]:
    for input_length, _ in [(20, 8), (60, 20), (128, 8)]:
      for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        try:
          print('Prefill')
          run_weight_stationary_layer(
              '  result',
              checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
              batch,
              cached_seqlen=0,
              gen_seqlen=input_length,
              quantized=False,
              attn_all_to_all=partitioning.AttnAllToAll.NONE,
              multihead=False,
              layout=Layout.WEIGHT_STATIONARY_2D,
              latency_collectives=latency_collectives,
              shard_seqlen_vs_batch=batch <= 16,
          )
          print('Generate')
          run_weight_stationary_layer(
              '  result',
              checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
              batch,
              cached_seqlen=input_length,
              gen_seqlen=1,  # multiplied by output length in post processing
              quantized=False,
              attn_all_to_all=partitioning.AttnAllToAll.NONE,
              layout=Layout.WEIGHT_STATIONARY_2D,
              multihead=False,
              latency_collectives=latency_collectives,
              shard_seqlen_vs_batch=batch <= 16,
          )
        except Exception as e:
          print(e)


def MT_NLG_benchmark():
  """Runs MT_NLG architecture on FasterTransformer benchmark lengths."""
  quantized = False
  for input_length, _ in [(20, 8), (60, 20), (128, 8)]:
    for batch in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
      try:
        print('Prefill')
        run_serial_layer(
            '  result',
            checkpoint.HParams.TURING_NLG.replace(layers=4),
            batch,
            cached_seqlen=0,
            gen_seqlen=input_length,
            quantized=quantized,
            attn_all_to_all=partitioning.AttnAllToAll.NONE,
            multihead=True,
            layout=Layout.WEIGHT_STATIONARY_2D,
            latency_collectives=False,
            swiglu=False,
        )
        print('Generate')
        run_serial_layer(
            '  result',
            checkpoint.HParams.TURING_NLG.replace(layers=4),
            batch,
            cached_seqlen=input_length,
            gen_seqlen=1,  # multiplied by output length in post processing
            quantized=quantized,
            attn_all_to_all=partitioning.AttnAllToAll.NONE,
            multihead=True,
            layout=Layout.WEIGHT_STATIONARY_2D,
            latency_collectives=True,
            swiglu=False,
        )
      except Exception as e:
        print(e)


def run_prefill_weight_stationary_vs_gathered_sweep():
  """Tries both weight stationary and weight gathered for prefill."""
  AttnAllToAll = partitioning.AttnAllToAll

  multihead = False
  quantized = False

  num_devices = len(jax.devices())

  print('-------- 2D Weight stationary ------------------')
  for batch in [1024, 2048]:
    # for batch in [1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for latency_collectives in [False, True]:
      if num_devices == 256 or (num_devices == 128 and not latency_collectives):
        model = checkpoint.HParams.PALM_540B_256HEADS
      if num_devices == 128 or (num_devices == 64 and not latency_collectives):
        model = checkpoint.HParams.PALM_540B_128HEADS
      else:
        model = checkpoint.HParams.PALM_540B_64HEADS

      for attn_all_to_all in [
          AttnAllToAll.AXIS_Z,
          AttnAllToAll.AXES_YZ,
          AttnAllToAll.AXES_YZX,
      ]:
        if (
            attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX
            and batch >= 512
        ):
          print('AlltoAll XYZ - batch> 512, oom')
        else:
          cached_seqlen = 0
          gen_seqlen = 2048
          layout = Layout.WEIGHT_STATIONARY_2D
          try:
            run_weight_stationary_layer(
                '  result',
                model.replace(layers=8),
                batch,
                cached_seqlen=cached_seqlen,
                gen_seqlen=gen_seqlen,
                quantized=quantized,
                attn_all_to_all=attn_all_to_all,
                multihead=multihead,
                layout=layout,
                latency_collectives=latency_collectives,
            )
          except BaseException as err:
            print('  failed: ', err)

  print('-------- 2D Weight gathered ------------------')

  for batch in [64, 128, 256, 512, 1024, 2048]:
    try:
      run_weight_gathered_xmap_layer(
          '  result',
          checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
          batch,
          cached_seqlen=0,
          gen_seqlen=2048,
          quantized=quantized,
      )
    except Exception as err:
      print('  failed: ', err)


def run_parallel_vs_serial():
  """Returns comparison of parallel vs non-parallel."""
  # Toggle checkpoint sizes as you will, but pattern
  # remains the same
  batch = 256
  run_weight_stationary_layer(
      '  result',
      checkpoint.HParams.PALM_540B_128HEADS.replace(layers=8),
      batch,
      cached_seqlen=0,
      gen_seqlen=1,
      quantized=False,
      attn_all_to_all=partitioning.AttnAllToAll.AXES_YZX,
      multihead=False,
      layout=Layout.WEIGHT_STATIONARY_2D,
      latency_collectives=True,
  )
  run_serial_layer(
      '  result',
      checkpoint.HParams.PALM_540B_128HEADS.replace(layers=8),
      batch,
      cached_seqlen=0,
      gen_seqlen=1,
      quantized=False,
      attn_all_to_all=partitioning.AttnAllToAll.AXES_YZX,
      multihead=False,
      layout=Layout.WEIGHT_STATIONARY_2D,
      latency_collectives=True,
  )
