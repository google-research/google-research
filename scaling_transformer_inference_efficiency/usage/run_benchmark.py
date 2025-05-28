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

"""Runs benchmarks against FasterTransformer benchmark."""

import jax
import pandas as pd
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency.usage import benchmarks

Layout = benchmarks.Layout


# TODO(sholto): Check correctness and fix in follow up cl - prioritising getting
# version into github
# pylint: disable = invalid-name
def run():
  """Runs table for FasterTransformer benchmark."""
  columns = [
      'batch', 'input_length', 'output_length', 'PaLM prefill - time',
      'PaLM generate - mfu', 'PaLM prefill - time', 'PaLM generate - mfu',
      'PaLM total - time', 'PaLM total - mfu', 'MT-NLG prefill - time',
      'MT-NLG generate - mfu', 'MT-NLG prefill - time', 'MT-NLG generate - mfu',
      'MT-NLG total - time', 'MT-NLG total - mfu'
  ]
  df = pd.DataFrame(columns=columns)
  PALM_LAYERS = 118
  PALM_PARAMS = 5.4e11
  MT_NLG_PARAMS = 5.3e11
  MT_NLG_LAYERS = 105
  CHIPS = 64
  TPU_V4_FLOPS = 2.75e14
  h = checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8)
  assert len(jax.local_devices()) == 64

  def latency(time, layers, non_stack, num_steps):
    return time * layers * num_steps + non_stack

  def mfu(batch, steps, params, chips, flops, latency):
    num = 2 * batch * steps * params
    denom = chips * flops
    return num / denom * (1000 / latency)

  for input_length, output_length in [(20, 8), (60, 20), (128, 8)]:
    for batch in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:

      prefill_non_layer_stack_time, _ = benchmarks.run_embed_unembed_topp(
          h, batch, input_length, sample=False)
      generate_non_layer_stack_time, _ = benchmarks.run_embed_unembed_topp(
          h, batch, 1, sample=False)

      palm_prefill_time, _ = benchmarks.run_weight_stationary_layer(
          '  result',
          checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
          batch,
          cached_seqlen=0,
          gen_seqlen=input_length,
          quantized=False,
          attn_all_to_all=partitioning.AttnAllToAll.NONE,
          multihead=False,
          layout=Layout.WEIGHT_STATIONARY_2D,
          latency_collectives=False,  # TODO(sholto): Confirm best config
          shard_seqlen_vs_batch=batch <= 16)

      palm_generate_time, _ = benchmarks.run_weight_stationary_layer(
          '  result',
          checkpoint.HParams.PALM_540B_64HEADS.replace(layers=8),
          batch,
          cached_seqlen=input_length,
          gen_seqlen=1,  # multiplied by output length in post processing
          quantized=False,
          attn_all_to_all=partitioning.AttnAllToAll.NONE,
          layout=Layout.WEIGHT_STATIONARY_2D,
          multihead=False,
          latency_collectives=True,
          shard_seqlen_vs_batch=batch <= 16)

      MT_NLG_prefill_time = benchmarks.run_serial_layer(
          '  result',
          checkpoint.HParams.TURING_NLG.replace(layers=4),
          batch,
          cached_seqlen=0,
          gen_seqlen=input_length,
          quantized=False,
          attn_all_to_all=partitioning.AttnAllToAll.NONE,
          multihead=True,
          layout=Layout.WEIGHT_STATIONARY_2D,
          latency_collectives=False,
          swiglu=False)

      MT_NLG_generate_time, _ = benchmarks.run_serial_layer(
          '  result',
          checkpoint.HParams.TURING_NLG.replace(layers=4),
          batch,
          cached_seqlen=input_length,
          gen_seqlen=1,  # multiplied by output length in post processing
          quantized=False,
          attn_all_to_all=partitioning.AttnAllToAll.NONE,
          multihead=True,
          layout=Layout.WEIGHT_STATIONARY_2D,
          latency_collectives=True,
          swiglu=False)

      palm_prefill_latency = latency(palm_prefill_time, PALM_LAYERS,
                                     prefill_non_layer_stack_time, input_length)
      palm_prefill_mfu = mfu(batch, input_length, PALM_PARAMS, CHIPS,
                             TPU_V4_FLOPS, palm_prefill_latency)

      # multiply non laye stack by output length for generate as it is done 1
      # by 1 as opposed to all at once for prefill, so we use 1 as the arg in
      # the timing function above instead of output_length
      palm_generate_latency = latency(
          palm_generate_time, PALM_LAYERS,
          generate_non_layer_stack_time * output_length, output_length)
      palm_generate_mfu = mfu(batch, output_length, PALM_PARAMS, CHIPS,
                              TPU_V4_FLOPS, palm_generate_latency)

      total_palm_latency = palm_prefill_latency + palm_generate_latency
      total_palm_mfu = mfu(batch, input_length + output_length, PALM_PARAMS,
                           CHIPS, TPU_V4_FLOPS, total_palm_latency)

      mt_nlg_prefill_latency = latency(MT_NLG_prefill_time, MT_NLG_LAYERS,
                                       prefill_non_layer_stack_time,
                                       input_length)
      mt_nlg_prefill_mfu = mfu(batch, input_length, MT_NLG_PARAMS, CHIPS,
                               TPU_V4_FLOPS, mt_nlg_prefill_latency)

      # multiply non laye stack by output length for generate as it is done 1
      # by 1 as opposed to all at once for prefill, so we use 1 as the arg in
      # the timing function above instead of output_length
      mt_nlg_generate_latency = latency(
          MT_NLG_generate_time, PALM_LAYERS,
          generate_non_layer_stack_time * output_length, output_length)
      mt_nlg_generate_mfu = mfu(batch, output_length, MT_NLG_PARAMS, CHIPS,
                                TPU_V4_FLOPS, mt_nlg_generate_latency)

      total_mt_nlg_latency = mt_nlg_prefill_latency + mt_nlg_generate_latency

      total_mt_nlg_latency = mt_nlg_prefill_latency + mt_nlg_generate_latency
      total_mt_nlg_mfu = mfu(batch, input_length + output_length, PALM_PARAMS,
                             CHIPS, TPU_V4_FLOPS, total_mt_nlg_latency)

      df.loc[len(df.index)] = [
          batch, input_length, output_length, palm_prefill_latency,
          palm_prefill_mfu, palm_generate_latency, palm_generate_mfu,
          total_palm_latency, total_palm_mfu, mt_nlg_prefill_latency,
          mt_nlg_prefill_mfu, mt_nlg_generate_latency, mt_nlg_generate_mfu,
          total_mt_nlg_latency, total_mt_nlg_mfu
      ]

      print(df)


if __name__ == '__main__':
  run()
