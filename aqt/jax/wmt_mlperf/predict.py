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

"""Shared implementation of prediction step for gen_hlo.py and train.py."""
import jax.numpy as jnp
from aqt.jax import quant_config
from aqt.jax.wmt_mlperf import decode
from aqt.jax.wmt_mlperf import models


def step(inputs, params, cache, state, eos_token, max_decode_len,
         transformer_kwargs, hparams,
         quant_context):
  """Predict translation with fast decoding beam search on a batch."""
  # If the state already has a cache, remove it in favor the 'cache' parameter
  # passed to this function.
  if 'cache' in state:
    raise ValueError(
        'cache was already specified in the `state` variable '
        'passed to `step`, which we disallow since it is ambiguous with the '
        '`cache` argument passed to `step`.')
  batch_size = inputs.shape[0]
  beam_size = 4

  # Prepare transformer fast-decoder call for beam search:
  # for beam search, we need to set up our decoder model
  # to handle a batch size equal to batch_size * beam_size,
  # where each batch item's data is expanded in-place rather
  # than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  src_padding_mask = decode.flat_batch_beam_expand((inputs > 0)[Ellipsis, None],
                                                   beam_size)
  tgt_padding_mask = decode.flat_batch_beam_expand(
      jnp.ones((batch_size, 1, 1)), beam_size)
  model = models.Transformer(
      **transformer_kwargs,
      train=False,
      quant_context=quant_context,
      hparams=hparams,
      should_decode=False,
      dropout_rate=0.0,
      attention_dropout_rate=0.0,
      use_bfloat16=False)
  encoded_inputs = decode.flat_batch_beam_expand(
      model.apply({
          'params': params,
          **state
      },
                  inputs,
                  method=model.encode,
                  mutable=False), beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    model = models.Transformer(
        **transformer_kwargs,
        train=False,
        quant_context=quant_context,
        hparams=hparams,
        should_decode=True,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        use_bfloat16=False)
    flat_logits, new_vars = model.apply(
        {
            'params': params,
            'cache': flat_cache,
            **state
        },
        encoded=encoded_inputs,
        src_padding_mask=src_padding_mask,
        targets=flat_ids,
        tgt_padding_mask=tgt_padding_mask,
        method=model.decode,
        mutable=['cache'])
    new_flat_cache = new_vars['cache']

    return flat_logits, new_flat_cache

  # using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_token=eos_token,
      max_decode_len=max_decode_len)

  # beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability
  # return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]
