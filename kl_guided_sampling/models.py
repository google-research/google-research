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

"""Contextual T5X Models.
"""

import functools
from typing import Any, Mapping, MutableMapping, Optional, Tuple
from absl import logging
import jax
import jax.numpy as jnp
from t5x import models
from kl_guided_sampling import decoding
from kl_guided_sampling import feature_converters

PyTree = Any


class ContextualEncoderDecoderModel(models.EncoderDecoderModel):
  """Wrapper class for the models.Transformer nn.module."""

  FEATURE_CONVERTER_CLS = feature_converters.ContextualEncDecFeatureConverter

  def __init__(
      self,
      *args,
      decode_fn = decoding.temperature_sample,
      **kwargs,
  ):
    super().__init__(
        *args,
        decode_fn=decode_fn,
        **kwargs,
    )

  def predict_batch_with_aux(
      self,
      params,
      batch,
      rng = None,
      decoder_params = None,
      return_all_decodes = False,
      num_decodes = 1,
      prompt_with_targets = False
  ):
    """Predict with fast decoding beam search on a batch.

    For ContextualEncoderDecoderModel, running two decoding sequences in
    parallel can be decoupled by two copies of encoders and hiding the rest of
    the complexity to the sampling algorithm. Two different inputs
    "encoder_input_tokens" and "encoder_input_tokens_wo" are fed to two
    copies of encoders (and encapsulated in tokens_ids_to_logits), respectively.
    The decoders are kept intact, except for connecting to two different
    encoders. The temperature sampling inputs parameters: inputs, cache, and
    initial_index are shared, since inputs differences only affects encoders,
    not decoders.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      prompt_with_targets: Whether the force decode decoder_inputs.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # [batch, input_len]
    encoder_input_tokens = batch['encoder_input_tokens']
    encoder_input_tokens_wo = batch['encoder_input_tokens_wo']
    decoder_input_tokens = batch['decoder_input_tokens']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs = decoding.flat_batch_beam_expand(
        self.module.apply(
            {'params': params},
            encoder_input_tokens,
            enable_dropout=False,
            method=self.module.encode,
        ),
        num_decodes,
    )
    encoded_inputs_wo = decoding.flat_batch_beam_expand(
        self.module.apply(
            {'params': params},
            encoder_input_tokens_wo,
            enable_dropout=False,
            method=self.module.encode,
        ),
        num_decodes,
    )

    # `decoder_prompt_inputs` is initialized from the batch's
    # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
    # after the prompt by matching to `output_vocabulary.eos_id`.
    # These inputs are ignored by the beam search decode fn.
    if prompt_with_targets:
      decoder_prompt_inputs = decoder_input_tokens
      decoder_prompt_inputs = decoder_prompt_inputs * (
          decoder_prompt_inputs != self.output_vocabulary.eos_id
      )
    else:
      decoder_prompt_inputs = jnp.zeros_like(decoder_input_tokens)

    # Prepare autoregressive cache.
    cache, initial_index = self._compute_kv_cache(
        params,
        encoded_inputs=encoded_inputs,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_prompt_inputs,
    )
    # Prepare autoregressive cache.
    cache_wo, initial_index_wo = self._compute_kv_cache(
        params,
        encoded_inputs=encoded_inputs_wo,
        encoder_input_tokens=encoder_input_tokens_wo,
        decoder_input_tokens=decoder_prompt_inputs,
    )

    # [batch * num_decodes, input_len]
    raw_inputs = decoding.flat_batch_beam_expand(
        encoder_input_tokens, num_decodes
    )
    raw_inputs_wo = decoding.flat_batch_beam_expand(
        encoder_input_tokens_wo, num_decodes
    )

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        encoded_inputs=encoded_inputs,
        raw_inputs=raw_inputs,
        max_decode_length=decoder_input_tokens.shape[1],
    )
    tokens_ids_to_logits_wo = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        encoded_inputs=encoded_inputs_wo,
        raw_inputs=raw_inputs_wo,
        max_decode_length=decoder_input_tokens.shape[1],
    )

    if decoder_params is None:
      decoder_params = {}
    if initial_index is not None:
      # We only set initial_index when it's non-None since it is not supported
      # by all decoders.
      decoder_params['initial_index'] = initial_index
    if initial_index_wo is not None:
      decoder_params['initial_index_wo'] = initial_index_wo

    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        inputs_wo=decoder_prompt_inputs,
        cache=cache,
        cache_wo=cache_wo,
        tokens_to_logits=tokens_ids_to_logits,
        tokens_to_logits_wo=tokens_ids_to_logits_wo,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        cache_offset_wo=1 if scanned else 0,
        **decoder_params)

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}


class ContextualDecoderOnlyModel(models.DecoderOnlyModel):
  """Model class for the decoder-only modules with contexts.
  """

  FEATURE_CONVERTER_CLS = feature_converters.ContextualPrefixLMFeatureConverter

  def __init__(
      self,
      *args,
      decode_fn = decoding.temperature_sample,
      **kwargs,
  ):
    super().__init__(
        *args,
        decode_fn=decode_fn,
        **kwargs,
    )

  def predict_batch_with_aux(
      self,
      params,
      batch,
      rng = None,
      *,
      return_all_decodes = False,
      num_decodes = 1,
      decoder_params = None,
  ):
    """Predict with prefix and contexts.

    For ContextualDecoderOnlyModel, running two decoding sequences in
    parallel involves two copies of inputs, cache, tokens_to_logits, and
    initial_index, since inputs differences affects all steps in decoders.

    Args:
      params: model parameters.
      batch: batch element with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      return_all_decodes: if True, will return all batch_size * num_decodes
        samples from the model as an array of shape [batch_size, num_decodes,
        sequence_length]. Otherwise returns only the most likely samples as an
        array of shape [batch_size, sequence_length].
      num_decodes: number of decoded sequences to be returned.
      decoder_params: additional (model-independent) parameters for the decoder.

    Returns:
      sampled_sequences: an array of shape [batch, max_decode_length].
    """
    if 'decoder_causal_attention' not in batch:
      raise ValueError(
          'Batch does not have the right format for text generation: probably '
          'because `task_feature_lengths` passed to the feature converter does '
          'not have both `inputs` and `targets`.'
      )

    # since decoder_input_tokens is shifted to the right and
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this masks out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']
    inputs_wo = batch[
        'decoder_input_tokens_wo'] * batch['decoder_causal_attention_wo']

    prefilled_cache, initial_index = self._compute_kv_cache(
        params, inputs, batch['decoder_causal_attention'])
    prefilled_cache_wo, initial_index_wo = self._compute_kv_cache(
        params, inputs_wo, batch['decoder_causal_attention_wo'])

    target_shape = batch['decoder_input_tokens'].shape
    max_decode_length = target_shape[1]

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        max_decode_length=max_decode_length)

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run temperature
    # sampling with the prefix.
    # [batch, max_decode_length]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs,
        inputs_wo=inputs_wo,
        cache=prefilled_cache,
        cache_wo=prefilled_cache_wo,
        tokens_to_logits=tokens_ids_to_logits,
        tokens_to_logits_wo=tokens_ids_to_logits,
        num_decodes=num_decodes,
        initial_index=initial_index,
        initial_index_wo=initial_index_wo,
        cache_offset=1 if scanned else 0,
        cache_offset_wo=1 if scanned else 0,
        **decoder_params)

    if not return_all_decodes:
      # Search returns [n_batch, n_beam/decodes, n_length] with the beam/decode
      # dimension sorted in increasing order of log-probability.
      # `scores` is [batch, beam/decode_size]
      # We take the highest scoring sequence (-1) and its score
      decoded_sequences = decoded_sequences[:, -1, :]
      # Beam search returns []
      aux = {'scores': scores[:, -1]}
    else:
      # We return all samples and scores, rather than just the top ones.
      aux = {'scores': scores}

    return models.remove_prefix(decoded_sequences, initial_index), aux
