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

"""PaDIR Models.

This module uses layers.py to build a higher-level model structure and define
methods for the loss computation as well as a train, prediction, and evaluation
steps.
"""

import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import clu.metrics as clu_metrics
from flax import linen as nn
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import seqio
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import models
from t5x import optimizers
from t5x.models import Array
from t5x.models import PyTree
import typing_extensions

from padir.padir import config_options
from padir.padir import decoding
from padir.padir.features import feature_converters
from padir.padir.utils import padir_utils

MetricsMap = metrics_lib.MetricsMap


class GeneralizedDecodeFnCallable(typing_extensions.Protocol):
  """Decoding function call signature."""

  def __call__(
      self,
      *,
      inputs,
      cache,
      tokens_to_logits,
      eos_id,
      num_decodes,
      decode_rng,
      cache_offset,
      **kwargs,
  ):
    """Decoding function interface.

    Support more flexible number and meaning of return values
    than DecodeFnCallable.

    Args:
      inputs: [batch_size, max_decode_len] int32 sequence of tokens, with non-0
        prefix tokens to be used as a forced prompt.
      cache: flax attention cache.
      tokens_to_logits: fast autoregressive decoder function taking single token
        slices and cache and returning next-token logits and updated cache.
      eos_id: end-of-sentence token for target vocabulary.
      num_decodes: number of decoded sequences to be returned.
      decode_rng: an optional JAX PRNG Key for stochastic sampling routines.
      cache_offset: axis offset for cache, arising from scanned layers.
      **kwargs: an optional kwargs. One common usecase of this is passing
        decoding parameters at the callsite.

    Returns:
      decodes: Array of sequences: [batch_size, num_decodes, max_decode_len].
        The `num_decodes` dimension is expected to be sorted by the `scores`,
        i.e., `decodes[:, -1, :] has the highest scores among `num_decodes`
        decoded sequences.
      scores: Array of log likelihood scores: [batch_size, num_decodes]
      decoder_self_attentions: of shape
        [batch, num_decodes, num_decoder_layers, num_heads, len, len].
        Each column in the matrix last indices [len, len] sums up to 1.
    """
    Ellipsis


class PadirEncoderDecoderModel(models.EncoderDecoderModel):
  """Wrapper class for the models.Transformer nn.module."""

  FEATURE_CONVERTER_CLS = feature_converters.PadirEncDecFeatureConverter  # pylint: disable=invalid-name

  def __init__(
      self,
      module,
      input_vocabulary,
      output_vocabulary,
      optimizer_def,
      decode_fn = decoding.temperature_sample,
      feature_converter_cls = None,
      label_smoothing = 0.0,
      z_loss = 0.0,
      loss_normalizing_factor = None,
      num_decode_iterations = 1,
      num_train_stages = 1,
      remask_stutter = False,
      decoder_input_scheme = config_options.DecoderInputScheme.RANDOM,
      remasking_scheme = config_options.RemaskingScheme.PREVIOUS_TOKENS,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    self._decode_fn = decode_fn
    self.num_decode_iterations = num_decode_iterations
    self.num_train_stages = num_train_stages
    self.remask_stutter = remask_stutter
    self.decoder_input_scheme = decoder_input_scheme
    self.remasking_scheme = remasking_scheme

  def get_initial_variables(
      self,
      rng,
      input_shapes,
      input_types = None,
  ):
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    encoder_shape = input_shapes['encoder_input_tokens']
    encoder_type = input_types.get('encoder_input_tokens', jnp.float32)
    decoder_shape = input_shapes['decoder_input_tokens_train']
    decoder_type = input_types.get('decoder_input_tokens_train', jnp.float32)
    initial_variables = self.module.init(
        rng,
        jnp.ones(encoder_shape, encoder_type),
        jnp.ones(decoder_shape, decoder_type),
        jnp.ones(decoder_shape, decoder_type),
        decode=False,
        enable_dropout=False,
        decode_iteration_idx=0,
        num_decode_iterations=self.num_train_stages,
    )
    return initial_variables

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params,
      batch,
      dropout_rng = None,
      mutable = False,
      other_variables = None,
  ):
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
    if other_variables is None:
      other_variables = {}
    decoder_logits = []
    rejecter_logits = []
    decoder_input_tokens = batch['decoder_input_tokens_train']
    decoder_target_tokens = batch['decoder_target_tokens']
    pred_tgt_len = None
    logits_sum = None
    for i in range(self.num_train_stages):
      decoding_iteration_outputs = self.module.apply(
          {'params': params, **other_variables},
          batch['encoder_input_tokens'],
          decoder_input_tokens,
          decoder_target_tokens,
          decode=False,
          enable_dropout=rngs is not None,
          rngs=rngs,
          mutable=mutable,
          decode_iteration_idx=i,
          num_decode_iterations=self.num_train_stages,
      )
      decoder_logits_i = decoding_iteration_outputs['decoder_logits']
      pred_tgt_len = decoding_iteration_outputs['raw_length_predictions']
      rejecter_logits_i = decoding_iteration_outputs['rejecter_logits']
      rejecter_preds_i = decoding_iteration_outputs['rejecter_predictions']

      decoder_logits.append(decoder_logits_i)
      if rejecter_logits_i is not None:
        rejecter_logits.append(rejecter_logits_i)

      if logits_sum is None:
        logits_sum = jnp.zeros_like(decoder_logits_i, dtype=jnp.float32)
      logits_sum += decoder_logits_i
      replacements = self.remasking_scheme.get_replacements(
          batch['decoder_input_tokens_train'],  # Partially masked.
          decoder_input_tokens,
          logits_sum,
      )
      decoder_input_tokens = padir_utils.replace_rejected_predictions(
          rejecter_preds_i,
          predictions=jnp.argmax(decoder_logits_i, axis=-1),
          replacements=replacements,
          ended=None,  # Results unused after last iteration.
          remask_stutter=self.remask_stutter,
      )

    decoder_logits = [logits[jnp.newaxis, Ellipsis] for logits in decoder_logits]
    rejecter_logits = [logits[jnp.newaxis, Ellipsis] for logits in rejecter_logits]
    return (
        jnp.concatenate(decoder_logits, axis=0),
        pred_tgt_len,
        jnp.concatenate(rejecter_logits, axis=0) if rejecter_logits else None,
    )

  def _compute_decoder_outputs_from_decoding_state(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      decoding_state,
      params,
      encoded_inputs,
      raw_inputs,
      output_self_attentions = False,
      output_cross_attentions = False,
  ):
    """Computes decoder outputs via a forward pass of `self.module_cls`."""
    decoded_dict, new_vars = self.module.apply(
        {'params': params, 'cache': decoding_state.cache},
        encoded_inputs,
        raw_inputs,
        decoding_state.sequences,
        decoding_state.sequences,
        enable_dropout=False,
        decode=True,
        output_self_attentions=output_self_attentions,
        output_cross_attentions=output_cross_attentions,
        mutable=['cache'],
        method=self.module.decode,
    )
    return decoded_dict, new_vars['cache']

  def _compute_rejecter_outputs_from_rejecter_state(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      rejecter_state,
      params,
      encoded_inputs,
      raw_inputs,
  ):
    """Computes rejecter outputs via a forward pass of `self.module_cls`."""
    reject_outputs, new_vars = self.module.apply(
        {'params': params, 'cache': rejecter_state.cache},
        encoded=encoded_inputs,
        encoder_input_tokens=raw_inputs,
        decoder_output_embeddings=rejecter_state.decoder_outputs['embeddings'],
        decoder_logits=rejecter_state.decoder_outputs['logits'],
        decoder_self_attentions=rejecter_state.decoder_outputs[
            'self_attentions'
        ],
        decoder_output_tokens=rejecter_state.sequences,
        decode_iteration_idx=rejecter_state.step,
        num_decode_iterations=self.num_decode_iterations,
        enable_dropout=False,
        decode=True,
        mutable=['cache'],
        method=self.module.reject,
    )
    return reject_outputs, new_vars['cache']

  def predict_batch_with_aux(
      self,
      params,
      batch,
      rng = None,
      decoder_params = None,
      return_all_decodes = False,
      num_decodes = 1,
      prompt_with_targets = False,
      output_self_attentions = False,
      output_cross_attentions = False,
  ):
    """Predict with fast decoding beam search on a batch.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    This method can be used with a customizable decoding function as long as it
    follows the signature of `DecodeFnCallable`. In order to provide a unified
    interface for the decoding functions, we use a generic names. For example, a
    beam size is a concept unique to beam search. Conceptually, it corresponds
    to the number of sequences returned by the beam search.  Therefore, the
    generic argument `num_decodes` corresponds to the beam size if
    `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
    corresponds to the number of independent sequences to be sampled. Typically
    `num_decodes = 1` is used for temperature sampling.

    If `return_all_decodes = True`, the return tuple contains the predictions
    with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
    probability of the generated sequence) with a shape [batch, num_decodes].
    The beam dimension is sorted in increasing order of log-probability.

    If `return_all_decodes = False`, the return tuple contains the predictions
    with a shape [batch, max_decode_len] and the scores with a shape [batch].

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
    from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
    decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      prompt_with_targets: Whether the force decode decoder_inputs.
      output_self_attentions: Whether to output decoder self attentions
      output_cross_attentions: Whether to output decoder cross attentions

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # [batch, input_len]
    _, encoder_input_len = batch['encoder_input_tokens'].shape
    encoder_input_tokens = batch['encoder_input_tokens']
    decoder_input_tokens = batch['decoder_input_tokens_infer']
    # We can use LengthOracle during inference if there is a gold target.
    decoder_target_tokens = batch.get('decoder_target_tokens', None)

    encoded_dict = self.module.apply(
        {'params': params},
        encoder_input_tokens,
        enable_dropout=False,
        method=self.module.encode,
    )
    encoded_inputs = encoded_dict['encoded']
    # Only present if length predictor requires it.
    length_embeddings = encoded_dict.get('length_embeddings', None)

    predicted_lengths, _ = self.module.apply(
        {'params': params},
        encoded=encoded_inputs,
        length_embeddings=length_embeddings,
        encoder_input_tokens=encoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
        method=self.module.predict_length,
    )

    # Parameters for temperature_sample decoding function.
    if decoder_params is None:
      decoder_params = {}
    decoder_params['num_decode_iterations'] = self.num_decode_iterations
    decoder_params['remask_stutter'] = self.remask_stutter
    decoder_params['decoder_input_scheme'] = self.decoder_input_scheme
    decoder_params['remasking_scheme'] = self.remasking_scheme

    tokens_ids_to_decoder_outputs = functools.partial(
        self._compute_decoder_outputs_from_decoding_state,
        params=params,
        encoded_inputs=encoded_inputs,
        raw_inputs=encoder_input_tokens,
        output_self_attentions=output_self_attentions,
        output_cross_attentions=output_cross_attentions,
    )

    tokens_ids_to_rejecter_outputs = functools.partial(
        self._compute_rejecter_outputs_from_rejecter_state,
        params=params,
        encoded_inputs=encoded_inputs,
        raw_inputs=encoder_input_tokens,
    )

    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and'
            " `decoder_params['decode_rng']`"
            f' ({decoder_params["decode_rng"]}). Please specify one or the'
            ' other.'
        )
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    decode_outputs = self._decode_fn(
        decoder_inputs=decoder_input_tokens,
        predicted_lengths=predicted_lengths,
        output_vocab=self.output_vocabulary,
        cache={},
        tokens_to_decoder_outputs=tokens_ids_to_decoder_outputs,
        tokens_to_rejecter_outputs=tokens_ids_to_rejecter_outputs,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        output_self_attentions=output_self_attentions,
        output_cross_attentions=output_cross_attentions,
        num_decoder_layers=self.module.config.num_decoder_layers,
        num_decoder_heads=self.module.config.num_heads,
        encoder_input_len=encoder_input_len,
        **decoder_params,
    )
    decodes = decode_outputs['decodes']
    scores = decode_outputs['log_prob']
    decoder_self_attentions = decode_outputs['decoder_self_attentions']
    cross_attentions = decode_outputs['cross_attentions']

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      out = {'scores': scores}
      if output_self_attentions:
        out['decoder_self_attentions'] = decoder_self_attentions
      if output_cross_attentions:
        out['cross_attentions'] = cross_attentions
      return decodes, out
    else:
      out = {'scores': scores[:, -1]}
      if output_self_attentions:
        out['decoder_self_attentions'] = decoder_self_attentions[:, -1, :]
      if output_cross_attentions:
        out['cross_attentions'] = cross_attentions[:, -1, :]
      return decodes[:, -1, :], out

  def _len_mse_loss_fn(self, tgt_len, pred_len, metrics):
    len_diff = tgt_len - pred_len
    len_loss = jnp.sum(jnp.power(len_diff, 2))
    metrics['len_loss'] = metrics_lib.AveragePerStep(total=len_loss)
    return len_loss

  def _len_xe_loss_fn(self, tgt_len, len_logits, metrics):
    tgt_idx = jnp.squeeze(tgt_len - 1)
    len_loss, _, _ = losses.compute_weighted_cross_entropy(
        len_logits,
        targets=tgt_idx,
        weights=None,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=None,
    )
    metrics['len_loss'] = metrics_lib.AveragePerStep(total=len_loss)
    return len_loss

  def _len_loss_fn(self, tgt_len, len_preds, metrics):
    if len_preds.shape[1] == 1:
      # Single float length prediction.
      return self._len_mse_loss_fn(tgt_len, len_preds, metrics)
    # Distribution over possible lengths.
    return self._len_xe_loss_fn(tgt_len, len_preds, metrics)

  def loss_fn(
      self,
      params,
      batch,
      dropout_rng,
  ):
    """Loss function used for training with a cross-entropy loss."""
    decoder_logits, pred_len, rejecter_logits = self._compute_logits(
        params, batch, dropout_rng
    )

    loss_normalizing_factor: Optional[
        Union[float, int, str, losses.SpecialLossNormalizingFactor]
    ]
    (loss_normalizing_factor, weights) = (
        losses.get_loss_normalizing_factor_and_weights(
            self._loss_normalizing_factor, batch
        )
    )
    assert weights is not None

    targets = batch['decoder_target_tokens']

    loss = jnp.zeros([])
    metrics = {}
    for i in range(self.num_train_stages):
      loss_i, z_loss_i, _ = losses.compute_weighted_cross_entropy(
          decoder_logits[i],
          targets=targets,
          weights=weights,
          label_smoothing=self._label_smoothing,
          z_loss=self._z_loss,
          loss_normalizing_factor=loss_normalizing_factor,
      )
      loss += loss_i
      if i == 0:
        metrics = self._compute_metrics(
            logits=decoder_logits[i],
            targets=targets,
            mask=weights,
            loss=loss_i,
            z_loss=z_loss_i,
        )
      else:
        metrics = compute_and_update_core_metrics(
            logits=decoder_logits[i],
            targets=targets,
            mask=weights,
            loss=loss_i,
            metrics=metrics,
            prefix='',
            i=i,
        )

      # Impose loss on all non-pad target positions for rejection and
      # subsequent stages.
      weights = jnp.asarray(targets > 0, jnp.float32)

      targets_mask = (targets > 0).astype(jnp.int32)
      predicted_tokens = jnp.argmax(decoder_logits[i], axis=-1) * targets_mask
      are_correct_tokens = (predicted_tokens == targets).astype(
          jnp.int32
      ) * targets_mask

      if rejecter_logits is not None:
        rejecter_loss_i, _, _ = losses.compute_weighted_cross_entropy(
            rejecter_logits[i],
            targets=are_correct_tokens,
            weights=weights,
            label_smoothing=self._label_smoothing,
            loss_normalizing_factor=loss_normalizing_factor,
        )
        loss += rejecter_loss_i
        metrics = compute_and_update_core_metrics(
            logits=rejecter_logits[i],
            targets=are_correct_tokens,
            mask=weights.astype(jnp.bool_),
            loss=rejecter_loss_i,
            metrics=metrics,
            prefix='rejecter_',
            i=i,
        )

    # [batch, 1]
    if pred_len is not None:
      tgt_len = jnp.sum(weights, axis=1, keepdims=True)
      loss += self._len_loss_fn(tgt_len, pred_len, metrics)
    return loss, metrics


def compute_and_update_core_metrics(
    logits,
    targets,
    mask,
    loss,
    metrics,
    prefix,
    i,
):
  """Compute core metrics and update metrics."""
  core_metrics = compute_core_metrics(
      logits=logits,
      targets=targets,
      mask=mask,
      loss=loss,
  )
  for k, v in core_metrics.items():
    metrics[f'{prefix}{k}_{i}'] = v
  return metrics


def compute_core_metrics(
    logits,
    targets,
    mask,
    loss,
):
  """Compute summary metrics.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
   mask: None or array of shape [batch, length]. Note: must consist of boolean
     values (float-valued weights not supported).
   loss: loss (float)

  Returns:
    Dict of metrics.
  """
  num_devices = jax.device_count()
  assert num_devices, 'JAX is reporting no devices, but it should.'
  # Note: apply mask again even though mask has already been applied to loss.
  # This is needed to divide by mask sum, but should not affect correctness of
  # the numerator.
  return {
      'accuracy': clu_metrics.Accuracy.from_model_output(
          logits=logits, labels=targets.astype(jnp.int32), mask=mask
      ),
      'loss': metrics_lib.AveragePerStep(total=loss),
  }
