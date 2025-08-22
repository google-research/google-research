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

"""Handy import wrappers."""

import dataclasses
from enum import Enum  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
import logging
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from flax import struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from seqio.vocabularies import Vocabulary
from t5x import losses
from t5x.models import DecoderOnlyModel

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import sampling
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import one_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap


PyTree = Any


@struct.dataclass
class TestVocab:
  eos_id = 0
  bos_id = 0
  pad_id = 0

  def encode_tf(self, text):
    chars = np.array([ord(c) for c in text]).astype(np.int32)
    return chars

  def decode_tf(self, tokens):
    results = np.split(tokens, tokens.shape[0])
    return np.array([[chr(r) for r in list(line[0])] for line in results])


class Layout(Enum):
  TWO_D = 'two_d'
  ONE_D = 'one_d'
  WEIGHT_GATHERED = 'weight_gathered'


@dataclasses.dataclass
class ModelConfig:
  """An object to make gin file input elegant.

  ckpt_path: typically cns path
  size: 8, 62, 540
  quantized:
  generate_steps: Amount of steps to do generation with
  kv_cache_sharding: the degree of kv cache sharding (0: None, 1: Z, 2: YZ, 3:
    YZX)
  latency_collectives: whether to use latency optimised forms (double compute
    per step, half the steps for collective matmuls)
  batch_unsharded:  whether to shard batch dim
  shard_seqlen_vs_batch: whether to shard seqlen vs batch
  stream: An object to facilitate streaming back to X (you defined the
    callbacks).
  transpose_scan_axis: transpose if layers was not saved as the leading axis
  bos_id: Optionally overwrite bos_id to the model.
  """

  ckpt_path: str
  size: int
  quantized: bool
  generate_steps: int
  kv_cache_sharding: int
  latency_collectives: bool
  batch_unsharded: bool
  shard_seqlen_vs_batch: bool
  stream: Optional[incremental.StreamClient] = None
  transpose_scan_axis: bool = True
  layout: Layout = Layout.TWO_D
  bos_id: Optional[int] = None


def return_minimal_palm(
    cfg,
    params_already_loaded=False,
    remat = None,
    devices = None,
):  # pylint: disable = g-bare-generic, line-too-long
  """Utility function to return a model.

  Args:
    cfg: A model configuration
    params_already_loaded: whether params have been loaded yet
    remat: Whether to remat the layer, used for training.
      jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      jax.checkpoint_policies.nothing_saveable
    devices: devices to make a mesh from

  Returns:
    model: A model wrapper
    params: The params
    prefill_fn: Function to pass as prefill (to ensure it is compilation cached)
    generate_fn: Function to pass as generation (to ensure it is compilation
    cached)
  """
  one_d = cfg.layout == Layout.ONE_D
  if cfg.shard_seqlen_vs_batch and cfg.batch_unsharded:
    raise NotImplementedError(
        "Either shard seqlen instead of batch or don't shard batch."
    )

  del remat  # for the moment, always remat
  # We have preset sizes
  if cfg.size == 0:
    hparams = checkpoint.HParams.TOY
  if cfg.size == 8:
    hparams = checkpoint.HParams.PALM_8B
  elif cfg.size == 62:
    hparams = checkpoint.HParams.PALM_62B
  elif cfg.size == 540:
    hparams = checkpoint.HParams.PALM_540B

  if cfg.quantized:
    ckpt = checkpoint.QuantizedCheckpoint
    params_spec = weights.QuantizedWeights
  else:
    ckpt = checkpoint.Checkpoint
    params_spec = weights.Weights

  if cfg.size == 0:
    loaded_ckpt = ckpt.init_zero(hparams)
  else:
    spec = checkpoint.CheckpointSpec(
        hparams=hparams,
        dir=cfg.ckpt_path,
        transpose_scan_axis=cfg.transpose_scan_axis,
    )
    loaded_ckpt = ckpt.load_spec(spec)

  if cfg.kv_cache_sharding == 0:
    attn_batch_sharding = partitioning.AttnAllToAll.NONE
  elif cfg.kv_cache_sharding == 1:
    attn_batch_sharding = partitioning.AttnAllToAll.AXIS_Z
  elif cfg.kv_cache_sharding == 2:
    attn_batch_sharding = partitioning.AttnAllToAll.AXES_YZ
  elif cfg.kv_cache_sharding == 3:
    attn_batch_sharding = partitioning.AttnAllToAll.AXES_YZX
  else:
    raise NotImplementedError

  if cfg.layout == Layout.TWO_D:
    rules = partitioning.make_rules_two_d(
        attn_batch_sharding, batch_unsharded=cfg.batch_unsharded
    )
    layer_fn = partial(
        two_d_parallel_xmap.transformer_layer_weight_stationary,
        attn_all_to_all=attn_batch_sharding,
        latency_collectives=cfg.latency_collectives,
        shard_seqlen_vs_batch=cfg.shard_seqlen_vs_batch,
        batch_unsharded=cfg.batch_unsharded,
    )
    # sample_fn = partial(sampling.sample_manual,
    # batch_unsharded=cfg.batch_unsharded)
    sample_fn = sampling.sample

  elif cfg.layout == Layout.ONE_D:
    rules = partitioning.make_rules_one_d()
    layer_fn = partial(
        one_d_parallel_xmap.weight_stationary_simple,
        latency_collectives=cfg.latency_collectives,
    )
    sample_fn = sampling.sample_manual_batch_unsharded
  elif cfg.layout == Layout.WEIGHT_GATHERED:
    rules = partitioning.make_rules_weight_gathered()
    sample_fn = sampling.sample
    raise NotImplementedError
  else:
    raise NotImplementedError

  if cfg.size == 0:
    the_vocab = TestVocab()
  else:
    the_vocab = checkpoint.load_vocab()

  mesh = partitioning.make_mesh(one_d=one_d, devices=devices)
  sharding_config = partitioning.ShardingConfig(
      mesh=mesh,
      attn_all_to_all=attn_batch_sharding,
      latency_collectives=cfg.latency_collectives,
      shard_seqlen_vs_batch=cfg.shard_seqlen_vs_batch,
      batch_unsharded=cfg.batch_unsharded,
  )

  embed_fn = partial(
      two_d_parallel_xmap.embed_manual,
      shard_seqlen_vs_batch=cfg.shard_seqlen_vs_batch,
      batch_unsharded=cfg.batch_unsharded,
      one_d=one_d,
  )

  unembed_fn = partial(
      two_d_parallel_xmap.unembed_manual,
      batch_unsharded=cfg.batch_unsharded,
      one_d=one_d,
  )

  forward_pass = partial(
      inference.manual_fwd_pass,
      hparams,
      sharding_config,
      embed_fn,
      layer_fn,
      unembed_fn,
  )

  infer_stack = partial(
      inference.infer_template,
      hparams,
      sharding_config,
      forward_pass,
  )

  model = incremental.InferenceModel(
      hparams,
      the_vocab.eos_id,
      infer_stack,
      sample_fn,
      mesh,
      rules,
      the_vocab,
      bos_id=cfg.bos_id,
  )

  generate_fn = model.instantiate_generating_fn(cfg.generate_steps)
  prefill_fn = model.instantiate_prefill_fn()

  if params_already_loaded:
    return model, None, prefill_fn, generate_fn
  else:
    # actually load the weights
    with model.mesh, model.rules:
      params = params_spec.from_checkpoint(hparams, model.mesh, loaded_ckpt)

    logging.info('Weights loaded.')

    # cs2 = cs.replace(hparams = cs.hparams.replace(heads=64, padded_heads=32))
    params = (
        model.rotate_weights(params, cfg.latency_collectives)
        if cfg.latency_collectives
        else params
    )
    logging.info('Weights formatted.')
  return model, params, prefill_fn, generate_fn


@jax.jit
def find_common_prefix(tokens):
  # find a common prefix
  base_case = tokens[0, :]
  is_equal = jnp.int8(tokens == base_case)  # broadcasts across the batch
  equal_at = jnp.prod(is_equal, axis=0)  # get a single dimensional array
  cp = jnp.cumprod(equal_at, 0)
  first_non_equal = jnp.argmin(cp)  # will get the first 0
  return first_non_equal


@jax.jit
def ce_loss(
    score_result, batch
):
  """Cross entropy loss."""
  token_scores = (
      -losses.cross_entropy_with_logits(
          score_result.logits,
          common_utils.onehot(
              batch['decoder_target_tokens'],
              score_result.logits.shape[-1],
              on_value=1,
              off_value=0,
          ),
          z_loss=0.0,
      )[0]
      * batch['decoder_loss_weights']
  )
  return token_scores


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
@dataclasses.dataclass
class InferenceT5X(DecoderOnlyModel):
  """Creates an API that fits T5X."""

  model: incremental.InferenceModel
  params: weights.Weights
  prefill_fn: Callable
  generate_fn: Callable
  _batch_size: int
  _input_vocabulary: Vocabulary
  _output_vocabulary: Vocabulary
  sample_ids: jax.Array
  max_input_length: int
  max_generate_length: int

  def __init__(
      self,
      cfg,
      _input_vocabulary,
      batch_size,
      task_feature_lengths,
  ):
    model, params, prefill_fn, generate_fn = return_minimal_palm(cfg)  # pylint: disable = unbalanced-tuple-unpacking
    self.model = model
    self.params = params
    self.prefill_fn = prefill_fn
    self.generate_fn = generate_fn
    self.get_logits_fn = model.instantiate_prefill_fn(return_full_chunk=True)
    self._batch_size = batch_size
    self._input_vocabulary = _input_vocabulary
    self._output_vocabulary = _input_vocabulary
    self.max_input_length = task_feature_lengths['inputs']
    self.max_generate_length = task_feature_lengths['targets']

    # make a custom model for the common_prefix / prefill sections
    # this is only function defs not params
    prefix_model_cfg = dataclasses.replace(
        cfg, kv_cache_sharding=0, batch_unsharded=True
    )
    prefix_model, _, prefix_prefill_fn, _ = return_minimal_palm(
        prefix_model_cfg, params_already_loaded=True
    )
    self.prefix_model = prefix_model
    self.prefix_prefill_fn = prefix_prefill_fn

  def predict_batch(self, params, batch):
    """Does an inference step.

    Args:
      params: Pytree definition of weights
      batch: assumed to have fields {'decoder_causal_attention': int [batch,
        length], 'decoder_input_tokens': same}

    Returns:
      inferences: (output.tokens, {'scores': output_result.per_token_scores})
      tokens is either [batch, tokens] or [batch, num_decodes, tokens]
    """

    return self.predict_batch_with_aux(params, batch)

  def predict_batch_with_aux(
      self,
      params,
      batch,
      rng = None,
      num_decodes = 1,
      temperature = 0.7,
      return_all_decodes = True,
      decoder_params=None,
  ):
    with jax.named_scope('make_batch'):
      prefix, prompt = self.make_batch(batch)
    processed_cache = self.process_cache(params, prompt, prefix)
    with jax.named_scope('generate'):
      sample_hyperparams = sampling.SamplingHyperParams(temperature=temperature)
      sample_ids = np.arange(self._batch_size * num_decodes)
      output, output_result = self.model.generate(
          params,
          self.generate_fn,
          processed_cache,
          sample_ids,
          sample_hyperparams,
      )

    if num_decodes > 1:
      tokens = output.tokens.reshape((self._batch_size, num_decodes, -1))
      scores = output_result.per_token_scores.sum(-1).reshape(
          (self._batch_size, num_decodes)
      )
    else:
      tokens = output.tokens
      scores = output_result.per_token_scores.sum(-1)

    inferences = tokens, {
        'scores': scores
    }  # none in place of scores for the moment

    return inferences

  def score_batch(
      self,
      params,
      batch,
      return_intermediates = False,
  ):
    inputs_lengths = np.sum(batch['decoder_causal_attention'], axis=1) - 1
    masked_inputs = (
        batch['decoder_input_tokens'] * batch['decoder_causal_attention']
    )
    score_chunk = chunk.Chunk(masked_inputs, inputs_lengths)  # [batch, time]

    # TODO(sholto): We could play the common prefix trick here too
    score_result = self.model.prefill(
        self.params, self.get_logits_fn, [], score_chunk
    )
    # TODO(sholto): Test if manual version made for cascades uses less memory
    token_scores = ce_loss(score_result, batch)
    sequence_scores = token_scores.sum(-1)
    return sequence_scores

  def make_batch(
      self,
      batch,
      extract_prefix = False,
      common_prefix_heuristic = 32,
  ):
    inputs_lengths = np.sum(batch['decoder_causal_attention'], axis=1) - 1
    masked_inputs = (
        batch['decoder_input_tokens'] * batch['decoder_causal_attention']
    )
    inputs = masked_inputs[:, : self.max_input_length]  # [batch, time]

    if extract_prefix:
      # NB: the below is not jax jittable.
      common_prefix = find_common_prefix(inputs)  # integer
      # Heuristic for whether prefix extraction is worth doing
      if (common_prefix > common_prefix_heuristic) and (
          self.max_input_length - common_prefix_heuristic > common_prefix
      ):
        logging.info('Detected common prefix of length %i', common_prefix)
        prefix = chunk.Chunk(
            jnp.expand_dims(inputs[0, :common_prefix], 0),
            jnp.array([common_prefix]),
        )
        prompt = chunk.Chunk(
            inputs[:, common_prefix:], inputs_lengths - common_prefix
        )
        return prefix, prompt
    # Default to no prefix extraction
    prompt = chunk.Chunk(inputs, inputs_lengths)
    prefix = None
    return prefix, prompt

  def process_cache(
      self, params, prompt, prefix=None
  ):
    processed_cache = []
    if prefix is not None:
      with jax.named_scope('common_prefill'):
        # the common prefix will be batch size 1, shard appropriately
        common_prefix = self.prefix_model.prefill(
            params, self.prefix_prefill_fn, [], prefix
        )
        processed_cache.append(common_prefix)
    with jax.named_scope('different_prefill'):
      prompt = self.model.prefill(
          params, self.prefill_fn, processed_cache, prompt
      )
      processed_cache.append(prompt)
    return processed_cache
