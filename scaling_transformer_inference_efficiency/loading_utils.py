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

"""Handy import wrappers."""
import dataclasses
from functools import partial  # pylint: disable=g-importing-member
import logging
from typing import Callable, Optional

import jax
import numpy as np
import seqio
import t5x

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap


@dataclasses.dataclass
class ModelConfig:
  """An object to make gin file input elegant.

  ckpt_path: typically cns path
  size: 8, 62, 540
  quantized:
  generate_steps: Amount of steps to do generation with
  temperature: sampling temperature
  kv_cache_sharding: the degree of kv cache sharding (0: None, 1: Z, 2: YZ, 3:
    YZX)
  latency_collectives: whether to use latency optimised forms (double compute
    per step, half the steps for collective matmuls)
  batch_unsharded:  whether to shard batch dim
  shard_seqlen_vs_batch: whether to shard seqlen vs batch
  stream: An object to facilitate streaming back to X (you defined the
    callbacks).
  transpose_scan_axis: transpose if layers was not saved as the leading axis
  """
  ckpt_path: str
  size: int
  quantized: bool
  generate_steps: int
  temperature: float
  kv_cache_sharding: int
  latency_collectives: bool
  batch_unsharded: bool
  shard_seqlen_vs_batch: bool
  stream: Optional[incremental.StreamClient] = None
  transpose_scan_axis: bool = True


def return_minimal_palm(cfg):
  """Utility function to return a model.

  Args:
    cfg: A model configuration

  Returns:
    model: A model wrapper
    params: The params
    prefill_fn: Function to pass as prefill (to ensure it is compilation cached)
    generate_fn: Function to pass as generation (to ensure it is compilation
    cached)
  """

  if cfg.shard_seqlen_vs_batch and cfg.batch_unsharded:
    raise NotImplementedError(
        "Either shard seqlen instead of batch or don't shard batch.")

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

  rules = partitioning.make_rules_two_d(
      attn_batch_sharding, batch_unsharded=cfg.batch_unsharded)

  the_vocab = checkpoint.load_vocab()
  model = incremental.XmapModel(
      hparams, the_vocab.eos_id,
      partial(
          inference.infer_xmap,
          hparams,
          two_d_parallel_xmap.transformer_layer_weight_stationary,
          attn_all_to_all=attn_batch_sharding,
          latency_collectives=cfg.latency_collectives,
          batch_unsharded=cfg.batch_unsharded,
          shard_seqlen_vs_batch=cfg.shard_seqlen_vs_batch),
      params_spec.logical_axes(), rules, the_vocab)

  # actually load the weights
  with model.mesh, model.rules:
    params = params_spec.from_checkpoint(hparams, model.mesh, loaded_ckpt)

  logging.info('Weights loaded.')

  # cs2 = cs.replace(hparams = cs.hparams.replace(heads=64, padded_heads=32))
  params = model.rotate_weights(
      params, cfg.latency_collectives) if cfg.latency_collectives else params
  # Prepares them for xmap mode
  params = model.prepare_params(params)
  logging.info('Weights formatted.')

  generate_fn = model.instantiate_generating_fn(
      cfg.generate_steps,
      incremental.Sampling(cfg.temperature),
      batch_unsharded=cfg.batch_unsharded,
      stream=cfg.stream)

  prefill_fn = model.instantiate_prefill_fn()

  return model, params, prefill_fn, generate_fn


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
@dataclasses.dataclass
class InferenceT5X(t5x.models.DecoderOnlyModel):
  """Creates an API that fits T5X."""
  model: incremental.XmapModel
  params: weights.Weights
  prefill_fn: Callable
  generate_fn: Callable
  _input_vocabulary: seqio.Vocabulary
  _output_vocabulary: seqio.Vocabulary
  sample_ids: jax.Array
  max_input_length: int
  max_generate_length: int

  def __init__(self, cfg, _input_vocabulary, batch_size,
               task_feature_lengths):
    model, params, prefill_fn, generate_fn = return_minimal_palm(cfg)
    self.model = model
    self.params = params
    self.prefill_fn = prefill_fn
    self.generate_fn = generate_fn
    self._input_vocabulary = _input_vocabulary
    self._output_vocabulary = _input_vocabulary
    self.sample_ids = model.prepare_sample_ids(np.arange(batch_size))
    self.max_input_length = task_feature_lengths['inputs']
    self.max_generate_length = task_feature_lengths['targets']

  def predict_batch(self, batch):
    """Does an inference step.

    Args:
      batch: assumed to have fields {'decoder_causal_attention': int [batch,
        length], 'decoder_input_tokens': same}

    Returns:
      chunk: Our model input datatype
    """
    with jax.named_scope('make_batch'):
      first_chunk = self.make_batch(batch)
    with jax.named_scope('prefill'):
      prompt = self.model.prefill(self.params, self.prefill_fn, [], first_chunk)
    with jax.named_scope('generate'):
      output, _ = self.model.generate(self.params, self.generate_fn, [prompt],
                                      self.sample_ids)
    inferences = output.tokens, {
        'scores': None
    }  # none in place of scores for the moment
    return inferences

  def predict_batch_with_aux(self,
                             batch,
                             rng,
                             num_decodes=1,
                             return_all_decodes=True,
                             decoder_params=None):
    raise NotImplementedError

  def score_batch(self, batch):
    raise NotImplementedError

  def make_batch(self, batch):
    inputs_lengths = np.sum(batch['decoder_causal_attention'], axis=1) - 1
    masked_inputs = batch['decoder_input_tokens'] * batch[
        'decoder_causal_attention']
    inputs = masked_inputs[:, :self.max_input_length]
    return chunk.Chunk(inputs, inputs_lengths)
