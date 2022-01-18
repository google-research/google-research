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

"""Generates ConfigDict instances for the WMT transformer.

Most users will just create a ConfigDict instance with 'get_config' and then
override its parameters to specialize the configuration.
"""

import copy
import enum
from typing import Optional

import ml_collections
from aqt.utils import config_schema_utils

float_ph = config_schema_utils.float_ph
int_ph = config_schema_utils.int_ph
str_ph = config_schema_utils.str_ph
bool_ph = config_schema_utils.bool_ph


def get_wmt_base_config(use_auto_acts,
                        fp_quant):
  """Return a base ConfigDict which does not yet have fields for individual layers."""
  base_config = config_schema_utils.get_base_config(
      use_auto_acts, fp_quant=fp_quant)
  base_config.update({
      "learning_rate_schedule": {
          "factors": str_ph(),
          "base_learning_rate": float_ph(),
          "warmup_steps": int_ph(),
          "decay_factor": float_ph(),
          "steps_per_decay": int_ph(),
          "steps_per_cycle": int_ph(),
      },
      "per_host_batch_size": int_ph(),
      "num_train_steps": int_ph(),
      "beta1": float_ph(),
      "beta2": float_ph(),
      "eps": float_ph(),
      "random_seed": int_ph(),
      "hardware_rng": bool_ph(),
      "weight_outlier_regularization": float_ph(),
      "weight_outlier_regularization_regex": str_ph(),
      "prefer_int8_to_int32_dot": bool_ph(),
      "model_hparams": {
          "emb_dim": int_ph(),
          "num_heads": int_ph(),
          "qkv_dim": int_ph(),
          "mlp_dim": int_ph(),
          "share_embeddings": bool_ph(),
          "logits_via_embedding": bool_ph(),
      },
  })

  base_config.dense = config_schema_utils.get_dense_config(base_config)
  base_config.mlp_block = get_mlp_block_config(base_config)
  base_config.embedding = get_embedding_config(base_config)
  base_config.attention = get_attention_config(base_config)

  return base_config


def get_mlp_block_config(
    parent_config):
  """Create a ConfigDict corresponding to wmt_mlperf.models.MlpBlock.HParams."""
  config = ml_collections.ConfigDict()
  config_schema_utils.set_default_reference(
      config, parent_config, ["dense_1", "dense_2"], parent_field="dense")
  config.dense_2.quant_act.input_distribution = "positive"
  config.lock()
  return config


def get_fp_quant_hparams_config(
    quantized,
    quantized_reductions):
  """Create appropriate setting for 'QuantHParams' field used by softmax and LayerNormAQT."""
  if quantized_reductions and not quantized:
    raise ValueError("If `quantized` is False, `quantized_reductions` "
                     "must also be False.")
  if quantized:
    quant_hparams = ml_collections.ConfigDict({
        "prec":
            config_schema_utils.get_fp_config(),
        "reduction_prec":
            config_schema_utils.get_fp_config()
            if quantized_reductions else None
    })
  else:
    quant_hparams = None
  return quant_hparams


def get_softmax_config(quantized,
                       quantized_reductions):
  """Create a ConfigDIct corresponding to flax_attention.SoftmaxHParams."""
  quant_hparams = get_fp_quant_hparams_config(
      quantized=quantized, quantized_reductions=quantized_reductions)
  config = ml_collections.ConfigDict({
      "exp_hparams": None,
      "reciprocal_hparams": None,
      "quant_hparams": quant_hparams
  })
  config.lock()
  return config


def get_layer_norm_config(
    quantized, quantized_reductions):
  """Create a ConfigDict corresponding to flax_layers.LayerNormAqt.HParams."""
  quant_hparams = get_fp_quant_hparams_config(
      quantized=quantized, quantized_reductions=quantized_reductions)
  config = ml_collections.ConfigDict({"quant_hparams": quant_hparams})
  config.lock()
  return config


def set_global_softmax_config(
    base_config,
    softmax_config):
  """Sets all layers' softmax configuration in `base_config` to `softmax_config`."""

  def set_softmax(cfg):
    # We unlock the configs since the 'softmax' field might not already exist.
    with cfg.unlocked():
      cfg.softmax = copy.deepcopy(softmax_config)

  base_config = copy.deepcopy(base_config)
  for block in base_config.model_hparams.encoder.encoder_1d_blocks:
    set_softmax(block.attention.attn_acts)
  for block in base_config.model_hparams.decoder.encoder_decoder_1d_blocks:
    set_softmax(block.self_attention.attn_acts)
    set_softmax(block.enc_dec_attention.attn_acts)
  return base_config


def set_global_layer_norm_config(
    base_config,
    layer_norm_config):
  """Sets all layers' softmax configuration in `base_config` to `layer_norm_config`."""
  def set_layer_norm(cfg):
    # We unlock the configs since the 'layer_norm' field might not already
    # exist.
    with cfg.unlocked():
      cfg.layer_norm = copy.deepcopy(layer_norm_config)

  base_config = copy.deepcopy(base_config)
  set_layer_norm(base_config.model_hparams.encoder)
  set_layer_norm(base_config.model_hparams.decoder)
  for block in base_config.model_hparams.encoder.encoder_1d_blocks:
    set_layer_norm(block)
  for block in base_config.model_hparams.decoder.encoder_decoder_1d_blocks:
    set_layer_norm(block)
  return base_config


def get_attention_config(
    parent_config):
  """Create a ConfigDict corresponding to aqt.flax_attention.MutliHeadDotProductAttentionAqt.HParams."""
  config = ml_collections.ConfigDict()
  config_schema_utils.set_default_reference(
      config, parent_config, ["dense_kqv", "dense_out"], parent_field="dense")

  config.attn_acts = ml_collections.ConfigDict({})

  config_schema_utils.set_default_reference(config, parent_config,
                                            ["quant_type", "quant_act"])
  config_schema_utils.set_default_reference(config.attn_acts, config,
                                            ["quant_type"])
  config_schema_utils.set_default_reference(
      config.attn_acts,
      config, ["attn_act_q", "attn_act_k", "attn_act_v"],
      parent_field="quant_act")
  config.attn_acts.attn_act_probs = ml_collections.ConfigDict({
      "input_distribution": "positive",
      "bounds": 1.0,
      "half_shift": False,  # Set half_shift to false for positive distribution
  })
  config_schema_utils.set_default_reference(config.attn_acts.attn_act_probs,
                                            parent_config.quant_act, "prec")
  config.lock()
  return config


class BlockKind(enum.Enum):
  """Enum to distinguish between an encoder and a decoder block."""
  encoder = enum.auto()
  decoder = enum.auto()


def get_block_config(parent_config,
                     block_kind):
  """Create a ConfigDict corresponding to wmt_mlperf.models.Encoder[Decoder]1DBlock.HParams."""
  config = ml_collections.ConfigDict()
  config_schema_utils.set_default_reference(config, parent_config, "mlp_block")
  if block_kind == BlockKind.encoder:
    config_schema_utils.set_default_reference(config, parent_config,
                                              "attention")
  elif block_kind == BlockKind.decoder:
    config_schema_utils.set_default_reference(
        config,
        parent_config, ["self_attention", "enc_dec_attention"],
        parent_field="attention")
  else:
    raise ValueError(f"Unknown block_kind {block_kind}")
  config.lock()
  return config


def get_embedding_config(
    parent_config):
  """Create a ConfigDict corresponding to aqt.flax_layers.Embedding.HParams."""
  config = ml_collections.ConfigDict()
  config_schema_utils.set_default_reference(
      config, parent_config,
      ["weight_prec", "quant_type", "quant_act", "weight_half_shift"])
  config.lock()
  return config


def get_config(n_layers, use_auto_acts,
               fp_quant):
  """Returns a ConfigDict instance for a WMT transformer.

  The ConfigDict is wired up so that changing a field at one level of the
  hierarchy changes the value of that field everywhere downstream in the
  hierarchy. For example, changing the top-level 'prec' parameter
  (eg, config.prec=4) will cause the precision of all layers to change.
  Changing the precision of a specific layer type
  (eg, config.mlp_block.dense_1.weight_prec=4) will cause the weight precision
  of all Dense1 layers to change, overriding the value of the global
  config.prec value.

  See config_schema_test.test_schema_matches_expected to see the structure
  of the ConfigDict instance this will return.

  Args:
    n_layers: Number of layers in the encoder and the decoder.
    use_auto_acts: Whether to use automatic clipping bounds for activations or
      fixed bounds. Unlike other properties of the configuration which can be
      overridden directly in the ConfigDict instance, this affects the immutable
      schema of the ConfigDict and so has to be specified before the ConfigDict
      is created.
    fp_quant: Whether to use floating point quantization. Defaults to False for
      integer quantization.

  Returns:
    A ConfigDict instance which parallels the hierarchy of TrainingHParams.
  """
  base_config = get_wmt_base_config(
      use_auto_acts=use_auto_acts, fp_quant=fp_quant)
  model_hparams = base_config.model_hparams
  model_hparams.encoder = {
      "encoder_1d_blocks": [
          get_block_config(base_config, BlockKind.encoder)
          for _ in range(n_layers)
      ]
  }
  config_schema_utils.set_default_reference(model_hparams.encoder, base_config,
                                            "embedding")
  model_hparams.decoder = {
      "encoder_decoder_1d_blocks": [
          get_block_config(base_config, BlockKind.decoder)
          for _ in range(n_layers)
      ]
  }
  config_schema_utils.set_default_reference(model_hparams.decoder, base_config,
                                            "embedding")

  config_schema_utils.set_default_reference(model_hparams.decoder, base_config,
                                            "logits", parent_field="dense")
  base_config.lock()
  return base_config
