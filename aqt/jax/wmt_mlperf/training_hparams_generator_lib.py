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

"""Convenience script to generate commonly used hparams configs.
"""

# TODO(b/175245107): Remove this file.

import dataclasses
import enum
import pathlib
import re
import time
import typing
from typing import Optional, Type, TypeVar

from absl import flags

from aqt.jax import flax_attention as aqt_flax_attention
from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import quant_config
from aqt.jax.flax import struct as flax_struct
from aqt.jax.quantization import ActsBoundT
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType
from aqt.jax.wmt_mlperf import models
from aqt.jax.wmt_mlperf import training_hparams
from aqt.utils import hparams_utils as os_hparams_utils

T = TypeVar('T')


# Default fixed bound and getbounds coefficients
# These values were determined empirically to be reasonable defaults.

DEFAULTS = {}
DEFAULTS['mlp_dense_1'] = {
    'fixed': 1.0,
    'absdev': 2.5,
    'stddev': 2.5,
    'mix': 1.0
}

DEFAULTS['mlp_dense_2'] = {
    'fixed': 6.0,
    'absdev': 13.0,
    'stddev': 13.0,
    'mix': 1.0
}

DEFAULTS['att_dense_kqv'] = {
    'fixed': 1.0,
    'absdev': 4.0,
    'stddev': 5.0,
    'mix': 0.0
}

DEFAULTS['att_dense_out'] = {
    'fixed': 2.0,
    'absdev': 0.1,
    'stddev': 0.3,
    'mix': 0.0
}

DEFAULTS['logits'] = {  #
    'fixed': 1.0,
    'absdev': 3.0,
    'stddev': 3.0,
    'mix': 0.0
}

DEFAULTS['attn_act_k'] = {
    'fixed': 1.0,
    'absdev': 2.0,
    'stddev': 1.0,
    'mix': 0.0
}

DEFAULTS['attn_act_q'] = {
    'fixed': 1.0,
    'absdev': 2.0,
    'stddev': 2.0,
    'mix': 0.0
}

DEFAULTS['attn_act_v'] = {
    'fixed': 6.0,
    'absdev': 6.0,
    'stddev': 5.5,
    'mix': 0.0
}

flags.DEFINE_string(
    'filename',
    default=None,
    help='Where to store the created hparams json file.')


# Base configuration flags
class BaseConfigSize(enum.Enum):
  """Specifies the size of the transformer."""

  MINIMAL_MODEL = enum.auto()
  SMALL_MODEL = enum.auto()
  FULL_MODEL = enum.auto()


# TODO(malmaud): Get rid of these flags that implicitly define a base config
# hparams instance (specifically, base_config_size, base_config_quant_target,
# and base_config_prec). Instead, only hparams_config_filename will be used.
flags.DEFINE_enum_class(
    'base_config_size',
    default=BaseConfigSize.SMALL_MODEL,
    enum_class=BaseConfigSize,
    help=(
        'Model size of the base configuration. '
        'FULL_MODEL creates a 6-layer model. '
        'SMALL_MODEL creates a smaller 2-layer model with channel dimensions '
        'and number of heads reduced by half compared to the full model.'
        'MINIMAL_MODEL creates a 1-layer model with minimal dimensions for '
        'testing. See training_hparams.create_from_base_config() for exact values'
        ' for each model size.'))


class BaseConfigQuantTarget(enum.Enum):
  """Specifies which part of the model to quantize."""

  NONE = enum.auto()
  WEIGHTS_ONLY = enum.auto()  # only weights get quantized
  WEIGHTS_AND_SOME_ACTS_1 = enum.auto()  # All matmuls in attention and MLP
  # blocks get quantized, except act-act matmuls i.e.: K*Q and softmax*V
  # fixed bound is used for activation quantization.
  WEIGHTS_AND_SOME_ACTS_2 = enum.auto()  # This is WEIGHTS_AND_SOME_ACTS_1 and
  # additionaly quantizes logits inputs for logits_via_embedding.
  WEIGHTS_AND_ACTS = enum.auto()  # This quantizes all the weights and
  # activations for matmul.


flags.DEFINE_enum_class(
    'base_config_quant_target',
    default=BaseConfigQuantTarget.NONE,
    enum_class=BaseConfigQuantTarget,
    help=(
        'Which aspects of the model to quantize. '
        'NONE creates a model with no quantization. '
        'WEIGHTS_ONLY creates a model where all weights in all layers are quantized. '
        'WEIGHTS_AND_SOME_ACTS_1 creates a model where all weights are quantized, and most '
        'activations are quantized (the MLP dense layers and the KQV and output '
        'operations within the attention layers.'
        'WEIGHTS_AND_SOME_ACTS_2 creates a "WEIGHTS_AND_SOME_ACTS_1" model and '
        'additionally quantizes logits inputs for logits_via_embedding.'
        'WEIGHTS_AND_ACTS creates a model where all weights and activations are'
        'quantized for matmul.'))

flags.DEFINE_string(
    'base_config_prec',
    default='-1',
    help=('Precision of the quantization. Could either be specified as an '
          '(1) integer, (2) a string describing a reduced precision '
          'floating-point type. Applies to all weights and '
          'to the activations specified by the base_config_quant_target flag.'
          'If base_config_quant_target is BaseConfigQuantTarget.NONE, please'
          ' set this flag to -1. Details of formats (1) and (2): '
          '(1) Number of bits to use for quantization. (2) A string of the '
          'form "fp_quant:scaled=<true|false>,exp_min=<integer>,exp_max='
          '<integer>,sig_bits=<integer>".'))


@dataclasses.dataclass
class BaseConfig:
  """Base config parameters."""
  size: BaseConfigSize
  prec: QuantOps.PrecT
  quant_target: BaseConfigQuantTarget

  @classmethod
  def create_from_flags(cls):
    return cls(
        size=FLAGS.base_config_size,
        prec=parse_base_config_prec(FLAGS.base_config_prec),
        quant_target=FLAGS.base_config_quant_target)


class ConfigGenerator(enum.Enum):
  """Specifies which configs to generate."""
  BFLOAT16 = enum.auto()  # will generate configs for bfloat16 models in three
  # model sizes.
  WEIGHTS_ONLY = enum.auto()
  WEIGHTS_AND_FIXED_ACTS = enum.auto()
  WEIGHTS_AND_AUTO_ACTS = enum.auto()


flags.DEFINE_enum_class(
    'config_generator',
    default=None,
    enum_class=ConfigGenerator,
    help='Name of config generation method to call.'
    'BFLOAT16 creates the bfloat16 configs in three model sizes.')


def parse_base_config_prec(prec):
  """Parse FLAGS.base_config_prec flag value.

  Examples prec values:
    - '-1' : Special value -1, is documented in FLAGS.base_config_prec and
      corresponds to (prec is None).
    - '4'
    - '8'
    - 'fp_quant:scaled=true,exp_min=-11,exp_max=4,sig_bits=3'

  Args:
    prec: a string representing the requested precision.

  Returns:
    Parsed string converted to a QuantOps.PrecT.
  """
  if prec is None:
    return None
  try:
    prec_val = int(prec)
    return None if prec_val == -1 else prec_val
  except ValueError:
    fp_quant_re = re.compile(
        r"""
    fp_quant:
    scaled=(true|false),
    exp_min=(-?[0-9]+),
    exp_max=(-?[0-9]+),
    sig_bits=([0-9]+)""", re.X)
    match_value = fp_quant_re.match(prec)
    if not match_value:
      raise ValueError(f'Failed to parse precision {prec}.')
    return QuantOps.FloatQuant(
        is_scaled=match_value.group(1) == 'true',
        fp_spec=QuantOps.FloatQuant.FloatPrec(
            exp_min=int(match_value.group(2)),
            exp_max=int(match_value.group(3)),
            sig_bits=int(match_value.group(4))))


FLAGS = flags.FLAGS

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


# TODO(wanglisa): Add additional parameters to this that are still currently
# defined in the transformer_kwargs dict in train.py.
def create_base_transformer_hparams(
    *,  #
    mlp_weight_prec,
    embedding_weight_prec,
    attention_weight_prec,
    mlp_pos_inputs_prec,
    mlp_pos_inputs_hyper,
    mlp_signed_inputs_prec,
    mlp_signed_inputs_hyper,
    attention_kqv_inputs_prec,
    attention_kqv_inputs_hyper,
    attention_out_inputs_prec,
    attention_out_inputs_hyper,
    attention_act_q_inputs_prec,
    attention_act_q_inputs_hyper,
    attention_act_k_inputs_prec,
    attention_act_k_inputs_hyper,
    attention_act_probs_inputs_prec,
    attention_act_v_inputs_prec,
    attention_act_v_inputs_hyper,
    logits_inputs_prec,
    logits_inputs_hyper,
    logits_via_embeddings,
    num_layers,
    num_heads,
    emb_dim,
    qkv_dim,
    mlp_dim,
    quant_type,
    half_shift = False,
):
  """Configure a transformer based on high-level configuration parameters.

  The encoder and decoder share hyperparameters.

  Args:
    mlp_weight_prec: Weight precision used by the MLP layers. `None` indicates
      no quantization.
    embedding_weight_prec: Weight precision used for weights in the embedding
      layers. `None` indicates no quantization.
    attention_weight_prec: Weight precision used for weights in the attention
      layers. `None` indicates no quantization.
    mlp_pos_inputs_prec: Positive input precision for the dense layers in the
      encoder and decoder.
    mlp_pos_inputs_hyper: Activation hyperparam for positive input quantization
      for the dense layers in the encoder and decoder.
    mlp_signed_inputs_prec: Signed input precision for the dense layers in the
      encoder and decoder.
    mlp_signed_inputs_hyper: Activation hyperparam for signed input quantization
      for the dense layers in the encoder and decoder.
    attention_kqv_inputs_prec: Input precision for the attention k in the
      encoder and decoder.
    attention_kqv_inputs_hyper: Activation hyperparam for input quantization for
      the attention k in the encoder and decoder.
    attention_out_inputs_prec: Input precision for the attention out in the
      encoder and decoder.
    attention_out_inputs_hyper: Activation hyperparam for input quantization for
      the attention out in the encoder and decoder.
    attention_act_q_inputs_prec: Input precision for the attention query(Q) for
      Q*K in the encoder and decoder.
    attention_act_q_inputs_hyper: Activation hyperparam for input query(Q)
      quantization for the attention Q*K in the encoder and decoder.
    attention_act_k_inputs_prec: Input precision for the attention key(K) for
      Q*K in the encoder and decoder.
    attention_act_k_inputs_hyper: Activation hyperparam for input key(K)
      quantization for the attention Q*K in the encoder and decoder.
    attention_act_probs_inputs_prec: Input precision for the attention weights
      i.e. Softmax(Q*K) for Softmax(Q*K)*V  in the encoder and decoder.
    attention_act_v_inputs_prec: Input precision for the attention value(V) for
      Softmax(Q*K)*V in the encoder and decoder.
    attention_act_v_inputs_hyper: Activation hyperparam for input value(V)
      quantization for the attention Softmax(Q*K)*V in the encoder and decoder.
    logits_inputs_prec: Input precision for the logits in decoder.
    logits_inputs_hyper: Activation hyperparam for input quantization for logits
      in decoder.
    logits_via_embeddings: Whether to compute logits using the embeddings
      weights.
    num_layers: Number of layers in the encoder and decoder.
    num_heads: number of heads.
    emb_dim: dimension of embedding.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    quant_type: quantization strategy for quantizing the model, one of
      `fake_quant` or `aqt`.
    half_shift: Half_shift flag used for both weights and inputs in the model.

  Returns:
    An instance of models.Transformer.HParams that contains a complete
      configuration of an encoder-decoder transformer.
  """
  embedding = aqt_flax_layers.EmbedAqt.HParams(
      weight_prec=embedding_weight_prec,
      weight_half_shift=half_shift,
      quant_act=QuantOps.ActHParams(
          input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
          prec=logits_inputs_prec,
          half_shift=half_shift,
          bounds=logits_inputs_hyper),
      quant_type=quant_type)
  # TODO(malmaud): Add flags to control weight_quant_granularity here and below.
  attention_kqv_dense = aqt_flax_layers.DenseAqt.HParams(
      weight_prec=attention_weight_prec,
      weight_half_shift=half_shift,
      quant_act=QuantOps.ActHParams(
          input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
          prec=attention_kqv_inputs_prec,
          half_shift=half_shift,
          bounds=attention_kqv_inputs_hyper),
      quant_type=quant_type,
      weight_quant_granularity=quant_config.QuantGranularity.per_channel)
  attention_out_dense = aqt_flax_layers.DenseAqt.HParams(
      weight_prec=attention_weight_prec,
      weight_half_shift=half_shift,
      quant_act=QuantOps.ActHParams(
          input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
          prec=attention_out_inputs_prec,
          half_shift=half_shift,
          bounds=attention_out_inputs_hyper),
      quant_type=quant_type,
      weight_quant_granularity=quant_config.QuantGranularity.per_channel)

  # TODO(b/169178846): We are quantizing attention_act_probs with unsigned
  # fixed bounds equal to [0.0, 1.0] to match the range of softmax, since
  # attn_act_probs is Softmax(Q*K) .
  attention_act_probs_inputs_hyper = 1.0
  dot_product_attention = aqt_flax_attention.DotProductAttnHParams(
      attn_act_q=QuantOps.ActHParams(
          QuantOps.ActHParams.InputDistribution.symmetric,
          prec=attention_act_q_inputs_prec,
          half_shift=half_shift,
          bounds=attention_act_q_inputs_hyper),
      attn_act_k=QuantOps.ActHParams(
          QuantOps.ActHParams.InputDistribution.symmetric,
          prec=attention_act_k_inputs_prec,
          half_shift=half_shift,
          bounds=attention_act_k_inputs_hyper),
      attn_act_probs=QuantOps.ActHParams(
          QuantOps.ActHParams.InputDistribution.positive,
          prec=attention_act_probs_inputs_prec,
          half_shift=half_shift,
          bounds=attention_act_probs_inputs_hyper),
      attn_act_v=QuantOps.ActHParams(
          QuantOps.ActHParams.InputDistribution.symmetric,
          prec=attention_act_v_inputs_prec,
          half_shift=half_shift,
          bounds=attention_act_v_inputs_hyper),
      quant_type=quant_type,
      softmax=None)
  # TODO(malmaud): Move these other create_from* methods to this file.

  attention = aqt_flax_attention.MultiHeadDotProductAttentionAqt.HParams(
      dense_kqv=attention_kqv_dense,
      dense_out=attention_out_dense,
      attn_acts=dot_product_attention)
  dense_1 = aqt_flax_layers.DenseAqt.HParams(
      weight_prec=mlp_weight_prec,
      weight_half_shift=half_shift,
      quant_act=QuantOps.ActHParams(
          input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
          prec=mlp_signed_inputs_prec,
          half_shift=half_shift,
          bounds=mlp_signed_inputs_hyper),
      quant_type=quant_type,
      weight_quant_granularity=quant_config.QuantGranularity.per_channel)
  dense_2 = aqt_flax_layers.DenseAqt.HParams(
      weight_prec=mlp_weight_prec,
      weight_half_shift=half_shift,
      quant_act=QuantOps.ActHParams(
          input_distribution=QuantOps.ActHParams.InputDistribution.positive,
          prec=mlp_pos_inputs_prec,
          half_shift=half_shift,
          bounds=mlp_pos_inputs_hyper),
      quant_type=quant_type,
      weight_quant_granularity=quant_config.QuantGranularity.per_channel)
  layer_norm = aqt_flax_layers.LayerNormAqt.HParams(quant_hparams=None)
  mlp_block = models.MlpBlock.HParams(dense_1=dense_1, dense_2=dense_2)
  encoder_block = models.Encoder1DBlock.HParams(
      mlp_block, attention, layer_norm=layer_norm)
  decoder_block = models.EncoderDecoder1DBlock.HParams(
      mlp_block,
      self_attention=attention,
      enc_dec_attention=attention,
      layer_norm=layer_norm)
  encoder = models.Encoder.HParams.create_from_block_template(
      embedding=embedding,
      block_template=encoder_block,
      num_layers=num_layers,
      layer_norm=layer_norm)
  if logits_via_embeddings:
    logits_hparams = None
  else:
    logits_hparams = aqt_flax_layers.DenseAqt.HParams(
        weight_prec=embedding_weight_prec,
        weight_half_shift=half_shift,
        quant_act=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
            prec=logits_inputs_prec,
            half_shift=half_shift,
            bounds=logits_inputs_hyper),
        quant_type=quant_type,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)
  decoder = models.Decoder.HParams.create_from_block_template(
      embedding=embedding,
      block_template=decoder_block,
      num_layers=num_layers,
      logits=logits_hparams,
      layer_norm=layer_norm)
  return models.Transformer.HParams(
      encoder=encoder,
      decoder=decoder,
      emb_dim=emb_dim,
      num_heads=num_heads,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      share_embeddings=True,
      logits_via_embedding=logits_via_embeddings,
  )


def create_training_hparams_from_base_config(
    base_config):
  """Creates TrainingHParams instance from BaseConfig instance."""

  num_layers, emb_dim, num_heads, qkv_dim, mlp_dim = None, None, None, None, None
  if base_config.size == BaseConfigSize.MINIMAL_MODEL:
    num_layers = 1
    emb_dim = 1
    num_heads = 1
    qkv_dim = 1
    mlp_dim = 1
  elif base_config.size == BaseConfigSize.SMALL_MODEL:
    num_layers = 3
    emb_dim = 512
    num_heads = 8
    qkv_dim = 512
    mlp_dim = 2048
  elif base_config.size == BaseConfigSize.FULL_MODEL:
    num_layers = 6
    emb_dim = 1024
    num_heads = 16
    qkv_dim = 1024
    mlp_dim = 4096
  else:
    raise ValueError(f'Unknown base config size {base_config.size}')

  prec = base_config.prec

  # Check that precision and quantization target are consistent
  if prec is None and base_config.quant_target != BaseConfigQuantTarget.NONE:
    raise ValueError(
        f'Cannot set quantization target to {base_config.quant_target} when requested precision is None.'
    )

  if base_config.quant_target == BaseConfigQuantTarget.NONE and prec is not None:
    raise ValueError(
        f'Cannot set quantization precision to {prec} when quantization target is NONE.'
    )

  # This variable keeps track of whether we know how to handle the
  # quantization target. If we don't, we want to raise an error to avoid
  # silently using defaults.
  recognized_quant_target = False
  if base_config.quant_target == BaseConfigQuantTarget.NONE:
    recognized_quant_target = True

  # Weight quantization
  mlp_weight_prec = None
  embedding_weight_prec = None
  attention_weight_prec = None
  if base_config.quant_target in [
      BaseConfigQuantTarget.WEIGHTS_ONLY,
      BaseConfigQuantTarget.WEIGHTS_AND_SOME_ACTS_1,
      BaseConfigQuantTarget.WEIGHTS_AND_SOME_ACTS_2,
      BaseConfigQuantTarget.WEIGHTS_AND_ACTS,
  ]:
    recognized_quant_target = True
    mlp_weight_prec = prec
    embedding_weight_prec = prec
    attention_weight_prec = prec

  # Activation quantization
  mlp_pos_inputs_prec = None
  mlp_pos_inputs_hyper_fixed_bound = None
  mlp_signed_inputs_prec = None
  mlp_signed_inputs_hyper_fixed_bound = None
  attention_kqv_inputs_prec = None
  attention_kqv_inputs_hyper_fixed_bound = None
  attention_out_inputs_prec = None
  attention_out_inputs_hyper_fixed_bound = None
  logits_inputs_prec = None
  logits_inputs_hyper_fixed_bound = None
  attention_act_q_inputs_prec = None
  attention_act_q_inputs_hyper_fixed_bound = None
  attention_act_k_inputs_prec = None
  attention_act_k_inputs_hyper_fixed_bound = None
  attention_act_probs_inputs_prec = None
  attention_act_v_inputs_prec = None
  attention_act_v_inputs_hyper_fixed_bound = None
  if base_config.quant_target in [
      BaseConfigQuantTarget.WEIGHTS_AND_SOME_ACTS_1,
      BaseConfigQuantTarget.WEIGHTS_AND_SOME_ACTS_2,
      BaseConfigQuantTarget.WEIGHTS_AND_ACTS,
  ]:
    recognized_quant_target = True
    mlp_pos_inputs_prec = prec
    mlp_signed_inputs_prec = prec
    attention_kqv_inputs_prec = prec
    attention_out_inputs_prec = prec

    mlp_signed_inputs_hyper_fixed_bound = DEFAULTS['mlp_dense_1']['fixed']
    mlp_pos_inputs_hyper_fixed_bound = DEFAULTS['mlp_dense_2']['fixed']
    attention_kqv_inputs_hyper_fixed_bound = DEFAULTS['att_dense_kqv']['fixed']
    attention_out_inputs_hyper_fixed_bound = DEFAULTS['att_dense_out']['fixed']

  if base_config.quant_target in [
      BaseConfigQuantTarget.WEIGHTS_AND_SOME_ACTS_2,
      BaseConfigQuantTarget.WEIGHTS_AND_ACTS,
  ]:
    logits_inputs_prec = prec
    logits_inputs_hyper_fixed_bound = DEFAULTS['logits']['fixed']

  if base_config.quant_target == BaseConfigQuantTarget.WEIGHTS_AND_ACTS:
    # TODO(shivaniagrawal): experimentally find reasonable default bounds.
    attention_act_q_inputs_prec = prec
    attention_act_q_inputs_hyper_fixed_bound = DEFAULTS['attn_act_q']['fixed']
    attention_act_k_inputs_prec = prec
    attention_act_k_inputs_hyper_fixed_bound = DEFAULTS['attn_act_k']['fixed']
    attention_act_probs_inputs_prec = prec
    attention_act_v_inputs_prec = prec
    attention_act_v_inputs_hyper_fixed_bound = DEFAULTS['attn_act_v']['fixed']

  if not recognized_quant_target:
    raise ValueError(f'Unknown quantization target {base_config.quant_target}')

  learning_rate_scheduler_hparams = training_hparams.LearningRateSchedulerHParams(
      factors='constant * linear_warmup * rsqrt_decay',
      base_learning_rate=0.0625,
      warmup_steps=1000,
      decay_factor=0.5,
      steps_per_decay=20000,
      steps_per_cycle=100000)
  batch_size = 256
  num_train_steps = 200000
  weight_decay = 0.0
  beta1 = .9
  beta2 = .98
  eps = 1e-9
  random_seed = 0
  hardware_rng = True

  if base_config.quant_target == BaseConfigQuantTarget.NONE:
    weight_outlier_regularization = 0.0
  else:
    weight_outlier_regularization = 1.0
  weight_outlier_regularization_regex = '^.*kernel$'

  prefer_int8_to_int32_dot = True
  quant_type = QuantType.fake_quant

  activation_bound_update_freq = -1
  activation_bound_start_step = -1

  model_hparams = create_base_transformer_hparams(
      mlp_weight_prec=mlp_weight_prec,
      embedding_weight_prec=embedding_weight_prec,
      attention_weight_prec=attention_weight_prec,
      num_layers=num_layers,
      num_heads=num_heads,
      emb_dim=emb_dim,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      mlp_pos_inputs_prec=mlp_pos_inputs_prec,
      mlp_pos_inputs_hyper=mlp_pos_inputs_hyper_fixed_bound,
      mlp_signed_inputs_prec=mlp_signed_inputs_prec,
      mlp_signed_inputs_hyper=mlp_signed_inputs_hyper_fixed_bound,
      attention_kqv_inputs_prec=attention_kqv_inputs_prec,
      attention_kqv_inputs_hyper=attention_kqv_inputs_hyper_fixed_bound,
      attention_out_inputs_prec=attention_out_inputs_prec,
      attention_out_inputs_hyper=attention_out_inputs_hyper_fixed_bound,
      attention_act_q_inputs_prec=attention_act_q_inputs_prec,
      attention_act_q_inputs_hyper=attention_act_q_inputs_hyper_fixed_bound,
      attention_act_k_inputs_prec=attention_act_k_inputs_prec,
      attention_act_k_inputs_hyper=attention_act_k_inputs_hyper_fixed_bound,
      attention_act_probs_inputs_prec=attention_act_probs_inputs_prec,
      attention_act_v_inputs_prec=attention_act_v_inputs_prec,
      attention_act_v_inputs_hyper=attention_act_v_inputs_hyper_fixed_bound,
      logits_inputs_prec=logits_inputs_prec,
      logits_inputs_hyper=logits_inputs_hyper_fixed_bound,
      logits_via_embeddings=True,
      quant_type=quant_type,
  )

  metadata = os_hparams_utils.HParamsMetadata(
      description='', last_updated_time=time.time())

  return training_hparams.TrainingHParams(
      metadata=metadata,
      model_hparams=model_hparams,
      learning_rate_schedule=learning_rate_scheduler_hparams,
      weight_decay=weight_decay,
      per_host_batch_size=batch_size,
      num_train_steps=num_train_steps,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      random_seed=random_seed,
      hardware_rng=hardware_rng,
      activation_bound_update_freq=activation_bound_update_freq,
      activation_bound_start_step=activation_bound_start_step,
      weight_outlier_regularization=weight_outlier_regularization,
      weight_outlier_regularization_regex=weight_outlier_regularization_regex,
      prefer_int8_to_int32_dot=prefer_int8_to_int32_dot,
  )


def create_training_hparams_from_flags():
  """Creates TrainingHParams instance from flags."""
  base_config = BaseConfig.create_from_flags()
  return create_training_hparams_from_base_config(base_config)
