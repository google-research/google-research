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

"""Generate HLOs for bits and pieces of the model."""
import logging
import sys

from absl import app
from absl import flags
from flax import serialization
import jax
import jax.numpy as jnp
from jax.tools import jax_to_ir
from ml_collections import config_dict
import tensorflow.compat.v2 as tf

from aqt.jax import hlo_utils
from aqt.jax import quant_config
from aqt.jax.quantization import QuantType
from aqt.jax.wmt_mlperf import models
from aqt.jax.wmt_mlperf import predict
from aqt.jax.wmt_mlperf import training_hparams
from aqt.jax.wmt_mlperf import training_hparams_generator_lib
from aqt.jax.wmt_mlperf.hparams_configs.experimental import full_model_8bit_weights_only_int8
from aqt.jax.wmt_mlperf.hparams_configs.experimental import small_model_8bit_weights_and_fixed_acts
from aqt.jax.wmt_mlperf.hparams_configs.experimental import small_model_bfloat16
from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_bfloat16
from aqt.utils import hparams_utils

flags.DEFINE_string(
    'hlo_output', '/tmp/hlo.pb', help='Destination file of the requested HLO.')

flags.DEFINE_string(
    'checkpoint', None, help='Checkpoint file saved by train.py.')

FLAGS = flags.FLAGS

EOS_TOKEN = 2  # Default Sentencepiece EOS token.


def _restore_from_checkpoint(model, checkpoint_file):
  with tf.io.gfile.GFile(checkpoint_file, 'rb') as fp:
    checkpoint = serialization.msgpack_restore(fp.read())
    if 'target' not in checkpoint:
      raise ValueError('Invalid checkpoint %s: no top-level "target".' %
                       checkpoint_file)
    checkpoint_model = checkpoint['target']
    checkpoint_model = jax.tree_map(jnp.array, checkpoint_model)
    return serialization.from_state_dict(model, checkpoint_model)


def encoder_from_file(config,
                      batch_size = 8,
                      encode_length = 16,
                      use_bfloat16 = True,
                      use_xla_optimizations = True):
  """Generates HLO for just the encoder of the WMT model.

  Args:
    config: A ConfigDict instance.
    batch_size: Batch size.
    encode_length: Max length of an input sentence.
    use_bfloat16: Use bfloat16 mixed precision training instead of float32.
    use_xla_optimizations: Whether to use xla optimizations.
  """
  if FLAGS.checkpoint:
    raise app.UsageError('Checkpoints not yet supported for WMT encoder.')

  input_shape = (batch_size, encode_length)
  rng = jax.random.PRNGKey(0)
  hparams = hparams_utils.load_dataclass_from_config_dict(
      training_hparams.TrainingHParams, config)
  model_hparams = hparams.model_hparams
  model = models.Encoder(
      vocab_size=32711,
      hparams=model_hparams.encoder,
      shared_embedding=None,
      use_bfloat16=use_bfloat16,
      emb_dim=model_hparams.emb_dim,
      num_heads=model_hparams.num_heads,
      qkv_dim=model_hparams.qkv_dim,
      mlp_dim=model_hparams.mlp_dim,
      max_len=encode_length,
      train=False,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=False, quantize_acts=True))
  init_state = model.init(rng, jnp.ones(input_shape, jnp.float32))

  def _fn(state, inputs):
    return model.apply(state, inputs, mutable=False)

  if not use_xla_optimizations:
    computation = jax.xla_computation(_fn)(init_state,
                                           jnp.ones(input_shape, jnp.float32))
    hlo_utils.output_hlo(computation, FLAGS.hlo_output)

  else:

    def _wrapped_fn(inputs):
      return _fn(init_state, inputs)

    def to_shape_str(shape_tuple):
      return 'f32[%s]' % ','.join(map(str, shape_tuple))

    hlo_module_proto_str, hlo_txt = jax_to_ir.jax_to_hlo(
        _wrapped_fn,
        [('inputs', jax_to_ir.parse_shape_str(to_shape_str(input_shape)))])
    hlo_utils.output_hlo_to_file(hlo_module_proto_str, hlo_txt,
                                 FLAGS.hlo_output)


def encoder_n_32(layers):
  """Generates HLO for just the encoder n-layer of the WMT model.

  Args:
    layers: The number of model layers.
  """
  if FLAGS.checkpoint:
    raise app.UsageError('Checkpoints not yet supported for WMT encoder.')

  input_shape = (1, 32)
  rng = jax.random.PRNGKey(0)
  model_hparams = training_hparams_generator_lib.create_base_transformer_hparams(
      embedding_weight_prec=None,
      attention_weight_prec=None,
      mlp_weight_prec=None,
      mlp_pos_inputs_prec=None,
      mlp_pos_inputs_hyper=None,
      mlp_signed_inputs_prec=None,
      mlp_signed_inputs_hyper=None,
      attention_kqv_inputs_prec=None,
      attention_kqv_inputs_hyper=None,
      attention_out_inputs_prec=None,
      attention_out_inputs_hyper=None,
      logits_inputs_prec=None,
      logits_inputs_hyper=None,
      logits_via_embeddings=True,
      attention_act_q_inputs_prec=None,
      attention_act_q_inputs_hyper=None,
      attention_act_k_inputs_prec=None,
      attention_act_k_inputs_hyper=None,
      attention_act_probs_inputs_prec=None,
      attention_act_v_inputs_prec=None,
      attention_act_v_inputs_hyper=None,
      num_layers=layers,
      emb_dim=1024,
      num_heads=1,
      qkv_dim=1024,
      mlp_dim=4096,
      quant_type=QuantType.fake_quant)
  model = models.Encoder(
      vocab_size=32711,
      hparams=model_hparams.encoder,
      shared_embedding=None,
      use_bfloat16=False,
      emb_dim=model_hparams.emb_dim,
      num_heads=model_hparams.num_heads,
      qkv_dim=model_hparams.qkv_dim,
      mlp_dim=model_hparams.mlp_dim,
      max_len=32,
      train=False,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=False, quantize_acts=True))
  init_state = model.init(rng, jnp.ones(input_shape, jnp.float32))

  def _fn(state, inputs):
    return model.apply(state, inputs)

  computation = jax.xla_computation(_fn)(init_state, jnp.ones(input_shape))
  hlo_utils.output_hlo(computation, FLAGS.hlo_output)


def encoder_full_model_opt_8_16():
  """Generates HLO for just the encoder of full transformer bfloat16."""
  encoder_from_file(config=full_model_bfloat16.get_config())


def encoder_full_model_quantized_8_16():
  """Generates HLO for the encoder of full transformer weights quantized."""
  encoder_from_file(config=full_model_8bit_weights_only_int8.get_config())


def encoder_small_model_quantized_1_32():
  """Generates HLO for just the encoder of small WMT 8-bit quantized model."""
  encoder_from_file(
      config=small_model_8bit_weights_and_fixed_acts.get_config(),
      batch_size=1,
      encode_length=32,
      use_bfloat16=False,
      use_xla_optimizations=False)


def encoder_1_32_big_model():
  """Generates HLO for just the encoder 1-layer of the WMT model."""
  encoder_n_32(1)


def encoder_2_32_big_model():
  """Generates HLO for just the encoder 2-layer of the WMT model."""
  encoder_n_32(2)


def transformer(config, batch_size,
                encode_length, decode_length):
  """Generates HLO for the WMT model.

  Args:
    config: A ConfigDict instance.
    batch_size: Batch size.
    encode_length: Max length of an input sentence.
    decode_length: Max length of an output sentence.
  """
  encode_shape = (batch_size, encode_length)
  decode_shape = (batch_size, decode_length)
  rng = jax.random.PRNGKey(0)

  hparams = hparams_utils.load_dataclass_from_config_dict(
      training_hparams.TrainingHParams, config)
  model_hparams = hparams.model_hparams

  transformer_kwargs = dict(
      vocab_size=33708,  # WMT en
      output_vocab_size=33708,  # WMT de
      max_len=encode_length)
  model = models.Transformer(
      **transformer_kwargs,
      hparams=model_hparams,
      train=False,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=False, quantize_acts=True),
      should_decode=True,
      use_bfloat16=False,
      dropout_rate=0.0,
      attention_dropout_rate=0.0)
  state = jax.jit(model.init)(rng, jnp.ones(encode_shape, jnp.float32),
                              jnp.ones(decode_shape, jnp.float32))
  if FLAGS.checkpoint:
    model = _restore_from_checkpoint(model, FLAGS.checkpoint)
  state, cache = state.pop('cache')  # pytype: disable=attribute-error
  state, params = state.pop('params')  # pytype: disable=attribute-error
  input_dummy = jnp.ones(encode_shape)
  if FLAGS.checkpoint:

    def _with_weights(inputs):
      return predict.step(
          inputs,
          params,
          cache,
          state,
          EOS_TOKEN,
          decode_length,
          transformer_kwargs=transformer_kwargs,
          hparams=model_hparams,
          quant_context=quant_config.QuantContext(
              update_bounds=False, quantize_acts=True))

    computation = jax.xla_computation(_with_weights)(input_dummy)
  else:

    def _without_weights(inputs, params):
      return predict.step(
          inputs,
          params,
          cache,
          state,
          EOS_TOKEN,
          decode_length,
          transformer_kwargs=transformer_kwargs,
          hparams=model_hparams,
          quant_context=quant_config.QuantContext(
              update_bounds=False, quantize_acts=True))

    computation = jax.xla_computation(_without_weights)(input_dummy, params)

  hlo_utils.output_hlo(computation, FLAGS.hlo_output)


def transformer_32_64_big_model():
  """Generates HLO that corresponds to a bigger version of the July demo."""
  transformer(
      config=full_model_bfloat16.get_config(),
      batch_size=1,
      encode_length=32,
      decode_length=64)


def transformer_32_64():
  """Generates HLO that corresponds to the July demo."""
  transformer(
      config=small_model_bfloat16.get_config(),
      batch_size=1,
      encode_length=32,
      decode_length=64)


# Corresponds to cl/306954847.
def transformer_16_97_147_big_model():
  """Generates HLO that mirrors the initial WMT transformer drop (cl/306954847)."""
  transformer(
      config=full_model_bfloat16.get_config(),
      batch_size=16,
      encode_length=97,
      decode_length=147)


def main(argv):
  if len(argv) != 2:
    raise app.UsageError('Too many command-line arguments.')
  variants = {
      'encoder_small_model_quantized_1_32': encoder_small_model_quantized_1_32,
      'encoder_full_model_opt_8_16': encoder_full_model_opt_8_16,
      'encoder_full_model_quantized_8_16': encoder_full_model_quantized_8_16,
      'encoder_1_32_big_model': encoder_1_32_big_model,
      'encoder_2_32_big_model': encoder_2_32_big_model,
      'transformer_32_64_big_model': transformer_32_64_big_model,
      'transformer_32_64': transformer_32_64,
      'transformer_16_97_147_big_model': transformer_16_97_147_big_model,
  }
  requested_variant = argv[1]
  if requested_variant not in variants:
    logging.error('Requested variant %s does not exist.', requested_variant)
    sys.exit(1)
  variants[requested_variant]()


if __name__ == '__main__':
  app.run(main)
