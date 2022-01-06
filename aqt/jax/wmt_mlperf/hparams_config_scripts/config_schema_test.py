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

"""Tests for config_schema.py."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.wmt_mlperf.hparams_config_scripts import config_schema


class BaseConfigTest(parameterized.TestCase):

  def test_precision_propagates(self):
    config = config_schema.get_config(
        n_layers=3, use_auto_acts=True, fp_quant=False)

    # Set the global precision to 4 bits.
    config.prec = 4
    # Test that this sets the weight and activation to 4 as well.
    self.assertEqual(config.weight_prec, 4)
    self.assertEqual(config.quant_act.prec, 4)
    # Test that propagates all the way down to the weight precision of layer
    # types and individual layers. As an example of an individual layer, we take
    # the dense1 matmul of the second block of the decoder.
    dense1_block2 = config.model_hparams.decoder.encoder_decoder_1d_blocks[
        1].mlp_block.dense_1
    # Meanwhile, 'dense1' represents the generic configuration of all dense1
    # layers throughout the model.
    dense1 = config.mlp_block.dense_1
    self.assertEqual(dense1.weight_prec, 4)
    self.assertEqual(dense1_block2.weight_prec, 4)

    # Test if we take the same config instance and alter the global precision to
    # 8, it automatically propagates to individual layers.
    config.prec = 8
    self.assertEqual(dense1.weight_prec, 8)
    self.assertEqual(dense1_block2.weight_prec, 8)

    # Test that the precision can be overridden for a specific layer type. We
    # want to verify that the change doesn't back-propagate back to the global
    # precision field but does propagate down to individual layers of that layer
    # type. We only want changes to fields to automatically propagate down the
    # parameter hierarchy, not up.
    dense1.weight_prec = 2
    self.assertEqual(dense1.weight_prec, 2)
    self.assertEqual(dense1_block2.weight_prec, 2)
    self.assertEqual(config.prec, 8)

    # Now update the precision for just a specific layer and check that it
    # doesn't propagate upwards.
    dense1_block2.weight_prec = 1
    self.assertEqual(dense1_block2.weight_prec, 1)
    self.assertEqual(dense1.weight_prec, 2)
    self.assertEqual(config.prec, 8)

  def test_fp_precision_param(self):
    config = config_schema.get_config(
        n_layers=3, use_auto_acts=True, fp_quant=True)
    prec_dict = {
        'is_scaled': False,
        'fp_spec': {
            'exp_min': -3,
            'exp_max': 5,
            'sig_bits': 2
        }
    }
    # Set the global precision to 4 bits.
    config.prec.update(prec_dict)

    # Test that this sets the weight and activation prec as well.
    self.assertEqual(config.weight_prec.to_dict(), prec_dict)
    self.assertEqual(config.quant_act.prec.to_dict(), prec_dict)

    # Test that propagates all the way down to the weight precision of layer
    # types and individual layers. As an example of an individual layer, we take
    # the dense1 matmul of the second block of the decoder.
    dense1_block2 = config.model_hparams.decoder.encoder_decoder_1d_blocks[
        1].mlp_block.dense_1
    # Meanwhile, 'dense1' represents the generic configuration of all dense1
    # layers throughout the model.
    dense1 = config.mlp_block.dense_1
    self.assertEqual(dense1.weight_prec.to_dict(), prec_dict)
    self.assertEqual(dense1_block2.weight_prec.to_dict(), prec_dict)

  @parameterized.parameters(
      dict(n_layers=1), dict(n_layers=3), dict(n_layers=6))
  def test_n_layers_parameter(self, n_layers):
    config = config_schema.get_config(
        n_layers=n_layers, use_auto_acts=True, fp_quant=False)
    self.assertLen(config.model_hparams.encoder.encoder_1d_blocks, n_layers)
    self.assertLen(config.model_hparams.decoder.encoder_decoder_1d_blocks,
                   n_layers)

  def test_auto_acts_parameter(self):
    # If use_auto_acts is False, then the bounds should be a single scalar that
    # specifies the fixed bound; 'None' by default.
    config = config_schema.get_config(
        n_layers=3, use_auto_acts=False, fp_quant=False)
    self.assertIsNone(config.quant_act.bounds)
    # If use_auto_acts is True, it should have the same structure as the
    # GetBounds.Hyper dataclass.
    config = config_schema.get_config(
        n_layers=3, use_auto_acts=True, fp_quant=False)
    self.assertIn('initial_bound', config.quant_act.bounds)

    # Because the config dict is locked, it shouldn't be possible to change it
    # back to fixed bounds if it was created with use_auto_acts=True.
    with self.assertRaises(TypeError):
      config.quant_act.bounds = 1.0

  @parameterized.parameters(
      dict(quantized_reductions=False), dict(quantized_reductions=True))
  def test_softmax_config(self, quantized_reductions):
    base_config = config_schema.get_config(
        n_layers=3, use_auto_acts=True, fp_quant=False)
    softmax_config = config_schema.get_softmax_config(
        quantized=True, quantized_reductions=quantized_reductions)
    softmax_config.quant_hparams.prec.update({
        'exp_min': -3,
        'exp_max': 5,
        'sig_bits': 2
    })
    if quantized_reductions:
      softmax_config.quant_hparams.reduction_prec.update({
          'exp_min': 1,
          'exp_max': 4,
          'sig_bits': 6
      })
    new_config = config_schema.set_global_softmax_config(
        base_config=base_config, softmax_config=softmax_config)
    quant_hparams = new_config.model_hparams.decoder.encoder_decoder_1d_blocks[
        1].enc_dec_attention.attn_acts.softmax.quant_hparams
    self.assertEqual(quant_hparams.prec.to_dict(), {
        'exp_min': -3,
        'exp_max': 5,
        'sig_bits': 2
    })
    if quantized_reductions:
      self.assertEqual(quant_hparams.reduction_prec.to_dict(), {
          'exp_min': 1,
          'exp_max': 4,
          'sig_bits': 6
      })
    else:
      self.assertIsNone(quant_hparams.reduction_prec)

  def test_softmax_raises_with_invalid_quantized_reduction(self):
    with self.assertRaises(ValueError):
      config_schema.get_softmax_config(
          quantized=False, quantized_reductions=True)

  @parameterized.parameters(dict(n_layers=1), dict(n_layers=3))
  def test_schema_matches_expected(self, n_layers):
    # This tests that the schema of the configdict returned by 'config_schema',
    # once all references are resolved, matches an expected schema. 'Schema'
    # here means the names and structure of fields at each level of the
    # configuration hierarchy. A value of 'None' in the expected schemas defined
    # below indicates a real configuration would have a concrete scalar value
    # there.

    quant_act_schema = {
        'bounds': {
            'initial_bound': None,
            'stddev_coeff': None,
            'absdev_coeff': None,
            'mix_coeff': None,
            'reset_stats': None,
            'ema_coeff': None,
            'use_cams': None,
            'exclude_zeros': None,
            'use_mean_of_max': None,
            'granularity': None
        },
        'input_distribution': None,
        'prec': None,
        'half_shift': None,
    }

    dense_schema = {
        'weight_prec': None,
        'weight_quant_granularity': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'weight_half_shift': None,
    }

    embedding_schema = {
        'weight_prec': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'weight_half_shift': None,
    }

    mlp_block_schema = {
        'dense_1': dense_schema,
        'dense_2': dense_schema,
    }

    fp_schema = {'exp_min': None, 'exp_max': None, 'sig_bits': None}

    layernorm_schema = {
        'quant_hparams': {
            'prec': fp_schema,
            'reduction_prec': fp_schema
        }
    }

    attention_schema = {
        'dense_kqv': dense_schema,
        'dense_out': dense_schema,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'attn_acts': {
            'quant_type': None,
            'attn_act_q': quant_act_schema,
            'attn_act_k': quant_act_schema,
            'attn_act_v': quant_act_schema,
            'attn_act_probs': {
                'bounds': None,
                'input_distribution': None,
                'prec': None,
                'half_shift': None,
            },
        },
    }

    expected_top_level_schema = {
        'metadata': {
            'description': None,
            'hyper_str': None
        },
        'learning_rate_schedule': {
            'factors': None,
            'base_learning_rate': None,
            'warmup_steps': None,
            'decay_factor': None,
            'steps_per_decay': None,
            'steps_per_cycle': None,
        },
        'per_host_batch_size': None,
        'num_train_steps': None,
        'weight_decay': None,
        'beta1': None,
        'beta2': None,
        'eps': None,
        'random_seed': None,
        'hardware_rng': None,
        'activation_bound_update_freq': None,
        'activation_bound_start_step': None,
        'weight_outlier_regularization': None,
        'weight_outlier_regularization_regex': None,
        'prefer_int8_to_int32_dot': None,
        'prec': None,
        'half_shift': None,
        'weight_prec': None,
        'weight_half_shift': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'weight_quant_granularity': None,
        'dense': dense_schema,
        'embedding': embedding_schema,
        'mlp_block': mlp_block_schema,
        'attention': attention_schema,
        'model_hparams': {
            'emb_dim': None,
            'num_heads': None,
            'qkv_dim': None,
            'mlp_dim': None,
            'share_embeddings': None,
            'logits_via_embedding': None,
            'encoder': {
                'encoder_1d_blocks': [{
                    'mlp_block': mlp_block_schema,
                    'attention': attention_schema,
                    'layer_norm': layernorm_schema
                }] * n_layers,
                'embedding':
                    embedding_schema,
                'layer_norm':
                    layernorm_schema
            },
            'decoder': {
                'encoder_decoder_1d_blocks': [{
                    'mlp_block': mlp_block_schema,
                    'self_attention': attention_schema,
                    'enc_dec_attention': attention_schema,
                    'layer_norm': layernorm_schema
                }] * n_layers,
                'embedding':
                    embedding_schema,
                'logits':
                    dense_schema,
                'layer_norm':
                    layernorm_schema
            },
        },
    }

    config = config_schema.get_config(
        n_layers=n_layers, use_auto_acts=True, fp_quant=False)
    layer_norm_config = config_schema.get_layer_norm_config(
        quantized=True, quantized_reductions=True)
    config = config_schema.set_global_layer_norm_config(config,
                                                        layer_norm_config)
    # This round-trip conversion from JSON forces all references to resolve to
    # concrete values.
    config_reified = json.loads(config.to_json())

    # This test is not interested in checking the specific values of fields in
    # the configuration, but only that the schema of the hierarchies
    # are the same. Thus we all set the value of leaf nodes in the config to
    # 'None' before checking that the actual and expected configuration
    # structures are the same.
    def set_leaves_to_none(config):
      # We are at an intermediate node in the tree-structured input, which could
      # either be in the form of a dictionary or a list of other nodes in the
      # tree.
      if isinstance(config, dict):
        return {key: set_leaves_to_none(value) for key, value in config.items()}
      elif isinstance(config, list):
        return [set_leaves_to_none(value) for value in config]

      # We are at a leaf node in the tree-structured input.
      else:
        return None

    self.assertSameStructure(
        set_leaves_to_none(config_reified), expected_top_level_schema)


if __name__ == '__main__':
  absltest.main()
