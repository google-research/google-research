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

"""Tests for config_schema.py for Imagenet."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.imagenet.configs_script import config_schema


class BaseConfigTest(parameterized.TestCase):

  def test_precision_propagates(self):
    config = config_schema.get_config(num_blocks=16, use_auto_acts=True)

    # Set the global precision to 4 bits.
    config.prec = 4
    # Test that this sets the weight and activation to 4 as well.
    self.assertEqual(config.weight_prec, 4)
    self.assertEqual(config.quant_act.prec, 4)
    # Test that propagates all the way down to the weight precision of layer
    # types and individual layers. As an example of an individual layer, we take
    # the dense1 matmul of the second block of the decoder.
    conv1_block3 = config.model_hparams.residual_blocks[2].conv_1
    # Meanwhile, 'conv1' represents the generic configuration of all conv1
    # layers throughout the model.
    conv1 = config.residual.conv_1
    self.assertEqual(conv1.weight_prec, 4)
    self.assertEqual(conv1_block3.weight_prec, 4)

    # Test if we take the same config instance and alter the global precision to
    # 8, it automatically propagates to individual layers.
    config.prec = 8
    self.assertEqual(conv1.weight_prec, 8)
    self.assertEqual(conv1_block3.weight_prec, 8)

    # Test that the precision can be overridden for a specific layer type. We
    # want to verify that the change doesn't back-propagate back to the global
    # precision field but does propagate down to individual layers of that layer
    # type. We only want changes to fields to automatically propagate down the
    # parameter hierarchy, not up.
    conv1.weight_prec = 2
    self.assertEqual(conv1.weight_prec, 2)
    self.assertEqual(conv1_block3.weight_prec, 2)
    self.assertEqual(config.prec, 8)

    # Now update the precision for just a specific layer and check that it
    # doesn't propagate upwards.
    conv1_block3.weight_prec = 1
    self.assertEqual(conv1_block3.weight_prec, 1)
    self.assertEqual(conv1.weight_prec, 2)
    self.assertEqual(config.prec, 8)

  @parameterized.parameters(dict(num_blocks=10), dict(num_blocks=20))
  def test_num_blocks(self, num_blocks):
    config = config_schema.get_config(num_blocks=num_blocks, use_auto_acts=True)
    self.assertLen(config.model_hparams.residual_blocks, num_blocks)
    self.assertLen(config.model_hparams.residual_blocks, num_blocks)

  def test_auto_acts_parameter(self):
    # If use_auto_acts is False, then the bounds should be a single scalar that
    # specifies the fixed bound; 'None' by default.
    config = config_schema.get_config(num_blocks=15, use_auto_acts=False)
    self.assertIsNone(config.quant_act.bounds)
    # If use_auto_acts is True, it should have the same structure as the
    # GetBounds.Hyper dataclass.
    config = config_schema.get_config(num_blocks=15, use_auto_acts=True)
    self.assertIn('initial_bound', config.quant_act.bounds)

    # Because the config dict is locked, it shouldn't be possible to change it
    # back to fixed bounds if it was created with use_auto_acts=True.
    with self.assertRaises(TypeError):
      config.quant_act.bounds = 1.0

  @parameterized.parameters(dict(num_blocks=16), dict(num_blocks=33))
  def test_schema_matches_expected(self, num_blocks):
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
    }

    dense_schema = {
        'weight_prec': None,
        'weight_quant_granularity': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
    }

    conv_schema = {
        'weight_prec': None,
        'weight_quant_granularity': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
    }

    residual_block_schema = {
        'conv_proj': conv_schema,
        'conv_1': conv_schema,
        'conv_2': conv_schema,
        'conv_3': conv_schema,
    }

    expected_top_level_schema = {
        'metadata': {
            'description': None,
            'hyper_str': None
        },
        'base_learning_rate': None,
        'momentum': None,
        'weight_decay': None,
        'activation_bound_update_freq': None,
        'activation_bound_start_step': None,
        'prec': None,
        'weight_prec': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'weight_quant_granularity': None,
        'dense_layer': dense_schema,
        'conv': conv_schema,
        'residual': residual_block_schema,
        'model_hparams': {
            'dense_layer': dense_schema,
            'conv_init': conv_schema,
            'residual_blocks': [residual_block_schema] * num_blocks,
            'filter_multiplier': None,
        },
    }

    config = config_schema.get_config(num_blocks=num_blocks, use_auto_acts=True)
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
