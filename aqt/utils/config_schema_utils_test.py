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

"""Tests for config_schema.py."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections

from aqt.utils import config_schema_utils


class MakeReferenceRecursiveTest(absltest.TestCase):

  def test_scalar_fields(self):
    config = ml_collections.ConfigDict({'parent_field': 1})
    config.child_field = config_schema_utils.make_reference(
        config, 'parent_field')

    # 'child_field' is a reference to 'parent_field'. Changes to
    # 'parent_field' propagate to 'child_field'.
    self.assertEqual(config.parent_field, 1)
    self.assertEqual(config.child_field, 1)

    config.parent_field = 2
    self.assertEqual(config.parent_field, 2)
    self.assertEqual(config.child_field, 2)

    # But changes to 'child_field' to NOT propagate back up to
    # 'parent_field'.
    config.child_field = 3
    self.assertEqual(config.parent_field, 2)
    self.assertEqual(config.child_field, 3)

    config.parent_field = 4
    self.assertEqual(config.parent_field, 4)
    # Reference is broken after 'child_field' was overridden earlier.
    self.assertEqual(config.child_field, 3)

  def test_nested_fields(self):
    config = ml_collections.ConfigDict({'parent': {'x': 1}})
    config.child = config_schema_utils.make_reference(config, 'parent')

    # In this case, 'config.child.x' is a reference to 'config.parent.x', but
    # note that 'config.child' is NOT a reference to 'config.parent'!

    self.assertEqual(config.parent.x, 1)
    self.assertEqual(config.child.x, 1)

    config.parent.x = 2
    self.assertEqual(config.parent.x, 2)
    self.assertEqual(config.child.x, 2)

    config.parent = ml_collections.ConfigDict({'x': 3})
    self.assertEqual(config.parent.x, 3)
    # In this case, config.parent is a new Python object unrelated to the old
    # config.parent. Since config.child is a reference to the old config.parent,
    # it has no connection to the new config.parent.
    self.assertEqual(config.child.x, 2)

    # However, this works as intended since the 'update' function assigns new
    # values to existing leaf nodes, preserving the reference structure between
    # parent and child internal nodes. Using this syntax is recommended for
    # updating many fields at once.
    config = ml_collections.ConfigDict({'parent': {'x': 1, 'y': 'hello'}})
    config.child = config_schema_utils.make_reference(config, 'parent')
    config.parent.update({'x': 3, 'y': 'goodbye'})
    self.assertEqual(config.parent.x, 3)
    self.assertEqual(config.parent.y, 'goodbye')

    self.assertEqual(config.child.x, 3)
    self.assertEqual(config.child.y, 'goodbye')


class SetDefaultReferenceTest(absltest.TestCase):

  def test_when_child_field_is_list(self):
    # Test when 'field' parameter of set_default_reference is a list
    # of specific fields. We expect a new reference to be created for each
    # element in the list.
    parent = ml_collections.ConfigDict({'x': 1, 'y': 2, 'z': 3})
    child = ml_collections.ConfigDict()
    config_schema_utils.set_default_reference(child, parent, ['x', 'y'])
    self.assertEqual((parent.x, parent.y), (1, 2))
    self.assertEqual((child.x, child.y), (1, 2))

    parent.y = 5
    self.assertEqual((parent.x, parent.y), (1, 5))
    self.assertEqual((child.x, child.y), (1, 5))

    child.y = 10
    self.assertEqual((parent.x, parent.y), (1, 5))
    self.assertEqual((child.x, child.y), (1, 10))

  def test_reference_to_self(self):
    # Test adding a new field to a configdict which is a reference to an
    # existing field in the same configdict instance.
    config = ml_collections.ConfigDict({'parent': 1})
    config_schema_utils.set_default_reference(
        config, config, 'child', parent_field='parent')
    self.assertEqual(config.child, 1)
    self.assertEqual(config.parent, 1)

    config.parent = 5
    self.assertEqual(config.parent, 5)
    self.assertEqual(config.child, 5)

    config.child = 10
    self.assertEqual(config.parent, 5)
    self.assertEqual(config.child, 10)


class BaseConfigTest(parameterized.TestCase):

  @parameterized.parameters(dict(use_auto_acts=True), dict(use_auto_acts=False))
  def test_precision_propagates(self, use_auto_acts):
    config = config_schema_utils.get_base_config(use_auto_acts, fp_quant=False)

    # Set the global precision to 4 bits.
    config.prec = 4
    # Set the global half_shift flag to False
    config.half_shift = False
    # Test that this sets the weight and activation to 4 as well.
    self.assertEqual(config.weight_prec, 4)
    self.assertEqual(config.quant_act.prec, 4)
    # Test that this sets the weight_half_shift and act half_shift to False
    self.assertEqual(config.weight_half_shift, False)
    self.assertEqual(config.quant_act.half_shift, False)

    # Set the global precision to None, checks whether referencing to None
    # works well.
    config.prec = None
    # Test that this sets the weight and activation to None as well.
    self.assertIsNone(config.weight_prec, None)
    self.assertIsNone(config.quant_act.prec, None)

  @parameterized.parameters(dict(use_auto_acts=True), dict(use_auto_acts=False))
  def test_fp_precision_propagates(self, use_auto_acts):
    config = config_schema_utils.get_base_config(use_auto_acts, fp_quant=True)

    config.prec.is_scaled = False
    # Set the global precision to 4 bits.
    config.prec.fp_spec.update({'exp_min': -3, 'exp_max': 5, 'sig_bits': 2})

    expected_prec_dict = {
        'is_scaled': False,
        'fp_spec': {
            'exp_min': -3,
            'exp_max': 5,
            'sig_bits': 2
        }
    }
    # Test that this sets the weight and activation to 4 as well.
    self.assertEqual(config.weight_prec.to_dict(), expected_prec_dict)
    self.assertEqual(config.quant_act.prec.to_dict(), expected_prec_dict)

  def test_auto_acts_parameter(self):
    # If use_auto_acts is False, then the bounds should be a single scalar that
    # specifies the fixed bound; 'None' by default.
    config = config_schema_utils.get_base_config(
        use_auto_acts=False, fp_quant=False)
    self.assertIsNone(config.quant_act.bounds)
    # If use_auto_acts is True, it should have the same structure as the
    # GetBounds.Hyper dataclass.
    config = config_schema_utils.get_base_config(
        use_auto_acts=True, fp_quant=False)
    self.assertIn('initial_bound', config.quant_act.bounds)

    # Because the config dict is locked, it shouldn't be possible to change it
    # back to fixed bounds if it was created with use_auto_acts=True.
    with self.assertRaises(TypeError):
      config.quant_act.bounds = 1.0

  @parameterized.parameters(
      dict(use_auto_acts=True, fp_quant=False),
      dict(use_auto_acts=False, fp_quant=False),
      dict(use_auto_acts=False, fp_quant=True))
  def test_schema_matches_expected(self, use_auto_acts, fp_quant):
    # This tests that the schema of the configdict returned by 'base_config',
    # once all references are resolved, matches an expected schema. 'Schema'
    # here means the names and structure of fields at each level of the
    # configuration hierarchy. A value of 'None' in the expected schemas defined
    # below indicates a real configuration would have a concrete scalar value
    # there.

    if fp_quant:
      prec = {
          'fp_spec': {
              'exp_min': None,
              'exp_max': None,
              'sig_bits': None,
          },
          'is_scaled': None,
      }
    else:
      prec = None

    if use_auto_acts:
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
          'prec': prec,
          'half_shift': None,
      }
    else:
      quant_act_schema = {
          'bounds': None,
          'input_distribution': None,
          'prec': prec,
          'half_shift': None,
      }

    expected_top_level_schema = {
        'metadata': {
            'description': None,
            'hyper_str': None
        },
        'weight_decay': None,
        'activation_bound_update_freq': None,
        'activation_bound_start_step': None,
        'prec': prec,
        'half_shift': None,
        'weight_prec': prec,
        'weight_half_shift': None,
        'quant_type': None,
        'quant_act': quant_act_schema,
        'weight_quant_granularity': None,
    }

    config = config_schema_utils.get_base_config(
        use_auto_acts=use_auto_acts, fp_quant=fp_quant)
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
