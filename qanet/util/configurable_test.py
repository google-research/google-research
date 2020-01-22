# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from qanet.util import configurable


class OuterConfigurable(configurable.Configurable):

  @staticmethod
  def _config():
    return {'nested': InnerConfigurable, 'output_val1': 1.0}


class InnerConfigurable(configurable.Configurable):

  @staticmethod
  def _config():
    return {'inner_val1': 1, 'inner_val2': '2'}


class SecondInnerConfigurable(configurable.Configurable):

  @staticmethod
  def _config():
    return {'second_inner_val1': 1, 'second_inner_val2': '2'}


class ConfigurableTest(tf.test.TestCase):

  def test_simple_nesting(self):
    built = OuterConfigurable.build_config(**{
        'nested': {
            'inner_val2': 'overridden'
        },
        'output_val1': 2
    })
    expected = {
        'nested': {
            'inner_val1': 1,
            'inner_val2': 'overridden',
            'fn': 'InnerConfigurable'
        },
        'output_val1': 2.0,
        'fn': 'OuterConfigurable'
    }

    self.assertAllEqual(expected, built)

  def test_nesting_override_fn(self):
    built = OuterConfigurable.build_config(**{
        'nested': SecondInnerConfigurable,
        'output_val1': 2
    })
    expected = {
        'nested': {
            'second_inner_val1': 1,
            'second_inner_val2': '2',
            'fn': 'SecondInnerConfigurable'
        },
        'output_val1': 2.0,
        'fn': 'OuterConfigurable'
    }

    self.assertAllEqual(expected, built)


class ConfigUtilsTest(tf.test.TestCase):

  def test_merge(self):
    a = {'a': 1, 'b': {'c': 2}}
    b = {'a': 3, 'b': {'c': 4, 'd': 5}}
    a_b_result = {'a': 3, 'b': {'c': 4, 'd': 5}}
    b_a_result = {'a': 1, 'b': {'c': 2, 'd': 5}}
    a_b_merge = configurable.merge(a, b)
    b_a_merge = configurable.merge(b, a)

    self.assertAllEqual(a_b_result, a_b_merge)
    self.assertAllEqual(b_a_result, b_a_merge)

  def test_flatten(self):
    flat_config_str = 'dataset.batch_size=32,dataset.epoch_size=96,dataset.fn=TokenCopyTask,dataset.min_length=10,dataset.vocab_size=10,model.cell.fn=DAGCell,model.optimizer.learning_rate=0.001'  # pylint: disable=line-too-long
    flat_config = {
        'dataset.min_length': 10,
        'dataset.fn': 'TokenCopyTask',
        'model.cell.fn': 'DAGCell',
        'model.optimizer.learning_rate': 0.001,
        'dataset.vocab_size': 10,
        'dataset.epoch_size': 96,
        'dataset.batch_size': 32
    }
    nested_config = {
        'dataset': {
            'batch_size': 32,
            'epoch_size': 96,
            'fn': 'TokenCopyTask',
            'min_length': 10,
            'vocab_size': 10
        },
        'model': {
            'optimizer': {
                'learning_rate': 0.001
            },
            'cell': {
                'fn': 'DAGCell'
            }
        }
    }
    self.assertAllEqual(flat_config, configurable.flatten_config(nested_config))
    self.assertAllEqual(flat_config_str,
                        configurable.config_to_string(nested_config))
    self.assertAllEqual(configurable.unflatten_dict(flat_config), nested_config)


class FindSubclassesTest(tf.test.TestCase):

  def test_find_subclasses(self):
    subclasses = configurable.all_subclasses(configurable.Configurable)
    expected = [OuterConfigurable, InnerConfigurable, SecondInnerConfigurable]
    self.assertAllEqual(expected, subclasses)


class ConvertTypeTest(tf.test.TestCase):

  def test_type_convert(self):
    convert = configurable._convert_type
    self.assertAllEqual(convert(0, bool), False)
    self.assertAllEqual(convert(1, bool), True)

    self.assertAllEqual(convert([1, 2, 3], tuple), (1, 2, 3))
    self.assertAllEqual(convert([1, 2, 3], list), [1, 2, 3])
    self.assertAllEqual(convert((1, 2, 3), list), [1, 2, 3])
    self.assertAllEqual(convert([1, 2, 3], list), [1, 2, 3])

  def test_try_numeric(self):
    self.assertAllEqual(configurable._try_numeric('0.0'), 0)
    self.assertAllEqual(configurable._try_numeric('0.1'), 0.1)
    self.assertAllEqual(configurable._try_numeric('1.0'), 1)


if __name__ == '__main__':
  tf.test.main()
