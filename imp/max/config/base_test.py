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

"""Tests for base."""

import dataclasses
import os

from absl.testing import absltest
# from jax import numpy as jnp
import tensorflow as tf
import yaml

from imp.max.config import base


@dataclasses.dataclass
class TestConfig(base.Config):
  name: str = 'name'
  param: int = 0


@dataclasses.dataclass
class NestedTestConfig(base.Config):
  a: TestConfig
  b: TestConfig
  opt: float | None = None
  l: list[TestConfig] = dataclasses.field(default_factory=lambda: [  # pylint:disable=g-long-lambda
      TestConfig(),
      TestConfig(name='n', param=1)
  ])
  l2: list[str] = dataclasses.field(default_factory=lambda: ['a', 'b', 'c'])
  d: dict[str, int] = dataclasses.field(default_factory=lambda: {'dd': 1})
  dl: list[dict[str, int]] = dataclasses.field(default_factory=lambda: [  # pylint:disable=g-long-lambda
      {
          'dl': 0
      }, {
          'ld': 0
      }
  ])


class BaseTest(absltest.TestCase):

  def test_as_dict(self):
    config = TestConfig()
    self.assertDictEqual(config.as_dict(), {'name': 'name', 'param': 0})

  def test_nested_as_dict(self):
    a = TestConfig()
    b = TestConfig('name2', 1)
    nested_config = NestedTestConfig(a, b)
    self.assertDictEqual(
        nested_config.as_dict(), {
            'a': {
                'name': 'name',
                'param': 0
            },
            'b': {
                'name': 'name2',
                'param': 1
            },
            'opt': None,
            'l': [{
                'name': 'name',
                'param': 0
            }, {
                'name': 'n',
                'param': 1
            }],
            'l2': ['a', 'b', 'c'],
            'd': {
                'dd': 1
            },
            'dl': [{
                'dl': 0
            }, {
                'ld': 0
            }]
        })

  def test_override(self):
    config = TestConfig()
    config.override({'name': 'n', 'param': -1})
    self.assertDictEqual(config.as_dict(), {'name': 'n', 'param': -1})
    b = TestConfig('name2', 1)
    nested_config = NestedTestConfig(config, b)
    self.assertDictEqual(
        nested_config.as_dict(), {
            'a': {
                'name': 'n',
                'param': -1
            },
            'b': {
                'name': 'name2',
                'param': 1
            },
            'opt': None,
            'l': [{
                'name': 'name',
                'param': 0
            }, {
                'name': 'n',
                'param': 1
            }],
            'l2': ['a', 'b', 'c'],
            'd': {
                'dd': 1
            },
            'dl': [{
                'dl': 0
            }, {
                'ld': 0
            }]
        })
    nested_config.override({
        'b': {
            'name': 'new_name'
        },
        'opt': 0,
        'l[1]': {
            'name': 'num'
        },
        'l2[2]': 'd',
        'd': {
            'dd': 2,
            'ddd': 3
        },
        'dl[0]': {
            'd': 0,
            'l': 1
        }
    })
    self.assertDictEqual(
        nested_config.as_dict(), {
            'a': {
                'name': 'n',
                'param': -1
            },
            'b': {
                'name': 'new_name',
                'param': 1
            },
            'opt': 0,
            'l': [{
                'name': 'name',
                'param': 0
            }, {
                'name': 'num',
                'param': 1
            }],
            'l2': ['a', 'b', 'd'],
            'd': {
                'dd': 2,
                'ddd': 3
            },
            'dl': [{
                'd': 0,
                'l': 1
            }, {
                'ld': 0
            }]
        })
    with self.assertRaises(ValueError):
      config.override({'num': 'n'})
    with self.assertRaises(ValueError):
      config.override({'name[0]': 'n'})
    with self.assertRaises(ValueError):
      nested_config.override({'l[0': {'name': 'num'}})
    with self.assertRaises(ValueError):
      nested_config.override({'l[g]': {'name': 'num'}})
    with self.assertRaises(NotImplementedError):
      nested_config.override({'l[1][0]': {'name': 'num'}})
    with self.assertRaises(ValueError):
      nested_config.override({'l[3]': {'name': 'num'}})

  def test_overried_copy(self):
    config = TestConfig()
    copied = config.copy_and_override({'name': 'n', 'param': -1})
    self.assertDictEqual(copied.as_dict(), {'name': 'n', 'param': -1})
    self.assertDictEqual(config.as_dict(), {'name': 'name', 'param': 0})

  def test_override_from_str(self):
    config = TestConfig()
    dict_str = yaml.safe_dump({'name': 'n', 'param': 1})
    config.override_from_str(dict_str)
    self.assertDictEqual(config.as_dict(), {'name': 'n', 'param': 1})

  def test_override_from_file(self):
    config = TestConfig()
    dict_str = yaml.safe_dump({'name': 'n', 'param': 1})
    tmp_file = self.create_tempfile('path/to/config.yaml', dict_str)
    config.override_from_file(tmp_file.full_path)
    self.assertDictEqual(config.as_dict(), {'name': 'n', 'param': 1})

  def test_export(self):
    path = self.create_tempdir('path/to/').full_path

    config = TestConfig()
    config = config.copy_and_override({'name': 'n', 'param': 1})
    config.export(path)
    self.assertTrue(tf.io.gfile.exists(os.path.join(path, 'config.yaml')))

    new_config = TestConfig()
    new_config.override_from_file(os.path.join(path, 'config.yaml'))
    self.assertDictEqual(new_config.as_dict(), {'name': 'n', 'param': 1})

  # def test_export_dtype(self):
  #   path = self.create_tempdir('path/to/').full_path
  #   config_path = os.path.join(path, 'config.yaml')

  #   @dataclasses.dataclass
  #   class BaseDtypes(base.Config):
  #     float32: jnp.generic = jnp.float32
  #     bfloat16: jnp.generic = jnp.bfloat16
  #     int32: jnp.generic = jnp.int32
  #     unit8: jnp.generic = jnp.uint8
  #     bool_: jnp.generic = jnp.bool_

  #   output_dict = {
  #       'float32': jnp.float32.dtype,
  #       'bfloat16': jnp.bfloat16.dtype,
  #       'int32': jnp.int32.dtype,
  #       'unit8': jnp.uint8.dtype,
  #       'bool_': jnp.bool_.dtype,
  #   }

  #   config = BaseDtypes()
  #   config.export(path)
  #   self.assertTrue(tf.io.gfile.exists(config_path))

  #   with tf.io.gfile.GFile(config_path) as f:
  #     yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

  #   self.assertDictEqual(yaml_dict, output_dict)

  #   @dataclasses.dataclass
  #   class DerivedDtypes(base.Config):
  #     float32: jnp.generic = jnp.float32.dtype
  #     bfloat16: jnp.generic = jnp.bfloat16.dtype
  #     int32: jnp.generic = jnp.int32.dtype
  #     unit8: jnp.generic = jnp.uint8.dtype
  #     bool_: jnp.generic = jnp.bool_.dtype

  #   config = DerivedDtypes()
  #   config.export(path)
  #   self.assertTrue(tf.io.gfile.exists(config_path))

  #   with tf.io.gfile.GFile(config_path) as f:
  #     yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

  #   self.assertDictEqual(yaml_dict, output_dict)


if __name__ == '__main__':
  absltest.main()
