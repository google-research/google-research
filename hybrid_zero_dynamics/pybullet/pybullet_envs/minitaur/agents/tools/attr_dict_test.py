# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the attribute dictionary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pybullet_envs.minitaur.agents.tools import attr_dict


class AttrDictTest(tf.test.TestCase):

  def test_construct_from_dict(self):
    initial = dict(foo=13, bar=42)
    obj = attr_dict.AttrDict(initial)
    self.assertEqual(13, obj.foo)
    self.assertEqual(42, obj.bar)

  def test_construct_from_kwargs(self):
    obj = attr_dict.AttrDict(foo=13, bar=42)
    self.assertEqual(13, obj.foo)
    self.assertEqual(42, obj.bar)

  def test_has_attribute(self):
    obj = attr_dict.AttrDict(foo=13)
    self.assertTrue('foo' in obj)
    self.assertFalse('bar' in obj)

  def test_access_default(self):
    obj = attr_dict.AttrDict()
    self.assertEqual(None, obj.foo)

  def test_access_magic(self):
    obj = attr_dict.AttrDict()
    with self.assertRaises(AttributeError):
      obj.__getstate__  # pylint: disable=pointless-statement

  def test_immutable_create(self):
    obj = attr_dict.AttrDict()
    with self.assertRaises(RuntimeError):
      obj.foo = 42

  def test_immutable_modify(self):
    obj = attr_dict.AttrDict(foo=13)
    with self.assertRaises(RuntimeError):
      obj.foo = 42

  def test_immutable_unlocked(self):
    obj = attr_dict.AttrDict()
    with obj.unlocked:
      obj.foo = 42
    self.assertEqual(42, obj.foo)


if __name__ == '__main__':
  tf.test.main()
