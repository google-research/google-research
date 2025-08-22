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

"""Tests for validators."""

import dataclasses
import functools

from absl.testing import absltest

from imp.max.config import base
from imp.max.config import validators


@dataclasses.dataclass
class TestConfig(base.Config):
  name: str = 'name'
  param: int = 0
  optional: int | None = 1


@validators.lock
@dataclasses.dataclass
class LockedTestConfig(TestConfig):
  pass


class ValidatorsTest(absltest.TestCase):

  def test_lock_set_attribute(self):
    @validators.validate
    @dataclasses.dataclass
    class ChildTestConfig(LockedTestConfig):
      name: str = 'newname'
      optional: int = 11  # optional unwraps the `Optional` type
    config = ChildTestConfig()

    # Ensure that the lock is based on the base class
    self.assertEqual(ChildTestConfig.get_attribute_lock(), LockedTestConfig)
    # Ensure that the decorator does not change the type of the config
    self.assertIsInstance(config, ChildTestConfig)
    # Ensure that the child class overrides the attribute
    self.assertEqual(config.name, 'newname')

  def test_lock_new_attribute(self):
    with self.assertRaises(AttributeError):
      @validators.validate
      @dataclasses.dataclass
      class ChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
        newattr: str = 'test'

  def test_lock_new_type(self):
    with self.assertRaises(AttributeError):
      @validators.validate
      @dataclasses.dataclass
      class ChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
        name: int = 0

  def test_lock_unwrap_wrong_type(self):
    with self.assertRaises(AttributeError):
      @validators.validate
      @dataclasses.dataclass
      class ChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
        optional: float = 11

  def test_lock_set_attribute_no_unwrap(self):
    @functools.partial(validators.validate, unwrap_types=False)
    @dataclasses.dataclass
    class PassingChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
      optional: int | None = None

    with self.assertRaises(AttributeError):
      @functools.partial(validators.validate, unwrap_types=False)
      @dataclasses.dataclass
      class ChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
        optional: int = 11

  def test_lock_non_locked(self):
    with self.assertRaises(AssertionError):
      @validators.validate
      @dataclasses.dataclass
      class ChildTestConfig(TestConfig):  # pylint:disable=unused-variable
        pass

  def test_lock_twice(self):
    @validators.lock
    @dataclasses.dataclass
    class ChildTestConfig(LockedTestConfig):  # pylint:disable=unused-variable
      pass

    # The lock should be overridden for the child config, but not the locked
    # base config.
    self.assertEqual(ChildTestConfig.get_attribute_lock(), ChildTestConfig)
    self.assertEqual(LockedTestConfig.get_attribute_lock(), LockedTestConfig)

if __name__ == '__main__':
  absltest.main()
