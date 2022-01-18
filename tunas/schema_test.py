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

"""Tests for schema."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow.compat.v1 as tf

from tunas import schema


class SchemaTest(tf.test.TestCase):

  def test_oneof_equality_simple(self):
    not_one_of = collections.namedtuple(
        'NotOneOf', ['choices', 'tag', 'mask'])
    tensor1 = tf.constant([3.0])
    tensor2 = tf.constant([4.0])

    self.assertEqual(
        schema.OneOf([1, 2], 'foo'),
        schema.OneOf([1, 2], 'foo'))
    self.assertEqual(
        schema.OneOf([1, 2], 'foo', tensor1),
        schema.OneOf([1, 2], 'foo', tensor1))

    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo'),
        schema.OneOf([1], 'foo'))
    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo'),
        schema.OneOf([1, 2], 'bar'))
    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo', tensor1),
        schema.OneOf([1, 2], 'foo', None))
    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo', tensor1),
        schema.OneOf([1, 2], 'foo', tensor2))

    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo', tensor1),
        not_one_of([1, 2], 'foo', tensor1))
    self.assertNotEqual(
        schema.OneOf([1, 2], 'foo'),
        {})
    self.assertNotEqual(
        {},
        schema.OneOf([1, 2], 'foo'))

  def test_oneof_equality_nested(self):
    self.assertEqual(
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([3], 'b')], 'c'),
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([3], 'b')], 'c'))
    self.assertNotEqual(
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([3], 'b')], 'c'),
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([4], 'b')], 'c'))
    self.assertNotEqual(
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([3], 'b')], 'c'),
        schema.OneOf([schema.OneOf([1, 5], 'a'), schema.OneOf([3], 'b')], 'c'))
    self.assertNotEqual(
        schema.OneOf([schema.OneOf([1, 2], 'a'), schema.OneOf([3], 'b')], 'c'),
        'Goooooooooooooooooooooooooooooooooooooooooooooogle')

  def test_oneof_repr(self):
    self.assertEqual(
        repr(schema.OneOf([1, 2], 'foo')),
        'OneOf(choices=[1, 2], tag=\'foo\')')
    self.assertStartsWith(
        repr(schema.OneOf([1, 2], 'foo', tf.constant([3.0]))),
        'OneOf(choices=[1, 2], tag=\'foo\', mask=')

  def test_map_oenofs_with_tuple_paths_trivial(self):
    structure = schema.OneOf([1, 2], 'tag')

    all_paths = []
    all_oneofs = []
    def visit(path, oneof):
      all_paths.append(path)
      all_oneofs.append(oneof)
      return schema.OneOf([x*10 for x in oneof.choices], oneof.tag)

    self.assertEqual(schema.map_oneofs_with_tuple_paths(visit, structure),
                     schema.OneOf([10, 20], 'tag'))
    self.assertEqual(all_paths, [()])
    self.assertEqual(all_oneofs, [schema.OneOf([1, 2], 'tag')])

  def test_map_oneofs_with_tuple_paths_simple(self):
    structure = [
        schema.OneOf([1, 2], 'tag1'),
        schema.OneOf([3, 4, 5], 'tag2'),
    ]

    all_paths = []
    all_oneofs = []
    def visit(path, oneof):
      all_paths.append(path)
      all_oneofs.append(oneof)
      return schema.OneOf([x*10 for x in oneof.choices], oneof.tag)

    self.assertEqual(schema.map_oneofs_with_tuple_paths(visit, structure), [
        schema.OneOf([10, 20], 'tag1'),
        schema.OneOf([30, 40, 50], 'tag2'),
    ])
    self.assertEqual(all_paths, [
        (0,),
        (1,)
    ])
    self.assertEqual(all_oneofs, [
        schema.OneOf([1, 2], 'tag1'),
        schema.OneOf([3, 4, 5], 'tag2'),
    ])

  def test_map_oneofs_with_tuple_paths_containing_arrays_and_dicts(self):
    structure = {
        'foo': [
            schema.OneOf([1, 2], 'tag1'),
            schema.OneOf([3, 4, 5], 'tag2'),
        ]}

    all_paths = []
    all_oneofs = []
    def visit(path, oneof):
      all_paths.append(path)
      all_oneofs.append(oneof)
      return schema.OneOf([x*10 for x in oneof.choices], oneof.tag)

    self.assertEqual(schema.map_oneofs_with_tuple_paths(visit, structure), {
        'foo': [
            schema.OneOf([10, 20], 'tag1'),
            schema.OneOf([30, 40, 50], 'tag2'),
        ]})
    self.assertEqual(all_paths, [
        ('foo', 0),
        ('foo', 1),
    ])
    self.assertEqual(all_oneofs, [
        schema.OneOf([1, 2], 'tag1'),
        schema.OneOf([3, 4, 5], 'tag2'),
    ])

  def test_map_oneofs_with_tuple_paths_containing_nested_oneofs(self):
    structure = {
        'root': schema.OneOf([
            schema.OneOf([
                {'leaf': schema.OneOf([1, 10], 'level2')},
                {'leaf': schema.OneOf([2, 20], 'level2')},
            ], 'level1'),
            schema.OneOf([
                {'leaf': schema.OneOf([3, 30], 'level2')},
                {'leaf': schema.OneOf([4, 40], 'level2')},
                {'leaf': schema.OneOf([5, 50], 'level2')},
            ], 'level1')
        ], 'level0')
    }

    all_paths = []
    all_oneofs = []
    def visit(path, oneof):
      all_paths.append(path)
      all_oneofs.append(oneof)
      return schema.OneOf([oneof.choices[0]], oneof.tag)

    self.assertEqual(
        schema.map_oneofs_with_tuple_paths(visit, structure),
        {
            'root': schema.OneOf([
                schema.OneOf([
                    {'leaf': schema.OneOf([1], 'level2')},
                ], 'level1'),
            ], 'level0')
        })
    self.assertEqual(all_paths, [
        ('root', 'choices', 0, 'choices', 0, 'leaf'),
        ('root', 'choices', 0, 'choices', 1, 'leaf'),
        ('root', 'choices', 0),
        ('root', 'choices', 1, 'choices', 0, 'leaf'),
        ('root', 'choices', 1, 'choices', 1, 'leaf'),
        ('root', 'choices', 1, 'choices', 2, 'leaf'),
        ('root', 'choices', 1),
        ('root',),
    ])
    # A OneOf node's children should already be updated by the time we visit it.
    self.assertEqual(all_oneofs, [
        schema.OneOf([1, 10], 'level2'),
        schema.OneOf([2, 20], 'level2'),
        schema.OneOf(
            [
                {'leaf': schema.OneOf([1], 'level2')},
                {'leaf': schema.OneOf([2], 'level2')},
            ], 'level1'),
        schema.OneOf([3, 30], 'level2'),
        schema.OneOf([4, 40], 'level2'),
        schema.OneOf([5, 50], 'level2'),
        schema.OneOf(
            [
                {'leaf': schema.OneOf([3], 'level2')},
                {'leaf': schema.OneOf([4], 'level2')},
                {'leaf': schema.OneOf([5], 'level2')},
            ], 'level1'),
        schema.OneOf(
            [
                schema.OneOf([
                    {'leaf': schema.OneOf([1], 'level2')},
                ], 'level1'),
                schema.OneOf([
                    {'leaf': schema.OneOf([3], 'level2')},
                ], 'level1')
            ], 'level0'),
    ])

  def test_map_oneofs_with_paths(self):
    structure = {
        'foo': [
            schema.OneOf([1, 2], 'tag1'),
            schema.OneOf([3, 4, 5], 'tag2'),
        ]}

    all_paths = []
    all_oneofs = []
    def visit(path, oneof):
      all_paths.append(path)
      all_oneofs.append(oneof)
      return schema.OneOf([x*10 for x in oneof.choices], oneof.tag)

    self.assertEqual(schema.map_oneofs_with_paths(visit, structure), {
        'foo': [
            schema.OneOf([10, 20], 'tag1'),
            schema.OneOf([30, 40, 50], 'tag2'),
        ]})
    self.assertEqual(all_paths, [
        'foo/0',
        'foo/1',
    ])
    self.assertEqual(all_oneofs, [
        schema.OneOf([1, 2], 'tag1'),
        schema.OneOf([3, 4, 5], 'tag2'),
    ])

  def test_map_oneofs(self):
    structure = {
        'foo': [
            schema.OneOf([1, 2], 'tag1'),
            schema.OneOf([3, 4, 5], 'tag2'),
        ]}

    all_oneofs = []
    def visit(oneof):
      all_oneofs.append(oneof)
      return schema.OneOf([x*10 for x in oneof.choices], oneof.tag)

    self.assertEqual(schema.map_oneofs(visit, structure), {
        'foo': [
            schema.OneOf([10, 20], 'tag1'),
            schema.OneOf([30, 40, 50], 'tag2'),
        ]})
    self.assertEqual(all_oneofs, [
        schema.OneOf([1, 2], 'tag1'),
        schema.OneOf([3, 4, 5], 'tag2'),
    ])


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
