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

"""Tests for utils."""

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from tf_agents.specs import array_spec

from caql import utils

tf.disable_v2_behavior()


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    seed = 199
    logging.info('Setting the numpy seed to %d', seed)
    np.random.seed(seed)

  def testStateAndActionSpecs(self):
    state_spec, action_spec = utils.get_state_and_action_specs(
        utils.create_env('Pendulum'))
    self.assertIsInstance(state_spec, array_spec.BoundedArraySpec)
    self.assertIsInstance(action_spec, array_spec.BoundedArraySpec)
    self.assertEqual((3,), state_spec.shape)
    self.assertEqual((1,), action_spec.shape)

    state_spec, action_spec = utils.get_state_and_action_specs(
        utils.create_env('Hopper'))
    self.assertIsInstance(state_spec, array_spec.BoundedArraySpec)
    self.assertIsInstance(action_spec, array_spec.BoundedArraySpec)
    self.assertEqual((11,), state_spec.shape)
    self.assertEqual((3,), action_spec.shape)

  def testStateAndActionSpecsWithActionBounds(self):
    _, action_spec = utils.get_state_and_action_specs(
        utils.create_env('Hopper'))
    self.assertEqual([-1, -1, -1], action_spec.minimum.tolist())
    self.assertEqual([1, 1, 1], action_spec.maximum.tolist())

    _, action_spec = utils.get_state_and_action_specs(
        utils.create_env('Hopper'), action_bounds=[-.5, .5])
    self.assertEqual([-.5, -.5, -.5], action_spec.minimum.tolist())
    self.assertEqual([.5, .5, .5], action_spec.maximum.tolist())

  def testActionProjection(self):
    action_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float32,
        minimum=[-1., -1., -1.], maximum=[1., 1., 1.])
    self.assertAllClose([-1., .7, 1.],
                        utils.action_projection([-1.5, .7, 1.5], action_spec))

  def testActionProjectionTFTensor(self):
    action_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float32,
        minimum=[-1., -1., -1.], maximum=[1., 1., 1.])
    with self.test_session() as sess:
      self.assertAllClose(
          [-1., .7, 1.],
          sess.run(utils.action_projection(tf.constant([-1.5, .7, 1.5]),
                                           action_spec)))

  def testCreatePlaceholdersForQNet(self):
    tf_vars = [tf.Variable([True], name='var1', dtype=tf.bool),
               tf.Variable([2, 3], name='var2', dtype=tf.int32),
               tf.Variable([4, 5, 6], name='var3', dtype=tf.float32)]
    ph_dict = utils.create_placeholders_for_q_net(tf_vars)

    self.assertEqual(['var1:0_ph', 'var2:0_ph', 'var3:0_ph'],
                     list(ph_dict.keys()))
    v = ph_dict['var1:0_ph']
    self.assertEqual([1], v.shape.as_list())
    self.assertEqual(tf.bool, v.dtype)
    v = ph_dict['var2:0_ph']
    self.assertEqual([2], v.shape.as_list())
    self.assertEqual(tf.int32, v.dtype)
    v = ph_dict['var3:0_ph']
    self.assertEqual([3], v.shape.as_list())
    self.assertEqual(tf.float32, v.dtype)

  def testBuildDummyQNetSingleLinearUnit(self):
    weights = np.array([[.1], [.2], [.3]])
    bias = np.array([-.5])
    tf_vars = [tf.Variable(weights, name='weights', dtype=tf.float32),
               tf.Variable(bias, name='bias', dtype=tf.float32)]
    ph_dict = utils.create_placeholders_for_q_net(tf_vars)
    output = utils.build_dummy_q_net([[3.]], [[2., 1.]], ph_dict, tf_vars)

    with self.test_session() as sess:
      self.assertAllClose([[.5]], sess.run(output, feed_dict={
          ph_dict['{}_ph'.format(tf_vars[0].name)]: weights,
          ph_dict['{}_ph'.format(tf_vars[1].name)]: bias,
      }))

  def testBuildDummyQNetTwoLayers(self):
    hidden_weights = np.array([[-.1, .1],
                               [-.2, .2],
                               [-.3, .3]])
    hidden_bias = np.array([.3, .3])
    output_weights = np.array([[.5], [.5]])
    output_bias = np.array([-.05])
    tf_vars = [
        tf.Variable(hidden_weights, name='hidden_weights', dtype=tf.float32),
        tf.Variable(hidden_bias, name='hidden_bias', dtype=tf.float32),
        tf.Variable(output_weights, name='output_weights', dtype=tf.float32),
        tf.Variable(output_bias, name='output_bias', dtype=tf.float32)
    ]
    ph_dict = utils.create_placeholders_for_q_net(tf_vars)
    output = utils.build_dummy_q_net([[3.]], [[2., 1.]], ph_dict, tf_vars)

    with self.test_session() as sess:
      self.assertAllClose([[.6]], sess.run(output, feed_dict={
          ph_dict['{}_ph'.format(tf_vars[0].name)]: hidden_weights,
          ph_dict['{}_ph'.format(tf_vars[1].name)]: hidden_bias,
          ph_dict['{}_ph'.format(tf_vars[2].name)]: output_weights,
          ph_dict['{}_ph'.format(tf_vars[3].name)]: output_bias,
      }))

  def testBuildDummyQNetIncorrectTFVars(self):
    tf_vars = [tf.Variable([[.1], [.2], [.3]], name='weights',
                           dtype=tf.float32)]
    ph_dict = utils.create_placeholders_for_q_net(tf_vars)
    with self.assertRaises(AssertionError):
      utils.build_dummy_q_net([[3.]], [[2., 1.]], ph_dict, tf_vars)


if __name__ == '__main__':
  tf.test.main()
