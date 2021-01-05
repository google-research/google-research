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

# Lint as: python2, python3
"""Test of the DeepQNetwork models.

Notes(zzp): The class of DeepQNetwork is tested in run_dqn_test.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from mol_dqn.chemgraph.dqn import deep_q_networks
from tensorflow.contrib import training as contrib_training


class DeepQNetworksTest(tf.test.TestCase):

  def test_multi_layer_model(self):
    hparams = contrib_training.HParams(
        dense_layers=[16, 8],
        activation='relu',
        num_bootstrap_heads=0,
        batch_norm=False)
    with tf.variable_scope('test'):
      out = deep_q_networks.multi_layer_model(tf.ones((5, 7)), hparams)
    self.assertListEqual([var.name for var in tf.trainable_variables()], [
        'test/dense_0/kernel:0', 'test/dense_0/bias:0', 'test/dense_1/kernel:0',
        'test/dense_1/bias:0', 'test/final/kernel:0', 'test/final/bias:0'
    ])
    self.assertListEqual(out.shape.as_list(), [5, 1])

  def test_get_fingerprint(self):
    hparams = deep_q_networks.get_hparams(fingerprint_length=64)
    fingerprint = deep_q_networks.get_fingerprint('c1ccccc1', hparams)
    self.assertListEqual(fingerprint.tolist(), [
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

  def test_get_fingerprint_with_steps_left(self):
    hparams = deep_q_networks.get_hparams(fingerprint_length=16)
    fingerprint = deep_q_networks.get_fingerprint_with_steps_left(
        'CC', steps_left=9, hparams=hparams)
    self.assertTupleEqual(fingerprint.shape, (17,))
    self.assertListEqual(fingerprint.tolist(), [
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 9.0
    ])


if __name__ == '__main__':
  tf.test.main()
