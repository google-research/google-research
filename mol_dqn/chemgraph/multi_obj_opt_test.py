# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for Optimizes QED of a molecule with DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import flagsaver

import tensorflow as tf
from dqn import deep_q_networks
from dqn.tensorflow_core import core
import multi_obj_naive


class MultiObjTest(tf.test.TestCase):

  def setUp(self):
    super(MultiObjTest, self).setUp()
    self.mount_point = tempfile.mkdtemp()
    self.model_dir = os.path.join(self.mount_point, 'model_dir')
    os.makedirs(self.model_dir)

  def test_run(self):
    hparams = deep_q_networks.get_hparams(
        replay_buffer_size=100,
        num_episodes=10,
        batch_size=10,
        update_frequency=1,
        save_frequency=1,
        dense_layers=[32],
        fingerprint_length=128,
        fingerprint_radius=2,
        num_bootstrap_heads=12,
        prioritized=True,
        double_q=True)
    hparams_file = os.path.join(self.mount_point, 'config.json')
    core.write_hparams(hparams, hparams_file)

    with flagsaver.flagsaver(
        model_dir=self.model_dir,
        start_molecule='c1ccccc1',
        target_molecule='c1ccccc1',
        similarity_weight=0.2,
        hparams=hparams_file):
      multi_obj_naive.main(None)


if __name__ == '__main__':
  tf.test.main()
