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

"""Tests for dql_grasping.train_collect_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import gin.tf
from dql_grasping import gin_imports  # pylint: disable=unused-import
from dql_grasping import grasping_env
from dql_grasping import train_collect_eval
from tensorflow.python.platform import test
FLAGS = flags.FLAGS


class TrainCollectEvalTest(test.TestCase):

  def setUp(self):
    super(TrainCollectEvalTest, self).setUp()
    self._root_dir = os.path.join(FLAGS.test_tmpdir, 'root')

  def testSynchronousTrainCollectEval(self):
    """End-to-end integration test.
    """
    env = grasping_env.KukaGraspingProceduralEnv(downsample_width=64,
                                                 downsample_height=64,
                                                 continuous=True,
                                                 remove_height_hack=True,
                                                 render_mode='DIRECT')
    data_dir = 'testdata'
    gin_config = os.path.join(FLAGS.test_srcdir, data_dir, 'random_collect.gin')
    # Collect initial data from random policy without training.
    with open(gin_config, 'r') as f:
      gin.parse_config(f)
    train_collect_eval.train_collect_eval(collect_env=env,
                                          eval_env=None,
                                          test_env=None,
                                          root_dir=self._root_dir,
                                          train_fn=None)
    # Run training (synchronous train, collect, & eval).
    gin_config = os.path.join(FLAGS.test_srcdir, data_dir, 'train_dqn.gin')
    with open(gin_config, 'r') as f:
      gin.parse_config(f)
    train_collect_eval.train_collect_eval(collect_env=env,
                                          eval_env=None,
                                          test_env=None,
                                          root_dir=self._root_dir)


if __name__ == '__main__':
  test.main()
