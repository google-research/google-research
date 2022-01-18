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

"""Tests for dql_grasping.run_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl.testing import parameterized
from dql_grasping import grasping_env
from dql_grasping import policies
from dql_grasping import run_env
from tensorflow.python.platform import test
FLAGS = flags.FLAGS


class RunEnvTest(parameterized.TestCase, test.TestCase):

  @parameterized.named_parameters(
      ('collect_1', 'collect', True),
      ('eval_1', 'eval', True),
      ('emptyRoot', 'collect', False))
  def testPolicyRun(self, tag, use_root_dir):
    env = grasping_env.KukaGraspingProceduralEnv(
        downsample_width=48, downsample_height=48,
        continuous=True, remove_height_hack=True, render_mode='DIRECT')
    policy = policies.RandomGraspingPolicyD4()
    root_dir = os.path.join(FLAGS.test_tmpdir, tag) if use_root_dir else None
    run_env.run_env(env,
                    policy=policy,
                    explore_schedule=None,
                    episode_to_transitions_fn=None,
                    replay_writer=None,
                    root_dir=root_dir,
                    tag=tag,
                    task=0,
                    global_step=0,
                    num_episodes=1)

if __name__ == '__main__':
  test.main()
