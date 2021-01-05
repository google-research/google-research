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

"""NeuTra tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import parameterized
import gin
import numpy as np
import tensorflow.compat.v2 as tf
from hmc_swindles import neutra

tf.enable_v2_behavior()


class NeutraTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NeutraTest, self).setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.test_german = os.path.join(self.temp_dir, "test_german")
    self.test_cloud = os.path.join(self.temp_dir, "test_cloud")
    with tf.io.gfile.GFile(self.test_german, "wb") as f:
      np.savetxt(f, np.random.rand(6, 4), delimiter=" ")
    with tf.io.gfile.GFile(self.test_cloud, "wb") as f:
      np.savetxt(f, np.random.rand(6, 4), delimiter=",")

  def tearDown(self):
    super(NeutraTest, self).tearDown()
    tf.io.gfile.rmtree(self.temp_dir)

  def testAffineBijector(self):
    bijector = neutra.MakeAffineBijectorFn(2)
    x = bijector.forward(tf.zeros([3, 2]))

    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testAffineBijectorTril(self):
    bijector = neutra.MakeAffineBijectorFn(2, use_tril=True)
    x = bijector.forward(tf.zeros([3, 2]))

    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testRNVPBijectorFn(self):
    bijector = neutra.MakeRNVPBijectorFn(2, 3, [4], learn_scale=True)
    x = bijector.forward(tf.zeros([3, 2]))

    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testIAFBijectorFn(self):
    bijector = neutra.MakeIAFBijectorFn(2, 3, [4], learn_scale=True)
    x = bijector.forward(tf.zeros([3, 2]))

    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testNeuTraExperiment(self):
    gin.clear_config()
    gin.bind_parameter("target_spec.name", "ill_conditioned_gaussian")
    gin.bind_parameter("chain_stats.compute_stats_over_time", True)
    exp = neutra.NeuTraExperiment(bijector="affine", log_dir=self.temp_dir)

    exp.Train(4, batch_size=2)
    exp.Eval(batch_size=2)
    exp.Benchmark(test_num_steps=100, test_batch_size=2, batch_size=2)
    exp.TuneObjective(
        1, 0.1, batch_size=2, test_num_steps=600, f_name="first_moment_mean")


if __name__ == "__main__":
  tf.test.main()
