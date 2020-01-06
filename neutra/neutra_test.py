# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
import shutil
from absl.testing import parameterized
import gin

from neutra import neutra
import tensorflow.compat.v1 as tf
import numpy as np


class NeutraTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.test_german = os.path.join(self.temp_dir, "test_german")
    self.test_cloud = os.path.join(self.temp_dir, "test_cloud")
    with tf.gfile.Open(self.test_german, "wb") as f:
      np.savetxt(f, np.random.rand(6, 4), delimiter=" ")
    with tf.gfile.Open(self.test_cloud, "wb") as f:
      np.savetxt(f, np.random.rand(6, 4), delimiter=",")

  def tearDown(self):
    tf.gfile.DeleteRecursively(self.temp_dir)

  def testAffineBijector(self):
    tf.reset_default_graph()
    bijector = neutra.MakeAffineBijectorFn(2)
    x = bijector.forward(tf.zeros([3, 2]))

    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testAffineBijectorTril(self):
    tf.reset_default_graph()
    bijector = neutra.MakeAffineBijectorFn(2, use_tril=True)
    x = bijector.forward(tf.zeros([3, 2]))

    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testRNVPBijectorFn(self):
    tf.reset_default_graph()
    bijector = neutra.MakeRNVPBijectorFn(2, 3, [4], learn_scale=True)
    x = bijector.forward(tf.zeros([3, 2]))

    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  def testIAFBijectorFn(self):
    tf.reset_default_graph()
    bijector = neutra.MakeIAFBijectorFn(2, 3, [4], learn_scale=True)
    x = bijector.forward(tf.zeros([3, 2]))

    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([3, 2], x.shape)

  @parameterized.parameters(
      "funnel", "ill_cond_gaussian", "new_ill_cond_gaussian", "ill_cond_t",
      "new_ill_cond_t", "logistic_reg", "easy_gaussian", "gp_reg")
  def testTargetSpec(self, target_name):
    gin.clear_config()
    gin.bind_parameter("cloud.path", self.test_cloud)
    gin.bind_parameter("german.path", self.test_german)

    target, spec = neutra.GetTargetSpec(
        target_name,
        num_dims=5,
        regression_dataset="german",
        regression_type="gamma_scales2")
    lp = self.evaluate(target.log_prob(tf.ones([2, spec.num_dims])))
    self.assertAllEqual([2], lp.shape)

  def testNeuTraExperiment(self):
    gin.clear_config()
    gin.bind_parameter("target_spec.name", "easy_gaussian")
    gin.bind_parameter("target_spec.num_dims", 2)
    exp = neutra.NeuTraExperiment(
        train_batch_size=2,
        test_chain_batch_size=2,
        bijector="affine",
        log_dir=self.temp_dir)

    with tf.Session() as sess:
      exp.Initialize(sess)
      exp.TrainBijector(sess, 1)
      exp.Eval(sess)
      exp.Benchmark(sess)
      exp.Tune(sess, method="random", max_num_trials=1)


if __name__ == "__main__":
  tf.test.main()
