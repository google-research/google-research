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

"""Tests for snlds.forward_backward_algo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from snlds import forward_backward_algo


class ForwardBackwardAlgoTest(tf.test.TestCase):
  """Testing the forward backward algorithm, against hand calculated examples.

  The example is adapted from the following Lecture Note by Imperial College
  https://ibug.doc.ic.ac.uk/media/uploads/documents/computing_em_hmm.pdf
  """

  def setUp(self):
    super(ForwardBackwardAlgoTest, self).setUp()

    # initial discrete state likelihood p(s[0])
    self.init_pi = tf.convert_to_tensor([0.5, 0.5])

    # matrix A is transition matrix p(s[t] | s[t-1], x[t-1])
    self.mat_a = tf.convert_to_tensor(
        np.array([[[[0.5, 0.5],
                    [0.5, 0.5]],
                   [[0.5, 0.5],
                    [0.5, 0.5]],
                   [[0.5, 0.5],
                    [0.5, 0.5]],
                   [[0.5, 0.5],
                    [0.5, 0.5]]]], dtype=np.float32))

    # matrix B is the emission matrix p(x[t](, z[t]) | s[t])
    self.mat_b = tf.convert_to_tensor(
        np.array([[[0.5, 0.75],
                   [0.5, 0.75],
                   [0.5, 0.25],
                   [0.5, 0.25]]], dtype=np.float32))

  def test_forward_pass(self):
    fwd_logprob, fwd_lognorm = forward_backward_algo.forward_pass(
        tf.math.log(self.mat_a), tf.math.log(self.mat_b),
        tf.math.log(self.init_pi))
    fwd_prob = tf.exp(fwd_logprob)
    fwd_norm = tf.math.cumprod(tf.exp(fwd_lognorm), axis=1)
    fwd_norm = fwd_norm[:, :, None]

    target_value = np.array([[[1./4., 3./8.],
                              [5./32., 15./64.],
                              [25./256., 25./512.],
                              [75./2048., 75./4096.]]], dtype=np.float32)
    self.assertAllClose(self.evaluate(fwd_prob * fwd_norm), target_value)

  def test_backward_pass(self):
    bwd_logprob, bwd_lognorm = forward_backward_algo.backward_pass(
        tf.math.log(self.mat_a), tf.math.log(self.mat_b),
        tf.math.log(self.init_pi))
    bwd_prob = tf.exp(bwd_logprob)
    bwd_norm = tf.math.cumprod(tf.exp(bwd_lognorm), axis=1, reverse=True)
    bwd_norm = bwd_norm[:, :, None]

    target_value = np.array([[[45./512., 45./512.],
                              [9./64., 9./64.],
                              [3./8., 3./8.],
                              [1., 1.]]], dtype=np.float32)

    self.assertAllClose(self.evaluate(bwd_prob * bwd_norm), target_value)

  def test_forward_backward(self):
    _, _, log_gamma1, log_gamma2 = forward_backward_algo.forward_backward(
        tf.math.log(self.mat_a), tf.math.log(self.mat_b),
        tf.math.log(self.init_pi))
    gamma1, gamma2 = tf.exp(log_gamma1), tf.exp(log_gamma2)
    gamma1_target = np.array([[[90./225., 135./225.],
                               [90./225., 135./225.],
                               [150./225., 75./225.],
                               [150./225., 75./225.]]], dtype=np.float32)
    gamma2_target = np.array([[[[1., 1.],
                                [1., 1.]],
                               [[36./225., 54./225.],
                                [54./225., 81./225.]],
                               [[60./225., 90./225.],
                                [30./225., 45./225.]],
                               [[100./225., 50./225.],
                                [50./225., 25./225.]]]], dtype=np.float32)

    self.assertAllClose(self.evaluate(gamma1), gamma1_target)
    self.assertAllClose(self.evaluate(gamma2), gamma2_target)


if __name__ == "__main__":
  tf.test.main()
