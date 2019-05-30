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

"""Tests for program transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import simple_probabilistic_programming.numpy as ed


class ProgramTransformationsTest(absltest.TestCase):

  def testMakeLogJointUnconditional(self):
    """Test `make_log_joint` works on unconditional model."""
    def normal_normal_model():
      loc = ed.norm.rvs(loc=0., scale=1., name='loc')
      x = ed.norm.rvs(loc=loc, scale=0.5, size=5, name='x')
      return x

    log_joint = ed.make_log_joint_fn(normal_normal_model)

    x = np.random.normal(size=5)
    loc = 0.3

    value = log_joint(loc=loc, x=x)
    true_value = np.sum(ed.norm.logpdf(loc, loc=0., scale=1.))
    true_value += np.sum(ed.norm.logpdf(x, loc=loc, scale=0.5))
    self.assertAlmostEqual(value, true_value)

  def testMakeLogJointConditional(self):
    """Test `make_log_joint` works on conditional model."""
    def linear_regression(features, prior_precision):
      beta = ed.norm.rvs(loc=0.,
                         scale=1. / np.sqrt(prior_precision),
                         size=features.shape[1],
                         name='beta')
      loc = np.einsum('ij,j->i', features, beta)
      y = ed.norm.rvs(loc=loc, scale=1., name='y')
      return y

    log_joint = ed.make_log_joint_fn(linear_regression)

    features = np.random.normal(size=[3, 2])
    prior_precision = 0.5
    beta = np.random.normal(size=[2])
    y = np.random.normal(size=[3])

    true_value = np.sum(ed.norm.logpdf(
        beta, loc=0., scale=1. / np.sqrt(prior_precision)))
    loc = np.einsum('ij,j->i', features, beta)
    true_value += np.sum(ed.norm.logpdf(y, loc=loc, scale=1.))

    # Test args as input.
    value = log_joint(features, prior_precision, beta, y)
    self.assertAlmostEqual(value, true_value)

    # Test kwargs as input.
    value = log_joint(features, prior_precision, y=y, beta=beta)
    self.assertAlmostEqual(value, true_value)

if __name__ == '__main__':
  np.random.seed(8327)
  absltest.main()
