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

"""Tests for train_proxy."""

from absl.testing import absltest
from flax import linen as nn
import jax
import numpy as np
from sklearn import linear_model

from proxy_rewards import train_proxy


class TrainProxyTest(absltest.TestCase):

  def setUp(self):
    super(TrainProxyTest, self).setUp()
    np.random.seed(0)
    self.data_x = np.array(
        np.random.binomial(1, 0.5, size=(1000, 2)), dtype=float)
    self.data_y = np.random.binomial(
        1, p=nn.sigmoid(self.data_x[:, 0] - self.data_x[:, 1]))
    self.data_a = np.random.choice(3, size=self.data_y.shape)
    self.data_t = np.random.choice(3, size=self.data_a.shape)

    self.data = {
        'a': self.data_a,
        'm': self.data_x,
        'y': self.data_y,
        't': self.data_t,
    }

  def test_match_sklearn(self):
    clf_sklearn = linear_model.LogisticRegression(penalty='none')
    clf_sklearn.fit(self.data_x, self.data_y)

    self.assertAlmostEqual(clf_sklearn.intercept_[0], -0.0271, places=3)
    self.assertAlmostEqual(clf_sklearn.coef_[0][0], 1.0987, places=3)
    self.assertAlmostEqual(clf_sklearn.coef_[0][1], -1.0216, places=3)

    model = train_proxy.LogisticReg()

    erm_loss = train_proxy.make_loss_func(
        model, self.data, erm_weight=1., bias_lamb=0.)

    init_params = train_proxy.initialize_params(
        model, mdim=self.data_x.shape[1], seed=0)

    erm_params, _ = train_proxy.train(erm_loss, init_params, lr=1., nsteps=1000)

    b, w = jax.tree_leaves(erm_params)
    self.assertAlmostEqual(w[0].item(), 1.0988, places=3)
    self.assertAlmostEqual(w[1].item(), -1.0216, places=3)
    self.assertAlmostEqual(b.item(), -0.0274, places=3)

  def test_l2_regularization(self):
    model = train_proxy.LogisticReg()

    erm_loss_reg = train_proxy.make_loss_func(
        model, self.data, erm_weight=1., l2_lamb=10., bias_lamb=0.)

    init_params = train_proxy.initialize_params(
        model, mdim=self.data_x.shape[1], seed=0)

    erm_params_reg, _ = train_proxy.train(
        erm_loss_reg, init_params, lr=1., nsteps=1000)

    b, w = jax.tree_leaves(erm_params_reg)
    self.assertAlmostEqual(w[0].item(), 0.0030, places=3)
    self.assertAlmostEqual(w[1].item(), -0.0028, places=3)
    self.assertAlmostEqual(b.item(), 0.0158, places=3)

  def test_policy_bias_regularization(self):
    model = train_proxy.LogisticReg()

    mix_loss = train_proxy.make_loss_func(
        model, self.data, erm_weight=1., bias_lamb=10.)

    init_params = train_proxy.initialize_params(
        model, mdim=self.data_x.shape[1], seed=0)

    mix_params, _ = train_proxy.train(
        mix_loss, init_params, lr=1., nsteps=1000)

    b, w = jax.tree_leaves(mix_params)
    self.assertAlmostEqual(w[0].item(), 1.4478, places=3)
    self.assertAlmostEqual(w[1].item(), -1.2915, places=3)
    self.assertAlmostEqual(b.item(), 0.1693, places=3)

if __name__ == '__main__':
  absltest.main()
