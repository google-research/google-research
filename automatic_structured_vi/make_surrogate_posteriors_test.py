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

"""Tests for make_surrogate_posteriors."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from automatic_structured_vi import make_surrogate_posteriors

tfd = tfp.distributions


def _make_prior_dist():

  def _prior_model_fn():
    innovation_noise = 0.1
    prior_loc = 0.
    new = yield tfd.Normal(loc=prior_loc, scale=innovation_noise)
    for _ in range(4):
      new = yield tfd.Normal(loc=new, scale=innovation_noise)

  return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)


class MakeSurrogatePosteriorsTest(tf.test.TestCase):

  def assertAllEqualNested(self, struct_a, struct_b):
    tf.nest.map_structure(self.assertAllEqual, struct_a, struct_b)

  def assertAllFinite(self, a):
    is_finite = np.isfinite(a.numpy())
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def test_make_flow_posterior(self):
    prior_dist = _make_prior_dist()

    iaf_surrogate_dist = make_surrogate_posteriors.make_flow_posterior(
        prior_dist, num_hidden_units=8, invert=True)

    iaf_log_prob = prior_dist.log_prob(iaf_surrogate_dist.sample())
    self.assertAllFinite(iaf_log_prob)

    self.assertAllEqualNested(prior_dist.event_shape_tensor(),
                              iaf_surrogate_dist.event_shape_tensor())

    maf_surrogate_dist = make_surrogate_posteriors.make_flow_posterior(
        prior_dist, num_hidden_units=8, invert=False)

    iaf_log_prob = prior_dist.log_prob(maf_surrogate_dist.sample())
    self.assertAllFinite(iaf_log_prob)

    self.assertAllEqualNested(prior_dist.event_shape_tensor(),
                              maf_surrogate_dist.event_shape_tensor())

  def test_make_mvn_posterior(self):
    prior_dist = _make_prior_dist()
    surrogate_dist = make_surrogate_posteriors.make_mvn_posterior(
        prior_dist)

    self.assertAllEqualNested(
        prior_dist.event_shape_tensor(),
        surrogate_dist.event_shape_tensor())

    self.assertAllFinite(prior_dist.log_prob(surrogate_dist.sample()))

  def test_build_autoregressive_surrogate_posterior(self):
    prior_dist = _make_prior_dist()
    surrogate_dist = make_surrogate_posteriors.build_autoregressive_surrogate_posterior(
        prior_dist, make_surrogate_posteriors.make_conditional_linear_gaussian)

    self.assertAllEqualNested(
        prior_dist.event_shape_tensor(),
        surrogate_dist.event_shape_tensor())

    self.assertAllFinite(prior_dist.log_prob(surrogate_dist.sample()))

if __name__ == '__main__':
  tf.test.main()
