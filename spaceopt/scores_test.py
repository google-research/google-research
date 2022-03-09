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

"""Tests for scores."""
import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from spaceopt import scores


def mocked_fantasize_y_values(key,
                              x_locations_for_p_min,
                              params,
                              x_obs,
                              y_obs,
                              y_fantasized_num_for_p_min=50):
  del params
  del x_obs
  del y_obs
  mu = jnp.zeros((x_locations_for_p_min.shape[0], 1))
  cov = jnp.eye(x_locations_for_p_min.shape[0])

  y_rand = jax.random.normal(key, (cov.shape[0], y_fantasized_num_for_p_min))
  chol = jscipy.linalg.cholesky(cov + jnp.eye(cov.shape[0]) * 1e-4, lower=True)
  fantasized_y = jnp.dot(chol, y_rand) + mu

  return fantasized_y


def draw_x_location_fn(key, search_space, budget):
  return jax.random.uniform(
      key,
      shape=(budget, 1),
      minval=search_space[:, 0],
      maxval=search_space[:, 1])


partial_fantasize_y_values = functools.partial(
    mocked_fantasize_y_values, params=None, x_obs=None, y_obs=None)


def draw_y_values_fn(key, x_locations, y_drawn_num):
  return partial_fantasize_y_values(
      key=key,
      x_locations_for_p_min=x_locations,
      y_fantasized_num_for_p_min=y_drawn_num)


class ScoresTest(absltest.TestCase):

  def setUp(self):

    super(ScoresTest, self).setUp()

    x_locations = jnp.linspace(-4., 4., 7)[:, None]
    self.x_obs = jnp.linspace(-1., 1., 2)[:, None]
    self.y_obs = jnp.array([[7.], [11.]])

    self.x_batch = jnp.linspace(1., 3., 4)[:, None]
    self.y_batch = jnp.array([[1., 5., 9.],
                              [2., 6., 10.],
                              [3., 7., 11.],
                              [4., 8., 12.]])
    key = jax.random.PRNGKey(0)
    self.utility_measure = scores.UtilityMeasure(
        incumbent=6.,
        x_locations_for_p_min=x_locations,
        params=None,
        fantasize_y_values_for_p_min=mocked_fantasize_y_values,
        y_fantasized_num_for_p_min=50,
        initial_entropy_key=key)

    self.key = jax.random.PRNGKey(0)
    self.budget = 10
    self.search_space = jnp.array([[-4., 4.]])

  def test_is_improvement_shape(self):
    """Test that the is_improvement output has the right shape."""
    is_improvement = self.utility_measure.is_improvement(self.y_batch)
    self.assertEqual(is_improvement.shape, (self.y_batch.shape[1],))

  def test_is_improvement_values(self):
    """Test that the is_improvement output has the right value."""
    is_improvement = self.utility_measure.is_improvement(self.y_batch)
    self.assertTrue((is_improvement == jnp.array([True, True, False])).all())

  def test_improvement_shape(self):
    """Test that the improvement output has the right shape."""
    improvement = self.utility_measure.improvement(self.y_batch)
    self.assertEqual(improvement.shape, (self.y_batch.shape[1],))

  def test_improvement_values(self):
    """Test that the improvement output has the right value."""
    improvement = self.utility_measure.improvement(self.y_batch)
    self.assertTrue((improvement == jnp.array([5., 1., 0.])).all())

  def test_information_gain_shape(self):
    """Test that the information_gain output has the right shape."""
    key = jax.random.PRNGKey(1)
    information_gain = self.utility_measure.information_gain(
        key=key,
        x_obs=self.x_obs,
        y_obs=self.y_obs,
        x_batch=self.x_batch,
        y_batch=self.y_batch)
    self.assertEqual(information_gain.shape, (self.y_batch.shape[1],))

  def test_score_values(self):
    """Test that the improvement-based score cannot be negative."""
    # pylint: disable=unused-argument
    def utility_is_imp_fn(key, x_batch, y_batch):
      return self.utility_measure.is_improvement(y_batch)

    def utility_imp_fn(key, x_batch, y_batch):
      return self.utility_measure.improvement(y_batch)
    # pylint: enable=unused-argument
    statistics_fns = [jnp.mean, jnp.median]
    key = jax.random.PRNGKey(1)

    mean_utility = scores.mean_utility(
        key,
        self.search_space,
        self.budget,
        utility_is_imp_fn,
        draw_y_values_fn,
        x_drawn_num=100,
        y_drawn_num=100)
    scores_dict = scores.scores(mean_utility, statistics_fns)

    for j in range(len(statistics_fns)):
      self.assertGreaterEqual(scores_dict[statistics_fns[j].__name__], 0.)

    mean_utility = scores.mean_utility(
        key,
        self.search_space,
        self.budget,
        utility_imp_fn,
        draw_y_values_fn,
        x_drawn_num=100,
        y_drawn_num=100)
    scores_dict = scores.scores(mean_utility, statistics_fns)

    for j in range(len(statistics_fns)):
      self.assertGreaterEqual(scores_dict[statistics_fns[j].__name__], 0.)


if __name__ == '__main__':
  absltest.main()
