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

# Lint as: python3
"""Tests for movie_lens."""
import copy
import functools
import os
import tempfile
from absl import flags
from absl.testing import absltest
import attr
import file_util
import test_util
from environments.recommenders import recsim_samplers
from environments.recommenders import recsim_wrapper
import numpy as np
from recsim.simulator import recsim_gym

from proxy_rewards import movie_lens_simulator as movie_lens
from proxy_rewards import utils as movie_lens_utils

FLAGS = flags.FLAGS


class MovieLensTestNoShift(absltest.TestCase):

  def _initialize_from_config(self, env_config):
    self.working_dir = tempfile.mkdtemp(dir='/tmp')

    self.initial_embeddings = movie_lens_utils.load_embeddings(env_config)
    self.genre_history = movie_lens_utils.load_genre_history(env_config)

    user_ctor = functools.partial(movie_lens.User,
                                  **attr.asdict(env_config.user_config))
    self.dataset = movie_lens_utils.Dataset(
        env_config.data_dir,
        user_ctor=user_ctor,
        movie_ctor=movie_lens.Movie,
        genre_history=self.genre_history,
        embeddings=self.initial_embeddings,
        genre_shift=env_config.genre_shift,
        bias_against_unseen=env_config.bias_against_unseen)

    self.document_sampler = recsim_samplers.SingletonSampler(
        self.dataset.get_movies(), movie_lens.Movie)

    self.user_sampler = recsim_samplers.UserPoolSampler(
        seed=env_config.seeds.user_sampler,
        users=self.dataset.get_users(),
        user_ctor=user_ctor)

    self.user_model = movie_lens.UserModel(
        user_sampler=self.user_sampler,
        seed=env_config.seeds.user_model,
        slate_size=env_config.slate_size,
    )

    env = movie_lens.MovieLensEnvironment(
        self.user_model,
        self.document_sampler,
        num_candidates=self.document_sampler.size(),
        slate_size=env_config.slate_size,
        resample_documents=False)
    env.reset()

    reward_aggregator = movie_lens.average_ratings_reward

    self.env = recsim_gym.RecSimGymEnv(env, reward_aggregator)

  def setUp(self):
    super(MovieLensTestNoShift, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 'test_data')
    self.env_config = movie_lens.EnvConfig(
        seeds=movie_lens.Seeds(0, 0),
        data_dir=self.data_dir,
        genre_history_path=os.path.join(self.data_dir, 'genre_history.json'),
        embeddings_path=os.path.join(self.data_dir, 'embeddings.json'),
        genre_shift=None,
        bias_against_unseen=0.)
    self._initialize_from_config(self.env_config)

  def tearDown(self):
    file_util.delete_recursively(self.working_dir)
    super(MovieLensTestNoShift, self).tearDown()

  def test_document_observation_space_matches(self):
    for doc in self.dataset.get_movies():
      self.assertIn(doc.create_observation(), doc.observation_space())

  def test_user_observation_space_matches(self):
    user = self.user_sampler.sample_user()
    self.assertIn(user.create_observation(), user.observation_space())

  def test_observations_in_observation_space(self):
    for slate in [[0], [1], [2]]:
      observation, _, _, _ = self.env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field],
                      self.env.observation_space.spaces[field])

  def test_diversity_seeking_impacts_score(self):
    n_diversity = 0
    user = self.user_sampler.get_user(1)

    user.diversity_seeking = True

    for doc in self.dataset.get_movies():
      if user.check_if_new_genre(doc) == 1.:
        n_diversity += 1
        self.assertNotEqual(user.score_document(doc), user.rate_document(doc))
      else:
        self.assertEqual(user.score_document(doc), user.rate_document(doc))

    # Make sure that the test data contains at least one movie that was diverse
    # for this user
    self.assertGreater(n_diversity, 0)

    user.diversity_seeking = False

    for doc in self.dataset.get_movies():
      self.assertEqual(user.score_document(doc), user.rate_document(doc))

  def test_user_can_rate_document(self):
    user = self.user_sampler.get_user(1)
    for doc in self.dataset.get_movies():
      self.assertBetween(
          user.rate_document(doc), movie_lens.MIN_RATING_SCORE,
          movie_lens.MAX_RATING_SCORE)

  def test_user_genre_can_shift(self):
    user = self.user_sampler.get_user(1)
    ratings_before = [
        user.rate_document(doc) for doc in self.dataset.get_movies()
    ]

    genre_shift = [2.] * len(movie_lens_utils.GENRES)
    user.shift_genre_preferences(genre_shift)
    ratings_after = [
        user.rate_document(doc) for doc in self.dataset.get_movies()
    ]

    # NOTE: This test can fail with 1.0 == 1.0 if you have modified
    # the original scores to the point that the genre shift does not
    # push a pre-clipped score about 1.0.  Similar for 5.0 == 5.0.
    for pair in zip(ratings_before, ratings_after):
      self.assertNotEqual(pair[0], pair[1])

  def test_environment_can_advance_by_steps(self):
    # Recommend some manual slates.
    for slate in [[0], [1], [3]]:
      # Tests that env.step completes successfully.
      self.env.step(slate)

  def test_environment_observation_space_is_as_expected(self):
    for slate in [[0], [1], [2]]:
      observation, _, _, _ = self.env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field],
                      self.env.observation_space.spaces[field])

  def test_gym_environment_builder(self):
    env = movie_lens.create_gym_environment(self.env_config)
    env.seed(100)
    env.reset()

    # Recommend some manual slates and check that the observations are as
    # expected.
    for slate in [[0], [0], [2]]:
      observation, _, _, _ = env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field], env.observation_space.spaces[field])

  def test_if_user_state_resets(self):
    observation = self.env.reset()
    curr_user_id = observation['user']['user_id']
    ta_vec = np.copy(self.env._environment.user_model._user_sampler
                     ._users[curr_user_id].topic_affinity)
    for i in range(3):
      self.env.step([i])
    self.env.reset()
    ta_new = self.env._environment.user_model._user_sampler._users[
        curr_user_id].topic_affinity
    self.assertTrue(np.all(ta_new == ta_vec))

  def test_user_order_is_shuffled(self):
    """Tests that user order does not follow a fixed pattern.

    We test this by checking that the list is not perioc for periods between
    0-10. Since there are only 5 unique users, this is enough to show that
    it's not following a simple pattern.
    """
    self.env.seed(100)

    user_list = []
    for _ in range(100):
      observation = self.env.reset()
      user_list.append(observation['user']['user_id'])

    def _is_periodic(my_list, period):
      for idx, val in enumerate(my_list[:-period]):
        if val != my_list[idx + period]:
          return False
      return True

    for period in range(1, 10):
      self.assertFalse(_is_periodic(user_list, period))

  def test_user_order_is_consistent(self):
    self.env.reset_sampler()
    first_list = []
    for _ in range(100):
      observation = self.env.reset()
      first_list.append(observation['user']['user_id'])

    self.env.reset_sampler()
    other_list = []
    for _ in range(100):
      observation = self.env.reset()
      other_list.append(observation['user']['user_id'])

    self.assertEqual(first_list, other_list)

    # Also check that changing the seed creates a new ordering.
    config = copy.deepcopy(self.env_config)
    config.seeds.user_sampler += 1
    env = movie_lens.create_gym_environment(config)
    other_list = []
    for _ in range(100):
      observation = env.reset()
      other_list.append(observation['user']['user_id'])
    self.assertNotEqual(first_list, other_list)

  def test_ml_fairness_gym_environment_can_run(self):
    ml_fairness_env = recsim_wrapper.wrap(self.env)
    test_util.run_test_simulation(env=ml_fairness_env, stackelberg=True)


class MovieLensTestShift(MovieLensTestNoShift):
  """Test with genre shifts."""

  def setUp(self):
    super(MovieLensTestShift, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 'test_data')
    self.env_config = movie_lens.EnvConfig(
        seeds=movie_lens.Seeds(0, 0),
        data_dir=self.data_dir,
        genre_history_path=os.path.join(self.data_dir, 'genre_history.json'),
        embeddings_path=os.path.join(self.data_dir, 'embeddings.json'),
        genre_shift=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bias_against_unseen=0.)
    self._initialize_from_config(self.env_config)


class MovieLensTestBiasAgainstUnseen(MovieLensTestNoShift):
  """Test with bias against unseen genres."""

  def setUp(self):
    super(MovieLensTestBiasAgainstUnseen, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 'test_data')
    self.env_config = movie_lens.EnvConfig(
        seeds=movie_lens.Seeds(0, 0),
        data_dir=self.data_dir,
        genre_history_path=os.path.join(self.data_dir, 'genre_history.json'),
        embeddings_path=os.path.join(self.data_dir, 'embeddings.json'),
        genre_shift=None,
        bias_against_unseen=-1.)
    self._initialize_from_config(self.env_config)

if __name__ == '__main__':
  absltest.main()
