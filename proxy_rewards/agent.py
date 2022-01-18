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

"""Class definitions for MovieLens agents."""

from typing import Sequence

from absl import logging
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing

from proxy_rewards import utils


class MovieLensAgent():
  """Generic MovieLens Agent.

  It is generically assumed that an agent will only take userId as input, since
  this is the canonical observation space for the simulator.  Agents that need
  to make use of user embeddings will need to have those provided, and use them
  internally to make recommendations.

  Attributes:
    movies: Set of movies (represented by MovieId) that can be recommended.
  """

  def __init__(self, movies):
    self.movies = movies

  def make_recommendation(self, user_id):
    """Recommend a movie from the set of all movies.

    In RecSim, an agent recommends a slate, and a user must choose from entries
    in that slate.  In our simulation, we want to recommend a single item, but
    still allow for users to choose a different movie from the universe of
    possible options.

    To implement this, all agents should return the full set of movies as the
    slate, which should always have the same elements: The first position in the
    slate corresponds to the "recommendation", and is treated as such by the
    user choice model, while the order of the remaining elements is ignored.

    Args:
      user_id: User for whom we make the recommendation.

    Returns:
      Slate of movies, where the first entry corresponds to the recommendation.

    """
    pass

  def _convert_rec_to_slate(self, movie_id):
    """Convert a single movieId to a slate.

    As described above, the agent controls the first element of the slate, which
    is interpreted as the recommendation of the agent.

    Args:
      movie_id: The movie to be placed first in the returned slate, i.e., the
        recommendation of the agent.

    Returns:
      A full slate containing all of the movies in self.movies.
    """
    assert movie_id in self.movies, 'Agent recommended a movie not in the set.'
    slate = [movie_id]
    slate.extend([m for m in self.movies if m != movie_id])
    slate = np.array(slate)

    return slate


class RandomAgent(MovieLensAgent):

  def __init__(self, movies, seed=0):
    super(RandomAgent, self).__init__(movies)
    self._rng = np.random.default_rng(seed)

  def make_recommendation(self, user_id):
    """Recommend a random movie from the slate."""
    return self._rng.permutation(self.movies)


class FittedAgent(MovieLensAgent):
  """Agent that is fitted using separate linear models for each action.

  Agent is initialized with the original user embeddings, and uses these to make
  predictions of the value for each movie.

  Note that user embeddings are given as *known* features.  The corresponding
  coefficients learned by each movie-specific model can be interpreted as an
  embedding for that movie.  In addition, we include a feature that indicates
  whether or not the movie is from a new genre for the user.
  """

  def __init__(self, movies, embed_path, genre_history_path):
    super(FittedAgent, self).__init__(movies)
    embed_dict = utils.load_json_pickle(embed_path)
    self.user_embeddings = np.array(
        embed_dict['users'], dtype=float)

    # Determine which movies contain new genres, for each user
    user_genres_seen = np.array(
        utils.load_json_pickle(genre_history_path), dtype=int)
    user_genres_unseen = (user_genres_seen * -1) + 1
    movie_genres = np.array(
        embed_dict['movies'], dtype=float)[:, -len(utils.GENRES):]

    self.user_by_movie_unseen_genre = user_genres_unseen.dot(movie_genres.T) > 0

    self.models = {}
    self.user_recs = {}
    # Track if outcome to predict is binary or continuous
    self._binary_outcome = True
    self._is_fitted = False
    self._is_cached = False

  def make_recommendation(self, user_id):
    """Recommend the movie predicted to have the highest score."""
    if self._is_cached:
      return self.user_recs[user_id]
    else:
      recommended_movie_id = None
      max_score = -1

      # Call score function, choose the top result.
      for movie_id in self.movies:
        this_score = self._score_movie(user_id, movie_id)
        if this_score > max_score:
          recommended_movie_id = movie_id
          max_score = this_score

      return self._convert_rec_to_slate(recommended_movie_id)

  def fit(self, data):
    """Fit the model to data.

    Args:
      data: Dictionary with keys 'recommendation', 'user_id', 'reward', each
        corresponding to a numpy array of the same first dimension.
    """
    self._validate_data(data)
    self._check_outcome_type(data)
    if self._binary_outcome:
      model = pipeline.make_pipeline(
          preprocessing.StandardScaler(),
          linear_model.LogisticRegression(penalty='none', max_iter=1000))
    else:
      model = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                     linear_model.LinearRegression())

    for movie_id in self.movies:
      idxs = data['recommendation'] == movie_id
      this_model = sklearn.base.clone(model)
      user_embeds = self.user_embeddings[data['user_id'][idxs]]
      is_new_genre = self.user_by_movie_unseen_genre[
          data['user_id'][idxs]][:, [movie_id]]
      features = np.hstack((user_embeds, is_new_genre))
      this_model.fit(features, data['reward'][idxs])
      self.models[movie_id] = this_model

    self._is_fitted = True
    self._cache_recs()

  def _cache_recs(self):
    """Cache recommendations for all possible users."""
    for user_id in range(self.user_embeddings.shape[0]):
      self.user_recs[user_id] = self.make_recommendation(user_id)
    self._is_cached = True

  def _score_movie(self, user_id, movie_id):
    """Look up the relevant model and score this movie."""
    # Get user embedding as a 2D array of shape (1, embedding_dim)
    assert self._is_fitted
    user_embed = self.user_embeddings[[user_id]]
    is_new_genre = self.user_by_movie_unseen_genre[user_id, movie_id]
    features = np.hstack((user_embed, np.array(is_new_genre, ndmin=2)))

    if self._binary_outcome:
      score = self.models[movie_id].predict_proba(features)[0, 1]
    else:
      score = self.models[movie_id].predict(features)[0]
    return score

  def _validate_data(self, data):
    """Validate the provided data."""
    for k in ['recommendation', 'user_id', 'reward']:
      if k not in data.keys():
        raise ValueError(f'Data is missing entry "{k}"')
      if not isinstance(data[k], np.ndarray):
        raise ValueError(f'Data[{k}] is not a numpy array')

    if data['recommendation'].shape[0] != data['user_id'].shape[0] or data[
        'user_id'].shape[0] != data['reward'].shape[0]:
      raise ValueError((f'Mismatch in shapes: {data["recommendations"].shape}, '
                        f'{data["user_id"].shape}, {data["reward"].shape}'))

    if set(np.unique(data['recommendation'])) != set(self.movies):
      raise ValueError(('Observed recommendation set does not equal '
                        'the set of movie_ids in agent.movies'))

  def _check_outcome_type(self, data):
    """Check if reward is binary or continuous."""
    if np.array_equal(data['reward'], data['reward'].astype(bool)):
      logging.info('Inferred a binary reward')
      self._binary_outcome = True
    else:
      logging.info('Inferred a continuous reward')
      self._binary_outcome = False


def load_agent(path):
  agent = utils.load_json_pickle(path)
  if not isinstance(agent, MovieLensAgent):
    raise TypeError('Loaded agent is not of class MovieLensAgent')

  return agent
