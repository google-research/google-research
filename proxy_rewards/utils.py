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

# Lint as: python3
"""Utilities for movielens simulation.

This project builds upon the
[ml-fairness-gym](https://github.com/google/ml-fairness-gym) recommender
environment. In particular, this file is a heavily modified version of
`ml-fairness-gym/environments/recommenders/movie_lens_utils.py`
"""

import json
import os
import pickle
import types
from typing import Optional
from absl import logging
import file_util
import numpy as np
import pandas as pd

GENRES = [
    'Other', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
GENRE_MAP = {genre: idx for idx, genre in enumerate(GENRES)}
OTHER_GENRE_IDX = GENRE_MAP['Other']

NUM_MOVIES = 3883
NUM_USERS = 6040


def find_top_movies_overall(num_movies, data_path):
  """Find the top movies overall, by number of ratings.

  Args:
    num_movies: Number of movies to return.
    data_path: Path to ratings.csv.

  Returns:
    Numpy array of unique movieIds.

  """
  ratings_df = pd.read_csv(file_util.open(f'{data_path}/ratings.csv', 'r'))

  top_movies = ratings_df.groupby('movieId').size().sort_values(
      ascending=False)[:num_movies].index.values

  top_movies = np.unique(top_movies)

  return top_movies


def find_top_movies_per_genre(movies_per_genre,
                              data_path):
  """Find the top movies of each genre, by number of ratings.

  Args:
    movies_per_genre: Number of top movies per genre.
    data_path: Path to movies.csv, ratings.csv.

  Returns:
    Numpy array of unique movieIds.

  """
  movies_df = pd.read_csv(file_util.open(f'{data_path}/movies.csv', 'r'))
  ratings_df = pd.read_csv(file_util.open(f'{data_path}/ratings.csv', 'r'))

  top_movies = np.array([], dtype=int)
  for genre in GENRES:
    filter_genre = movies_df['genres'].str.contains(genre)
    movies_in_genre = movies_df[filter_genre]['movieId'].values  # pylint: disable=unused-variable
    movies_in_genre_by_num_ratings = ratings_df.query(
        'movieId in @movies_in_genre').groupby('movieId').size()
    top_movies_in_genre = movies_in_genre_by_num_ratings.sort_values(
        ascending=False)[:movies_per_genre].index.values
    top_movies = np.append(top_movies, top_movies_in_genre)

  return np.unique(top_movies)


def load_json_pickle(path):
  """Attempt to load an array-like object from a file path.

  Args:
    path: File to load.

  Returns:
    File loaded via either json or pickle.

  Raises:
    ValueError: If file could not be loaded as either json or pickle
  """
  try:
    return json.load(file_util.open(path, 'rb'))
  except ValueError:
    logging.debug(('File could not be loaded as json, falling back to pickle: '
                   '%s'), path)

  try:
    return pickle.load(file_util.open(path, 'rb'))
  except ValueError:
    raise ValueError(f'File could not be loaded as json or pickle: {path}')


def load_genre_history(env_config):
  """Attempt to load genre history from json or pickle file.

  Args:
    env_config: `EnvConfig` class from movie_lens_simulator.py

  Returns:
    genre_history, a numpy array of shape NUM_USERS x NUM_GENRES

  Raises:
    ValueError: If filetype is neither json nor pickle
  """
  path = env_config.genre_history_path
  genre_history = load_json_pickle(path)
  return np.array(genre_history, dtype=int)


def load_embeddings(env_config):
  """Attempt to loads user and movie embeddings from a json or pickle file.

  Args:
    env_config: `EnvConfig` class from movie_lens_simulator.py

  Returns:
    embedding_dict containing embeddings for movies and users
  """
  path = env_config.embeddings_path
  embedding_dict = load_json_pickle(path)
  return types.SimpleNamespace(
      movies=np.array(
          embedding_dict[env_config.embedding_movie_key], dtype=float),
      users=np.array(
          embedding_dict[env_config.embedding_user_key], dtype=float))


class Dataset(object):
  """Class to represent all of the movielens data together."""

  def __init__(self,
               data_dir,
               genre_history,
               embeddings,
               user_ctor,
               movie_ctor,
               genre_shift=None,
               bias_against_unseen=0.):
    """Initializes data from the data directory.

    Args:
      data_dir: Path to directory with {movies, users, ratings}.csv files.
      genre_history: A numpy array containing genre histories
      embeddings: An object containing embeddings with attributes `users` and
        `movies`.
      user_ctor: User constructor.
      movie_ctor: Movie constructor.
      genre_shift: Optional list of length len(GENRES), which is globally added
        to all user preferences.
      bias_against_unseen: A negative float that reduces the affinity to
        previously seen genres (at the start of the episode)
    """
    self._user_ctor = user_ctor
    self._movie_ctor = movie_ctor
    self._movie_path = os.path.join(data_dir, 'movies.csv')
    self._user_path = os.path.join(data_dir, 'users.csv')
    self._rating_path = os.path.join(data_dir, 'ratings.csv')
    self._movies = self._read_movies(self._movie_path)
    self._users = self._read_users(self._user_path)
    self._populate_embeddings(embeddings)
    self._shift_genre_preferences(genre_shift)
    self._populate_genre_history(data=genre_history)
    # Additional shift in genre preferences, where users give lower ratings to
    # genres that they have not watched before.  This is introduced to create a
    # more meaningful tradeoff between diversity and rating.
    self._update_affinity_for_unseen_genres(weight=bias_against_unseen)

  def _read_movies(self, path):
    """Returns a dict of Movie objects."""
    movies = {}
    movie_df = pd.read_csv(file_util.open(path))

    for _, row in movie_df.iterrows():
      genres = [
          GENRE_MAP.get(genre, OTHER_GENRE_IDX)
          for genre in row.genres.split('|')
      ]
      if not isinstance(row.movieId, int):
        raise ValueError('MovieId is not an integer')
      movie_id = row.movieId
      # `movie_vec` is left as None, and will be filled in later in the init
      # of this Dataset.
      movies[movie_id] = self._movie_ctor(
          movie_id,
          row.title,
          genres,
          vec=None,
      )

    return movies

  def _read_users(self, path):
    """Returns a dict of User objects."""
    users = {}
    for _, row in pd.read_csv(file_util.open(path)).iterrows():
      users[row.userId] = self._user_ctor(user_id=row.userId)

    return users

  def _populate_embeddings(self, initial_embeddings):
    """Modifies stored Users and Movies with learned vectors."""

    for movie_ in self.get_movies():
      movie_.movie_vec = initial_embeddings.movies[movie_.doc_id()]
      if len(movie_.movie_vec) <= len(GENRES):
        raise ValueError('The embeddings must include genre dimensions.')
      if not np.all(
          np.array_equiv(movie_.movie_vec[-len(GENRES):], movie_.genre_vec)):
        raise ValueError('Embedding dims for genre do not match genres!')
    for user_ in self.get_users():
      user_.topic_affinity = np.copy(initial_embeddings.users[user_.user_id])
      # Since users' topic affinities can change over time, store the initial
      # value as well.
      user_.initial_topic_affinity = np.copy(
          initial_embeddings.users[user_.user_id])

  def _shift_genre_preferences(self, genre_shift):
    """Modifies stored user preferences, based on the provided shift.

    Note that this modifies the initial topic affinity, so that resetting the
    user state does not over-write the shift.

    Args:
      genre_shift: None or list of length len(GENRES).
    """
    for user_ in self.get_users():
      user_.shift_genre_preferences(genre_shift)
      user_.initial_topic_affinity = np.copy(user_.topic_affinity)

  def _populate_genre_history(self,
                              data = None,
                              save_path = None,
                              min_freq = None):
    """Modifies stored users with watch history.

    Most users have watched at least one movie from each genre.  With that in
    mind, a minimum threshold is used to make this a non-trivial user feature.

    Because users vary in the number of movies they have watched, a genre
    is included in the genre history if and only if movies from this genre make
    up at least a certain percentage (specified by `min_freq`) of movies rated
    by a given user.

    Args:
      data: Numpy array specifying genre history.  If None, then this is
        generated from the raw data files.
      save_path: If specified (and `data` is None), then generate and save genre
        history to this path.
      min_freq: Minimum percentage of watches from a genre to quality as part of
        history, given as a float in [0., 1.]
    Updates: For each user, populates the genre history as a multi-hot encoding,
      a numpy array of length `NUM_GENRES`.
    """

    if data is not None:
      if save_path:
        raise ValueError('If data is given, save path must be None')

      if not isinstance(data, np.ndarray):
        raise TypeError('Provided data is not a numpy array')

      user_genre_history = np.copy(data)

    else:
      if min_freq < 0 or min_freq > 1:  # pytype: disable=unsupported-operands
        raise ValueError(f'Expected min_freq in [0, 1], got {min_freq:.3f}')
      movie_df = pd.read_csv(file_util.open(self._movie_path))
      ratings_df = pd.read_csv(file_util.open(self._rating_path))

      # Populate multi-hot encoding of genres for each movie
      genre_vecs = np.zeros((NUM_MOVIES, len(GENRES)))

      for idx, row in movie_df.iterrows():
        genres = [
            GENRE_MAP.get(genre, OTHER_GENRE_IDX)
            for genre in row.genres.split('|')
        ]

        genre_vecs[idx, genres] = 1

      if not np.all(genre_vecs.sum(axis=1) > 0):
        raise ValueError('Some movies have no genres')

      # For each user, track the total number of movies of each genre,
      # where a movie can count for multiple genres.
      user_history = np.zeros((NUM_USERS, len(GENRES)))

      for _, row in ratings_df.iterrows():
        user_history[row.userId] += genre_vecs[row.movieId]

      # Normalize by the total number of watches for each user
      num_watches = ratings_df.groupby('userId').size()
      if not np.array_equal(num_watches.index.values, np.arange(NUM_USERS)):
        raise ValueError('userId in provided ratings file has gaps')
      user_genre_per_watch = user_history / np.array(num_watches)[:, None]
      user_genre_history = np.array(user_genre_per_watch > min_freq, dtype=int)

      if save_path:
        with file_util.open(save_path, 'wb') as f:
          pickle.dump(user_genre_history, f)

    for user_ in self.get_users():
      user_.initial_genre_history = np.copy(user_genre_history[user_.user_id])
      user_.genre_history = np.copy(user_genre_history[user_.user_id])

  def _update_affinity_for_unseen_genres(self, weight=-2.):
    """Modifies stored user preferences, based on their history.

    For movies that are NOT part of the genre history, we reduce the embedding
    weight by the amount specified.

    Args:
      weight: Negative float, by which affinity is reduced.
    """
    if weight > 0:
      raise ValueError('Weight should be a non-positive float')

    for user_ in self.get_users():
      # Genre history is 1 for seen, 0 for unseen.  We flip that here.
      unseen_genres = (-1 * user_.genre_history) + 1
      shift = (unseen_genres * weight).tolist()
      user_.shift_genre_preferences(genre_shift=shift)
      user_.initial_topic_affinity = np.copy(user_.topic_affinity)

  def get_movies(self):
    """Returns an iterator over movies."""
    return list(self._movies.values())

  def get_users(self):
    """Returns an iterator over users."""
    return list(self._users.values())
