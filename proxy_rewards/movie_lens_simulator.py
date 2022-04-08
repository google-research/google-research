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

"""Dynamic user simulation using movielens data.

This project builds upon the
[ml-fairness-gym](https://github.com/google/ml-fairness-gym) recommender
environment. In particular, this file is a heavily modified version of
`ml-fairness-gym/environments/recommenders/movie_lens_dynamic.py`
"""
import collections
import functools
from typing import List

from absl import flags
import attr
import core
from environments.recommenders import recsim_samplers
from gym import spaces
import numpy as np
from recsim import choice_model as recsim_choice
from recsim import document
from recsim import user
from recsim.simulator import environment as recsim_environment
from recsim.simulator import recsim_gym

from proxy_rewards import utils

FLAGS = flags.FLAGS

MIN_RATING_SCORE = 1
MAX_RATING_SCORE = 5

MIN_DIVERSITY_SCORE = 0
MAX_DIVERSITY_SCORE = 1


@attr.s
class UserConfig(core.Params):
  affinity_update_delta = attr.ib(default=1.0)
  topic_affinity_update_threshold = attr.ib(default=3.0)
  accept_prob = attr.ib(default=0.5)
  diversity_prob = attr.ib(default=0.5)


@attr.s
class Seeds(core.Params):
  user_sampler = attr.ib(default=None)
  user_model = attr.ib(default=None)
  train_eval_test = attr.ib(default=None)


@attr.s
class EnvConfig(core.Params):
  """Config object for MovieLensEnvironment."""
  data_dir = attr.ib()
  genre_history_path = attr.ib()
  # Path to a json or pickle file with a dict containing user and movie
  # embeddings
  embeddings_path = attr.ib()
  # Dictionary key to access the movie embeddings.
  embedding_movie_key = attr.ib(default='movies')
  # Dictionary key to access the user embeddings.
  embedding_user_key = attr.ib(default='users')
  train_eval_test = attr.ib(factory=lambda: [0.7, 0.15, 0.15])
  user_config = attr.ib(factory=UserConfig)
  seeds = attr.ib(factory=Seeds)
  slate_size = attr.ib(default=1)
  # The genre bias is used to simulate dataset shift, and contributes to the
  # rating of movies independently of the topic affinity.
  genre_shift = attr.ib(factory=list)

  @genre_shift.validator
  def _check_genre_shift(self, attribute, value):
    if value and len(value) != len(utils.GENRES):
      raise ValueError(('genre_shift must be empty or len(genre_shift) must '
                        'match the number of genres'))

  bias_against_unseen = attr.ib(default=0.)


class Movie(document.AbstractDocument):
  """Class to represent a MovieLens Movie.

  Attributes:
    _doc_id: A unique integer (accessed with the doc_id() method).
    title: A string.
    genres: A list of ints between 0 and len(GENRES).
    genre_vec: A binary vector of size len(GENRES).
    movie_vec: An embedding vector for this movie.
  """

  def __init__(self, doc_id, title, genres,
               vec):
    super(Movie, self).__init__(int(doc_id))
    self.title = title
    self.genres = genres
    self.genre_vec = np.zeros(len(utils.GENRES), dtype=np.int)
    self.genre_vec[self.genres] = 1
    self.movie_vec = vec

  def create_observation(self):
    """Returns an observation dictionary."""
    return {'genres': self.genre_vec, 'doc_id': self.doc_id()}

  @classmethod
  def observation_space(cls):
    """Returns a gym.Space describing observations."""
    return spaces.Dict({
        'genres': spaces.MultiBinary(len(utils.GENRES)),
        'doc_id': spaces.Discrete(utils.NUM_MOVIES)
    })


class Response(user.AbstractResponse):
  """Class to represent a user's response to a document."""

  def __init__(self, rating=0, diversity=0, doc_id=-1):
    self.rating = rating
    self.diversity = diversity
    if doc_id < 0:
      raise ValueError('Document ID not captured in response')
    self.doc_id = doc_id

  def create_observation(self):
    # Ratings are cast into numpy floats to be consistent with the space
    # described by `spaces.Box` (see the response_space description below).
    return {
        'rating': np.float_(self.rating),
        'diversity': np.float_(self.diversity),
        'doc_id': np.int_(self.doc_id),
    }

  @classmethod
  def response_space(cls):
    return spaces.Dict({
        'rating':
            spaces.Box(MIN_RATING_SCORE, MAX_RATING_SCORE, tuple(), np.float32),
        'diversity':
            spaces.Box(MIN_DIVERSITY_SCORE, MAX_DIVERSITY_SCORE, tuple(),
                       np.float32),
        'doc_id':
            spaces.Discrete(utils.NUM_MOVIES),
    })


class User(user.AbstractUserState):
  """Class to represent a movielens user."""

  def __init__(self,
               user_id,
               affinity_update_delta=1.0,
               topic_affinity_update_threshold=3.0,
               accept_prob=0.5,
               diversity_prob=0.5):
    """Initializes the dynamic user.

    Args:
      user_id: Integer identifier of the user.
      affinity_update_delta: Delta for updating user's preference for genres
        whose movies are rated >= topic_affinity_update_threshold.
      topic_affinity_update_threshold: Rating threshold above which user's
        preferences for the genre's is updated.
      accept_prob: Probability that this user will accept our recommendation.
      diversity_prob: Probability that this user is diversity-seeking.
    """
    self.user_id = user_id
    self.affinity_update_delta = affinity_update_delta
    self.topic_affinity_update_threshold = topic_affinity_update_threshold
    # The topic_affinity vector is None until the _populate_embeddings call
    # assigns a vector to it.
    self.topic_affinity = None
    # Also store a pristine initial value so that the user can be reset after
    # topic affinity changes during the simulation.
    # NOTE: If there is a genre_shift, then this is included in the initial
    # topic affinity.
    self.initial_topic_affinity = None
    # Multi-hot encoding representing genres watched by the user, populated
    # by _populate_genre_history in utils.Dataset
    self.initial_genre_history = np.zeros(len(utils.GENRES))
    self.genre_history = np.zeros(len(utils.GENRES))
    self.accept_prob = accept_prob
    # TODO(moberst@): The user_ids are fixed (from the original dataset), and so
    # there is no option to change these random seeds across runs (unlike the
    # other random seeds that are passed to the Seeds object).  We may want to
    # change this in the future.
    self._rng = np.random.default_rng(seed=user_id)
    self.diversity_prob = diversity_prob
    self.diversity_seeking = bool(self._rng.binomial(1, diversity_prob))

  def rate_document(self, doc):
    """Returns the user's rating for a document."""
    return np.clip(
        np.dot(doc.movie_vec, self.topic_affinity), MIN_RATING_SCORE,
        MAX_RATING_SCORE)

  def check_if_new_genre(self, doc):
    """Check if the movie is from a new genre.

    If the movie contains a genre that is not represented in the users genre
    history, then this returns 1, otherwise zero.

    Args:
      doc: Movie to be scored.

    Returns:
      Diversity score for the movie (1 or 0).
    """
    for genre in doc.genres:
      if self.genre_history[genre] == 0:
        return 1.

    return 0.

  def score_document(self, doc, lambda_diversity = 3.):
    """Returns the user affinity for choosing a document.

    Note that the score is distinct from the rating:  In the fairness gym
    movielens simulator, the score is equal to the rating, but in our case we
    incorporate diversity-seeking behavior.

    Args:
      doc: Movie to be scored.
      lambda_diversity: Amount to weight diversity in the scoring objective, if
        the user is diversity-seeking.

    Returns:
      Score for the movie.
    """
    return (self.rate_document(doc) + self.diversity_seeking *
            lambda_diversity * self.check_if_new_genre(doc))

  def check_accept(self):
    """Check whether or not the user accepts our recommendation.

    This is currently a fixed probability that is the same across all movies,
    so the user does not take the "quality" of the recommendation into account.

    Returns:
      Whether or not the user accepts the recommendation.
    """
    return bool(self._rng.binomial(1, self.accept_prob))

  def create_observation(self):
    """Returns a user observation."""
    return {'user_id': self.user_id}

  def _update_affinity_vector(self, doc):
    embedding_dim = len(self.topic_affinity)
    if embedding_dim < len(utils.GENRES):
      raise ValueError('Embedding dimension is smaller than number of genres')
    offset_index = embedding_dim - len(utils.GENRES)
    genre_indices_to_update = [
        genre_id + offset_index for genre_id in doc.genres
    ]
    self.topic_affinity[genre_indices_to_update] *= (1 +
                                                     self.affinity_update_delta)

  def shift_genre_preferences(self, genre_shift=None):
    """Alter genre preferences in line with the provided shift.

    Note that this is called as part of the initialization of utils.Dataset,
    in Dataset._shift_genre_preferences()

    Args:
      genre_shift: List of length len(utils.GENRES) or None
    """
    embedding_dim = len(self.topic_affinity)
    if embedding_dim < len(utils.GENRES):
      raise ValueError('Embedding dimension is smaller than number of genres')
    if genre_shift and len(genre_shift) != len(utils.GENRES):
      raise ValueError('Genre shift vector is not the correct length')
    if genre_shift:
      self.topic_affinity[-len(utils.GENRES):] += genre_shift

  def _update_genre_history(self, doc):
    for genre in doc.genres:
      self.genre_history[genre] = 1.

  def update_state(self, doc, response):
    if response.rating >= self.topic_affinity_update_threshold:
      self._update_affinity_vector(doc)

    self._update_genre_history(doc)

  def reset_state(self):
    self.topic_affinity = np.copy(self.initial_topic_affinity)
    self.genre_history = np.copy(self.initial_genre_history)
    # Diversity seeking behavior can vary across instances of this user.
    self.diversity_seeking = bool(self._rng.binomial(1, self.diversity_prob))

  @classmethod
  def observation_space(cls):
    return spaces.Dict({'user_id': spaces.Discrete(utils.NUM_USERS)})


class UserModel(user.AbstractUserModel):
  """Dynamic Model of a user responsible for generating responses."""

  def __init__(self,
               user_sampler,
               seed=None,
               slate_size=1,
               affinity_update_delta=1.0,
               topic_affinity_update_threshold=3.0):
    """Defines the dynamic user model.

    Args:
      user_sampler: Object of Class UserSampler.
      seed: Random seed for the user model.
      slate_size: Maximum number of items that can be presented to the user.
      affinity_update_delta: Delta for updating user's preference for genres
        whose movies are rated >= topic_affinity_update_threshold.
      topic_affinity_update_threshold: Rating threshold above which user's
        preferences for the genre's is updated.
    """
    super().__init__(
        slate_size=slate_size,
        user_sampler=user_sampler,
        response_model_ctor=Response)
    self._response_model_ctor = Response
    self.affinity_update_delta = affinity_update_delta
    self.topic_affinity_update_threshold = topic_affinity_update_threshold
    self._rng = np.random.RandomState(seed)
    self.choice_model = recsim_choice.MultinomialLogitChoiceModel(
        choice_features={'no_click_mass': -float('Inf')})

  def update_state(self, doc, response):
    """Updates the user state for the current user.

    Updates the topic_affinity vector for the current user based on the
    response to the chosen document.

    Args:
      doc: A Movie object, representing the document chosen by the user.
      response: A Response object, including the user rating of this movie.
    Updates: The user's topic affinity in self._user_state.topic_affinity.
    """
    self._user_state.update_state(doc, response)

  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents.

    Users choose to follow our recommendation with a fixed probability.
    Otherwise, they choose among the remaining members of the slate according
    to the multinomial choice model.

    Note that, by our convention, the first item is the slate is our
    recommendation, while the remaining N-1 items in the slate are the remaining
    universe of possible movie choices.  The recommendation policy should only
    control which movie is shown in the first entry, but not control the other
    items in the slate.

    Args:
      documents: a list of Movie objects in the slate.

    Returns:
      A Response object for the selected document.
    """
    if len(documents) > 1:
      recommendation = documents[0]
      other_docs = documents[1:]
      if self._user_state.check_accept():
        selected_doc = recommendation
      else:
        self.choice_model.score_documents(self._user_state, other_docs)
        selected_index = self.choice_model.choose_item()
        selected_doc = other_docs[selected_index]
    elif not documents:
      raise ValueError('Document list contains no movies')
    else:
      selected_doc = documents[0]

    return self._response_model_ctor(
        rating=self._user_state.rate_document(selected_doc),
        diversity=self._user_state.check_if_new_genre(selected_doc),
        doc_id=selected_doc.doc_id())

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return False

  def reset(self):
    """Resets the current user to their initial state and samples a new user."""
    self._user_state.reset_state()
    self._user_state = self._user_sampler.sample_user()


class MovieLensEnvironment(recsim_environment.Environment):
  """MovieLensEnvironment with some modifications to recsim.Environment."""

  USER_POOLS = {'train': 0, 'eval': 1, 'test': 2}

  def reset(self):
    self._user_model.reset()
    user_obs = self._user_model.create_observation()
    if self._resample_documents:
      self._do_resample_documents()
    if self._resample_documents or not hasattr(self, '_current_documents'):
      # Since _candidate_set.create_observation() is an expensive operation
      # for large candidate sets, this step only needs to be done when the
      # _current_documents attribute hasn't been defined yet or
      # _resample_documents is set to True.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())
    return (user_obs, self._current_documents)

  def step(self, slate):
    """Executes the action, returns next state observation and reward.

    Args:
      slate: An integer array of size slate_size, where each element is an index
        into the set of current_documents presented.

    Returns:
      user_obs: A gym observation representing the user's next state.
      doc_obs: A list of observations of the documents.
      responses: A list of AbstractResponse objects for each item in the slate.
      done: A boolean indicating whether the episode has terminated.
    """
    if len(slate) > self._slate_size:
      raise ValueError(('Received unexpectedly large slate size: '
                        f'expecting {self._slate_size}, got {len(slate)}'))

    # Get the documents associated with the slate
    doc_ids = list(self._current_documents)  # pytype: disable=attribute-error
    mapped_slate = [doc_ids[x] for x in slate]
    documents = self._candidate_set.get_documents(mapped_slate)
    # Simulate the user's response
    response = self._user_model.simulate_response(documents)
    # Unpacks the single document chosen by the user
    [selected_doc] = self._candidate_set.get_documents([response.doc_id])

    # Update the user's state.
    self._user_model.update_state(selected_doc, response)

    # Obtain next user state observation.
    user_obs = self._user_model.create_observation()

    # Check if reaches a terminal state and return.
    done = self._user_model.is_terminal()

    # Optionally, recreate the candidate set to simulate candidate
    # generators for the next query.
    if self._resample_documents:
      self._do_resample_documents()

      # Create observation of candidate set.
      # Compared to the original recsim environment code, _current_documents
      # needs to be done only for each step when resample_docuemnts is set to
      # True.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())

    # Recsim gym expects a list of responses
    return (user_obs, self._current_documents, [response], done)

  def set_active_pool(self, pool_name):
    self._user_model._user_sampler.set_active_pool(self.USER_POOLS[pool_name])  # pylint: disable=protected-access


def average_ratings_reward(responses):
  """Calculates the average rating for the slate from a list of responses."""
  if not responses:
    raise ValueError('Empty response list')
  return np.mean([response.rating for response in responses])


def create_gym_environment(env_config):
  """Returns a RecSimGymEnv with specified environment parameters.

  Args:
    env_config: an `EnvConfig` object.

  Returns:
    A RecSimGymEnv object.
  """

  user_ctor = functools.partial(User, **attr.asdict(env_config.user_config))

  initial_embeddings = utils.load_embeddings(env_config)
  genre_history = utils.load_genre_history(env_config)

  dataset = utils.Dataset(
      env_config.data_dir,
      user_ctor=user_ctor,
      movie_ctor=Movie,
      embeddings=initial_embeddings,
      genre_history=genre_history,
      genre_shift=env_config.genre_shift,
      bias_against_unseen=env_config.bias_against_unseen)

  document_sampler = recsim_samplers.SingletonSampler(dataset.get_movies(),
                                                      Movie)

  user_sampler = recsim_samplers.UserPoolSampler(
      seed=env_config.seeds.user_sampler,
      users=dataset.get_users(),
      user_ctor=user_ctor,
      partitions=env_config.train_eval_test,
      partition_seed=env_config.seeds.train_eval_test)

  user_model = UserModel(
      user_sampler=user_sampler,
      seed=env_config.seeds.user_model,
      slate_size=env_config.slate_size,
  )

  env = MovieLensEnvironment(
      user_model,
      document_sampler,
      num_candidates=document_sampler.size(),
      slate_size=env_config.slate_size,
      resample_documents=False,
  )

  reward_aggregator = average_ratings_reward

  return recsim_gym.RecSimGymEnv(env, reward_aggregator)
