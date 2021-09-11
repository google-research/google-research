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

"""Generate simulator data and save to file.

Uses a randomized exploration policy to collect simulated data, or use a
provided agent (loaded from file).
"""

import os
from typing import Sequence, Optional

from absl import app
from absl import flags
from absl import logging
import file_util
import numpy as np
import pandas as pd
import scipy

from proxy_rewards import agent
from proxy_rewards import movie_lens_simulator
from proxy_rewards import utils

DEFAULT_DATA_DIRECTORY = None

flags.DEFINE_string(
    name='data_directory',
    default=DEFAULT_DATA_DIRECTORY,
    help='Location to find MovieLens files {movies, ratings}.csv')
flags.DEFINE_string(
    name='embedding_file',
    default='movielens_factorization.json',
    help='Name of embedding file (should end in .json or .pkl)')
flags.DEFINE_string(
    name='genre_history_file',
    default='user_genre_history.pkl',
    help='Name of genre history file (should end in .json or .pkl)')
flags.DEFINE_string(
    name='agent_directory',
    default=None,
    help='Directory where agent .pkl files are stored')
flags.DEFINE_string(
    name='agent_file_str',
    default=None,
    help='String identifier for learned agent')
flags.DEFINE_float(
    name='shift',
    default=0.,
    help=(
        'Shift genre preferences: Increase the rating (prior to clipping) by '
        'this amount for each genre in [Comedy, Drama, Childrens] and decrease '
        'by this amount for each genre in [Sci-Fi, Fantasy, War]'))
flags.DEFINE_string(
    name='output_directory', default=None, help='Location to save csv output')
flags.DEFINE_integer(name='global_seed', default=0, help='Random Seed')
flags.DEFINE_integer(
    name='n_samples', default=10000, help='Number of samples to generate')
flags.DEFINE_float(
    name='accept_prob',
    default=0.5,
    help='Probability that users accept our recommendation',
    lower_bound=0.,
    upper_bound=1.)
flags.DEFINE_float(
    name='diversity_prob',
    default=0.5,
    help='Probability that a user is diversity-seeking',
    lower_bound=0.,
    upper_bound=1.)
flags.DEFINE_enum(
    name='slate_type',
    default='all',
    enum_values=['all', 'top1genre', 'top2genre', 'top20', 'test'],
    help='Movies to include')
flags.DEFINE_enum(
    name='user_pool',
    default='train',
    enum_values=['train', 'eval', 'test'],
    help='Pool of users to sample from, in [train, eval, test]')

flags.DEFINE_float(
    name='intercept',
    default=-5.,
    help='Intercept of the ground truth model for the long-term outcome.')
flags.DEFINE_float(
    name='rating_coef',
    default=1.,
    help='Coefficient for rating in the ground truth model.')
flags.DEFINE_float(
    name='div_seek_coef',
    default=3.,
    help='Coefficient for diversity_seeker in the ground truth model.')
flags.DEFINE_float(
    name='diversity_coef',
    default=-0.5,
    help='Coefficient for diversity in the ground truth model.')

FLAGS = flags.FLAGS


def generate_data(env_config,
                  slate_type,
                  n_samples,
                  seed,
                  intercept,
                  rating_coef,
                  div_seek_coef,
                  diversity_coef,
                  recommender_agent = None,
                  shift=0.,
                  user_pool='train'):
  """Generate synthetic data.

  Args:
    env_config: Environment Configuration.
    slate_type: Type of slate, should be one of ['all', 'top1genre',
      'top2genre', 'test'].
    n_samples: Number of samples to generate.
    seed: Random seed.
    intercept: Intercept of the ground truth logReg model P(Y).
    rating_coef: Coefficient for rating in the ground truth model.
    div_seek_coef: Coefficient for diversitySeeker in the ground truth model.
    diversity_coef: Coefficient for diversity in the ground truth model.
    recommender_agent: If None, will make recommendations uniformly at random.
    shift: Increase the rating (prior to clipping) by this amount for each genre
      in [Comedy, Drama, Childrens] and decrease by this amount for each genre
      in [Sci-Fi, Fantasy, War]'))
    user_pool: Train/Eval/Test pool of users.

  Returns:
    Pandas DataFrame containing the samples.
  """
  if not env_config.embeddings_path.endswith(('.json', '.pkl')):
    raise ValueError('Embedding path should end in .json or .pkl')
  if not env_config.genre_history_path.endswith(('.json', '.pkl')):
    raise ValueError('Genre history path should end in .json or .pkl')
  if (env_config.user_config.accept_prob < 0 or
      env_config.user_config.accept_prob > 1):
    raise ValueError('Accept probability should be in [0, 1]')

  if slate_type == 'all':
    top_movie_slate = np.arange(utils.NUM_MOVIES)
  elif slate_type == 'top1genre':
    # Most popular movie of each genre, with de-duplication
    top_movie_slate = utils.find_top_movies_per_genre(
        movies_per_genre=1, data_path=env_config.data_dir)
  elif slate_type == 'top2genre':
    # Top 2 most popular movies of each genre, with de-duplication
    top_movie_slate = utils.find_top_movies_per_genre(
        movies_per_genre=2, data_path=env_config.data_dir)
  elif slate_type == 'top20':
    # Top 20 most popular movies
    top_movie_slate = utils.find_top_movies_overall(
        num_movies=20, data_path=env_config.data_dir)
  elif slate_type == 'test':
    top_movie_slate = np.arange(5)
  else:
    raise ValueError('Slate type not recognized')

  env_config.slate_size = len(top_movie_slate)

  # Comedy and drama are the most commonly watched (and therefore least
  # "diverse") genres
  pos_shift = ['Comedy', 'Drama', 'Children\'s']
  # These happen to be more "diverse" genres, and in our top1 from each
  # genre tends to hit Star Wars movies in particular.
  neg_shift = ['Sci-Fi', 'Fantasy', 'War']

  genre_shift = [0.] * len(utils.GENRES)
  for genre in pos_shift:
    genre_shift[utils.GENRE_MAP[genre]] = shift
  for genre in neg_shift:
    genre_shift[utils.GENRE_MAP[genre]] = -shift
  env_config.genre_shift = genre_shift

  # TODO(moberst): All randomization should be through rngs, but currently the
  # multinomial choice model (imported from RecSim) does not accept an rng,
  # so this is a work-around to ensure consistent user choices.
  np.random.seed(seed)
  rng = np.random.default_rng(seed)
  if recommender_agent is None:
    recommender_agent = agent.RandomAgent(movies=top_movie_slate)

  ml_env = movie_lens_simulator.create_gym_environment(env_config)
  ml_env.environment.set_active_pool(user_pool)
  res = []

  for i in range(n_samples):
    if i % 100 == 0:
      logging.info('Iteration: %d / %d', i, n_samples)

    # Generate data, one row per user interaction
    row = {}

    initial_obs = ml_env.reset()
    user_id = initial_obs['user']['user_id']

    row['user'] = user_id
    row['diversity_seeker'] = (
        ml_env.environment.user_model._user_state.diversity_seeking)  # pylint: disable=protected-access

    slate = recommender_agent.make_recommendation(user_id)
    obs, _, _, _ = ml_env.step(slate)
    [response] = obs['response']
    row['rec'] = slate[0]
    row['watched'] = response['doc_id']
    row['rating'] = response['rating']
    row['diversity'] = response['diversity']

    res.append(row)

  df = pd.DataFrame.from_dict(res).sort_values('user')

  logits = intercept + df['rating'] * rating_coef + df[
      'diversity_seeker'] * div_seek_coef + df['diversity'] * diversity_coef

  # Calculate probability of the long-term reward (p_ltr) and sample it.
  df['p_ltr'] = scipy.special.expit(logits)
  df['ltr'] = rng.binomial(1, df['p_ltr'])

  return df


def write_csv_output(dataframe, filename,
                     directory):
  """Write dataframe to CSV.

  Args:
    dataframe: pandas DataFrame
    filename: name of the file (should end in ".csv")
    directory: directory to write to
  """
  if not filename.endswith('.csv'):
    raise ValueError('Filename does not end in .csv')
  file_util.makedirs(directory)

  dataframe.to_csv(
      file_util.open(os.path.join(directory, filename), 'w'), index=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  seeds = movie_lens_simulator.Seeds(
      user_model=FLAGS.global_seed,
      user_sampler=FLAGS.global_seed,
      train_eval_test=FLAGS.global_seed)

  user_config = movie_lens_simulator.UserConfig(
      accept_prob=FLAGS.accept_prob, diversity_prob=FLAGS.diversity_prob)

  # TODO(moberst): genre_shift should be passed in here, not calculated
  # within the generate_data function
  env_config = movie_lens_simulator.EnvConfig(
      data_dir=FLAGS.data_directory,
      embeddings_path=f'{FLAGS.data_directory}/{FLAGS.embedding_file}',
      genre_history_path=f'{FLAGS.data_directory}/{FLAGS.genre_history_file}',
      seeds=seeds,
      user_config=user_config,
      bias_against_unseen=0.)

  if FLAGS.agent_file_str:
    agent_file = f'{FLAGS.agent_directory}/{FLAGS.agent_file_str}.pkl'
    recommender_agent = agent.load_agent(agent_file)
    agent_str = FLAGS.agent_file_str
  else:
    # If None, this will be populated with a random agent in generate_data
    recommender_agent = None
    agent_str = 'random-agent'

  df = generate_data(
      env_config=env_config,
      slate_type=FLAGS.slate_type,
      n_samples=FLAGS.n_samples,
      seed=FLAGS.global_seed,
      intercept=FLAGS.intercept,
      rating_coef=FLAGS.rating_coef,
      div_seek_coef=FLAGS.div_seek_coef,
      diversity_coef=FLAGS.diversity_coef,
      recommender_agent=recommender_agent,
      shift=FLAGS.shift,
      user_pool=FLAGS.user_pool)
  write_csv_output(
      dataframe=df,
      filename=(f'simulation_results'
                f'_{FLAGS.n_samples}'
                f'_{FLAGS.slate_type}'
                f'_{FLAGS.accept_prob}'
                f'_{FLAGS.user_pool}'
                f'_{agent_str}'
                f'{"_shift" if FLAGS.shift != 0. else ""}'
                '.csv'),
      directory=FLAGS.output_directory)


if __name__ == '__main__':
  app.run(main)
