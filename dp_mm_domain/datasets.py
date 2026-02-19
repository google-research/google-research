# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""This file contains example code for loading and preprocessing datasets."""

import re

import nltk
import pandas as pd
import tqdm


def clean_text(text):
  """Returns cleaned and tokenized version of input string text.

  This function is renamed from reddit_preprocess from
  https://github.com/heyyjudes/differentially-private-set-union/blob/master/utils.py.

  Args:
    text: The input text to clean.
  """
  text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  text = re.sub(r'\n', ' ', text, flags=re.MULTILINE)
  text = re.sub(r'\[removed\]', ' ', text, flags=re.MULTILINE)
  text = re.sub(r'\[deleted\]', ' ', text, flags=re.MULTILINE)
  sentences = nltk.tokenize.sent_tokenize(text)
  sentences = [' '.join(nltk.tokenize.word_tokenize(s)) for s in sentences]
  return ' '.join(sentences)


def preprocess_reddit(data_path):
  """Loads and preprocess the Reddit dataset.

  This dataset can be downloaded from
  https://github.com/heyyjudes/differentially-private-set-union/blob/master/sample_data.csv.

  Args:
    data_path: The path to the Reddit dataset

  Returns:
    A list of lists of tokens, where each sublist is the set of tokens used by
    a single user across their posts.
  """
  with open(data_path, 'r') as f:
    df_reddit = pd.read_csv(f, index_col=0).dropna()

  reddit_data = []
  # note that the reddit dataset is already cleaned
  for _, group in tqdm(df_reddit.groupby('author'), position=0, leave=True):
    posts = group['clean_text']
    posts = [p.split(' ') for p in posts]
    words = []
    for p in posts:
      for tokens in p:
        words.append(tokens)
    all_grams = words
    reddit_data.append(all_grams)

  return reddit_data


def preprocess_amazon_games(data_path):
  """Loads and preprocess the Amazon Games dataset.

  This dataset can be downloaded from
  https://nijianmo.github.io/amazon/index.html.


  Args:
    data_path: The path to the Amazon Games dataset.

  Returns:
    A list of lists of games, where each sublist is a single user's reviewed
    games.
  """
  with open(data_path, 'r') as f:
    df_amz_games = pd.read_csv(
        f, header=None, names=['name', 'user', 'something', 'something else']
    )

  amz_data = []
  for _, group in tqdm(df_amz_games.groupby('user'), position=0, leave=True):
    ids = list(set(group['name'].tolist()))
    amz_data.append(ids)

  return amz_data


def preprocess_movie_reviews(data_path):
  """Loads and preprocess the Movie Reviews dataset.

  This dataset can be downloaded from
  https://grouplens.org/datasets/movielens/25m/.

  Args:
    data_path: The path to the Movie Reviews dataset.

  Returns:
    A list of lists of movies, where each sublist is a single user's reviewed
    movies.
  """
  with open(data_path, 'r') as f:
    df_imdb = pd.read_csv(f, index_col=0).dropna()
  # make the review index a column
  df_imdb = df_imdb.reset_index()
  df_imdb['clean_text'] = df_imdb['review'].apply(clean_text)
  df_imdb.drop(['review', 'sentiment'], axis=1, inplace=True)
  df_imdb['author'] = range(df_imdb.shape[0])
  df_imdb['author'] = df_imdb['author'].apply(lambda x: 'a' + str(x))

  imdb_data = []
  for _, group in tqdm(df_imdb.groupby('author'), position=0, leave=True):
    posts = group['clean_text']
    posts = [p.split(' ') for p in posts]
    words = []
    for p in posts:
      for tokens in p:
        words.append(tokens)
    all_grams = words
    imdb_data.append(all_grams)

  return imdb_data


def preprocess_steam_games(data_path):
  """Loads and preprocess the Steam Games dataset.

  This dataset can be downloaded from
  https://www.kaggle.com/datasets/tamber/steam-video-games/data.

  Args:
    data_path: The path to the Steam Games dataset.

  Returns:
    A list of lists of games, where each sublist is a single user's purchased
    games.
  """
  with open(data_path, 'rb') as f:
    df_games = pd.read_csv(
        f,
        header=None,
        names=['id', 'name', 'interaction', 'something', 'something else'],
    )

  games_data = []
  for _, group in tqdm(df_games.groupby('id'), position=0, leave=True):
    games = list(set(group['name'].tolist()))
    games_data.append(games)

  return games_data


def preprocess_amazon_magazine(data_path):
  """Loads and preprocess the Amazon Magazine dataset.

  This dataset can be downloaded from
  https://nijianmo.github.io/amazon/index.html.

  Args:
    data_path: The path to the Amazon Magazine dataset.

  Returns:
    A list of lists of magazines, where each sublist is a single user's reviewed
    magazines.
  """
  with open(data_path, 'r') as f:
    df_mag = pd.read_csv(
        f, header=None, names=['name', 'user', 'something', 'something else']
    )

  mag_data = []
  for _, group in tqdm(df_mag.groupby('user'), position=0, leave=True):
    ids = list(set(group['name'].tolist()))
    mag_data.append(ids)

  return mag_data


def preprocess_amazon_pantry(data_path):
  """Loads and preprocess the Amazon Pantry dataset.

  This dataset can be downloaded from
  https://nijianmo.github.io/amazon/index.html.

  Args:
    data_path: The path to the Amazon Pantry dataset.

  Returns:
    A list of lists of pantry items, where each sublist is a single user's
    reviewed pantry items.
  """
  with open(data_path, 'r') as f:
    df_pantry = pd.read_csv(
        f, header=None, names=['name', 'user', 'something', 'something else']
    )

  pantry_data = []
  for _, group in tqdm(df_pantry.groupby('user'), position=0, leave=True):
    ids = list(set(group['name'].tolist()))
    pantry_data.append(ids)

  return pantry_data
