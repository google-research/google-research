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

"""Acquires "real" datasets from scikit-learn.

Obtains 20 Newsgroups, Sentiment Labelled Sentences, and MNIST.
"""

import csv
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle as shuffle_arrays

import tensorflow.compat.v1 as tf

FILE_PATH = 'trees/raw_data'
RANDOM_STATE = 109971161161043253 % 8085


class TwentyNewsgroups(object):
  """The 20 Newsgroups text classification dataset.

  Very informative text (headers, footers, and quotes) has been removed to make
  the classification problem more challenging.
  """

  def __init__(self, vectorizer='tf-idf'):
    """Initialize the dataset.

    Args:
      vectorizer: str text vectorization method; values = 'bow' (bag of words),
        'binary-bow' (binary bag of words), 'tf-idf'
    """
    self.vectorizer = vectorizer

  def get(self):
    """Gets the 20 Newsgroups dataset (sparse).

    Returns:
      x_train: scipy.sparse.*matrix
        array of features of training data
      y_train: np.array
        1-D array of class labels of training data
      x_test: scipy.sparse.*matrix
        array of features of test data
      y_test: np.array
          1-D array of class labels of the test data
    """
    # fix doesn't work
    data_path = '{}/{}'.format(FILE_PATH, '20news')
    with tf.gfile.GFile('{}/{}'.format(data_path, 'train.pkl'), 'r') as f:
      train = pickle.load(f)
    with tf.gfile.GFile('{}/{}'.format(data_path, 'test.pkl'), 'r') as f:
      test = pickle.load(f)

    x_train = train.data
    y_train = train.target

    x_test = test.data
    y_test = test.target

    x_train, x_test = vectorize_text(x_train, x_test, method=self.vectorizer)

    return x_train, y_train, x_test, y_test


class SentimentSentences(object):
  """The Sentiment Labelled Sentences text classification dataset."""

  def __init__(self, vectorizer='tf-idf'):
    """Initialize the dataset.

    Args:
      vectorizer: str text vectorization method; values = 'bow' (bag of words),
        'binary-bow' (binary bag of words), 'tf-idf'
    """
    self.vectorizer = vectorizer

  def get(self):
    """Gets the Sentiment Labelled Sentences dataset (sparse).

    Returns:
      x_train: scipy.sparse.*matrix
        array of features of training data
      y_train: np.array
        1-D array of class labels of training data
      x_test: scipy.sparse.*matrix
        array of features of test data
      y_test: np.array
          1-D array of class labels of the test data
    """
    data_path = '{}/{}'.format(FILE_PATH, 'sentiment_sentences')
    with tf.gfile.GFile('{}/{}'.format(data_path, 'amazon_cells_labelled.txt'),
                        'r') as f:
      amazon_df = pd.read_csv(f, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    with tf.gfile.GFile('{}/{}'.format(data_path, 'imdb_labelled.txt'),
                        'r') as f:
      imdb_df = pd.read_csv(f, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    with tf.gfile.GFile('{}/{}'.format(data_path, 'yelp_labelled.txt'),
                        'r') as f:
      yelp_df = pd.read_csv(f, sep='\t', header=None, quoting=csv.QUOTE_NONE)

    df = pd.concat([amazon_df, imdb_df, yelp_df])

    x = df[0].values
    y = df[1].values

    x, y = shuffle_arrays(x, y, random_state=RANDOM_STATE)

    train_test_split = 1000
    x_train = x[train_test_split:]
    y_train = y[train_test_split:]
    x_test = x[:train_test_split]
    y_test = y[:train_test_split]

    x_train, x_test = vectorize_text(x_train, x_test, method=self.vectorizer)

    return x_train, y_train, x_test, y_test


class MNIST(object):
  """The classic MNIST image classification dataset."""

  def get(self):
    """Gets the MNIST dataset (dense).

    Returns:
      x_train: np.array
        array of features of training data
      y_train: np.array
        1-D array of class labels of training data
      x_test: np.array
        array of features of test data
      y_test: np.array
        1-D array of class labels of the test data
    """
    data_path = '{}/{}'.format(FILE_PATH, 'mnist')
    with tf.gfile.GFile('{}/{}'.format(data_path, 'data.pkl'), 'r') as f:
      raw = pickle.load(f)

    x = raw.data / 255.  # standard across experiments
    y = raw.target
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return x_train, y_train, x_test, y_test


def vectorize_text(x_train, x_test, method='tf-idf'):
  """Vectorize text data.

  Args:
    x_train: [string] list of training data
    x_test: [string] list of training data
    method: string text vectorization method; values = 'bow' (bag of words),
      'binary-bow' (binary bag of words), 'tf-idf'hod

  Returns:
    x_train: scipy.sparse.*matrix
      array of features of training data
    x_test: scipy.sparse.*matrix
      array of features of test data

  Raises:
    ValueError: if unknown `method` arg is given
  """
  if method == 'bow':
    vec = CountVectorizer()
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)
  elif method == 'binary-bow':
    vec = CountVectorizer()
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)
    x_train = (x_train > 0).astype(int)
    x_test = (x_test > 0).astype(int)
  elif method == 'tf-idf':
    vec = TfidfVectorizer()
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)
  else:
    raise ValueError('Unknown vectorizer method: {}'.format(method))

  return x_train, x_test
