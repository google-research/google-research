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

"""Script for running single experiment and saving its plots."""

import numpy as np
import pandas as pd

from dp_topk import experiment
from dp_topk.differential_privacy import NeighborType

input_methods = [
    experiment.TopKEstimationMethod.JOINT,
    experiment.TopKEstimationMethod.CDP_PEEL,
    experiment.TopKEstimationMethod.PNF_PEEL
]
k_range = np.arange(5, 205, 10)
eps = 1
delta = 1e-6
num_trials = 50

books = pd.read_csv("books.csv", usecols=["ratings_count"])
counts = np.array(books["ratings_count"][1:]).astype(int)
results = experiment.compare(counts, input_methods, -1, k_range, eps, delta,
                             num_trials, NeighborType.ADD_REMOVE)
experiment.plot("books", input_methods, results, k_range, True, True)

# Instructions for the other five datasets in the paper appear below.

# foods: https://jmcauley.ucsd.edu/data/amazon/
# Save the file as foods.csv and use
# column_names = ['user', 'item', 'rating', 'timestamp']
# food = pd.read_csv("foods.csv", names = column_names, skipinitialspace=False)
# counts = np.asarray(food['item'].value_counts())

# games: https://www.kaggle.com/tamber/steam-video-games/version/3
# Save the file as games.csv and use
# column_names = ['user', 'game', 'behavior', 'hours_or_bool', '?']
# games = pd.read_csv("games.csv", names = column_names, skipinitialspace=False)
# counts = np.asarray(
#    games.loc[games['behavior'] == 'purchase']['game'].value_counts())

# movies: https://grouplens.org/datasets/movielens/25m/
# Save the file as movies.csv and use
# movies = pd.read_csv("movies.csv", usecols=["movieId"])
# counts = movies.value_counts()

# news: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
# Save the file as news.csv and use
# news = pd.read_csv("news.csv", usecols=[" shares"])
# counts = np.array(news[" shares"]).astype(int)

# tweets: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JBXKFD/F4FULO&version=2.2
# tweets = tweets = pd.read_csv("tweets.csv", usecols=["number_of_likes"])
# counts = np.array(tweets["number_of_likes"]).astype(int)
