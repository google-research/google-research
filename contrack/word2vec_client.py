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

"""Client for looking up word2vec Embeddings."""

from typing import List

from gensim.models import KeyedVectors
import numpy as np


class Word2VecClient(object):
  """API for looking up word2vec vectors for words."""

  def __init__(self, path):
    self.path = path

    self.keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=True)

    test_vec = self.keyed_vectors['test']
    assert test_vec is not None

    self.vec_len = len(test_vec)
    self.empty_vec = np.array([0.0] * self.vec_len)

  def lookup(self, words):
    """Look up a list of words and return list of wordvecs."""
    res = []

    for word in words:
      try:
        vec = self.keyed_vectors[word]
      except KeyError:
        vec = self.empty_vec
      res.append(vec)

    return res
