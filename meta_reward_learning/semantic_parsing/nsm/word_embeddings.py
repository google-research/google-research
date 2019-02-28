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

import json
import numpy as np
# import gensim
from tensorflow import gfile


class EmbeddingModel(object):

  def __init__(self, vocab_file, embedding_file, normalize_embeddings=True):
    with gfile.Open(embedding_file, 'rb') as f:
      self.embedding_mat = np.load(f)
    if normalize_embeddings:
      self.embedding_mat = self.embedding_mat / np.linalg.norm(
          self.embedding_mat, axis=1, keepdims=True)
    with gfile.Open(vocab_file, 'r') as f:
      tks = json.load(f)
    self.vocab = dict(zip(tks, range(len(tks))))

  def __contains__(self, word):
    return word in self.vocab

  def __getitem__(self, word):
    if word in self.vocab:
      index = self.vocab[word]
      return self.embedding_mat[index]
    else:
      raise KeyError
