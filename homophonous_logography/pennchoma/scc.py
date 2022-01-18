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

"""Computes sample correlation coefficients following Penn & Choma.

Penn, Gerald and Travis Choma. (2006). "Quantitative methods for classifying
writing systems." Proceedings of the North American Chapter of the Association
for Computational Linguistics, pages 117--120.
"""

import collections
import math
import unicodedata


class Document(object):
  """Holds a single "document" of text.
  """

  def __init__(self, text, prepro=None):
    """Produces a document from text.

    Args:
      text: UTF8-encoded Unicode text
      prepro: Optional preprocessor function to apply instead of _clean().
        The preprocessor must take text as input and return a sequence
        as output.
    """
    self._text = text
    self._clean(prepro)
    self._counts = collections.defaultdict(int)
    self._size = 0
    for c in self._text:
      self._counts[c] += 1
      self._size += 1

  def _clean(self, prepro=None):
    """Cleans the input text possibly using the preprocessor.

    Args:
      prepro: a preprocessor or None.
    """
    newtext = []
    for c in self._text:
      if unicodedata.category(c)[0] == "P":
        continue
      newtext.append(c)
    self._text = "".join(newtext).lower()
    if prepro:
      self._text = prepro(self._text)
    else:
      self._text = "".join(self._text.split())

  @property
  def size(self):
    return self._size

  @property
  def counts(self):
    return self._counts

  @property
  def characters(self):
    return self._counts.keys()


class Corpus(object):
  """A corpus of Documents.
  """

  def __init__(self, documents):
    """Initialize a corpus of Documents.

    Args:
       documents: a list of Documents
    """
    self._documents = documents
    self._characters = set()
    self._size = 0
    self._ndocs = 0
    for document in self._documents:
      for character in document.characters:
        self._characters.add(character)
      self._size += document.size
      self._ndocs += 1
    self._means = collections.defaultdict(float)
    self._compute_means()
    self._std_dev = {}
    self._cov = {}

  @property
  def size(self):
    return self._size

  @property
  def ndocs(self):
    return self._ndocs

  @property
  def characters(self):
    return self._characters

  @property
  def nchars(self):
    return len(self._characters)

  def _compute_means(self):
    """Computes means of character counts over documents.

    Returns:
       Mean of character counts over documents.
    """
    for c in self._characters:
      for d in self._documents:
        self._means[c] += d.counts[c]
      self._means[c] /= self._ndocs

  def std_dev(self, c):
    """Computes standard deviation for a character, memoizing the result.

    Args:
       c: a character.
    Returns:
       Standard deviation for c.
    """
    if c not in self._std_dev:
      tot = 0
      for d in self._documents:
        tot += (d.counts[c] - self._means[c]) ** 2
      self._std_dev[c] = math.sqrt(1.0 / (self._ndocs - 1) * tot)
    return self._std_dev[c]

  def cov(self, c1, c2):
    """Computes covariance of c1, c2, memoizing the result.

    Args:
       c1: a character.
       c2: a character.
    Returns:
       cov(c1, c2).
    """
    if (c1, c2) not in self._cov:
      tot = 0
      for d in self._documents:
        tot += ((d.counts[c1] - self._means[c1]) *
                (d.counts[c2] - self._means[c2]))
      self._cov[c1, c2] = 1.0 / (self._ndocs - 1) * tot
    return self._cov[c1, c2]

  def corr(self, c1, c2):
    """Computes correlation of c1, c2.

    Args:
       c1: a character.
       c2: a character.
    Returns:
       cor(c1, c2), or 0 if one of the standard deviations is 0.
    """
    try:
      return (self.cov(c1, c2) /
              (self.std_dev(c1) * self.std_dev(c2)))
    except ZeroDivisionError:
      return 0.0
