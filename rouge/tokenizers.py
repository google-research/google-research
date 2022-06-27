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

"""Library containing Tokenizer definitions.

The RougeScorer class can be instantiated with the tokenizers defined here. New
tokenizers can be defined by creating a subclass of the Tokenizer abstract class
and overriding the tokenize() method.
"""
import abc
from nltk.stem import porter
from rouge import tokenize


class Tokenizer(abc.ABC):
  """Abstract base class for a tokenizer.

  Subclasses of Tokenizer must implement the tokenize() method.
  """

  @abc.abstractmethod
  def tokenize(self, text):
    raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
  """Default tokenizer which tokenizes on whitespace."""

  def __init__(self, use_stemmer=False):
    """Constructor for DefaultTokenizer.

    Args:
      use_stemmer: boolean, indicating whether Porter stemmer should be used to
      strip word suffixes to improve matching.
    """
    self._stemmer = porter.PorterStemmer() if use_stemmer else None

  def tokenize(self, text):
    return tokenize.tokenize(text, self._stemmer)
