# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Abstract classes for identity annotation."""

import abc
from typing import Any, Tuple, Union

import pandas as pd


class Lexicon(abc.ABC):
  """Lexicon abstract class."""

  @abc.abstractmethod
  def match(
      self, tokens_df, spans_df
  ):
    raise NotImplementedError('Abstract method not implemented.')

  # Needed for calling this class within beam.
  def __reduce_ex__(self, protocol):
    return (self.__class__, ())


class Tokenizer(abc.ABC):
  """Tokenizer abstract class."""

  @abc.abstractmethod
  def tokenize(self, text):
    raise NotImplementedError('Abstract method not implemented.')

  # Needed for calling this class within beam.
  def __reduce_ex__(self, protocol):
    return (self.__class__, ())


class EntityAnnotator(abc.ABC):
  """EntityAnnotator abstract class."""

  @abc.abstractmethod
  def annotate(
      self,
      text,
      tokens_df,
      token_span_matches_df,
  ):
    raise NotImplementedError('Abstract method not implemented.')

  # Needed for calling this class within beam.
  def __reduce_ex__(self, protocol):
    return (self.__class__, ())
