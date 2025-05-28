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

"""Matching functions."""

import abc
from typing import Union, List

from bleurt import score as bleurt_score
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
import sacrebleu

from rouge_score import rouge_scorer


class MatchingFunction(metaclass=abc.ABCMeta):
  """Interface for matching function APIs."""

  @abc.abstractmethod
  def __call__(
      self,
      reference,
      candidate,
  ):
    raise NotImplementedError()


class RougeMatchingFunction(MatchingFunction):
  """ROUGE matching function."""

  def __init__(self, rouge_type):
    self._rouge = rouge_scorer.RougeScorer(rouge_types=[rouge_type])
    self.rouge_type = rouge_type

  def __call__(
      self,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          self._rouge.score(r, c)[self.rouge_type].fmeasure
          for r, c in zip(reference, candidate)
      ]
    else:
      return self._rouge.score(reference, candidate)[self.rouge_type].fmeasure


class MeteorMatchingFunction(MatchingFunction):
  """METEOR matching function."""

  def __call__(
      self,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          meteor_score.single_meteor_score(word_tokenize(r), word_tokenize(c))
          for r, c in zip(reference, candidate)
      ]
    else:
      return meteor_score.single_meteor_score(
          word_tokenize(reference), word_tokenize(candidate))


class ChrfMatchingFunction(MatchingFunction):
  """CHRF matching function."""

  def __call__(
      self,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          sacrebleu.sentence_chrf(c, r) for r, c in zip(reference, candidate)]
    else:
      return sacrebleu.sentence_chrf(candidate, reference)


class BleuMatchingFunction(MatchingFunction):
  """BLEU matching function."""

  def __call__(
      self,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          sacrebleu.sentence_bleu(c, r) for r, c in zip(reference, candidate)]
    else:
      return sacrebleu.sentence_bleu(candidate, reference)


class BleurtMatchingFunction(MatchingFunction):
  """BLEURT matching function."""

  def __init__(self, bleurt_ckpt):
    self._bleurt = bleurt_score.BleurtScorer(bleurt_ckpt)

  def __call__(
      self,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return self._bleurt.score(references=reference, candidates=candidate)
    else:
      return self._bleurt.score(references=[reference], candidates=[candidate])
