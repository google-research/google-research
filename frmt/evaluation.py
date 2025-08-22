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

"""Library for computing metrics on the FRMT dataset."""
from __future__ import annotations

import collections
from collections.abc import Collection, MutableMapping, Sequence
import enum
from typing import Optional, Type

import attrs
import bleurt.score as bleurt_lib
from etils import epath
import sacrebleu

BleurtScorer = bleurt_lib.LengthBatchingBleurtScorer


@attrs.frozen(eq=True, kw_only=True)
class TranslationPair:
  """Container class for a source/translation pair.

  Attributes:
    source: The (English) source sentence. May be `None` if this information is
      not tracked.
    translation: The gold or model translation. Unlike `source`, it is required.
  """

  source: Optional[str]
  translation: str


@attrs.define()
class BleurtScorerCache:
  """Container class for a cache of Bleurt models."""

  bleurt_checkpoint_path: epath.Path = attrs.field(converter=epath.Path)
  _cache: MutableMapping[epath.Path, BleurtScorer] = attrs.field(factory=dict)

  def __getitem__(self, bleurt_name):
    bleurt_checkpoint = self.bleurt_checkpoint_path / bleurt_name
    if bleurt_checkpoint in self._cache:
      return self._cache[bleurt_checkpoint]
    else:
      bleurt_scorer = BleurtScorer(str(bleurt_checkpoint))
      self._cache[bleurt_checkpoint] = bleurt_scorer
      return bleurt_scorer


@attrs.define(eq=True, kw_only=True, order=True, slots=True)
class Metrics:
  """Container class for the computed evaluation metrics."""

  bleu: Optional[float] = None
  chrf: Optional[float] = None
  bleurt: Optional[float] = None
  bleurt_d12: Optional[float] = None
  bleurt_d6: Optional[float] = None
  bleurt_d3: Optional[float] = None

  def as_dict(self):
    d = attrs.asdict(self)
    ordered_dict = collections.OrderedDict()
    for metric_name in self.__slots__:  # pytype: disable=attribute-error
      if metric_name not in d:  # E.g. '__weakref__'
        continue
      if d[metric_name] is not None:
        ordered_dict[metric_name] = d[metric_name]
    return ordered_dict


class MetricType(enum.Enum):
  """Supported metric types."""

  UNDEFINED = 0
  BLEU = 1
  CHRF = 2
  BLEURT = 3
  BLEURT_D12 = 4
  BLEURT_D6 = 5
  BLEURT_D3 = 6

  @staticmethod
  def _validate_predictions_and_references(
      predictions,
      references,
  ):
    """Ensures that the predictions and references look okay.

    Args:
      predictions: A sequence of TranslationPair objects containing model
        translations.
      references: A sequence of TranslationPair objects containing gold
        references.

    Raises:
      ValueError: If the list of predictions or references is empty.
      ValueError: If the predictions and references have different lengths.
      ValueError: If the predictions and references are misaligned (requires
        the predictions to be in .tsv format).
    """
    if not predictions:
      raise ValueError('List of predictions is empty.')

    if not references:
      raise ValueError('List of references is empty.')

    if len(predictions) != len(references):
      raise ValueError(
          f'Number of predictions ({len(predictions)}) != '
          f'number of references ({len(references)})'
      )

    for i, (prediction, reference) in enumerate(zip(predictions, references)):
      if (
          prediction.source is not None
          and prediction.source != reference.source
      ):
        raise ValueError(
            f'Predictions and references are misaligned at index {i}.'
            f'\nPrediction: {prediction}\nReference: {reference}'
        )

  @classmethod
  def _compute_bleu(
      cls,
      *,
      predictions,
      references,
      language,
  ):
    """Computes the BLEU score for a file pair."""
    cls._validate_predictions_and_references(predictions, references)

    if language.startswith('zh'):
      tokenizer = 'zh'
    else:
      tokenizer = sacrebleu.DEFAULT_TOKENIZER

    return (
        sacrebleu.corpus_bleu(
            [prediction.translation for prediction in predictions],
            [[reference.translation for reference in references]],
            tokenize=tokenizer,
        ).score
        / 100
    )

  @classmethod
  def _compute_chrf(
      cls,
      *,
      predictions,
      references,
  ):
    """Computes the ChRF score for predictions and references."""
    cls._validate_predictions_and_references(predictions, references)

    return sacrebleu.corpus_chrf(
        [prediction.translation for prediction in predictions],
        [reference.translation for reference in references],
    )

  @classmethod
  def _compute_bleurt(
      cls,
      *,
      predictions,
      references,
      bleurt_scorer,
  ):
    """Computes the BLEURT score for predictions and references."""
    cls._validate_predictions_and_references(predictions, references)

    bleurt_scores = bleurt_scorer.score(
        candidates=[prediction.translation for prediction in predictions],
        references=[reference.translation for reference in references],
    )
    return sum(bleurt_scores) / len(bleurt_scores)

  def compute(
      self,
      *,
      predictions,
      references,
      language = None,
      bleurt_scorer_cache = None,
  ):
    """Computes the metric on predictions and references."""

    if self is MetricType.UNDEFINED:
      raise ValueError('Cannot compute UNDEFINED metric.')
    if self is MetricType.BLEU and language is None:
      raise ValueError(
          '`language` keyword must be non-None when computing BLEU.'
      )
    elif self.name.startswith('BLEURT') and bleurt_scorer_cache is None:
      raise ValueError(
          '`bleurt_scorer_cache` must be non-None when computing BLEURT.'
      )

    if self is MetricType.BLEU:
      return self._compute_bleu(
          predictions=predictions, references=references, language=language
      )
    elif self is MetricType.CHRF:
      return self._compute_chrf(predictions=predictions, references=references)
    elif self is MetricType.BLEURT:
      return self._compute_bleurt(
          predictions=predictions,
          references=references,
          bleurt_scorer=bleurt_scorer_cache['BLEURT-20'],
      )
    elif self is MetricType.BLEURT_D12:
      return self._compute_bleurt(
          predictions=predictions,
          references=references,
          bleurt_scorer=bleurt_scorer_cache['BLEURT-20-D12'],
      )
    elif self is MetricType.BLEURT_D6:
      return self._compute_bleurt(
          predictions=predictions,
          references=references,
          bleurt_scorer=bleurt_scorer_cache['BLEURT-20-D6'],
      )
    elif self is MetricType.BLEURT_D3:
      return self._compute_bleurt(
          predictions=predictions,
          references=references,
          bleurt_scorer=bleurt_scorer_cache['BLEURT-20-D3'],
      )
    else:
      raise ValueError(f'Cannot compute {self} metric.')


def evaluate(
    *,
    predictions,
    references,
    eval_metrics,
    language,
    bleurt_scorer_cache,
):
  """Runs the specified evaluation metrics."""
  metrics = Metrics()
  for eval_metric in eval_metrics:
    value = eval_metric.compute(
        predictions=predictions,
        references=references,
        language=language,
        bleurt_scorer_cache=bleurt_scorer_cache,
    )
    metrics.__setattr__(eval_metric.name.lower(), value)

  return metrics
