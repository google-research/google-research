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

"""Text augmentation utils for AP parsing.

Implements augmentations to take text and labels and create synthetic AP
sections. These augmentations include changing order, changing delemiters,
interleaving labels etc.
"""
import abc
import copy
import dataclasses
import random
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar

from absl import logging
import numpy as np

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import ap_parsing_utils
from assessment_plan_modeling.ap_parsing import tokenizer_lib


@dataclasses.dataclass(frozen=True)
class _DefaultDelims:
  PROBLEM_TITLE_PREFIX = "\n*. "
  PROBLEM_TITLE_SUFFIX = ":"
  PROBLEM_DESCRIPTION_PREFIX = "\n- "
  ACTION_ITEM_PREFIX = "\n- "


@dataclasses.dataclass(eq=True, order=True)
class ProblemClusterFragment:
  """Container for the text and delimiters of part of a problem cluster.

  For example, a specific action item.
  """
  labeled_char_span: ap_parsing_lib.LabeledCharSpan
  text: str
  prefix_delim: str
  suffix_delim: str

  def __str__(self):
    return self.prefix_delim + self.text + self.suffix_delim


@dataclasses.dataclass(eq=True, order=True)
class ProblemCluster:
  """Container of cluster fragments."""

  fragments: List[ProblemClusterFragment] = dataclasses.field(
      default_factory=list)

  def sort(self):
    self.fragments.sort(key=lambda x: x.labeled_char_span.span_type)

  def __str__(self):
    return "".join(str(fragment) for fragment in self.fragments)


def _to_cluster_fragment(
    labeled_char_span,
    ap_text = "",
    prefix_delim = "",
    suffix_delim = "",
):
  return ProblemClusterFragment(
      labeled_char_span=labeled_char_span,
      prefix_delim=prefix_delim,
      suffix_delim=suffix_delim,
      text=(ap_text[labeled_char_span.start_char:labeled_char_span.end_char]
            if ap_text else ""))


T = TypeVar("T", bound="StructuredAP")


class StructuredAP:
  """Container for structured A&P section.

  Structured as a sequence of problem clusters with some preceding text.
  """

  def __init__(self, problem_clusters,
               prefix_text):
    self.problem_clusters = problem_clusters
    self.prefix_text = prefix_text

  @classmethod
  def build(cls, text,
            labeled_char_spans):
    """Builds structured AP inplace from text and labels.

    Bundles together labels into clusters based on problem titles.

    Args:
      text: str, text of A&P section
      labeled_char_spans: LabeledCharSpans, which are converted to cluster
        fragments.

    Returns:
      An instance of StructuredAP.
    """

    tokens = tokenizer_lib.tokenize(text)

    labeled_char_spans = ap_parsing_utils.normalize_labeled_char_spans_iterable(
        labeled_char_spans, tokens)
    labeled_char_spans.sort(key=lambda x: x.start_char)

    structured_ap = cls(problem_clusters=list(), prefix_text="")
    structured_ap._parse_problem_clusters(labeled_char_spans, text)  # pylint: disable=protected-access

    prefix_text_span = ap_parsing_utils.normalize_labeled_char_span(
        ap_parsing_lib.LabeledCharSpan(
            start_char=0,
            end_char=labeled_char_spans[0].start_char,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE), tokens)
    structured_ap.prefix_text = (
        text[prefix_text_span.start_char:prefix_text_span.end_char]
        if prefix_text_span else "")

    return structured_ap

  def __eq__(self, o):
    if o.__class__ is self.__class__:
      return (self.problem_clusters, self.prefix_text) == (o.problem_clusters,
                                                           o.prefix_text)
    return NotImplemented

  def _parse_problem_clusters(
      self, labeled_char_spans,
      ap_text):
    """Parses labeled character spans to problem clusters.

    Clusters are generated based on problem titles.
    Items before first problem title are not considered as part of any cluster.

    Args:
      labeled_char_spans: Labels as labeled character spans.
      ap_text: Text of the AP section.
    """

    cur_cluster = None
    self.problem_clusters = []
    for labeled_char_span in labeled_char_spans:
      if (labeled_char_span.span_type ==
          ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE):
        self.problem_clusters.append(
            ProblemCluster(fragments=[
                _to_cluster_fragment(
                    labeled_char_span,
                    ap_text,
                    prefix_delim=_DefaultDelims.PROBLEM_TITLE_PREFIX,
                    suffix_delim=_DefaultDelims.PROBLEM_TITLE_SUFFIX)
            ]))
        cur_cluster = self.problem_clusters[-1]

      elif (labeled_char_span.span_type
            == ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION and
            cur_cluster):
        cur_cluster.fragments.append(
            _to_cluster_fragment(
                labeled_char_span,
                ap_text,
                prefix_delim=_DefaultDelims.PROBLEM_DESCRIPTION_PREFIX))
      elif (labeled_char_span.span_type
            == ap_parsing_lib.LabeledSpanType.ACTION_ITEM and cur_cluster):
        cur_cluster.fragments.append(
            _to_cluster_fragment(
                labeled_char_span,
                ap_text,
                prefix_delim=_DefaultDelims.ACTION_ITEM_PREFIX))
      else:
        logging.info("Invalid labeled span type (%s) or cur_cluster",
                     labeled_char_span.span_type)

  def compile(self):
    """Compiles structured ap back to text.

    Returns:
      A tuple of (ap text, labeled character spans).
    """
    char_index = len(self.prefix_text)
    text = self.prefix_text
    labels = []
    for problem_cluster in self.problem_clusters:
      for fragment in problem_cluster.fragments:
        char_index += len(fragment.prefix_delim)
        cur_labeled_char_span = copy.deepcopy(fragment.labeled_char_span)
        cur_labeled_char_span.start_char = char_index
        cur_labeled_char_span.end_char = char_index + len(fragment.text)
        labels.append(cur_labeled_char_span)
        text += str(fragment)
        char_index += len(fragment.text) + len(fragment.suffix_delim)

    return text, labels


class BaseAugmentation(abc.ABC):
  """Base class for augmentations.

  Defines the interface required for augmentations - an init and call functions.
  Defines a from config function for all child classes which builds
  a class instance based on a dictionary of kwargs.
  """

  @abc.abstractmethod
  def __call__(self, ap, seed):
    Ellipsis


class ShuffleClusters(BaseAugmentation):
  """Shuffles problem clusters in random order."""

  def __call__(self, ap, seed):
    random.seed(seed)
    ap.problem_clusters = random.sample(
        population=ap.problem_clusters, k=len(ap.problem_clusters))
    return ap


class ShuffleFragments(BaseAugmentation):
  """Shuffles fragments per cluster in random order."""

  def __call__(self, ap, seed):
    random.seed(seed)
    for cluster in ap.problem_clusters:
      # Ensures first fragment is problem title.
      cluster.sort()
      assert (cluster.fragments[0].labeled_char_span.span_type ==
              ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE), (
                  "First fragment should be the problem title! " +
                  repr(cluster.fragments))

      cluster.fragments = ([cluster.fragments[0]] + random.sample(
          population=cluster.fragments[1:], k=len(cluster.fragments) - 1))
    return ap


def verify_probability_weights(delims, probs,
                               class_name):
  """Verifies probability weights match a single delim each (if specified)."""
  if probs and len(probs) == len(delims):
    return probs
  elif not probs:
    return None
  else:
    logging.warning(
        ("Tried to initialize augmentation (%s) with an incompatible "
         "number of delimiters and probability weights: (%d!=%d), "
         "defaults to no probability weighting"), class_name, len(delims),
        len(probs))
  return None


class ChangeDelimAugmentation(BaseAugmentation):
  """Samples delimiters for a specific fragment type.

  Can operate on multiple fragment types and perform weighted sampling.
  Can sample either prefix or suffix delimiters.
  """

  def __init__(self,
               fragment_types,
               delims,
               probs = None,
               is_prefix = True):
    self._delims = delims
    self._probs = verify_probability_weights(delims, probs, str(self.__class__))
    self._fragment_types = fragment_types
    self._is_prefix = is_prefix

  def __call__(self, ap, seed):
    random.seed(seed)
    for cluster in ap.problem_clusters:
      for fragment in cluster.fragments:
        if fragment.labeled_char_span.span_type in self._fragment_types:
          delim = random.choices(
              population=self._delims, weights=self._probs, k=1)[0]
          if self._is_prefix:
            fragment.prefix_delim = delim
          else:
            fragment.suffix_delim = delim
    return ap


class NumberTitlesAugmentation(BaseAugmentation):
  """Samples number delimiters for a problem titles."""

  def __init__(self,
               delims,
               probs = None,
               first_index = 1):
    self._delims = delims
    self._probs = verify_probability_weights(delims, probs, str(self.__class__))
    self._first_index = first_index

  def __call__(self, ap, seed):
    random.seed(seed)
    for i, cluster in enumerate(ap.problem_clusters):
      for fragment in cluster.fragments:
        if (fragment.labeled_char_span.span_type ==
            ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE):
          fragment.prefix_delim = random.choices(
              population=self._delims, weights=self._probs, k=1)[0]
          fragment.prefix_delim = fragment.prefix_delim.format(
              self._first_index + i)
    return ap


@dataclasses.dataclass
class AugmentationSequence:
  name: str
  augmentation_sequence: List[BaseAugmentation]


@dataclasses.dataclass
class AugmentationConfig:
  """Container for the configurations of augmentations to be applied.

  Defines the augmentations and their sampling procedure. Number of augmentation
  sequences to be applied is determined by either a poisson distribution
  (paramaterized by augmentation_number_poisson_lambda) or deterministically
  by augmentation_number_deterministic. Only one of them should be specified.

  Attributes:
    augmentation_sequences: list of AugmentationSequence. Each contain a
      determenistic sequence of atomic augmentations to be applied.
    augmentation_sample_probabilities: Probability weights for each of the
      sequences in augmentation_sequences. Optional, defaults to equal
      weighting.
    augmentation_number_poisson_lambda: parametrization for sampling number of
      augmentations from a poisson distribution.
    augmentation_number_deterministic: number of augmentation sequences to
      sample.
  """
  augmentation_sequences: List[AugmentationSequence]
  augmentation_sample_probabilities: Optional[Sequence[float]] = None
  augmentation_number_poisson_lambda: Optional[float] = None
  augmentation_number_deterministic: Optional[int] = 0

  def get_n_augmentations(self, rng):
    if self.augmentation_number_poisson_lambda and rng:
      return rng.poisson(lam=self.augmentation_number_poisson_lambda)
    assert self.augmentation_number_deterministic is not None
    return self.augmentation_number_deterministic


def apply_augmentations(structured_ap,
                        augmentation_sequence,
                        seed):
  """Sequentially applies augmentations by order.

  Args:
    structured_ap: AP section as a structured AP object.
    augmentation_sequence: Sequence of objects of type Augmentation.
    seed: int, seed for augmentations.

  Returns:
    An updated StructuredAP after the augmentations were applied.
  """

  for i, augmentation in enumerate(augmentation_sequence.augmentation_sequence):
    structured_ap = augmentation(structured_ap, seed=seed + i)
  return structured_ap
