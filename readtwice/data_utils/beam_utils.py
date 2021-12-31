# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Apache Beam utilities."""
import collections
import copy
import math
from typing import (Any, Iterable, Iterator, List, Mapping, MutableSequence,
                    Optional, Sequence, Text, Tuple, Union)

import apache_beam as beam
import attr
import sortedcontainers
import tensorflow.compat.v1 as tf

from readtwice.data_utils import data_utils
from readtwice.data_utils import tokenization

PCollection = beam.pvalue.PCollection


def get_feature(feature_name,
                example):
  """Gets Tensorflow feature by name.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The Tensorflow feature with the given feature name in the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example.
  """
  if feature_name in example.features.feature:
    return example.features.feature[feature_name]
  else:
    raise ValueError('Feature name {} is not in the example {}'.format(
        feature_name, example))


def get_repeated_values(
    feature_name,
    example):
  """Gets the underlying repeated values of a feature by feature name.

  The return type depends on which oneof `kind` is populated for the feature.
  Whichever one is populated is returned.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The underlying repeated values for the given feature name in the example.
    Modifying these repeated values will modify the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example or
      none of the oneof `kind` fields is populated.
  """
  feature = get_feature(feature_name, example)
  which_oneof = feature.WhichOneof('kind')
  if which_oneof is None:
    raise ValueError(
        'No field populated in oneof `kind` for feature name {} in example '
        '{}'.format(feature_name, example))
  return getattr(feature, which_oneof).value


def get_int64_list(feature_name,
                   example):
  """Gets Tensorflow int64 list in the feature by feature name.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The Tensorflow int64 list in the feature with the given feature name in the
    example.
  """
  return get_feature(feature_name, example).int64_list


def get_bytes_list(feature_name,
                   example):
  """Gets Tensorflow bytes list in the feature by feature name.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The Tensorflow bytes list in the feature with the given feature name in the
    example.
  """
  return get_feature(feature_name, example).bytes_list


@attr.s
class SummaryStatistics(object):
  """Summary statistics for a set of numbers."""

  count = attr.ib()  # type: int
  min = attr.ib()  # type: float
  mean = attr.ib()  # type: float
  max = attr.ib()  # type: float

  # TODO(jainslie): Implement stddev calculation also.
  # stddev = attr.ib()  # type: float

  # A list of (probability, value) tuples, where `probability` is in the
  # interval [0, 1] and `value` is the corresponding quantile. Quantiles may be
  # approximate when calculated in Beam.
  quantiles = attr.ib()  # type: List[Tuple[float, float]]

  @classmethod
  def from_dict(cls, dictionary):
    """Constructs a `SummaryStatistics` object from a Python dictionary.

    If `stats` is an instance of `SummaryStatistics`,
    `SummaryStatistics.from_dict(attr.asdict(stats))` should equal `stats`.

    Args:
      dictionary: Dictionary resulting from `stats.to_dict_for_json()` or
        `attr.asdict(stats)` for some `SummaryStatistics` instance `stats`.

    Returns:
      The `SummaryStatistics` instance.
    """
    if isinstance(dictionary['quantiles'], collections.Mapping):
      # Handles dictionaries made via `to_dict_for_json`.
      quantiles = [
          (float(key), value) for key, value in dictionary['quantiles'].items()
      ]
      quantiles.sort(key=lambda x: x[0])
    else:
      # Handles dictionaries made via `attr.asdict`.
      quantiles = [tuple(pair) for pair in dictionary['quantiles']]

    return cls(
        count=dictionary['count'],
        min=dictionary['min'],
        mean=dictionary['mean'],
        max=dictionary['max'],
        quantiles=quantiles,
    )

  def to_dict_for_json(self):
    """Returns a dictionary with the statistics in nicer formatting for JSON."""
    result = collections.OrderedDict()
    result['count'] = self.count
    result['min'] = self.min
    result['mean'] = self.mean
    result['max'] = self.max
    result['quantiles'] = collections.OrderedDict(self.quantiles)
    return result


class ReadFilesToTokenizedDocuments(beam.PTransform):
  """PTransform for reading text files into tokenized documents."""

  def __init__(self,
               vocab_path = None,
               do_lower_case = True,
               spm_model_path = None,
               generate_document_ids = False):
    """Initialization.

    Args:
      vocab_path: Path to the BERT vocabulary file to use with the BERT
        tokenizer. Leave as None or set to empty string if using
        `spm_model_path` instead.
      do_lower_case: Whether to lowercase all text for BERT tokenization
        (default True). Must match assumption in `vocab_path`. Ignored if using
        `spm_model_path` instead of `vocab_path`.
      spm_model_path: Instead of using the options above, this argument may
        specify a path to a SentencePiece model file to use with the ALBERT
        tokenizer. Leave as None or set to empty string if using `vocab_path`
        instead.
      generate_document_ids: If True, then every output `TokenizedBertDocument`
        will have a random (not necessary unique) `document_id`.
    """
    if bool(vocab_path) == bool(spm_model_path):
      raise ValueError(
          'Exactly 1 of `vocab_path` or `spm_model_path` must be specified, '
          'not both.')
    self.vocab_path = vocab_path
    self.do_lower_case = do_lower_case
    self.spm_model_path = spm_model_path
    self.generate_document_ids = generate_document_ids

  def expand(
      self, file_paths
  ):
    """Converts text file paths to tokenized documents.

    Args:
      file_paths: PCollection of text file paths to read from.

    Returns:
      PCollection of tokenized documents.
    """

    def file_path_to_documents(
        file_path):
      lines = data_utils.read_text_file_lines(file_path)
      for document in data_utils.parse_bert_pretraining_text(
          lines, self.generate_document_ids):
        yield document

    return (file_paths
            | 'ReadDocuments' >> beam.FlatMap(file_path_to_documents)
            | 'TokenizeDocuments' >> beam.ParDo(
                _TokenizeDocumentFn(self.vocab_path, self.do_lower_case,
                                    self.spm_model_path)))


class _TokenizeDocumentFn(beam.DoFn):
  """DoFn for tokenizing a `data_utils.BertDocument`."""

  def __init__(self,
               vocab_path = None,
               do_lower_case = True,
               spm_model_path = None):
    self.vocab_path = vocab_path
    self.do_lower_case = do_lower_case
    self.spm_model_path = spm_model_path

  def setup(self):
    self.tokenizer = tokenization.FullTokenizer(
        self.spm_model_path, self.vocab_path, do_lower_case=self.do_lower_case)

  def process(
      self, element
  ):
    yield data_utils.tokenize_document_for_bert(element, self.tokenizer)


class CalculateStatistics(object):
  """PTransforms for generating statistics over a PCollection of numbers."""

  class Globally(beam.PTransform):
    """Global application that returns a PCollection with a single summary."""

    def expand(self,
               numbers):
      count = numbers | 'Count' >> beam.combiners.Count.Globally()
      minimum = (
          numbers | 'Min' >> beam.combiners.Top.Smallest(1)
          | beam.Map(lambda x: x[0] if x else float('inf')))
      mean = numbers | 'Mean' >> beam.combiners.Mean.Globally()
      maximum = (
          numbers | 'Max' >> beam.combiners.Top.Largest(1)
          | beam.Map(lambda x: x[0] if x else float('-inf')))

      # We ultimately want quantiles for every 0.1% but we ask for quantiles for
      # every 0.05% in case this makes the approximation more precise.
      def pick_quantiles(
          all_2001_quantiles):
        """Picks informative quantiles out of the 2001 raw quantiles."""

        # If `numbers` is empty then `all_2001_quantiles` will be empty, so we
        # just return an empty list.
        if not all_2001_quantiles:
          return []

        probabilities = ([.001, .002, .003, .004, .005, .01, .02, .03, .04] +
                         [(x + 1) / 20 for x in range(19)] +
                         [.96, .97, .98, .99, .995, .996, .997, .998, .999])
        result = []
        for prob in probabilities:
          result.append((prob, all_2001_quantiles[int(round(prob * 2000))]))
        return result

      quantiles = (
          numbers |
          'Quantiles' >> beam.ApproximateQuantiles.Globally(num_quantiles=2001)
          | 'PickQuantiles' >> beam.Map(pick_quantiles))

      return (  #
          singletons_to_dict(
              count=count,
              min=minimum,
              mean=mean,
              max=maximum,
              quantiles=quantiles)
          | beam.Map(SummaryStatistics.from_dict))

  # TODO(jainslie): Implement this version.
  class PerKey(beam.PTransform):
    """Unimplemented."""

    def __init__(self):
      raise NotImplementedError


class BaseExamplePacker(object):
  """Base class for TensorFlow Example packing logic.

  See the `PackExamples` PTransform for more details.
  """

  def add_example(self, example):
    """Adds a new example to pack and returns any completed examples.

    Args:
      example: A TensorFlow Example to pack.

    Returns:
      A (possibly empty) list of examples for which packing is finished.
    """
    raise NotImplementedError

  def flush_examples(self):
    """Returns any remaining (partially packed) examples and clears state."""
    raise NotImplementedError


class PackExamples(beam.PTransform):
  """PTransform for packing TensorFlow Examples together to fill padding.

  This is used for examples that contain variable-length features that
  ultimately need to be padded to a maximum length when forming a batch.
  "Packing" means to concatenate multiple short examples into one longer
  example (usually maintaining features to mark separation "breakpoints"
  between the original examples). This reduces the amount of padding necessary.

  Packing is done sequentially within each Beam bundle, so it may be helpful to
  shuffle ahead of time to diversify each bundle's composition.
  """

  def __init__(self, example_packer):
    """Initialization.

    Args:
      example_packer: Instance of `BaseExamplePacker` to use for packing logic.
    """
    self.example_packer = example_packer

  def expand(
      self,
      examples):
    if not examples.windowing.is_default():
      raise ValueError('`PackExamples` only works for global windowing.')
    return (examples
            |
            'PackExamples' >> beam.ParDo(_PackExamplesFn(self.example_packer)))


class _PackExamplesFn(beam.DoFn):
  """DoFn for packing TensorFlow Examples."""

  def __init__(self, example_packer):
    self.example_packer = example_packer

  def process(self, element):
    for example in self.example_packer.add_example(element):
      yield example

  def finish_bundle(self):
    for example in self.example_packer.flush_examples():
      yield beam.transforms.window.GlobalWindows.windowed_value(example)


class PriorityExamplePacker(BaseExamplePacker):
  """Example packer that maintains partially packed examples by priority.

  Partially packed examples are kept in a cache up to `max_cache_len` and
  prioritized from longest to shortest based on the `priority_feature`.
  When a new example is added, it's packed with the first partial example that
  can fit it.  Once an example is packed to a certain `min_packing_fraction` in
  any of its features, it's padded and returned.

  All examples added to the packer must define the same set of features.
  """

  def __init__(self,
               priority_feature,
               max_lengths,
               breakpoint_features,
               cumulative_features,
               min_packing_fraction = 0.95,
               max_cache_len = 32,
               packing_status_feature = 'packing_status',
               padding_token_ids = None):
    """Initialization.

    Args:
      priority_feature: Name of the feature to prioritize packing for. Partially
        packed examples will be prioritized based on the highest length for this
        feature.
      max_lengths: Dictionary of maximum lengths for every feature in each
        example. All features must be specified here. An error will be thrown if
        an example has any feature that already exceeds its max length.
      breakpoint_features: Dictionary mapping names of features for which we
        want to track breakpoints to the names of the new breakpoint features
        that will be added during packing. Breakpoint features will be int64
        features containing only 0 and 1 values, where 1 marks the end of an
        example and 0 is used everywhere else.
      cumulative_features: Set of int64_list features to increase ids for
        cumulatively when packing.
      min_packing_fraction: Minimum fraction any feature of a partially packed
        example needs to attain (relative to its max length) before the example
        is considered fully packed.
      max_cache_len: Maximum number of partially packed examples to keep in the
        cache. When the cache is full, the highest priority example is emitted
        to make room for new examples as needed. Must be positive.
      packing_status_feature: Feature name to use for recording the packing
        status before outputting each example. This will be a BytesList feature
        with a single value.
      padding_token_ids: Dictionary of token IDs used for padding
        for integer valued features. If None (default) then 0 is assumed to be
        a padding value. However, if it's not None then the dictionary must
        contain values for ALL of the integer-value features.
    """
    self._priority_feature = priority_feature
    self._breakpoint_features = dict(breakpoint_features)
    self._cumulative_features = set(cumulative_features)
    self._min_packing_fraction = min_packing_fraction
    self._max_cache_len = max_cache_len
    self._packing_status_feature = packing_status_feature

    self._max_lengths = dict(max_lengths)
    for name, breakpoint_name in breakpoint_features.items():
      self._max_lengths[breakpoint_name] = self._max_lengths[name]

    self._min_lengths = {}
    for name, max_len in self._max_lengths.items():
      self._min_lengths[name] = int(math.ceil(max_len * min_packing_fraction))

    self._is_initialized = False
    self._padding_token_ids = padding_token_ids

  def add_example(self, example):
    self._ensure_initialized()

    example = copy.deepcopy(example)
    self._add_breakpoint_features(example)
    self._validate_num_features(example)

    lengths = self._get_lengths(example)
    self._validate_max_len_is_not_exceeded(lengths)

    if self._is_min_len_satisfied(lengths):
      self._finalize_example(example, b'untouched')
      return [example]

    for key, partial_example in list(self._cache.items()):
      combined_lengths = self._get_lengths(partial_example) + lengths
      if self._is_max_len_exceeded(combined_lengths):
        continue
      del self._cache[key]
      self._extend_example(partial_example, example)
      assert combined_lengths == self._get_lengths(partial_example)
      if self._is_min_len_satisfied(combined_lengths):
        self._finalize_example(partial_example)
        return [partial_example]
      else:
        self._insert_into_cache(partial_example)
        return []

    self._insert_into_cache(example)
    if len(self._cache) > self._max_cache_len:
      first_key = next(iter(self._cache))
      evicted_example = self._cache[first_key]
      del self._cache[first_key]
      self._finalize_example(evicted_example, b'evicted')
      return [evicted_example]
    return []

  def flush_examples(self):
    self._ensure_initialized()

    result = list(self._cache.values())
    for example in result:
      self._finalize_example(example, b'flushed')
    self._clear_state()
    return result

  def _ensure_initialized(self):
    if not self._is_initialized:
      self._clear_state()
      self._is_initialized = True

  def _clear_state(self):
    self._cache = sortedcontainers.SortedDict()
    self._cache_insertions = 0

  def _add_breakpoint_features(self, example):
    for name, breakpoint_name in self._breakpoint_features.items():
      target_len = len(get_repeated_values(name, example))
      breakpoint_values = [0] * target_len
      breakpoint_values[-1] = 1
      example.features.feature[breakpoint_name].int64_list.value[:] = (
          breakpoint_values)

  def _validate_num_features(self, example):
    num_features = len(example.features.feature)
    expected_num_features = len(self._max_lengths)
    if num_features != expected_num_features:
      raise ValueError(
          'Example contains {} features (after adding breakpoint features) but '
          'expected {} features total.'.format(num_features,
                                               expected_num_features))

  def _get_lengths(self, example):
    result = collections.Counter()
    for name in example.features.feature:
      result[name] = len(get_repeated_values(name, example))
    return result

  def _is_max_len_exceeded(self, lengths):
    """Returns true if any feature in `lengths` exceeds its max length."""
    for name, length in lengths.items():
      if length > self._max_lengths[name]:
        return True
    return False

  def _validate_max_len_is_not_exceeded(self, lengths):
    """Asserts that all features in `lengths` do not exceed their max length."""
    for name, length in lengths.items():
      if length > self._max_lengths[name]:
        raise ValueError(
            'Feature %s has length %d which exceeds its max length %d' %
            (name, length, self._max_lengths[name]))

  def _is_min_len_satisfied(self, lengths):
    """Returns true if any feature in `lengths` satisfies its min length."""
    for name, length in lengths.items():
      if length >= self._min_lengths[name]:
        return True
    return False

  def _extend_example(self, partial_example,
                      extension):
    """Extends a partially packed example with an `extension` example.

    Args:
      partial_example: Example which will be extended (mutated in place).
      extension: Example to extend `partial_example` with (not mutated).
    """
    for name in partial_example.features.feature:
      repeated_values = get_repeated_values(name, partial_example)
      extension_values = list(get_repeated_values(name, extension))
      if name in self._cumulative_features:
        adder = repeated_values[-1] + 1 if repeated_values else 0
        extension_values = [x + adder for x in extension_values]
      repeated_values.extend(extension_values)

  def _insert_into_cache(self, example):
    remaining_len = (
        self._max_lengths[self._priority_feature] -
        len(get_repeated_values(self._priority_feature, example)))

    # Use `_cache_insertions` as the tiebreaker for the priority so that no 2
    # priority keys are equal.
    priority_key = (remaining_len, self._cache_insertions)
    self._cache[priority_key] = example
    self._cache_insertions += 1

  def _get_padding_id(self, feature):
    if self._padding_token_ids is None:
      return 0
    else:
      return self._padding_token_ids[feature]

  def _finalize_example(self,
                        example,
                        packing_status = b'packed'):
    """Finalizes the example (modifying in place) so it's ready to be emitted.

    This includes the following steps:
    1. Padding all features to their maximum lengths.
    2. Adding a feature for the packing status.

    Args:
      example: Example to finalize.
      packing_status: Status to record for this example.
    """
    for name, feature in example.features.feature.items():
      which_oneof = feature.WhichOneof('kind')
      pad_value = '' if which_oneof == 'bytes_list' else self._get_padding_id(
          name)
      repeated_values = getattr(feature, which_oneof).value
      remaining_len = self._max_lengths[name] - len(repeated_values)
      repeated_values.extend([pad_value] * remaining_len)
    self._validate_max_len_is_not_exceeded(self._get_lengths(example))

    example.features.feature[
        self._packing_status_feature].bytes_list.value[:] = [packing_status]


def singletons_to_list(
    singletons,
    beam_label = 'SingletonsToList'):
  """Combines a list of singleton PCollections into a single PCollection.

  Args:
    singletons: A list of singleton PCollection arguments to combine.
    beam_label: A string label for this Beam transform. A unique label may need
      to be supplied instead of the default if this function is called multiple
      times in the same scope.

  Returns:
    A singleton PCollection containing a list of the combined results.

  Raises:
    ValueError: If `singletons` is empty.
  """
  singletons = list(singletons)
  if not singletons:
    raise ValueError('`singletons` must not be empty.')
  first_arg = singletons[0]

  def make_list(unused_element, *args):
    return list(args)

  wrapped_singletons = [beam.pvalue.AsSingleton(x) for x in singletons]

  # `first_arg` is just used to have something to apply `beam.Map` to, but
  # it corresponds to `unused_element` in `make_list()`.
  return first_arg | beam_label >> beam.Map(make_list, *wrapped_singletons)


def singletons_to_dict(
    beam_label = 'SingletonsToDict',
    **kwargs):
  """Combines multiple singleton PCollections into a dictionary PCollection.

  Args:
    beam_label: A string label for this Beam transform. A unique label may need
      to be supplied instead of the default if this function is called multiple
      times in the same scope.
    **kwargs: Singleton PCollection arguments to combine. The argument names
      will become the keys in the resulting dictionary singleton PCollection.

  Returns:
    A singleton PCollection containing a dictionary of the combined results.

  Raises:
    ValueError: If `kwargs` is empty.
  """
  if not kwargs:
    raise ValueError('`kwargs` must not be empty.')
  first_arg = kwargs[next(iter(kwargs))]
  key_ordering = list(kwargs.keys())

  def make_dict(unused_element, key_ordering, **kwargs):
    result = collections.OrderedDict()
    for key in key_ordering:
      result[key] = kwargs[key]
    return result

  singletons = {k: beam.pvalue.AsSingleton(v) for k, v in kwargs.items()}

  # `first_arg` is just used to have something to apply `beam.Map` to, but
  # it corresponds to `unused_element` in `make_dict()`.
  return (first_arg
          | beam_label >> beam.Map(make_dict, key_ordering, **singletons))


def combine_dictionaries(
    dictionaries):
  """Utility for merging entries from all dictionaries in `dictionaries`.

  If there are multiple dictionaries with the same key, the entry from the
  latest dictionary is used for that key.

  Args:
    dictionaries: Iterable of dictionaries to combine entries of.

  Returns:
    A dictionary with the combined results.
  """
  result = {}
  for dictionary in dictionaries:
    result.update(dictionary)
  return result
