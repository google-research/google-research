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

"""Library for Beam token_ids reidentification."""

from collections.abc import Callable
import json
from typing import Any, Iterable, List, Optional, Tuple, TypeVar

import apache_beam as beam
import numpy as np
import tensorflow as tf

Example = tf.train.Example

_QUERY_CLIENT_TOKENS = 'client_2_tokens'
_TARGET_CLIENT_TOKENS = 'client_1_tokens'
_USER_ID = 'user_id'
_TOKEN_ID = 'token_id'
_MATCHING_WEIGHT = 'match_weight'
_MISMATCHING_WEIGHT = 'mismatch_weight'

_IntOrFloatOrBytes = TypeVar('_IntOrFloatOrBytes', int, float, bytes)


def _get_feature_and_check_type(
    example, name, feature_type
):
  """Gets the feature with given name from the example and checks the type.

  Args:
    example: The input example.
    name: The name of the feature.
    feature_type: The expected type of the feature.

  Returns:
    The corresponding feature of the example.

  Raises:
    ValueError if the input example does not contain the corresponding feature
    or with wrong type.
  """
  if name not in example.features.feature.keys():
    raise ValueError(f"The input example must have feature with key '{name}'.")
  if example.features.feature[name].WhichOneof('kind') != feature_type:
    raise ValueError(
        f"The feature '{name}' of input example must be {feature_type}."
    )

  if feature_type == 'bytes_list':
    return example.features.feature[name].bytes_list.value
  elif feature_type == 'float_list':
    return example.features.feature[name].float_list.value
  elif feature_type == 'int64_list':
    return example.features.feature[name].int64_list.value
  else:
    raise ValueError(f'Unsupported feature type: {feature_type}')


def _aggregate_output(
    num_matches, num_queries, num_targets
):
  """Aggregates the output numbers into a single string.

  Args:
    num_matches: The number of correct matchings.
    num_queries: The number of sampled queries.
    num_targets: The number of sampled target data points.

  Returns:
    A string corresponds to the output.
  """
  return f'{num_matches},{num_queries},{num_targets}'


def _get_weights(line):
  """Parses the token_id, mismatching weight and matching weight from a string.

  NOTE: we assume the weights from the input file are used for Hamming distance
  while this library requires weights for Hamming similarity. So we negate the
  input weights.

  Args:
    line: A string corresponding to a json object which contains _TOKEN_ID,
      _MATCHING_WEIGHT and _MISMATCHING_WEIGHT.

  Returns:
    A tuple containing token_id (int) and a tuple of floats indicating
    mismatching weight and matching weight respectively.

  Raises:
    ValueError if the input string does not contain required features.
  """
  dic = json.loads(line)
  if _TOKEN_ID not in dic:
    raise ValueError(f"The input '{line}' does not contain '{_TOKEN_ID}'.")
  if _MATCHING_WEIGHT not in dic:
    raise ValueError(
        f"The input '{line}' does not contain '{_MATCHING_WEIGHT}'."
    )
  if _MISMATCHING_WEIGHT not in dic:
    raise ValueError(
        f"The input '{line}' does not contain '{_MISMATCHING_WEIGHT}'."
    )

  return (
      int(dic[_TOKEN_ID]),
      (-float(dic[_MISMATCHING_WEIGHT]), -float(dic[_MATCHING_WEIGHT])),
  )


def _side_input_iterable_as_list(element, iterable):
  """Converts an iterable to a list.

  Args:
    element: None. It is not used but required by the beam.Map.
    iterable: An iterable that is going to convert to a list.

  Returns:
    A list converted from the input iterable.
  """
  del element
  return list(iterable)


def _asymmetric_weighted_hamming_similarity(
    queries,
    targets,
    mismatching_weights,
    matching_weights,
):
  """Computes the asymmetric weighted hamming similarity.

  This function requires queries, targets, mismatching_weights and
  matching_weights to have equal length. When mismatching_weights are all 0s and
  matching_weights are all 1s, the function computes the standard hamming
  (intersection) similarity.

  Args:
    queries: A sequence of token_ids from the query user.
    targets: A sequence of token_ids from the target user.
    mismatching_weights: A list of floats where each corresponds to a weight
      contributed to the final similarity score if the corresponding token_ids
      are mismatched.
    matching_weights: A list of floats where each corresponds to a weight
      contributed to the final similairty score if the corresponding token_ids
      are matched.

  Returns:
    A float corresponding to the similarity between queries and targets.

  Raises:
    ValueError if the lengths of queries, targets, mismatching_weights and
    matching_weights are not consistent.
  """
  n = len(queries)
  if (
      len(targets) != n
      or len(mismatching_weights) != n
      or len(matching_weights) != n
  ):
    raise ValueError(
        'queries, targets, mismatching_weights and matching_weights should have'
        ' the same length.'
    )
  return np.sum(
      np.where(
          # For each entry, return matching weight if query is equal to the
          # target, return mismatching weight otherwise.
          np.array(queries) == np.array(targets),
          matching_weights,
          mismatching_weights,
      )
  )


def _aggregate_weights(
    example,
    extracted_weights,
):
  """Returns corresponding mismatching and matching weights for token_ids.

  Args:
    example: The input example containing token_ids from the query client. In
      particular, the example should contain a int64_list feature with key
      _QUERY_CLIENT_TOKENS which corresponds to the token_ids from the query
      client.
    extracted_weights: The weights corresponding to all possible token_ids. If
      the corresponding weight is missing, the default matching weight is 1 and
      the default mismatching weight is 0.

  Returns:
    A tuple containing input example and the mismatching/matching weights
    corresponding to its token_ids from query client.
  """
  dict_weights = dict()
  for token_id_and_weights in extracted_weights:
    dict_weights[token_id_and_weights[0]] = token_id_and_weights[1]
  mismatching_weights = []
  matching_weights = []

  query_client_tokens = _get_feature_and_check_type(
      example, _QUERY_CLIENT_TOKENS, 'int64_list'
  )
  for token_id in query_client_tokens:
    mismatching_weights.append(
        dict_weights[token_id][0] if token_id in dict_weights else 0
    )
    matching_weights.append(
        dict_weights[token_id][1] if token_id in dict_weights else 1
    )
  return (example, mismatching_weights, matching_weights)


class _AllPairComparisonFn(beam.DoFn):
  """Run all-pair comparisons between queries and targets."""

  def process(
      self,
      example,
      queries_with_weights,
  ):
    """Run all-pair comparisons between queries and targets.

    Args:
      example: A target example containing token ids. In particular, the example
        should contain a byte_list feature with key _USER_ID which corresponds
        to the user id, and a int64_list feature with key _TARGET_CLIENT_TOKENS
        which corresponds to the token ids from the target client.
      queries_with_weights: An iterable containing query examples and
        corresponding mismatching weights and matching weights. The query
        examples should contain a byte_list feature with key _USER_ID which
        corresponds to the user id, and a int64_list feature with key
        _QUERY_CLIENT_TOKENS which corresponds to the token ids from the query
        client.

    Yields:
      (query user id, (similarity, target user id))
    """
    for query_and_weight in queries_with_weights:
      query = query_and_weight[0]
      mismatching_weights = query_and_weight[1]
      matching_weights = query_and_weight[2]

      query_user_id = _get_feature_and_check_type(
          query, _USER_ID, 'bytes_list'
      )[0]
      query_client_tokens = _get_feature_and_check_type(
          query, _QUERY_CLIENT_TOKENS, 'int64_list'
      )
      target_user_id = _get_feature_and_check_type(
          example, _USER_ID, 'bytes_list'
      )[0]
      target_client_tokens = _get_feature_and_check_type(
          example, _TARGET_CLIENT_TOKENS, 'int64_list'
      )
      similarity = _asymmetric_weighted_hamming_similarity(
          query_client_tokens,
          target_client_tokens,
          mismatching_weights,
          matching_weights,
      )
      yield (query_user_id, (similarity, target_user_id))


def user_tokens_nearest_neighbor_search(
    targets,
    sampled_queries,
    extracted_weights,
):
  """For each query user tokens, find the closest target user tokens.

  Args:
    targets: Examples containing target user tokens.
    sampled_queries: Examples containing query user tokens.
    extracted_weights: Tuples including the matching weight and mismatching
      weight of each token.

  Returns:
    The closest target user and the corresponding similarity for each query
    user.
  """
  sampled_query_with_weights = sampled_queries | 'AggregateWeights' >> beam.Map(
      _aggregate_weights, beam.pvalue.AsIter(extracted_weights)
  )

  return (
      targets
      | 'AllPairComparison'
      >> beam.ParDo(
          _AllPairComparisonFn(),
          beam.pvalue.AsIter(sampled_query_with_weights),
      )
      | 'Max' >> beam.CombinePerKey(max)
  )


def reidentification(
    input_file_pattern,
    output_dir,
    query_size,
    weight_file_pattern,
):
  """Returns a pipeline that measures reidentification risk.

  Args:
    input_file_pattern: The file pattern of tfrecordio file(s) to read.
    output_dir: The directory to write the output to.
    query_size: The number of sampled queries.
    weight_file_pattern: The file containing the mismatching weights and
      matching weights of token ids.

  Returns:
     A pipeline measures the reidentification risk.
  """

  def pipeline(root):
    if weight_file_pattern:
      lines = root | 'ReadWeights' >> beam.io.ReadFromText(weight_file_pattern)
      extracted_weights = lines | 'ExtractWeights' >> beam.Map(_get_weights)
    else:
      extracted_weights = root | 'CreateEmptyWeights' >> beam.Create([])

    # A pcollection of examples where each example contains user id, query
    # client tokens and target client tokens.
    user_and_tokens = root | 'Read' >> beam.io.tfrecordio.ReadFromTFRecord(
        input_file_pattern, coder=beam.coders.ProtoCoder(Example)
    )

    # A subset of user_and_tokens as queries
    sampled_queries = (
        user_and_tokens
        | 'SampleQueries' >> beam.combiners.Sample.FixedSizeGlobally(query_size)
        | 'FlattenSampledQueries' >> beam.FlatMap(lambda elements: elements)
    )

    user_matches = user_tokens_nearest_neighbor_search(
        user_and_tokens, sampled_queries, extracted_weights
    )

    num_correct_matches = (
        user_matches
        | 'FilterCorrectMatching'
        >> beam.Filter(lambda match: match[0] == match[1][1])
        | 'CountCorrectMatches' >> beam.combiners.Count.Globally()
    )

    num_targets = (
        user_and_tokens | 'CountAllTargets' >> beam.combiners.Count.Globally()
    )

    num_queries = (
        user_matches | 'CountAllQueries' >> beam.combiners.Count.Globally()
    )

    output_result = num_correct_matches | 'AggregateOutput' >> beam.Map(
        _aggregate_output,
        beam.pvalue.AsSingleton(num_queries),
        beam.pvalue.AsSingleton(num_targets),
    )

    _ = output_result | 'WriteResult' >> beam.io.WriteToText(
        output_dir + '/result.txt'
    )

  return pipeline
