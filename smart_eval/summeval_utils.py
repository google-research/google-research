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

"""Functions and classes used for the SummEval experiments."""

import collections
import dataclasses
import enum
import json
from typing import Any, Callable, Dict, List

from nltk.tokenize import sent_tokenize
import numpy as np
import scipy.stats

from smart_eval import scorer


@dataclasses.dataclass
class SummevalExample:
  instance_key: str
  system_key: str
  source: List[str]
  references: List[List[str]]
  candidate: List[str]
  scores: Dict[str, float]


class CorrType(enum.Enum):
  # System-level correlation (average-then-correlate).
  SYSTEM = 1
  # Example-level correlation (correlate-then-average).
  EXAMPLE = 2


def open_file(filename, mode='r'):
  return open(filename, mode)  # pylint: disable=unreachable


def transform_bartscore_data(
    bartscore_data):
  """Transform BARTScore data into a list of SummevalExample data objects.

  BARTScore data can be downloaded here: https://github.com/neulab/BARTScore

  Args:
    bartscore_data: data dictionary from the BARTScore paper.

  Returns:
    BARTScore data transformed into a list of SummevalExample objects.
  """
  examples = []
  for inst_key in bartscore_data:
    inst = bartscore_data[inst_key]
    src = inst['src']
    if 'ref_summs' not in inst:
      refs = [inst['ref_summ']]
    else:
      refs = inst['ref_summs']

    src = sent_tokenize(src)
    for i in range(len(refs)):
      assert isinstance(refs[i], str)
      refs[i] = sent_tokenize(refs[i])

    for sys_key in inst['sys_summs']:
      system = inst['sys_summs'][sys_key]

      can = system['sys_summ']
      can = sent_tokenize(can)

      scores = system['scores']

      examples.append(
          SummevalExample(
              instance_key=inst_key,
              system_key=sys_key,
              source=src,
              references=refs,
              candidate=can,
              scores=scores))

  return examples


def get_data(bartscore_file, summeval_file):
  """Gets data from BARTScore and SummEval papers and combine their results."""
  with open_file(bartscore_file, 'rb') as f:
    summeval_data = json.load(f)

  summeval_examples = transform_bartscore_data(summeval_data)

  with open_file(summeval_file, 'r') as f:
    for line in f:
      inst = json.loads(line.strip())
      inst_key = inst['id']
      sys_key = inst['model_id']
      # Search for the example in summeval_examples.
      example = [
          summeval_example for summeval_example in summeval_examples
          if summeval_example.instance_key == inst_key and
          summeval_example.system_key == sys_key
      ]
      if len(example) != 1:
        # There is one system not used in BARTScore paper. Discard this.
        continue
      example = example[0]

      for score_key in inst['metric_scores_11']:
        if score_key == 'rouge' or 'bert_score' in score_key:
          # We already have ROUGE/BertScore scores from BARTScore data.
          continue
        if score_key == 'supert':
          example.scores[score_key] = inst['metric_scores_11'][score_key][0]
        else:
          example.scores[score_key] = inst['metric_scores_11'][score_key]

  return summeval_examples


def calculate_smart_score(
    examples,
    matcher_name,
    matcher = None,
    pairwise_scores = None):
  """Calculate all kinds of SMART with several default matchers.

  Pre-calculated pairwise scores, one src-specific and one ref-specific, can
    also be passed to speed up calculation. This will be transformed into
    score matrices.

  Args:
    examples: A list of SummevalExample data object.
    matcher_name: The name of the matching function used/to be used.
    matcher: Matching function.
    pairwise_scores: Pre-calculated pairwise scores to speed up calculation.

  Returns:
    The 'examples' list with SMART scores.
  """
  print(f'calculating scores using matcher {matcher_name}')

  if matcher:
    smart_scorer = scorer.SmartScorer(matching_fn=matcher)
  elif pairwise_scores:
    src_score_id = 0
    ref_score_id = 0
    smart_scorer = scorer.SmartScorer()
  else:
    raise ValueError('Either matcher or pairwise_scores should be provided.')

  for i, example in enumerate(examples):
    if i % 100 == 0:
      print(f'processing example {i}')

    src = example.source
    can = example.candidate
    refs = example.references

    # Get score matrix if pairwise scores are provided.
    if pairwise_scores:
      src_pairwise_scores = []
      for _ in src:
        for _ in can:
          src_pairwise_scores.append(
              float(pairwise_scores['src'][src_score_id]))
          src_score_id += 1
      src_score_matrix = np.reshape(src_pairwise_scores, (len(src), len(can)))

      ref_score_matrices = []
      for idx, ref in enumerate(refs):
        ref_pairwise_scores = []
        for _ in ref:
          for _ in can:
            ref_pairwise_scores.append(
                float(pairwise_scores['ref'][ref_score_id]))
            ref_score_id += 1
        ref_score_matrices.append(
            np.reshape(ref_pairwise_scores, (len(ref), len(can))))

    # Get SMART using either input src or output ref as reference.
    if pairwise_scores:
      smart_input = smart_scorer.smart_score_precomputed(
          src_score_matrix)
    else:
      smart_input = smart_scorer.smart_score(src, can)
    smart_outputs = []
    for idx, ref in enumerate(refs):
      if pairwise_scores:
        smart_output = smart_scorer.smart_score_precomputed(
            ref_score_matrices[idx])
      else:
        smart_output = smart_scorer.smart_score(ref, can)
      smart_outputs.append(smart_output)

    # SMART using src as reference.
    for smart_type in smart_input:
      score_name = f'src_{smart_type}_fmeasure_{matcher_name}'
      example.scores[score_name] = smart_input[smart_type]['fmeasure']

    # SMART using ref as reference.
    for smart_type in smart_outputs[0]:
      scores = [
          smart_output[smart_type]['fmeasure']
          for smart_output in smart_outputs
      ]
      score = np.max(scores)
      score_name = f'ref_{smart_type}_fmeasure_{matcher_name}'
      example.scores[score_name] = score

    # max(src SMART, ref SMART)
    for smart_type in smart_outputs[0]:
      ref_scores = [
          smart_output[smart_type]['fmeasure']
          for smart_output in smart_outputs
      ]
      ref_score = np.max(ref_scores)
      score = max(ref_score, smart_input[smart_type]['fmeasure'])
      score_name = f'max_sent_{smart_type}_fmeasure_{matcher_name}'
      example.scores[score_name] = score

    return examples


def get_correlation(examples,
                    score_names,
                    comparisons,
                    correlator=scipy.stats.kendalltau,
                    corr_type = CorrType.SYSTEM):
  """Calculate correlation coefficients.

  Args:
    examples: The list of SummevalExample examples.
    score_names: The list of score names we want to get correlation values from.
    comparisons: The list of quality dimensions we are interested.
    correlator: The correlation calculation function from scipy.stats.
    corr_type: The correlation type, which can either be (a) system-level
      (CorrType.SYSTEM), where we first take the average score for each system
      and then calculate correlation. (b) example-level (CorrType.EXAMPLE),
      where we first calculate the correlation for each system and then take the
      average.

  Returns:
    A dictionary containing correlation coefficients for each example of score
      name and comparison.
  """
  comparison_scores = {}
  for comparison in comparisons:
    comparison_scores[comparison] = collections.defaultdict(list)
    for example in examples:
      if corr_type == CorrType.SYSTEM:
        key = example.system_key
      elif corr_type == CorrType.EXAMPLE:
        key = example.instance_key
      else:
        raise NotImplementedError(
            f'Correlation type {corr_type} not implemented.')
      comparison_scores[comparison][key].append(example.scores[comparison])

  # If score_names is not provided, calculate all available metric scores.
  if score_names is None:
    score_names = sorted(list(examples[0].scores.keys()))

  ret_dict = collections.defaultdict(dict)
  for score_name in score_names:
    if score_name in comparisons:
      continue

    metric_scores = collections.defaultdict(list)
    for example in examples:
      if corr_type == CorrType.SYSTEM:
        key = example.system_key
      elif corr_type == CorrType.EXAMPLE:
        key = example.instance_key
      else:
        raise NotImplementedError(
            f'Correlation type {corr_type} not implemented.')
      metric_scores[key].append(example.scores[score_name])

    for comparison in comparisons:
      if corr_type == CorrType.SYSTEM:
        comp_avg_scores = [
            np.mean(comparison_scores[comparison][sys_key])
            for sys_key in comparison_scores[comparison]
        ]
        avg_scores = [
            np.mean(metric_scores[sys_key]) for sys_key in metric_scores
        ]
        corr = correlator(comp_avg_scores, avg_scores)[0]
      elif corr_type == CorrType.EXAMPLE:
        corr_list = []
        for src_key in metric_scores:
          if len(set(metric_scores[src_key])) == 1 or len(
              set(comparison_scores[comparison][src_key])) == 1:
            continue
          c = correlator(comparison_scores[comparison][src_key],
                         metric_scores[src_key])[0]
          corr_list.append(c)
        corr = np.mean(corr_list)
      else:
        raise NotImplementedError(
            f'Correlation type {corr_type} not implemented.')

      ret_dict[score_name][comparison] = corr

  return ret_dict
