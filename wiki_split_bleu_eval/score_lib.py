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

"""Evaluation library for split+rephrase sentence decompostion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging
from nltk.translate import bleu_score as nltk_bleu_score
import numpy as np


def MacroAvgSentBLEU(ref_str_lists, hyp_strs):
  """Compute multi-reference BLEU (macro-averaged over sentences) using NLTK.

  Contents must already be split into tokens.

  Args:
    ref_str_lists: list of reference lists
    hyp_strs: list of hypothesis strings

  Returns:
    (float) BLEU score
  """
  assert len(hyp_strs) == len(ref_str_lists)
  scores = []
  sentence_bleu_fn = MaybeEmulateMultiBleu(nltk_bleu_score.sentence_bleu)
  for references, hypothesis in zip(ref_str_lists, hyp_strs):
    scores.append(sentence_bleu_fn(references, hypothesis))
  return np.mean(scores)


def GetTokenLists(ref_str_lists, hyp_strs, tokenize_fn=lambda s: s.split()):
  """Split tokenized strings into lists of tokens.

  Args:
    ref_str_lists: list(list(str)) of multi-reference items
    hyp_strs: list(str) hypotheses
    tokenize_fn: a function that splits a string into a list of tokens.

  Returns:
    references: tokenized references as a list(list(list(str))).
    hypotheses: tokenized hypotheses as a list(list(str)).
  """

  ref_str_lists_tokenized = [
      list(map(tokenize_fn, ref_list)) for ref_list in ref_str_lists
  ]
  hyp_strs_tokenized = list(map(tokenize_fn, hyp_strs))

  return ref_str_lists_tokenized, hyp_strs_tokenized


def ReadParcels(line_iterator,
                parcel_sep='<::::>',
                reduce_to_single_analysis=False):
  r"""Parse one or more decompositions from each line in line_iterator.

  Each input item is split by tab and then by parcel_sep.

  Args:
    line_iterator: iterable over strings of the format
      First parcel . <::::> Second parcel.\tOther <:::> option .
    parcel_sep: the string symbol between two simple sentences (parcels).
    reduce_to_single_analysis: if True, assume each line has a single
      decomposition and reduce the return type accordingly (see below).

  Returns:
    parceled_instances: a list(list(list(str))). The example string above would
      yield the list item
        [["First parcel .", "Second parcel."], ["Other", "option ."]]
      When reduce_to_single_analysis=True, one dimension is stripped out such
      that the return value is a list(list(str)).
  """

  def SplitParcels(analysis):
    """Split one analysis string into list of non-empty parcels."""
    parcels = [parcel.strip() for parcel in analysis.split(parcel_sep)]
    return [p for p in parcels if p]

  # Parse input lines to multi-analysis parcel lists.
  parceled_instances = []
  for line in line_iterator:
    analyses = line.strip().split('\t')
    assert analyses
    parceled_instances.append([SplitParcels(analysis) for analysis in analyses])

  if reduce_to_single_analysis:
    assert all([len(analyses) == 1 for analyses in parceled_instances])
    parceled_instances = [analyses[0] for analyses in parceled_instances]

  return parceled_instances


def MaybeEmulateMultiBleu(nltk_target_fn):
  """Includes emulate_multibleu argument into nltk_target_fn if necessary.

  The signature of the NLTK functions corpus_bleu and sentence_bleu depend on
  the NLTK version. This function works around version differences encountered
  in the public and internal environments.

  Args:
    nltk_target_fn: a function that computes BLEU given arguments gold and
      predicted.

  Returns:
    a function that takes arguments gold and predicted, in the format
    expected by NLTK's corpus_bleu and sentence_bleu functions.
  """
  fn = nltk_target_fn

  return fn


def ComputeMetrics(pred, gold):
  """Calculates metrics and returns scores as a dict.

  Computes the following metrics:
    - corpus-level BLEU
      - multi-reference, the standard way.
    - macro-averaged
      - sentence-level BLEU

  Args:
    pred: hypotheses as a list of strings
    gold: references as list of list of strings

  Returns:
    dict(string -> float) metrics
  """
  results = {}
  tok_gold, tok_pred = GetTokenLists(gold, pred)

  # Legacy tag.
  field = 'decomp'

  # Sentence-level BLEU.
  macro_avg_sent_bleu = MacroAvgSentBLEU(tok_gold, tok_pred) * 100.0
  results['bleu.macro_avg_sent.' + field] = macro_avg_sent_bleu

  # Corpus-level BLEU.
  corpus_bleu_fn = MaybeEmulateMultiBleu(nltk_bleu_score.corpus_bleu)
  corpus_bleu = corpus_bleu_fn(tok_gold, tok_pred) * 100.0
  results['bleu.corpus.' + field] = corpus_bleu
  logging.info('BLEU %s: %05.02f', field, corpus_bleu)

  return results


def NumTokens(s):
  return len(s.split())


def LengthStatistics(data):
  """Updates results with simple length-based statistics.

  parcels / input_sentence - (S/C metric in paper) macro averaged num
  tokens per parcel (Tokens/S in paper)

  Example of an item in data: ['parcel1 here .', 'parcel 2 here .']

  Args:
    data: list of parcel lists.

  Returns:
    dictionary of results
  """

  results = {}

  # Average number of parcels per decomposed instance.
  parcel_counts = [len(instance) for instance in data]
  results['lengths.simple_per_complex'] = np.mean(parcel_counts)

  # Token counts.
  token_counts = []
  for instance in data:
    token_counts.append([NumTokens(parcel) for parcel in instance])

  # Macro averaged number of tokens per parcel.
  results['lengths.tokens_per_simple'] = np.mean(
      [np.mean(counts) for counts in token_counts])

  # Micro averaged number of tokens per parcel.
  total_tokens = np.sum(list(itertools.chain.from_iterable(token_counts)))
  total_parcels = np.sum(parcel_counts)
  results['lengths.tokens_per_simple_micro'] = total_tokens / total_parcels

  return results


def GoldLengthStatistics(data):
  """Updates results with simple length-based statistics over multi-ref data.

  Example of an item in data: [['parcel1 here .', 'parcel 2 here .'], [alt..]]

  Args:
    data: list of list of parcel lists.

  Returns:
    dictionary of results
  """

  results = {}

  # Macro-average number of parcels per decomposed instance.
  parcel_counts = []
  for instance in data:
    parcel_counts.append([len(analysis) for analysis in instance])

  results['ref_lengths.simple_per_complex'] = np.mean(
      [np.mean(counts) for counts in parcel_counts])

  # Token counts.
  token_counts = []
  for instance in data:
    instance_counts = []
    for analysis in instance:
      instance_counts.append([NumTokens(parcel) for parcel in analysis])
    token_counts.append(instance_counts)

  # Macro averaged number of tokens per parcel.
  token_means_per_analysis = []
  for instance in token_counts:
    token_means_per_analysis.append(
        [np.mean(analysis_counts) for analysis_counts in instance])

  results['ref_lengths.tokens_per_simple'] = np.mean(
      [np.mean(counts) for counts in token_means_per_analysis])

  return results


def PerformEval(gold, pred, debug=False):
  """Runs evaluation of predictions relative to references.

  Args:
    gold: gold references; each item is a list of one or more analyses, and each
      analysis is a list of parcel strings.
    pred: system predictions as a list of parcel lists.
    debug: debug mode prints out sample of data.

  Returns:
    dictionary of results
  """

  logging.info('Gold labels: read %d rows', len(gold))
  logging.info('Predicted labels: read %d rows', len(pred))

  assert len(gold) == len(pred), (
      'Got unequal number of gold items ({}) and predictions ({})'.format(
          len(gold), len(pred)))

  if debug:
    print(gold[:2])
    print(pred[:2])

  results = {}

  # Calculate some stats on predictions.
  results.update(LengthStatistics(pred))
  results.update(GoldLengthStatistics(gold))

  # Collapse each analysis from a list of parcels into a single string,
  # since that is what we calculate metrics over.
  gold_decompositions = []
  for gold_instance in gold:
    gold_decompositions.append(
        [' '.join(parcel_list) for parcel_list in gold_instance])

  pred_decompositions = [' '.join(parcel_list) for parcel_list in pred]
  if debug:
    print(gold_decompositions[:2])
    print(pred_decompositions[:2])

  # Number of unique references per input.
  counts = [
      len(set(instance_references))
      for instance_references in gold_decompositions
  ]
  results['uniq_refs_per_input.avg'] = np.mean(counts)
  results['uniq_refs_per_input.min'] = np.min(counts)
  results['uniq_refs_per_input.max'] = np.max(counts)

  # Number of references per input.
  counts = [
      len(instance_references) for instance_references in gold_decompositions
  ]
  results['refs_per_input.avg'] = np.mean(counts)
  results['refs_per_input.min'] = np.min(counts)
  results['refs_per_input.max'] = np.max(counts)

  # Number of items in input data.
  results['counts.pred_inputs'] = len(pred)
  results['counts.gold_inputs'] = len(gold)

  # Number of individual items in input data (across analyses)
  results['counts.references'] = len(list(itertools.chain.from_iterable(gold)))
  results['counts.predictions'] = len(pred)

  # Calculate scoring metrics.
  results.update(
      ComputeMetrics(pred=pred_decompositions, gold=gold_decompositions))

  return results
