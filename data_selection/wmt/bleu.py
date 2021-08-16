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

r"""Parallel BLEU score calculation.

This version of BLEU calculation is derived from the MLPerf transformer
reference.
Tries to match SacreBLEU metric reasonably well, but is not identical.

Refs:
    tokenizer at:
    https://github.com/tensorflow/models/blob/master/official/transformer/utils/tokenizer.py
    original preprocessing tokenizer:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    original t2t code:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py

Usage:
    refs = '''food bar brown cow
    blee bloo dog sat
    or please take me out
    '''
    hyps = '''foo bar brown cow
    blee bloo dog sit
    please do take me out
    '''
    bleu_local(refs.split("\n"), hyps.split("\n"))  # 39.65
"""

import collections
import math
import re
import sys
import unicodedata
import numpy as np
import six


class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(
        six.unichr(x)
        for x in range(sys.maxunicode)
        if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.

  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).

  Note that a number (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized, i.e. the dot stays with the number because
  `s/(\p{P})(\P{N})/ $1 $2/g` does not match this case (unless we add a
  space after each sentence). However, this error is already in the
  original mteval-v14.pl and we want to be consistent with it.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this methods.

  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu_matches(reference_corpus, translation_corpus, max_order=4):
  """Computes BLEU match stats of translations against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each reference
      should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation should
      be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.

  Returns:
    Aggregated n-gram stats for BLEU calculation.
  """
  reference_length = 0
  translation_length = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = dict((ngram, min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) -
                                1] += translation_ngram_counts[ngram]

  return (np.array(matches_by_order),
          np.array(possible_matches_by_order),
          np.array(reference_length),
          np.array(translation_length))


def bleu_partial(ref_lines, hyp_lines, case_sensitive=False):
  """Compute n-gram statistics for two lists of references and translations."""
  if len(ref_lines) != len(hyp_lines):
    raise ValueError("Reference and translation lists have different "
                     "numbers of lines.")
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return compute_bleu_matches(ref_tokens, hyp_tokens)


def complete_bleu(matches_by_order,
                  possible_matches_by_order,
                  reference_length,
                  translation_length,
                  max_order=4,
                  use_bp=True):
  """Compute BLEU score from aggregated n-gram statistics."""
  precisions = [0] * max_order
  smooth = 1.0
  geo_mean = 0.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1. / ratio)
  bleu = geo_mean * bp
  return float(bleu) * 100.0


def bleu_local(ref_lines, hyp_lines, case_sensitive=False):
  """Compute BLEU for two lists of reference and hypothesis translations."""
  stats = bleu_partial(ref_lines, hyp_lines, case_sensitive=case_sensitive)
  return complete_bleu(*stats) * 100
