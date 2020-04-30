# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""WT5 postprocessors."""

import collections

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import spacy
spacy_nlp = spacy.load("en_core_web_sm")
"""  # GOOGLE-INTERNAL pylint: disable=pointless-string-statement


def abstractive_explanations(output,
                             separator=" explanation: ",
                             **unused_kwargs):
  """Get label and explanations/rationales from predictions and targets.

  Args:
    output: str, target or prediction in text format.
    separator: str, separator used to delimit rationales/explanations.
  Returns:
    a dict, containing the label and a list of explanations.
  """
  parts = output.split(separator)
  label, expls = parts[0], parts[1:]

  ex = {
      "label": label,
      "explanations": expls,
  }

  return ex


def extractive_explanations(output,
                            separator=" explanation: ",
                            match_fn=None,
                            tokenizer_fn=None,
                            example=None,
                            is_target=False):
  """Get labels and a list of overlap spans from targets/predictions.

  Args:
    output: str, target or predictions in text format.
    separator: str, delimiter used for separating multiple explanations.
    match_fn: function, used to find rationale in input text. Default None uses
      an exact match.
    tokenizer_fn: function, takes in a string and returns a list of tokens.
      Default None just uses characters as tokens.
    example: dict, input example.
    is_target: bool, whether the input was ground truth (True) or
      prediction (False).
  Returns:
    a dict, with the label list of overlap spans, and span array. Each span is
      a tuple with the start and end (inclusive) token position. The span array
      is a binary array whose length is the same as the (tokenized) input text,
      where a 1 indicates that that token is part of an explanation and a 0
      indicates it isn't.
  """

  # If no tokenizer was provided, convert string to list of characters
  tokenizer_fn = tokenizer_fn or list

  def spacy_tokenizer(text):
    doc = spacy_nlp(text)
    tokens = [token.text for token in doc]
    return tokens

  tokenizer_fn = spacy_tokenizer
  """  # GOOGLE-INTERNAL  pylint:disable=pointless-string-statement

  def _find_exact_match(haystack, needle):
    for start_pos in range(len(haystack) - len(needle) + 1):
      if haystack[start_pos:start_pos + len(needle)] == needle:
        return (start_pos, start_pos + len(needle))
      start_pos += 1
    return None

  match_fn = match_fn or _find_exact_match

  label_with_explanations = abstractive_explanations(
      output, separator=separator)

  explanations = label_with_explanations["explanations"]

  spans = []
  prediction_not_found_in_input = 0
  input_text = tf.compat.as_text(example["inputs_plaintext"])
  input_text = input_text.replace("\n", " ")

  # Removing duplicates from explanations since extractive models can produce
  # duplicate explanations.
  explanations = list(collections.OrderedDict.fromkeys(explanations))
  final_explanations = []

  tokenized_input = tokenizer_fn(input_text)
  span_array = np.zeros(len(tokenized_input), np.int)
  for e in explanations:
    overlap_tuple = match_fn(tokenized_input, tokenizer_fn(e))
    if overlap_tuple:
      spans.append(overlap_tuple)
      span_array[overlap_tuple[0]:overlap_tuple[1]] = 1
      final_explanations.append(e)
    else:
      if is_target:
        raise ValueError("Rationale: %s from ground truth not found in "
                         "input example." % e)
      else:
        prediction_not_found_in_input += 1

  if not is_target and prediction_not_found_in_input:
    logging.info("Number of times the rationale generated from the model was "
                 "not present in the input: %d ", prediction_not_found_in_input)

  return {
      "label": label_with_explanations["label"],
      "overlap_spans": spans,
      "span_array": span_array.tolist(),
      "explanations": final_explanations,
  }
