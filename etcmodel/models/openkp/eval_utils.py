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

"""Utilities for evaluation and inference."""

import json
from typing import List, Optional, Set, Text, Tuple

import attr
import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models.openkp import generate_examples_lib as lib


@attr.s(auto_attribs=True)
class KpPositionPrediction:
  """Predictions of positions for key phrases."""

  # Index of the first word in the predicted key phrase.
  start_idx: int

  # Number of words in the predicted key phrase.
  phrase_len: int

  # Logit of the prediction
  logit: float


@attr.s(auto_attribs=True)
class OpenKpTextExample:
  """A text-only representation of an OpenKP example for eval/inference."""

  # Url for the example.
  url: Text

  # A list of all the words in the entire text.
  words: List[Text]

  # This is the label (up to 3 key phrases), if present. Lowercased.
  key_phrases: Optional[Set[Text]] = None

  @classmethod
  def from_json(cls, json_str: Text) -> 'OpenKpTextExample':
    """Constructs a `OpenKpTextExample` from a json string."""
    json_obj = json.loads(json_str)
    if 'KeyPhrases' in json_obj:
      key_phrases = set(' '.join(lib.KeyPhrase(x).words).lower()
                        for x in json_obj['KeyPhrases'])
      assert len(key_phrases) <= 3
    else:
      key_phrases = None
    return cls(
        url=json_obj['url'],
        words=json_obj['text'].split(' '),
        key_phrases=key_phrases)

  def to_json_string(self) -> Text:
    """Serializes this instance to a JSON string."""
    d = {'url': self.url, 'text': ' '.join(self.words)}
    if self.key_phrases is not None:
      d['KeyPhrases'] = [[kp] for kp in sorted(self.key_phrases)]
    return json.dumps(d)

  @classmethod
  def from_openkp_example(cls,
                          example: lib.OpenKpExample) -> 'OpenKpTextExample':
    """Constructs a `OpenKpTextExample` from a `OpenKpExample`."""
    if example.key_phrases is None:
      key_phrases = None
    else:
      key_phrases = set(' '.join(key_phrase.words).lower()
                        for key_phrase in example.key_phrases)
      assert len(key_phrases) <= 3
    return cls(
        url=example.url, words=example.text.split(' '), key_phrases=key_phrases)

  def get_key_phrase_predictions(
      self,
      position_predictions: List[KpPositionPrediction],
      max_predictions: int = 5) -> List[Text]:
    """Returns key phrases for the given position predictions.

    Args:
      position_predictions: An unsorted list of position predictions.
      max_predictions: Maximum number of predicted key phrases to return.

    Returns:
       A list of the top key phrase predictions in descending order. The key
       phrases are lowercased and deduplicated. Empty phrases are skipped.
       Predictions with invalid indices are skipped.
    """
    sorted_predictions = sorted(
        position_predictions,
        key=lambda prediction: prediction.logit,
        reverse=True)
    key_phrases = []
    for prediction in sorted_predictions:
      if prediction.phrase_len <= 0:
        continue
      key_phrase = ' '.join(
          self.words[prediction.start_idx:prediction.start_idx +
                     prediction.phrase_len]).lower()
      if not key_phrase:  # Empty string because start_idx was out of range.
        continue
      if key_phrase not in key_phrases:
        key_phrases.append(key_phrase)
      if len(key_phrases) >= max_predictions:
        return key_phrases
    return key_phrases

  def get_score_full(
      self,
      candidates: List[Text],
      max_depth: int = 5) -> Tuple[List[float], List[float], List[float]]:
    """Scores the candidate predictions.

    Follows the official evaluate.py script.

    Args:
      candidates: Predicted key phrases (sorted by descending confidence and
        deduped), will be lower cased.
      max_depth: Maximum k for precision@k to return.

    Returns:
      Three lists (precision@k, recall@k, and F1@k), for k = 1...max_depth.
    """
    assert self.key_phrases is not None
    precision = []
    recall = []
    f1 = []
    true_positive = 0.0
    referencelen = float(len(self.key_phrases))
    for i in range(max_depth):
      if len(candidates) > i:
        kp_pred = candidates[i]
        if kp_pred.lower() in self.key_phrases:
          true_positive += 1
      p = true_positive / (i + 1)
      r = true_positive / referencelen
      if p + r > 0:
        f = 2 * p * r / (p + r)
      else:
        f = 0.0
      precision.append(p)
      recall.append(r)
      f1.append(f)
    return precision, recall, f1


def score_examples(
    reference_examples: List[OpenKpTextExample],
    predictions: List[List[KpPositionPrediction]]) -> List[float]:
  """Scores the candidate predictions for all examples.

  Args:
    reference_examples: List of reference examples.
    predictions: List of the same length with the predictions.

  Returns:
    List with nine scores: average precision @1, @3, @5, average recall @1, @3,
    @5, and average F1@1, @3, @5.
  """
  assert len(reference_examples) == len(predictions)
  summary = []  #   precision@k, recall@k, and F1@k in this order for k=1, 3, 5
  for i in range(len(reference_examples)):
    example = reference_examples[i]
    pred_key_phrases = example.get_key_phrase_predictions(predictions[i], 5)
    precision, recall, f1 = example.get_score_full(pred_key_phrases, 5)
    summary.append([
        precision[0], precision[2], precision[4], recall[0], recall[2],
        recall[4], f1[0], f1[2], f1[4]
    ])
  return list(np.mean(summary, axis=0))


def read_text_examples(jsonl_file: Text) -> List[OpenKpTextExample]:
  """Reads reference key phrases from jsonl file for eval.

  We assume that the urls are already deduped.

  Args:
    jsonl_file: Input jsonl file with OpenKP labels.

  Returns:
    List of `OpenKpTextExample`.
  """
  with tf.gfile.GFile(jsonl_file, 'r') as reader:
    examples = [
        OpenKpTextExample.from_json(line) for line in reader.readlines()
    ]
  return examples


def write_text_examples(examples: List[OpenKpTextExample],
                        output_path: Text) -> None:
  """Writes examples to a jsonl file."""
  with tf.gfile.GFile(output_path, 'w') as writer:
    for example in examples:
      writer.write(example.to_json_string() + '\n')


def logits_to_predictions(
    logits: np.ndarray,
    max_predictions: int = 30) -> List[KpPositionPrediction]:
  """Converts highest logit predictions to the corresponding word indices.

  Args:
    logits: Numpy ndarray of logits with shape `[max_ngram_size,
      long_max_length]`, where `logits[i, j]` is the logit for `(i+1)`-gram
      starting at the `j`th *word* (not wordpiece; 0 based index).
    max_predictions: Max number of predictions to return.

  Returns:
    Indices (of words, not wordpieces) of predicted keyphrases as a list of
    `KpPositionPrediction`. The list is not sorted by logits, and may include
    out of range indices (for short documents).
  """
  assert len(logits.shape) == 2
  top_indices_flat = np.argpartition(
      logits, -max_predictions, axis=None)[-max_predictions:]
  top_indices = np.unravel_index(top_indices_flat, logits.shape)
  predictions = []
  for multi_index in zip(*top_indices):
    i, j = multi_index
    predictions.append(
        KpPositionPrediction(start_idx=j, phrase_len=i + 1, logit=logits[i, j]))
  return predictions
