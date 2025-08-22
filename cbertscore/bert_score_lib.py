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

"""Implementation of BERTScore."""

import collections
from typing import Optional, Sequence

import numpy as np
from cbertscore import bert_lib

# TODO(agnesbi): Change this to a dataclass.
CBERTScore = collections.namedtuple('CBERTScore',
                                    ['precision', 'recall', 'f_score'])


class CBertScorer(object):
  """Class for scoring the meaning-similarity between two translations."""

  def __init__(self, bert_model_dir=None, bert_model=None,
               special_words = None):
    """Initialize a BERT_Score scorer using default BERT model configuration.

    Args:
      bert_model_dir: string specifing a directory containing a BERT model. This
        model will be loaded using default configuration.
      bert_model: A bert_lib.BertModel object. If specified, overrides
        bert_model_dir. One of bert_model_dir or bert_model must be specified.
      special_words: If provided, these are the only words that matter.
    """
    assert bert_model_dir or bert_model

    if bert_model is not None:
      self.bert_model = bert_model
    else:
      self.bert_model = bert_lib.BertModel(model_dir=bert_model_dir)
    self.medical_tokens = []
    if special_words:
      for w in special_words:
        self.medical_tokens.extend(self.bert_model.tokenizer.tokenize(w))
    self.medical_tokens = set(self.medical_tokens)

  def score(self,
            candidates,
            references,
            skip_empty_sentences_after_tokenization = False):
    """Calculates the BERTScore of the given sentences.

    Args:
      candidates ([str]): List of strings representing the hypothesis sentences.
      references ([str]): List of strings representing the reference strings.
      skip_empty_sentences_after_tokenization: If `False`, the function will
        raise an error, if a candidate or reference is empty after tokenization.
        If `True`, instead of raising an error, the BertScore won't be computed
        and (-1, -1, -1) will be returned for this entry.

    Returns:
      list of named tuples, one for each candidate/reference pair, each of which
      has 3 components:
        precision: how much of the candidate's meaning is in the reference
        recall:    how much of the reference's meaning is in the candidate
        f_score:   overall score (harmonic mean of precision & recall)

    Raises:
      AssertionError: If one of the candidate or reference sentence are empty,
        or if they are empty after tokenization.
    """
    candidates, references = list(candidates), list(references)

    assert len(candidates) and len(references), \
      ('You must provide at least one candidate and reference.')
    assert len(candidates) == len(references), \
      ('The number of candidate sentences must equal the number of reference '
       'sentences!')
    assert all(len(c.strip()) for c in candidates), 'Empty candidate sentence.'
    assert all(len(r.strip()) for r in references), 'Empty reference sentence.'

    n_examples = len(candidates)

    all_text = candidates + references
    print('About to get activations...')
    tokens, layers = self.bert_model.get_activations(all_text)
    print('Got activations.')

    all_cand_tokens = tokens[:n_examples]
    all_ref_tokens = tokens[-n_examples:]

    scores = collections.defaultdict(list)

    for example_idx in range(n_examples):
      # Ignore the first and last tokens when cheking similarity because
      # every sentence starts and ends with [CLS] and [SEP] tokens.
      cand_tokens = all_cand_tokens[example_idx][1:-1]
      ref_tokens = all_ref_tokens[example_idx][1:-1]

      if self.medical_tokens:
        # Only care about these words.
        idfs = {}
        for token in np.concatenate([cand_tokens, ref_tokens], axis=0):
          idfs[token] = 1 if token in self.medical_tokens else 0

      # pylint: disable=g-explicit-length-test
      if (skip_empty_sentences_after_tokenization and
          (len(cand_tokens) == 0 or len(ref_tokens) == 0)):
        scores[-1] = [CBERTScore(precision=-1, recall=-1, f_score=-1)]
        continue

      if len(cand_tokens) == 0:
        raise AssertionError(
            'You have an empty hypothesis sentence for the index {}. '
            'candidates[idx] was {}, in bytes: {}, all_cand_tokens[idx] was {} '
            ''.format(example_idx, candidates[example_idx],
                      candidates[example_idx].encode(),
                      all_cand_tokens[example_idx]))
      if len(ref_tokens) == 0:
        raise AssertionError(
            'You have an empty reference sentence for the index {}. '
            'references[idx] was {}, in bytes: {} all_ref_tokens[idx] was {} '
            'and '.format(example_idx, references[example_idx],
                          references[example_idx].encode(),
                          all_ref_tokens[example_idx]))
      # pylint: enable=g-explicit-length-test

      all_cand_embeddings = layers[:n_examples]
      all_ref_embeddings = layers[-n_examples:]

      layer_numbers = [int(k) for k in all_ref_embeddings[0].keys()]

      for ln in layer_numbers:
        cand_embeddings = np.vstack(all_cand_embeddings[example_idx][ln])[1:-1]
        ref_embeddings = np.vstack(all_ref_embeddings[example_idx][ln])[1:-1]

        # Calculate cosine similarity by normalizing then dot product of all
        # pairs.
        cand_embeddings /= np.sqrt(
            np.square(cand_embeddings).sum(axis=1, keepdims=True))
        ref_embeddings /= np.sqrt(
            np.square(ref_embeddings).sum(axis=1, keepdims=True))

        sim_matrix = np.matmul(cand_embeddings, ref_embeddings.T)

        precision = 0
        recall = 0

        if self.medical_tokens:
          idf_sum = 0
          for cand_idx in range(len(cand_tokens)):
            idf_val = idfs[cand_tokens[cand_idx]]
            precision += idf_val * np.max(sim_matrix[cand_idx])
            idf_sum += idf_val
          if idf_sum == 0:
            precision = np.nan
          else:
            precision /= idf_sum

          idf_sum = 0
          for ref_idx in range(len(ref_tokens)):
            idf_val = idfs[ref_tokens[ref_idx]]
            recall += idf_val * np.max(sim_matrix[:, ref_idx])

            idf_sum += idf_val
          if idf_sum == 0:
            recall = np.nan
          else:
            recall /= idf_sum
        else:
          precision = np.sum(np.max(sim_matrix, axis=1))
          precision /= len(cand_tokens)

          recall = np.sum(np.max(sim_matrix, axis=0))
          recall /= len(ref_tokens)

        if precision == 0 and recall == 0:
          f_score = 0
        else:
          f_score = 2 * precision * recall / (precision + recall)

        scores[ln].append(
            CBERTScore(precision=precision, recall=recall, f_score=f_score))

    return scores
