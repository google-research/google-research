# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""This file holds the algorithm for the CSR based implementation os Personalized Page Rank."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'fact_score_type', 'FREQ_SCORE',
    'Scoring method for facts. One in ["FREQ_SCORE", "MIN_SCORE"]')


def csr_personalized_pagerank(seeds, adj_mat, alpha, max_iter=20):
  """Return the PPR Scores vector for the given seed and adjacency matrix.

  Algorithm :
      https://pdfs.semanticscholar.org/a4df/5ff749d823905ff9c1a23b522d3f426a1bb6.pdf
      (Figure 1)

  Args:
    seeds: A sparse matrix of size E x 1.
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    alpha: Probability of staying at current node [0-1]
    max_iter: Maximum iterations to run ppr for

  Returns:
    s_ovr: A vector of size E, ppr scores for every entity
  """

  restart_prob = alpha
  r = restart_prob * seeds
  s_ovr = np.copy(r)
  for _ in range(max_iter):
    if FLAGS.verbose_logging:
      tf.logging.info('Performing PPR Matrix Multiplication')
    r_new = (1. - restart_prob) * (adj_mat.dot(r))
    s_ovr = s_ovr + r_new
    delta = abs(r_new.sum())
    if delta < 1e-5:
      break
    r = r_new
  return np.squeeze(s_ovr)


def get_fact_score(extracted_scores,
                   subj,
                   obj,
                   freq_dict,
                   score_type='FREQ_SCORE'):
  """Return score for a subj, obj pair of entities.

  Args:
    extracted_scores: A score vector of size E
    subj: subj entity id
    obj: obj entity id
    freq_dict: frequency of every entity in passage
    score_type: string for type of scoring used

  Returns:
      score: A float score for a subj, obj entity pair
  """
  score_types = set('FREQ_SCORE', 'MIN_SCORE')
  # Min of Page Rank scores of both Entities
  # Upweight facts where both have high scores
  min_score = min(
      extracted_scores[subj], extracted_scores[obj]
  )

  # Freq Score - If both entities are present - sum of frequencies
  # Upweight facts where both entities are in passage
  if subj in freq_dict and obj in freq_dict:
    freq_score = freq_dict[subj] + freq_dict[obj]
  else:
    freq_score = min(extracted_scores[subj],
                     extracted_scores[obj])
  if score_type == 'FREQ_SCORE':
    return freq_score
  elif score_type == 'MIN_SCORE':
    return min_score
  else:
    ValueError(
        'The score_type should be one of: %s' + ', '.join(list(score_types)))


def csr_topk_fact_extractor(adj_mat, rel_dict, freq_dict, entity_names,
                            extracted_ents, extracted_scores):
  """Return facts for selected entities.

  Args:
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    rel_dict: A sparse matrix of size E x E whose values are rel_ids between
          entities
    freq_dict: A dictionary with frequency of every entity in passage
    entity_names: A dictionary of entity and relation ids to their surface
          form names
    extracted_ents: A list of selected topk entities
    extracted_scores: A list of selected topk entity scores

  Returns:
      facts: A list of ((subj_id, subj_name), (obj_id, obj_name), (rel_id,
          rel_name), score)
  """

  # Slicing adjacency matrix to subgraph of all extracted entities
  submat = adj_mat[extracted_ents, :]
  submat = submat[:, extracted_ents]
  # Extracting non-zero entity pairs
  col_idx, row_idx = submat.nonzero()

  facts = []
  for ii in range(row_idx.shape[0]):
    subj_id = extracted_ents[row_idx[ii]]
    obj_id = extracted_ents[col_idx[ii]]
    fwd_dir = (subj_id, obj_id)
    rev_dir = (obj_id, subj_id)
    rel_id = rel_dict[fwd_dir]
    if rel_id == 0:  # no relation from subj to obj
      # Checking for relation from obj to subj
      rel_id = rel_dict[rev_dir]
      if rel_id == 0:
        continue
      subj_id, obj_id = obj_id, subj_id
    score = get_fact_score(
        extracted_scores,
        row_idx[ii],
        col_idx[ii],
        freq_dict,
        score_type=FLAGS.fact_score_type)
    subj_name = entity_names['e'][str(subj_id)]['name']
    obj_name = entity_names['e'][str(obj_id)]['name']
    rel_name = entity_names['r'][str(rel_id)]['name']
    facts.append(((subj_id, subj_name),
                  (obj_id, obj_name),
                  (rel_id, rel_name), score))
  return facts
