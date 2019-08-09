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

"""This file contains a class which acts as a wrapper around the PPR algorithm.

This class has the following functionality:
1. Load the KB graph,
2. Given list of seed entities, get topk entities from PPR.
3. Get unique facts between all extracted entities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from fat.fat_bert_nq.ppr.apr_algo import csr_personalized_pagerank
from fat.fat_bert_nq.ppr.apr_algo import csr_topk_fact_extractor
from fat.fat_bert_nq.ppr.kb_csr_io import CsrData

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal NQ evaluation.')


class ApproximatePageRank(object):
  """APR main lib which is used to wrap functions around ppr algo."""

  def __init__(self):
    self.data = CsrData()
    self.data.load_csr_data(
        full_wiki=FLAGS.full_wiki, files_dir=FLAGS.apr_files_dir)

  def get_topk_extracted_ent(self, seeds, alpha, topk):
    """Extract topk entities given seeds.

    Args:
      seeds: An Ex1 vector with weight on every seed entity
      alpha: probability for PPR
      topk: max top entities to extract
    Returns:
      extracted_ents: list of selected entities
      extracted_scores: list of scores of selected entities
    """
    ppr_scores = csr_personalized_pagerank(seeds, self.data.adj_mat_t_csr,
                                           alpha)
    sorted_idx = np.argsort(ppr_scores)[::-1]
    extracted_ents = sorted_idx[:topk]
    extracted_scores = ppr_scores[sorted_idx[:topk]]

    # Check for really low values
    # Get idx of First value < 1e-6, limit extracted ents till there
    zero_idx = np.where(ppr_scores[extracted_ents] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
      extracted_ents = extracted_ents[:zero_idx[0]]

    return extracted_ents, extracted_scores

  def get_facts(self, entities, topk, alpha, seed_weighting=True):
    """Get subgraph describing a neighbourhood around given entities.

    Args:
      entities: A list of Wikidata entities
      topk: Max entities to extract from PPR
      alpha: Node probability for PPR
      seed_weighting: Boolean for performing weighting seeds by freq in passage

    Returns:
      unique_facts: A list of unique facts around the seeds.
    """

    if FLAGS.verbose_logging:
      tf.logging.info('Getting subgraph')
    entity_ids = [
        int(self.data.ent2id[x]) for x in entities if x in self.data.ent2id
    ]
    if FLAGS.verbose_logging:
      tf.logging.info(
          str([self.data.entity_names['e'][str(x)]['name'] for x in entity_ids
              ]))
    freq_dict = {x: entity_ids.count(x) for x in entity_ids}

    seed = np.zeros((self.data.adj_mat.shape[0], 1))
    if not seed_weighting:
      seed[entity_ids] = 1. / len(set(entity_ids))
    else:
      for x, y in freq_dict.items():
        seed[x] = y
      seed = seed / seed.sum()

    extracted_ents, extracted_scores = self.get_topk_extracted_ent(
        seed, alpha, topk)
    if FLAGS.verbose_logging:
      tf.logging.info('Extracted ents: ')
      tf.logging.info(
          str([
              self.data.entity_names['e'][str(x)]['name']
              for x in extracted_ents
          ]))

    facts = csr_topk_fact_extractor(self.data.adj_mat_t_csr, self.data.rel_dict,
                                    freq_dict, self.data.entity_names,
                                    extracted_ents, extracted_scores)
    if FLAGS.verbose_logging:
      tf.logging.info('Extracted facts: ')
      tf.logging.info(str(facts))

    # Extract 1 unique fact per pair of entities (fact with highest score)
    # Sort by scores
    unique_facts = {}
    for (sub, obj, rel, score) in facts:
      fwd_dir = (sub, obj)
      rev_dir = (obj, sub)
      if fwd_dir in unique_facts and score > unique_facts[fwd_dir][1]:
        unique_facts[fwd_dir] = (rel, score)
      elif rev_dir in unique_facts and score > unique_facts[rev_dir][1]:
        unique_facts[fwd_dir] = (rel, score)
        del unique_facts[rev_dir]  # Remove existing entity pair
      else:
        unique_facts[(sub, obj)] = (rel, score)
    unique_facts = list(unique_facts.items())
    return unique_facts
