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

"""This file does create, save and load of the KB graph in Scipy CSR format."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import gzip
import json
import os
import tempfile

import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
import tensorflow as tf

from fat.fat_bert_nq.ppr import sling_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('apr_files_dir', 'None', 'Read and Write apr data')
flags.DEFINE_bool('full_wiki', False, '')
flags.DEFINE_bool('decompose_ppv', True, '')
flags.DEFINE_integer('total_kb_entities', 29610404,
                     'Total entities in processed sling KB')


class CsrData(object):
  """Class to perform IO operations on KB data in CSR format."""

  def __init__(self):
    self.adj_mat = None  # CSR ExE sparse weighted adjacency matrix
    self.adj_mat_t_csr = None  # adj_mat transposed and back to CSR form
    self.rel_dict = None  # CSR ExE sparse matrix where values are rel_id
    self.ent2id = None  # Python dictionary mapping entities to integer ids.
    self.entity_names = None  # dict mapping entity ids to entity surface forms

  def get_file_names(self, full_wiki, files_dir):
    """Return filenames depending on full KB or subset of KB."""
    sub_file_names = {
        'ent2id_fname': 'csr_ent2id_sub.json.gz',
        'id2ent_fname': 'csr_id2ent_sub.json.gz',
        'rel2id_fname': 'csr_rel2id_sub.json.gz',
        'rel_dict_fname': 'csr_rel_dict_sub.npz',
        'kb_fname': 'kb.sling',
        'entity_names_fname': 'csr_entity_names_sub.json.gz',
        'adj_mat_fname': 'csr_adj_mat_sparse_matrix_sub.npz'
    }
    full_file_names = {
        'ent2id_fname': 'csr_ent2id_full.json.gz',
        'id2ent_fname': 'csr_id2ent_full.json.gz',
        'rel2id_fname': 'csr_rel2id_full.json.gz',
        'rel_dict_fname': 'csr_rel_dict_full.npz',
        'kb_fname': 'kb.sling',
        'entity_names_fname': 'csr_entity_names_full.json.gz',
        'adj_mat_fname': 'csr_adj_mat_sparse_matrix_full.npz',
    }
    files = full_file_names if full_wiki else sub_file_names
    file_paths = {k: os.path.join(files_dir, v) for k, v in files.items()}
    return file_paths

  def create_and_save_csr_data(self, full_wiki, decompose_ppv, files_dir):
    """Return the PPR vector for the given seed and adjacency matrix.

      Algorithm : Parses sling KB - extracts subj, obj, rel triple and stores
        as sparse matrix.
      Data Store :
          ent2id = json {'Q123':1}
          rel2id = json {'P123':1}
          entity_names = json { 'e':{ 'Q123':'abc'}, 'r':{ 'P123':'abc'} }
          adj_mat = ExE scipy CSR matrix reldict = ExE scipy DOK matrix
    Args:
      full_wiki : boolean True which Parses entire Wikidata
      decompose_ppv : boolean True which
                  Creates Relation level SP Matrices and then combines them
      files_dir : Directory to save KB data in

    Returns:
      None
    """
    file_paths = self.get_file_names(full_wiki, files_dir)
    tf.logging.info('KB Related filenames: %s'%(file_paths))

    tf.logging.info('Loading KB')
    kb = sling_utils.get_kb(file_paths['kb_fname'])

    # Initializing Dictionaries
    ent2id = dict()
    rel2id = dict()
    rel2id['NoRel'] = len(rel2id)
    entity_names = dict()
    entity_names['e'] = dict()
    entity_names['r'] = dict()
    num_entities = FLAGS.total_kb_entities
    # Using pre-computed entity count - since we need pre-created matrix
    rel_dict = sparse.dok_matrix((num_entities, num_entities), dtype=np.int16)
    relation_map = {}
    all_row_ones, all_col_ones = [], []
    count = 0

    tf.logging.info('Processing KB')
    for x in kb:
      count += 1
      if not full_wiki and count == 100000:
        break  # For small KB Creation
      if sling_utils.is_subj(x, kb):
        subj = x.id
        properties = sling_utils.get_properties(x, kb)
        for (rel, obj) in properties:

          if subj not in ent2id:
            ent2id[subj] = len(ent2id)
          if obj not in ent2id:
            ent2id[obj] = len(ent2id)
          if rel not in rel2id:
            rel2id[rel] = len(rel2id)
          subj_id = ent2id[subj]
          obj_id = ent2id[obj]
          rel_id = rel2id[rel]

          entity_names['e'][subj_id] = dict()
          entity_names['e'][subj_id]['name'] = str(kb[subj]['name'])
          entity_names['e'][obj_id] = dict()
          entity_names['e'][obj_id]['name'] = str(kb[obj]['name'])
          entity_names['r'][rel_id] = dict()
          entity_names['r'][rel_id]['name'] = str(kb[rel]['name'])

          rel_dict[(subj_id, obj_id)] = rel_id

          if rel_id not in relation_map:
            relation_map[rel_id] = [[], []]

          if decompose_ppv:
            relation_map[rel_id][0].append(subj_id)
            relation_map[rel_id][1].append(obj_id)
          else:
            all_row_ones.append(subj_id)
            all_col_ones.append(obj_id)
            # Add the below for forcing bidirectional graphs
            # all_row_ones.append(obj_id)
            # all_col_ones.append(subj_id)

    kb = None
    gc.collect()

    id2ent = {idx: ent for ent, idx in ent2id.items()}
    loaded_num_entities = len(id2ent)
    assert loaded_num_entities == num_entities  # Sanity check for processing
    tf.logging.info('%d entities loaded' % loaded_num_entities)
    tf.logging.info('Building Sparse Matrix')

    if decompose_ppv:  # Relation Level Sparse Matrices to weight accordingly
      for rel in relation_map:
        row_ones, col_ones = relation_map[rel]
        m = sparse.csr_matrix((np.ones(
            (len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
                              shape=(num_entities, num_entities))
        relation_map[rel] = normalize(m, norm='l1', axis=1)

        # TODO(vidhisha) : Add this during Relation Training
        # if RELATION_WEIGHTING:
        #   if rel not in relation_embeddings:
        #     score = NOTFOUNDSCORE
        #   else:
        #     score = np.dot(question_embedding, relation_embeddings[rel]) / (
        #         np.linalg.norm(question_embedding) *
        #         np.linalg.norm(relation_embeddings[rel]))
        #   relation_map[rel] = relation_map[rel] * np.power(score, EXPONENT)
      adj_mat = sum(relation_map.values()) / len(relation_map)

    else:  # KB Level Adjacency Matrix
      adj_mat = sparse.csr_matrix(
          (np.ones((len(all_row_ones),)),
           (np.array(all_row_ones), np.array(all_col_ones))),
          shape=(num_entities, num_entities))
      adj_mat = normalize(adj_mat, 'l1', axis=1)

    tf.logging.info('Saving all files')
    # FIX for Bad Magic Header bug
    self.save_safe_npz(file_paths['adj_mat_fname'], adj_mat)
    self.save_safe_npz(file_paths['rel_dict_fname'], rel_dict.tocsr())
    self.save_json_gzip(file_paths['ent2id_fname'], ent2id)
    self.save_json_gzip(file_paths['id2ent_fname'], id2ent)
    self.save_json_gzip(file_paths['rel2id_fname'], rel2id)
    self.save_json_gzip(file_paths['entity_names_fname'], entity_names)

  def save_safe_npz(self, fname, obj):
    with tempfile.TemporaryFile() as tmp, tf.gfile.Open(fname, 'wb') as f:
      sparse.save_npz(tmp, obj)
      tmp.seek(0)
      f.write(tmp.read())
      f.close()

  def save_json_gzip(self, fname, obj):
    with gzip.GzipFile(fileobj=tf.gfile.Open(fname, 'w')) as op1:
      op1.write(json.dumps(obj).encode('utf8'))
      op1.close()

  def safe_load_npz(self, fname):
    with tf.gfile.Open(fname, 'rb') as fp1:
      obj = sparse.load_npz(fp1)
      fp1.close()
    return obj

  def load_json_gz(self, fname):
    with gzip.GzipFile(fileobj=tf.gfile.Open(fname, 'rb')) as op4:
      obj = json.load(op4)
      op4.close()
    return obj

  def load_csr_data(self, full_wiki, files_dir):
    """Load saved KB data in CSR format."""
    if files_dir == 'None':
      return
    tf.logging.info("""Load saved KB files.""")
    file_paths = self.get_file_names(full_wiki, files_dir)
    tf.logging.info('KB Related filenames: %s'%(file_paths))

    tf.logging.info('Loading adj_mat')
    self.adj_mat = self.safe_load_npz(file_paths['adj_mat_fname'])
    tf.logging.info('Loading rel_dict')
    self.rel_dict = self.safe_load_npz(file_paths['rel_dict_fname'])

    tf.logging.info('Loading ent_name')
    self.entity_names = self.load_json_gz(file_paths['entity_names_fname'])

    tf.logging.info('Loading ent2id')
    self.ent2id = self.load_json_gz(file_paths['ent2id_fname'])

    # Performing this once instead of for every iteration
    self.adj_mat_t_csr = self.adj_mat.transpose().tocsr()
    del self.adj_mat
    tf.logging.info('Entities loaded: %d', len(list(self.ent2id.keys())))

if __name__ == '__main__':
  csr_data = CsrData()
  csr_data.create_and_save_csr_data(full_wiki=FLAGS.full_wiki,
                                    decompose_ppv=FLAGS.decompose_ppv,
                                    files_dir=FLAGS.apr_files_dir)
