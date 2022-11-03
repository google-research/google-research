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

"""Analyzes the linked QA data with indexed corpus."""

import collections
import itertools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
from language.google.drfact import index_corpus
from language.labs.drkit import search_utils
import networkx as nx
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_init_concept2freq", None,
                     "Whether to run mention counting.")
flags.DEFINE_boolean("do_meantion_counting", None,
                     "Whether to run paragraph preprocessing.")
flags.DEFINE_boolean("do_generate_entity_networks", None,
                     "Whether to run networkx graph of entities.")
flags.DEFINE_boolean("do_qa_hop_analysis", None,
                     "Whether to analyze the hops between q and a concepts.")

flags.DEFINE_string("index_data_dir", "",
                    "The path to the folder of indexed files.")
flags.DEFINE_string("linked_qa_file", "",
                    "The path to the folder of linked qa files.")
flags.DEFINE_string("analysis_linked_qa_file", "",
                    "The path to the folder of linked qa files.")

flags.DEFINE_string("drkit_format_corpus_path", "",
                    "The path to the drkit-format processed corpus.")
flags.DEFINE_string("concept_frequency_dict_path", "",
                    "The path to save the concept2freq.")
flags.DEFINE_string("corpus_concept_vocab", "",
                    "The path to save the concept2freq.")


def do_init_concept2freq():
  """Save the number of mentions of each concept."""
  with tf.gfile.Open(FLAGS.drkit_format_corpus_path) as f:
    logging.info("Reading the corpus from %s ...", f.name)
    jsonlines = f.read().split("\n")
  data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]
  concept2freq = collections.defaultdict(lambda: 0)
  for instance in tqdm(data[:], desc="Computing concept2freq"):
    for mention in instance["mentions"]:
      concept2freq[mention["kb_id"]] += 1
  with tf.gfile.Open(FLAGS.concept_frequency_dict_path, "w") as f:
    logging.info("Saving the concept2freq to %s ...", f.name)
    json.dump(concept2freq, f)
    logging.info("# of non-empty concepts: %d. ", len(concept2freq))
  concepts = [
      k for k, _ in sorted(
          concept2freq.items(), key=lambda item: item[1], reverse=True)
  ]
  with tf.gfile.Open(FLAGS.corpus_concept_vocab, "w") as f:
    logging.info("Saving the concept2freq to %s ...", f.name)
    f.write("\n".join(concepts))


def load_entity2mention():
  """Loads the entity2mention data."""
  e2m_checkpoint = os.path.join(FLAGS.index_data_dir, "ent2ment.npz")
  with tf.device("/cpu:0"):
    logging.info("Reading %s", e2m_checkpoint)
    tf_e2m_data, tf_e2m_indices, tf_e2m_rowsplits = (
        search_utils.load_ragged_matrix("ent2ment", e2m_checkpoint))
    with tf.name_scope("RaggedConstruction"):
      e2m_ragged_ind = tf.RaggedTensor.from_row_splits(
          values=tf_e2m_indices, row_splits=tf_e2m_rowsplits, validate=False)
      e2m_ragged_val = tf.RaggedTensor.from_row_splits(
          values=tf_e2m_data, row_splits=tf_e2m_rowsplits, validate=False)
  return e2m_ragged_ind, e2m_ragged_val


def do_meantion_counting(concept2freq):
  """Executes the mention counting process for a linked QA file."""
  with tf.gfile.Open(FLAGS.linked_qa_file) as f:
    logging.info("Reading linked QA data from %s ...", f.name)
    jsonlines = f.read().split("\n")
  data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]
  data_analysis = []
  for instance in tqdm(data, desc="Mention Counting"):
    question_concepts = [e["kb_id"] for e in instance["entities"]]
    answer_concepts = [sf["kb_id"] for sf in instance["supporting_facts"]]
    question_concepts_num_mentions = [
        (concept, concept2freq.get(concept, 0)) for concept in question_concepts
    ]
    answer_concepts_num_mentions = [
        (concept, concept2freq.get(concept, 0)) for concept in answer_concepts
    ]
    analysis = dict()
    analysis["id"] = instance["_id"]
    analysis["q"] = instance["question"]
    analysis["a"] = instance["answer"]
    analysis["question_concepts_analysis"] = question_concepts_num_mentions
    analysis["answer_concepts_analysis"] = answer_concepts_num_mentions
    analysis["avg_num_mentions_question_concepts"] = float(
        np.mean([c[1] for c in question_concepts_num_mentions]))
    analysis["avg_num_mentions_answer_concepts"] = float(
        np.mean([c[1] for c in answer_concepts_num_mentions]))
    data_analysis.append(analysis)
  with tf.gfile.Open(FLAGS.analysis_linked_qa_file, "w") as f_out:
    logging.info("Writing analysis to output file...%s", f_out.name)
    f_out.write("\n".join(json.dumps(q) for q in data_analysis))


def do_generate_entity_networks(entity2id, e2m_ragged_ind, mentions):
  """Generates and saves the networkx graph object to file."""
  entity_connection_nxgraph = nx.Graph()
  for concept_i_id in tqdm(range(len(entity2id))):
    cooccured_mentions = e2m_ragged_ind[concept_i_id]
    for mid in cooccured_mentions.numpy():
      concept_j_id = mentions[mid][0]
      if concept_i_id != concept_j_id:
        entity_connection_nxgraph.add_edge(concept_i_id, concept_j_id)
  with tf.gfile.Open(
      os.path.join(FLAGS.index_data_dir, "entity_network.gpickle"), "wb") as f:
    logging.info("Reading %s", f.name)
    pickle.dump(f)


def find_shortest_path(entity_networks,
                       entity2id,
                       concept_i,
                       concept_j,
                       max_hops=3,
                       only_length=True):
  """Finds shortest paths between two nodes."""
  assert max_hops >= 1
  concept_i_id, concept_j_id = entity2id[concept_i], entity2id[concept_j]
  print("concept_ids:", concept_i_id, ",", concept_j_id)

  k = nx.shortest_path_length(
      entity_networks, source=concept_i_id, target=concept_j_id)
  if only_length:
    return k
  else:
    max_hops = min(max_hops, k)
    print("shortest_path_length: ", k)
    paths = nx.all_simple_paths(
        entity_networks,
        source=concept_i_id,
        target=concept_j_id,
        cutoff=max_hops)
    return k, list(paths)


def do_qa_hop_analysis(entity2id):
  """Analyze the hops between question concepts and answer concepts."""
  with tf.gfile.Open(
      os.path.join(FLAGS.index_data_dir, "entity_networks.gpickle"), "rb") as f:
    entity_networks = pickle.load(f)

  with tf.gfile.Open(FLAGS.analysis_linked_qa_file) as f:
    logging.info("Reading analysis linked QA data from %s ...", f.name)
    jsonlines = f.read().split("\n")
  data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]
  data_analysis = []
  for instance in tqdm(data, desc="Hop analysis"):
    question_concepts = [e[0] for e in instance["question_concepts_analysis"]]
    answer_concepts = [e[0] for e in instance["answer_concepts_analysis"]]
    hops = []
    for qc, ac in itertools.product(question_concepts, answer_concepts):
      print(qc, ac)
      k = find_shortest_path(entity_networks, entity2id, qc, ac)
      if k >= 1:
        hops.append(k)
    instance["qa_hop_min"] = min(hops)
    instance["qa_hop_max"] = max(hops)
    instance["qa_hop_avg"] = np.mean(hops)
    data_analysis.append(instance)
  with tf.gfile.Open(FLAGS.analysis_linked_qa_file, "w") as f_out:
    logging.info("Writing hop analysis to output file...%s", f_out.name)
    f_out.write("\n".join(json.dumps(q) for q in data_analysis))


def main(_):
  # Load all basic data.

  if FLAGS.do_init_concept2freq:
    do_init_concept2freq()
    return

  entity2id = index_corpus.load_concept_vocab(FLAGS.corpus_concept_vocab)
  if FLAGS.do_meantion_counting:
    with tf.gfile.Open(FLAGS.concept_frequency_dict_path) as f:
      logging.info("Reading %s", f.name)
      concept2freq = json.load(f)
    do_meantion_counting(concept2freq)
  if FLAGS.do_generate_entity_networks:

    with tf.gfile.Open(
        os.path.join(FLAGS.index_data_dir, "mentions.npy"), "rb") as f:
      logging.info("Reading %s", f.name)
      mentions = np.load(f)
    e2m_ragged_ind, _ = load_entity2mention()
    do_generate_entity_networks(entity2id, e2m_ragged_ind, mentions)
  if FLAGS.do_qa_hop_analysis:
    do_qa_hop_analysis(entity2id)


if __name__ == "__main__":
  app.run(main)
