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

# Lint as: python3
"""Script to pre-process fact2fact sparse tensor."""
import collections
import gc
import json
import os

from absl import app
from absl import flags
from language.labs.drkit import search_utils
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string("wiki_file", None, "Path to corpus.")

flags.DEFINE_string("entity_file", None, "Path to entities.")

flags.DEFINE_string("multihop_output_dir", None, "Path to output files.")

flags.DEFINE_string("fact2fact_index_dir", None, "Path to output files.")

flags.DEFINE_boolean("do_preprocess", None,
                     "Whether to run paragraph preprocessing.")

flags.DEFINE_boolean("do_combine", None,
                     "Whether to combine all shards into one.")

flags.DEFINE_integer("num_shards", 1,
                     "Number of shards to store mention embeddings in.")

flags.DEFINE_integer("max_follow", 300, "Maximum of the following facts.")

flags.DEFINE_integer("my_shard", None,
                     "Shard number for this process to run over.")

flags.DEFINE_integer("shards_to_combine", None,
                     "Max number of shards to combine.")


def _load_lines():
  tf.logging.info("Reading paragraphs from %s", FLAGS.wiki_file)
  with tf.gfile.Open(FLAGS.wiki_file) as f:
    lines = f.read().split("\n")
  if not lines[-1]:
    lines = lines[:-1]
  return lines


def do_preprocess():
  """Loads and processes the data."""
  lines = _load_lines()
  fact2entity_set = []
  facts = []
  entity_freq = collections.defaultdict(lambda: 0)
  for line in tqdm(lines, desc="preprocessing lines"):
    orig_para = json.loads(line.strip())
    facts.append(orig_para)
    concept_set = set([m["kb_id"] for m in orig_para["mentions"]])
    fact2entity_set.append(concept_set)
    for c in concept_set:
      entity_freq[c] += 1
  k = 100  # For avoiding the bridge concepts being too common.
  most_frequent_concepts = set([
      c for c, _ in sorted(
          entity_freq.items(), key=lambda x: x[1], reverse=True)[:k]
  ])
  num_facts = len(facts)
  start_fact_index, end_fact_index = 0, None
  if FLAGS.num_shards >= 2:
    num_facts_per_shard = num_facts / FLAGS.num_shards
    assert FLAGS.my_shard >= 0 and FLAGS.my_shard <= FLAGS.num_shards
    start_fact_index = int(FLAGS.my_shard * num_facts_per_shard)
    end_fact_index = int(start_fact_index + num_facts_per_shard)
    tf.logging.info("batch_mode! num_facts_per_shard: %d start:end=[%d:%d]",
                    num_facts_per_shard, start_fact_index, end_fact_index)

  rows, cols, vals = [], [], []
  indices = list(range(num_facts))
  for fact_i in tqdm(
      indices[start_fact_index:end_fact_index], desc="Processing Fact2Fact"):
    following_facts = []
    for fact_j in indices:
      if fact_i == fact_j:
        # Never jump back to itself.
        continue
      concept_set_i = fact2entity_set[fact_i]
      concept_set_j = fact2entity_set[fact_j]
      intersection = concept_set_i & concept_set_j
      if not intersection:
        # Must have at least one bridging concepts.
        continue
      if len(intersection) < 2:
        # Must have at least two bridging concepts.
        continue
      if intersection.issubset(most_frequent_concepts):
        # The bridging concepts are all very common.
        continue
      if len(intersection) == len(concept_set_i):
        # Never jump to a fact that convers exactly same set of concepts.
        continue
      num_new_concepts = len(concept_set_j) - len(intersection)
      if num_new_concepts < 2:
        # Must jump to a fact with two more new concepts.
        continue
      following_facts.append((fact_j, float(num_new_concepts)))
    # Sort and then cut following_facts.
    if FLAGS.max_follow > 0:
      following_facts = sorted(
          following_facts, key=lambda x: x[1], reverse=True)
      following_facts = following_facts[:FLAGS.max_follow]
    rows += [fact_i] * len(following_facts)
    cols += [x[0] for x in following_facts]
    vals += [1.0 for x in following_facts]  # Note: consider using TF-IDF later.
    print(fact_i, len(following_facts))
  tf.logging.info("Done! NNZ in fact2fact = %d", len(rows))
  if not tf.gfile.Exists(FLAGS.fact2fact_index_dir):
    tf.gfile.MakeDirs(FLAGS.fact2fact_index_dir)

  tf.logging.info("Saving part-of-matrix of shard: %d", FLAGS.my_shard)
  with tf.gfile.Open(
      os.path.join(FLAGS.fact2fact_index_dir, "f2f_%d.json" % FLAGS.my_shard),
      "w") as f:
    json.dump(dict(rows=rows, cols=cols, vals=vals), f)
  tf.logging.info("Saving Done!")


def do_combine():
  """Combines all shards of the fact2fact matrix."""
  lines = _load_lines()
  num_facts = len(lines)

  rows, cols, vals = [], [], []
  shard_range = list(range(FLAGS.num_shards + 1))
  for shard_id in tqdm(shard_range, desc="loading shard files"):
    tf.logging.info("Appending %d-th shard ...", shard_id)
    with tf.gfile.Open(
        os.path.join(FLAGS.fact2fact_index_dir, "f2f_%d.json" % shard_id)) as f:
      cur_data = json.load(f)
    rows += cur_data["rows"]
    cols += cur_data["cols"]
    vals += cur_data["vals"]
    del cur_data
    gc.collect()

  tf.logging.info("Creating sp.csr_matrix")
  sp_fact2fact = sp.csr_matrix((vals, (rows, cols)),
                               shape=[num_facts, num_facts])
  tf.logging.info("Num nonzero in f2f = %d", sp_fact2fact.getnnz())
  tf.logging.info("Saving as ragged f2f tensor %s.", str(sp_fact2fact.shape))
  search_utils.write_ragged_to_checkpoint(
      "fact2fact", sp_fact2fact,
      os.path.join(FLAGS.fact2fact_index_dir, "fact2fact.npz"))


def main(_):
  # Initialize tokenizer.
  if FLAGS.do_preprocess:
    do_preprocess()
  elif FLAGS.do_combine:
    do_combine()


if __name__ == "__main__":
  app.run(main)
