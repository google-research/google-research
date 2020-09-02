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

# Lint as: python3
r"""Script to pre-process a commonsense knowledge corpus (e.g., GenericsKB).

Example usage:

# 1) pre-processing the corpus.
index_corpus \
--do_preprocess=True \
--entity_file /path/to/nscskg_data/gkb_best.vocab.txt \
--wiki_file /path/to/nscskg_data/gkb_best.drkit_format.jsonl \
--multihop_output_dir /path/to/nscskg_data/drfact_output/ \
--max_seq_length 128 --doc_stride=128  \
--alsologtostderr

# 2) indexing the mention embeddings and saving to num_shard files.
# 3) combining all the shard files.
"""
import collections
import json
import os

from absl import app
from absl import flags
from garcon.albert import tokenization as albert_tokenization
from bert import tokenization as bert_tokenization
from language.labs.drkit import bert_utils_v2
from language.labs.drkit import search_utils
from language.labs.drkit.hotpotqa import index as index_util
from nltk.corpus import stopwords
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("wiki_file", None, "Path to corpus.")

flags.DEFINE_string("entity_file", None, "Path to entities.")

flags.DEFINE_string("multihop_output_dir", None, "Path to output files.")

flags.DEFINE_string(
    "bert_ckpt_dir",
    "/path/to/bert/pretrained_models/wwm_uncased_L-24_H-1024_A-16/",
    "Directory with pre-trained BERT model.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("tokenizer_type", "bert_tokenization",
                    "The tokenizier type that the BERT model was trained on.")
flags.DEFINE_string("tokenizer_model_file", None,
                    "The tokenizier model that the BERT was trained with.")

flags.DEFINE_boolean("do_preprocess", None,
                     "Whether to run paragraph preprocessing.")

flags.DEFINE_boolean("do_embed", None, "Whether to run mention embedding.")

flags.DEFINE_boolean("do_embed_mentions", None,
                     "Whether to run mention embedding.")
flags.DEFINE_boolean("do_embed_facts", None, "Whether to run fact embedding.")

flags.DEFINE_string("ckpt_name", "bert_model.ckpt",
                    "The name of the BERT ckpt point.")

flags.DEFINE_string("embed_prefix", "bert_large",
                    "The prefix of the embedding files.")

flags.DEFINE_boolean("do_combine", None,
                     "Whether to combine all shards into one.")

flags.DEFINE_integer("max_total_paragraphs", None,
                     "Maximum number of paragraphs to process.")

flags.DEFINE_integer("predict_batch_size", 32, "Batch size for embedding.")

flags.DEFINE_integer("max_mentions_per_doc", 20,
                     "Maximum number of mentions to retrieve per doc.")

flags.DEFINE_integer("max_facts_per_entity", 500,
                     "Maximum number of mentions to retrieve per doc.")

flags.DEFINE_integer("max_entity_length", 5,
                     "Maximum number of tokens per entity.")

flags.DEFINE_integer("num_shards", 1,
                     "Number of shards to store mention embeddings in.")

flags.DEFINE_integer("my_shard", None,
                     "Shard number for this process to run over.")

flags.DEFINE_integer("shards_to_combine", None,
                     "Max number of shards to combine.")


def simple_match(lemmas,
                 concept_vocab,
                 max_len=4,
                 disable_overlap=False,
                 stop_words=None):
  """Extracts mentions of concepts in a sentence.

  Args:
    lemmas: list (List of lemmas, which are processed by lemmatize_sentence.
    concept_vocab: dict (The key is a concept and the value is its id.)
    max_len: int (The maximum length of concept.)
    disable_overlap: bool (Whether we want only non-overlapped mentions.)
    stop_words: set of string, which are used for skip single words.

  Returns:
    instance: a dict consisting of concept mentions and other info.
  """
  if stop_words is None:
    stop_words = stopwords.words("english")
  labels = [0] * len(lemmas)  # 0 means not-mentioned, 1 means mentioned.
  concept_mentions = []
  for span_length in range(max_len, 0, -1):
    for start in range(len(lemmas) - span_length + 1):
      end = start + span_length
      if disable_overlap and any(labels[start:end]):
        # At least one token was already labeled.
        continue
      span = " ".join(lemmas[start:end])
      if span_length == 1 and span in stop_words:
        # Skip single stop-word
        continue
      concept_id = concept_vocab.get(span.lower(), -1)
      if concept_id >= 0:
        concept_mentions.append({
            "start": start,
            "end": end,
            "mention": span.lower(),
            "concept_id": concept_id,
        })
        labels = labels[:start] + [1] * span_length + labels[end:]
  return concept_mentions


def load_concept_vocab(path_to_vocab):
  """Loads concept vocabulary from a file at the input path.

  Args:
    path_to_vocab: The path to the vocab text file. Each line is a concept.

  Returns:
    concept_vocab: a concept-to-id dict.
  """
  concept_vocab = {}
  print("Loading:", path_to_vocab, "...")
  with tf.gfile.Open(path_to_vocab, "r") as gfo:
    lines = gfo.read().split("\n")
  for word in lines:
    if word.lower() not in concept_vocab:
      concept_vocab[word.lower()] = len(concept_vocab)
  return concept_vocab


def do_preprocess(tokenizer):
  """Loads and processes the data."""
  # Read concepts.
  tf.logging.info("Reading entities.")
  entity2id = load_concept_vocab(FLAGS.entity_file)
  entity2name = {concept: concept for (concept, _) in entity2id.items()}
  # print("print # concepts:", len(entity2id))
  tf.logging.info("# concepts: %d", len(entity2id))

  if not tf.gfile.Exists(FLAGS.multihop_output_dir):
    tf.gfile.MakeDirs(FLAGS.multihop_output_dir)
  # Read paragraphs, mentions and entities.

  mentions = []
  ent_rows, ent_cols, ent_vals = [], [], []
  ent2fact_rows, ent2fact_cols, ent2fact_vals = [], [], []
  fact2ent_rows, fact2ent_cols, fact2ent_vals = [], [], []
  ent2num_facts = collections.defaultdict(lambda: 0)
  mention2text = {}
  total_sub_paras = [0]
  all_sub_paras = []
  num_skipped_mentions = 0.

  tf.logging.info("Reading paragraphs from %s", FLAGS.wiki_file)
  with tf.gfile.Open(FLAGS.wiki_file) as f:
    lines = f.read().split("\n")
  if not lines[-1]:
    lines = lines[:-1]
  fact2entity = []
  for ii, line in tqdm(
      enumerate(lines[:]), total=len(lines), desc="preprocessing lines"):
    if ii == FLAGS.max_total_paragraphs:
      tf.logging.info("Processed maximum number of paragraphs, breaking.")
      break
    orig_para = json.loads(line.strip())
    if orig_para["kb_id"].lower() not in entity2id:
      tf.logging.info("%s not in entities. Skipping %s para",
                      orig_para["kb_id"], orig_para["title"])
      continue
    sub_para_objs = index_util.get_sub_paras(orig_para, tokenizer,
                                             FLAGS.max_seq_length,
                                             FLAGS.doc_stride, total_sub_paras)
    assert len(sub_para_objs) == 1  # each doc is a single paragraph.
    for para_obj in sub_para_objs:
      # Add mentions from this paragraph.
      local2global = {}
      # title_entity_mention = None
      assert para_obj["id"] == len(fact2entity)
      fact2entity.append([])
      for im, mention in enumerate(
          para_obj["mentions"][:FLAGS.max_mentions_per_doc]):
        if mention["kb_id"].lower() not in entity2id:
          # tf.logging.info("%s not in entities. Skipping mention %s",
          #                 mention["kb_id"], mention["text"])
          num_skipped_mentions += 1
          continue
        mention2text[len(mentions)] = mention["text"]
        # Map the index of a local mention to the global mention id.
        local2global[im] = len(mentions)
        # if mention["kb_id"] == orig_para["kb_id"]:
        #   title_entity_mention = len(mentions)
        # The shape of 'mentions.npy' is thus #mentions * 4.
        mentions.append((entity2id[mention["kb_id"].lower()], para_obj["id"],
                         mention["start_token"], mention["end_token"]))
        # fact_id 2 entity_ids
        fact2entity[para_obj["id"]].append(entity2id[mention["kb_id"].lower()])
      # Note: each pair of mention in this paragraph is recorded.
      local_mentioned_entity_ids = set(
          [mentions[gm][0] for _, gm in local2global.items()])

      # Creating sparse entries for entity2mention matrix.
      for _, gm in local2global.items():
        for cur_entity_id in local_mentioned_entity_ids:
          if cur_entity_id != gm:
            ent_rows.append(cur_entity_id)
            ent_cols.append(gm)
            ent_vals.append(1.)

      # Creating sparse entries for entity2fact matrix.
      for cur_entity_id in local_mentioned_entity_ids:
        fact2ent_rows.append(ii)  # doc_id
        fact2ent_cols.append(cur_entity_id)
        fact2ent_vals.append(1.)
        # Note: Use tf-idf to limit this in the future.
        if ent2num_facts[cur_entity_id] >= FLAGS.max_facts_per_entity:
          # We want to limit the number of the facts in ent2fact.
          # Otherwise, the init_fact will be huge.
          continue
        ent2fact_rows.append(cur_entity_id)
        ent2fact_cols.append(ii)  # doc_id
        ent2fact_vals.append(1.)
        ent2num_facts[cur_entity_id] += 1

      all_sub_paras.append(para_obj["tokens"])
    assert len(all_sub_paras) == total_sub_paras[0], (len(all_sub_paras),
                                                      total_sub_paras)

  tf.logging.info("Num paragraphs = %d, Num mentions = %d", total_sub_paras[0],
                  len(mentions))

  tf.logging.info("Saving mention2entity coreference map.")
  search_utils.write_to_checkpoint(
      "coref", np.array([m[0] for m in mentions], dtype=np.int32), tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "coref.npz"))

  tf.logging.info("Creating ent2men matrix with %d entries.", len(ent_vals))
  # Fill a zero-inited sparse matrix for entity2mention.
  sp_entity2mention = sp.csr_matrix((ent_vals, (ent_rows, ent_cols)),
                                    shape=[len(entity2id),
                                           len(mentions)])
  tf.logging.info("Num nonzero in e2m = %d", sp_entity2mention.getnnz())
  tf.logging.info("Saving as ragged e2m tensor %s.",
                  str(sp_entity2mention.shape))
  search_utils.write_ragged_to_checkpoint(
      "ent2ment", sp_entity2mention,
      os.path.join(FLAGS.multihop_output_dir, "ent2ment.npz"))
  tf.logging.info("Saving mentions metadata.")
  np.save(
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "mentions.npy"), "w"),
      np.array(mentions, dtype=np.int64))
  json.dump(
      mention2text,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "mention2text.json"), "w"))
  tf.logging.info("Saving entities metadata.")

  assert len(lines) == len(all_sub_paras)
  num_facts = len(all_sub_paras)
  # Fill a zero-inited sparse matrix for entity2fact.
  sp_entity2fact = sp.csr_matrix(
      (ent2fact_vals, (ent2fact_rows, ent2fact_cols)),
      shape=[len(entity2id), num_facts])
  tf.logging.info("Num nonzero in e2f = %d", sp_entity2fact.getnnz())
  tf.logging.info("Saving as ragged e2f tensor %s.", str(sp_entity2fact.shape))
  search_utils.write_ragged_to_checkpoint(
      "ent2fact", sp_entity2fact,
      os.path.join(FLAGS.multihop_output_dir,
                   "ent2fact_%d.npz" % FLAGS.max_facts_per_entity))

  # Fill a zero-inited sparse matrix for fact2entity.
  sp_fact2entity = sp.csr_matrix(
      (ent2fact_vals, (ent2fact_cols, ent2fact_rows)),  # Transpose.
      shape=[num_facts, len(entity2id)])
  tf.logging.info("Num nonzero in f2e = %d", sp_fact2entity.getnnz())
  tf.logging.info("Saving as ragged f2e tensor %s.", str(sp_fact2entity.shape))
  search_utils.write_ragged_to_checkpoint(
      "fact2ent", sp_fact2entity,
      os.path.join(FLAGS.multihop_output_dir, "fact_coref.npz"))

  json.dump([entity2id, entity2name],
            tf.gfile.Open(
                os.path.join(FLAGS.multihop_output_dir, "entities.json"), "w"))
  tf.logging.info("Saving split paragraphs.")
  json.dump(
      all_sub_paras,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "subparas.json"), "w"))

  # Store entity tokens.
  tf.logging.info("Processing entities.")
  entity_ids = np.zeros((len(entity2id), FLAGS.max_entity_length),
                        dtype=np.int32)
  entity_mask = np.zeros((len(entity2id), FLAGS.max_entity_length),
                         dtype=np.float32)
  num_exceed_len = 0.
  for entity in tqdm(entity2id):
    ei = entity2id[entity]
    entity_tokens = tokenizer.tokenize(entity2name[entity])
    entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
    if len(entity_token_ids) > FLAGS.max_entity_length:
      num_exceed_len += 1
      entity_token_ids = entity_token_ids[:FLAGS.max_entity_length]
    entity_ids[ei, :len(entity_token_ids)] = entity_token_ids
    entity_mask[ei, :len(entity_token_ids)] = 1.
  tf.logging.info("Saving %d entity ids. %d exceed max-length of %d.",
                  len(entity2id), num_exceed_len, FLAGS.max_entity_length)
  search_utils.write_to_checkpoint(
      "entity_ids", entity_ids, tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "entity_ids"))
  search_utils.write_to_checkpoint(
      "entity_mask", entity_mask, tf.float32,
      os.path.join(FLAGS.multihop_output_dir, "entity_mask"))


def do_embed(tokenizer, do_embed_mentions, do_embed_facts, embed_prefix):
  """Gets mention embeddings from BERT."""
  # Start Embedding.
  bert_ckpt = os.path.join(FLAGS.bert_ckpt_dir, FLAGS.ckpt_name)
  with tf.gfile.Open(
      os.path.join(FLAGS.multihop_output_dir, "mentions.npy"), "rb") as f:
    mentions = np.load(f)
  with tf.gfile.Open(os.path.join(FLAGS.multihop_output_dir,
                                  "subparas.json")) as f:
    all_sub_paras = json.load(f)

  if do_embed_mentions:
    tf.logging.info("Computing embeddings for %d mentions over %d paras.",
                    len(mentions), len(all_sub_paras))
    shard_size = len(mentions) // FLAGS.num_shards
    # Note that some FLAGS args are passed to the init function here.
    tf.logging.info("Loading BERT from %s", bert_ckpt)
    bert_predictor = bert_utils_v2.BERTPredictor(tokenizer, bert_ckpt)
    if FLAGS.my_shard is None:
      shard_range = range(FLAGS.num_shards + 1)
    else:
      shard_range = [FLAGS.my_shard]
    for ns in shard_range:
      min_ = ns * shard_size
      max_ = (ns + 1) * shard_size
      if min_ >= len(mentions):
        break
      if max_ > len(mentions):
        max_ = len(mentions)
      min_subp = mentions[min_][1]  # the start sentence id
      max_subp = mentions[max_ - 1][1]  # the end sentence id
      tf.logging.info("Processing shard %d of %d mentions and %d paras.", ns,
                      max_ - min_, max_subp - min_subp + 1)
      # Get the embeddings of all the sentences.
      # Note: this is always the last layer of the BERT.
      para_emb = bert_predictor.get_doc_embeddings(
          all_sub_paras[min_subp:max_subp + 1])
      assert para_emb.shape[2] == 2 * FLAGS.projection_dim
      mention_emb = np.empty((max_ - min_, 2 * bert_predictor.emb_dim),
                             dtype=np.float32)
      for im, mention in enumerate(mentions[min_:max_]):
        # mention[1] is the sentence id
        # mention[2/3] is the start/end index of the token
        mention_emb[im, :] = np.concatenate([
            para_emb[mention[1] - min_subp, mention[2], :FLAGS.projection_dim],
            para_emb[mention[1] - min_subp, mention[3],
                     FLAGS.projection_dim:2 * FLAGS.projection_dim]
        ])
      del para_emb
      tf.logging.info("Saving %d mention features to tensorflow checkpoint.",
                      mention_emb.shape[0])
      with tf.device("/cpu:0"):
        search_utils.write_to_checkpoint(
            "db_emb_%d" % ns, mention_emb, tf.float32,
            os.path.join(FLAGS.multihop_output_dir,
                         "%s_mention_feats_%d" % (embed_prefix, ns)))
  if do_embed_facts:
    tf.logging.info("Computing embeddings for %d facts with %d mentions.",
                    len(all_sub_paras), len(mentions))
    fact2mentions = collections.defaultdict(list)
    for m in mentions:
      fact2mentions[int(m[1])].append(m)
    shard_size = len(all_sub_paras) // FLAGS.num_shards
    # Note that some FLAGS args are passed to the init function here.
    bert_predictor = bert_utils_v2.BERTPredictor(tokenizer, bert_ckpt)
    if FLAGS.my_shard is None:
      shard_range = range(FLAGS.num_shards + 1)
    else:
      shard_range = [FLAGS.my_shard]
    for ns in shard_range:
      min_ = ns * shard_size
      max_ = (ns + 1) * shard_size
      if min_ >= len(all_sub_paras):
        break
      if max_ > len(all_sub_paras):
        max_ = len(all_sub_paras)
      min_subp = min_  # the start sentence id
      max_subp = max_ - 1  # the end sentence id

      tf.logging.info("Processing shard %d of %d facts and %d paras.", ns,
                      max_ - min_, max_subp - min_subp + 1)
      # Get the embeddings of all the sentences.
      para_emb = bert_predictor.get_doc_embeddings(
          all_sub_paras[min_subp:max_subp + 1])
      assert para_emb.shape[2] == 2 * FLAGS.projection_dim
      fact_emb = np.empty((max_ - min_, 2 * bert_predictor.emb_dim),
                          dtype=np.float32)
      for ii, _ in enumerate(all_sub_paras[min_:max_]):
        fact_id = min_ + ii
        local_mentions = fact2mentions[fact_id]

        mention_agg_emb = np.empty(
            (len(local_mentions), 2 * bert_predictor.emb_dim), dtype=np.float32)
        for jj, m in enumerate(local_mentions):
          mention_agg_emb[jj, :] = np.concatenate([
              para_emb[ii, m[2], :FLAGS.projection_dim],
              para_emb[ii, m[3], FLAGS.projection_dim:2 * FLAGS.projection_dim]
          ])

        fact_emb[ii, :] = np.mean(mention_agg_emb, axis=0)
      del para_emb
      tf.logging.info("Saving %d fact features to tensorflow checkpoint.",
                      fact_emb.shape[0])
      with tf.device("/cpu:0"):
        search_utils.write_to_checkpoint(
            "fact_db_emb_%d" % ns, fact_emb, tf.float32,
            os.path.join(FLAGS.multihop_output_dir,
                         "%s_fact_feats_%d" % (embed_prefix, ns)))


def do_combine(do_embed_mentions, do_embed_facts, embed_prefix):
  """Combines sharded DB into one single file."""
  if FLAGS.shards_to_combine is None:
    shard_range = range(FLAGS.num_shards + 1)
  else:
    shard_range = range(FLAGS.shards_to_combine)
  if do_embed_mentions:
    db_emb_str = "db_emb"
  elif do_embed_facts:
    db_emb_str = "fact_db_emb"
  with tf.device("/cpu:0"):
    all_db = []
    for i in shard_range:
      if do_embed_mentions:
        embed_str = "%s_mention_feats_%d" % (embed_prefix, i)
      elif do_embed_facts:
        embed_str = "%s_fact_feats_%d" % (embed_prefix, i)
      else:
        tf.logging.info("Error choice")
        return
      ckpt_path = os.path.join(FLAGS.multihop_output_dir, embed_str)
      if not tf.gfile.Exists(ckpt_path + ".meta"):
        tf.logging.info("%s does not exist", ckpt_path)
        continue
      reader = tf.train.NewCheckpointReader(ckpt_path)
      var_to_shape_map = reader.get_variable_to_shape_map()
      tf.logging.info("Reading %s from %s with shape %s",
                      db_emb_str + "_%d" % i, ckpt_path,
                      str(var_to_shape_map[db_emb_str + "_%d" % i]))
      tf_db = search_utils.load_database(
          db_emb_str + "_%d" % i, var_to_shape_map[db_emb_str + "_%d" % i],
          ckpt_path)
      all_db.append(tf_db)
    tf.logging.info("Reading all variables.")
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    np_db = session.run(all_db)
    tf.logging.info("Concatenating and storing.")
    np_db = np.concatenate(np_db, axis=0)
    if do_embed_mentions:
      embed_feats_str = "%s_mention_feats" % embed_prefix
    elif do_embed_facts:
      embed_feats_str = "%s_fact_feats" % embed_prefix
    search_utils.write_to_checkpoint(
        db_emb_str, np_db, tf.float32,
        os.path.join(FLAGS.multihop_output_dir, embed_feats_str))


def main(_):
  # Initialize tokenizer.
  if FLAGS.tokenizer_type == "bert_tokenization":
    tokenizer = bert_tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)
  elif FLAGS.tokenizer_type == "albert_tokenization":
    tokenizer = albert_tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=False,
        spm_model_file=FLAGS.tokenizer_model_file)
  if FLAGS.do_preprocess:
    do_preprocess(tokenizer)
  elif FLAGS.do_embed:
    do_embed(
        tokenizer,
        do_embed_mentions=FLAGS.do_embed_mentions,
        do_embed_facts=FLAGS.do_embed_facts,
        embed_prefix=FLAGS.embed_prefix)
  elif FLAGS.do_combine:
    if FLAGS.do_embed_mentions:
      do_combine(
          do_embed_mentions=True,
          do_embed_facts=False,
          embed_prefix=FLAGS.embed_prefix)
    if FLAGS.do_embed_facts:
      do_combine(
          do_embed_mentions=False,
          do_embed_facts=True,
          embed_prefix=FLAGS.embed_prefix)


if __name__ == "__main__":
  app.run(main)
