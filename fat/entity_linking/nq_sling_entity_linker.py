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

"""This file performs Sling based entity linking on NQ.

The file iterates through entire train and dev set of NQ.
For every example it does entity linking on long answer candidates,
  annotated long and short answer and questiopn.
Every paragraph in the dataset is augmented with an entity map from
  every token to it's entity id.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gzip
import json
import os
import sling
import sling.flags as flags
import sling.task.entity as entity
import sling.task.workflow as workflow
import tensorflow as tf

# Calling these 'args' to avoid conflicts with sling flags
args = tf.flags
ARGS = args.FLAGS
args.DEFINE_string("nq_dir", "", "NQ data location")
args.DEFINE_string("files_dir", "", "Preprocess files location")
args.DEFINE_string("output_data_dir", "", "Location to write augmented data to")
args.DEFINE_boolean("annotate_candidates", True, "Flag to annotate candidates")
args.DEFINE_boolean("annotate_long_answers", True,
                    "Flag to annotate long answer")
args.DEFINE_boolean("annotate_short_answers", True,
                    "Flag to annotate short answers")
args.DEFINE_boolean("annotate_question", True, "Flag to annotate questions")


def extract_and_tokenize_text(item, tokens):
  """Extracts the tokens in passage, tokenizes them using sling tokenizer."""
  start_token = item["start_token"]
  end_token = item["end_token"]
  if start_token >= 0 and end_token >= 0:
    non_html_tokens = [
        x
        for x in tokens[start_token:end_token]
        if not x["html_token"]
    ]
    answer = " ".join([x["token"] for x in non_html_tokens])
    answer_map = [idx for idx, x in enumerate(non_html_tokens)]
    doc = sling.tokenize(str(answer))
    return answer, answer_map, doc
  return "", [], None


def is_sling_entity(item):
  return isinstance(
      item[0]) == sling.Frame and "id" in item[0] and item[0]["id"].startswith(
          "Q")


def prepare_sling_input_corpus(nq_data, sling_input_corpus):
  """Parse each paragrapgh in NQ (LA candidate, LA, SA, question).

     Prepare a sling corpus to do entity linking.

  Args:
    nq_data: A python dictionary containint NQ data of 1 train/dev shard
    sling_input_corpus: A filename string to write the sling format documents
        into
  """

  corpus = sling.RecordWriter(sling_input_corpus)
  for i in nq_data.keys():
    tokens = nq_data[i]["document_tokens"]
    if ARGS.annotate_candidates:
      for idx, la_cand in enumerate(nq_data[i]["long_answer_candidates"]):
        answer, answer_map, doc = extract_and_tokenize_text(la_cand, tokens)
        if answer:
          nq_data[i]["long_answer_candidates"][idx]["text_answer"] = answer
          nq_data[i]["long_answer_candidates"][idx]["answer_map"] = answer_map
          key = i + "|candidate|" + str(idx) + "|i"
          corpus.write(key, doc.frame.data(binary=True))
    if ARGS.annotate_short_answers:
      for idx, ann in enumerate(nq_data[i]["annotations"]):
        short_ans = ann["short_answers"]
        if not short_ans:
          continue
        for sid in range(len(short_ans)):
          ans = short_ans[sid]
          answer, answer_map, doc = extract_and_tokenize_text(ans, tokens)
          if answer:
            nq_data[i]["annotations"][idx]["short_answers"][sid][
                "text_answer"] = answer
            nq_data[i]["annotations"][idx]["short_answers"][sid][
                "answer_map"] = answer_map
            key = i + "|annotated_short_answer|" + str(idx) + "|" + str(sid)
            corpus.write(key, doc.frame.data(binary=True))
    if ARGS.annotate_long_answers:
      for idx, ann in enumerate(nq_data[i]["annotations"]):
        long_ans = ann["long_answer"]
        answer, answer_map, doc = extract_and_tokenize_text(long_ans, tokens)
        if answer:
          nq_data[i]["annotations"][idx]["long_answer"]["text_answer"] = answer
          nq_data[i]["annotations"][idx]["long_answer"][
              "answer_map"] = answer_map
          key = i + "|annotated_long_answer|" + str(idx) + "|i"
          corpus.write(key, doc.frame.data(binary=True))
    if ARGS.annotate_question:
      doc = sling.tokenize(str(nq_data[i]["question_text"]))
      key = i + "|question|i|i"
      corpus.write(key, doc.frame.data(binary=True))
  corpus.close()


def sling_entity_link(sling_input_corpus, sling_output_corpus):
  """Does sling entity linking and created linked output corpus."""
  labeler = entity.EntityWorkflow("wiki-label")
  unannotated = labeler.wf.resource(
      sling_input_corpus, format="records/document")
  annotated = labeler.wf.resource(
      sling_output_corpus, format="records/document")
  labeler.label_documents(indocs=unannotated, outdocs=annotated)
  workflow.run(labeler.wf)


def extract_entity_mentions(nq_data, labelled_record):
  """Parse ourput corpus and create map from tokens to entity ids.

  Args:
    nq_data: A python dictionary containint NQ data of 1 train/dev shard
    labelled_record: Sling output document with labelled paragraphs

  Returns:
    nq_data: Original object augmented with entity maps
  """
  recin = sling.RecordReader(labelled_record)
  commons = sling.Store()
  docschema = sling.DocumentSchema(commons)
  commons.freeze()
  cnt = 1

  for key, value in recin:
    store = sling.Store(commons)
    doc = sling.Document(store.parse(value), store, docschema)
    index, ans_type, idx, ans_id = key.decode("utf-8").split("|")
    cnt += 1
    entity_map = {}

    # Parse entity mentions labelled by sling
    for m in doc.mentions:
      e = [i["is"] for i in m.evokes()]
      if not e:
        continue
      if is_sling_entity(e):
        e_val = e[0]["id"]
        if m.begin in entity_map:
          entity_map[m.begin].append((m.end, e_val))
        else:
          entity_map[m.begin] = [(m.end, e_val)]

    if ans_type == "annotated_long_answer":
      nq_data[index]["annotations"][int(
          idx)]["long_answer"]["entity_map"] = entity_map
    elif ans_type == "question":
      nq_data[index]["question_entity_map"] = entity_map
    elif ans_type == "annotated_short_answer":
      nq_data[index]["annotations"][int(idx)]["short_answers"][int(
          ans_id)]["entity_map"] = entity_map
    else:
      nq_data[index]["long_answer_candidates"][int(
          idx)]["entity_map"] = entity_map
  return nq_data


def extract_nq_data(nq_file):
  """Read nq shard file and return dict of nq_data."""
  fp = gzip.GzipFile(fileobj=tf.gfile.Open(nq_file, "rb"))
  lines = fp.readlines()
  data = {}
  counter = 0
  for line in lines:
    data[str(counter)] = json.loads(line.decode("utf-8"))
    tok = []
    for j in data[str(counter)]["document_tokens"]:
      tok.append(j["token"])
    data[str(counter)]["full_document_long"] = " ".join(tok)
    counter += 1
  return data


def get_shard(mode, task_id, shard_id):
  return "nq-%s-%02d%02d" % (mode, task_id, shard_id)


def get_full_filename(data_dir, mode, task_id, shard_id):
  return os.path.join(
      data_dir, "%s/%s.jsonl.gz" % (mode, get_shard(mode, task_id, shard_id)))


def get_examples(data_dir, mode, task_id, shard_id):
  """Reads NQ data, does sling entity linking and returns augmented data."""
  file_path = get_full_filename(data_dir, mode, task_id, shard_id)
  tf.logging.info("Reading file: %d" % (file_path))
  if not os.path.exists(file_path):
    return None
  nq_data = extract_nq_data(file_path)
  tf.logging.info("NQ data Size: " + str(len(nq_data.keys())))

  tf.logging.info("Preparing sling corpus: ")
  sling_input_corpus = os.path.join(ARGS.files_dir, "sling_input_corpus.rec")
  sling_output_corpus = os.path.join(ARGS.files_dir, "nq_labelled_output.rec")
  prepare_sling_input_corpus(nq_data, sling_input_corpus)

  tf.logging.info("Performing Sling NER Labeling")
  sling_entity_link(sling_input_corpus, sling_output_corpus)
  fact_extracted_data = extract_entity_mentions(nq_data, sling_output_corpus)

  return fact_extracted_data


def main(_):
  workflow.startup()
  max_tasks = {"train": 50, "dev": 5}
  max_shards = {"train": 6, "dev": 16}
  for mode in ["train", "dev"]:
    # Parse all shards in each mode
    # Currently sequentially, can be parallelized later
    for task_id in range(0, max_tasks[mode]):
      for shard_id in range(0, max_shards[mode]):
        nq_augmented_data = get_examples(ARGS.nq_dir, mode, task_id, shard_id)
        if nq_augmented_data is None:
          continue
        path = get_full_filename(ARGS.output_data_dir, mode, task_id, shard_id)
        with gzip.GzipFile(fileobj=tf.gfile.Open(path, "w")) as output_file:
          for idx in nq_augmented_data.keys():
            json_line = nq_augmented_data[idx]
            output_file.write(json.dumps(json_line) + "\n")
  workflow.shutdown()


if __name__ == "__main__":
  # This will fail if non-sling CMDLine Args are given.
  # Will modify sling separately to parse known args
  flags.parse()
  tf.app.run()
