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
r"""Links the concepts in the questions and reformat the dataset to OpenCSR.

Example usage:
ORI_DATA_DIR=/path/to/datasets/CSQA/; \
DATA_DIR=/path/to/nscskg_data/xm_drfact_output_bert200/; \
VOCAB=/path/to/nscskg_data/gkb_best.vocab.txt \

SPLIT=csqa_train NUM_CHOICES=1;  \
link_questions \
  --csqa_file $ORI_DATA_DIR/${SPLIT}_processed.jsonl \
  --output_file $DATA_DIR/linked_${SPLIT}.jsonl \
  --indexed_concept_file  ${VOCAB} \
  --do_filtering ${NUM_CHOICES} \
  --alsologtostderr

SPLIT=csqa_dev; \
link_questions \
  --csqa_file $ORI_DATA_DIR/${SPLIT}_processed.jsonl \
  --output_file $DATA_DIR/linked_${SPLIT}.jsonl \
  --indexed_concept_file ${VOCAB} \
  --disable_overlap --alsologtostderr

"""
import itertools
import json

from absl import app
from absl import flags
from absl import logging
from language.google.drfact import index_corpus
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("csqa_file", None, "Path to dataset file.")

flags.DEFINE_string("index_data_dir", None,
                    "Path to Entity co-occurrence directory.")

flags.DEFINE_string("vocab_file", None, "Path to vocab for tokenizer.")
flags.DEFINE_string("indexed_concept_file", None, "Path to indexed vocab.")

flags.DEFINE_string("output_file", None, "Path to Output file.")
flags.DEFINE_boolean("tfidf", False, "Whether to use tfidf for linking.")
flags.DEFINE_boolean("disable_overlap", None,
                     "Whether to use tfidf for linking.")
flags.DEFINE_integer(
    "do_filtering", -1,
    "Whether to ignore the examples where at least one choice"
    " is not in the vocab.")

entity2id = None


def remove_intersection(dict_of_sets):
  """Removes the intersection of any two sets in a list."""
  for i, j in itertools.combinations(list(dict_of_sets.keys()), 2):
    set_i = dict_of_sets[i]
    set_j = dict_of_sets[j]
    dict_of_sets[i] = set_i - set_j
    dict_of_sets[j] = set_j - set_i
  return dict_of_sets


def main(_):
  global entity2id
  logging.set_verbosity(logging.INFO)

  logging.info("Reading CSQA(-formatted) data...")
  with tf.gfile.Open(FLAGS.csqa_file) as f:
    jsonlines = f.read().split("\n")
  data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]
  logging.info("Done.")

  logging.info("Entity linking %d questions...", len(data))

  all_questions = []
  entity2id = index_corpus.load_concept_vocab(
      FLAGS.indexed_concept_file)
  linked_question_entities = []

  for item in tqdm(data, desc="Matching concepts in the questions."):
    concept_mentions = index_corpus.simple_match(
        lemmas=item["question"]["lemmas"],
        concept_vocab=entity2id,
        max_len=4,
        disable_overlap=True)  # Note: we want to limit size of init facts.
    qry_concept_set = set()
    qry_concept_list = []
    for m in concept_mentions:
      c = m["mention"].lower()
      if c not in qry_concept_set:
        qry_concept_set.add(c)
        qry_concept_list.append({"kb_id": c, "name": c})
    linked_question_entities.append(qry_concept_list)

  num_empty_questions = 0
  num_empty_choices = 0
  num_empty_answers = 0

  for ii, item in tqdm(enumerate(data), desc="Processing", total=len(data)):
    # pylint: disable=g-complex-comprehension
    if item["answerKey"] in "ABCDE":
      truth_choice = ord(item["answerKey"]) - ord("A")  # e.g., A-->0
    elif item["answerKey"] in "12345":
      truth_choice = int(item["answerKey"]) - 1
    choices = item["question"]["choices"]
    assert choices[truth_choice]["label"] == item["answerKey"]
    correct_answer = choices[truth_choice]["text"].lower()

    # Check the mentioned concepts in each choice.
    choice2concepts = {}
    for c in choices:
      mentioned_concepts = []
      for m in index_corpus.simple_match(
          c["lemmas"],
          concept_vocab=entity2id,
          max_len=4,
          disable_overlap=FLAGS.disable_overlap):
        mentioned_concepts.append(m["mention"])
      choice2concepts[c["text"].lower()] = set(mentioned_concepts)

    choice2concepts = remove_intersection(choice2concepts)

    non_empty_choices = sum([bool(co) for _, co in choice2concepts.items()])
    num_empty_choices += len(choices) - non_empty_choices
    if not linked_question_entities[ii]:
      num_empty_questions += 1
      continue
    if not choice2concepts[correct_answer]:
      # the correct answer does not contain any concepts, skip it.
      num_empty_answers += 1
      continue

    if FLAGS.do_filtering > 0:
      if non_empty_choices < FLAGS.do_filtering:
        continue
    choice2concepts = {
        k: sorted(list(v), key=lambda x: -len(x))  # Sort concepts by len.
        for k, v in choice2concepts.items()
    }
    sup_facts = choice2concepts[correct_answer]
    all_questions.append({
        "question": item["question"]["stem"],
        "entities": linked_question_entities[ii],
        "answer": correct_answer,
        "_id": item["id"],
        "level": "N/A",  # hotpotQA-specific keys: hard/medium/easy
        "type": "N/A",  # hotpotQA-specific keys: comparison, bridge, etc.
        "supporting_facts": [{
            "kb_id": c,
            "name": c
        } for c in sup_facts],  # hotpotQA-specific keys
        "choice2concepts": choice2concepts,
    })

  with tf.gfile.Open(FLAGS.output_file, "w") as f_out:
    logging.info("Writing questions to output file...%s", f_out.name)
    logging.info("Number of questions %d", len(all_questions))
    f_out.write("\n".join(json.dumps(q) for q in all_questions))

  logging.info("===============================================")
  logging.info("%d questions without entities (out of %d)", num_empty_questions,
               len(data))
  logging.info("%d answers not IN entities (out of %d)", num_empty_answers,
               len(data))
  logging.info("%d choices not IN entities (out of %d)", num_empty_choices,
               5 * len(data))


if __name__ == "__main__":
  app.run(main)
