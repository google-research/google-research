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

"""Tests for language.question_answering.bert_joint.run_nq."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import os
import tempfile
import tensorflow as tf

from fat.fat_bert_nq import nq_data_utils
from fat.fat_bert_nq import run_nq

flags = tf.flags
FLAGS = flags.FLAGS


class RunNqTest(tf.test.TestCase):

  VOCAB_TOKENS = [
      "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[Q]", "the", "docum", "##ent",
      "question", "[ContextId=-1]", "[NoLongAnswer]", "[ContextId=0]",
      "[Paragraph=1]", "[ContextId=1]", "[Paragraph=2]"
  ]

  def write_examples(self, examples):
    tf.gfile.MakeDirs(self.test_tmpdir)
    path = os.path.join(self.test_tmpdir, "nq-unittest.jsonl.gz")
    with gzip.GzipFile(fileobj=tf.gfile.Open(path, "w")) as output_file:
      for e in examples:
        output_file.write((json.dumps(e) + "\n").encode("utf-8"))

  def make_tf_examples(self, example, is_training):
    passages = []
    spans = []
    token_maps = []
    tf_example_creator = run_nq.CreateTFExampleFn(is_training=is_training)
    for record in list(tf_example_creator.process(example)):
      tfexample = tf.train.Example()
      tfexample.ParseFromString(record)
      tokens = [
          self.VOCAB_TOKENS[x]
          for x in (tfexample.features.feature["input_ids"].int64_list.value)
      ]
      passages.append(" ".join(tokens).replace(" ##", ""))
      if is_training:
        start = tfexample.features.feature["start_positions"].int64_list.value[
            0]
        end = tfexample.features.feature["end_positions"].int64_list.value[0]
        spans.append(" ".join(tokens[start:end + 1]).replace(" ##", ""))
      else:
        token_maps.append(
            tfexample.features.feature["token_map"].int64_list.value)

    return passages, spans, token_maps

  def setUp(self):
    super(RunNqTest, self).setUp()
    self.test_tmpdir = tf.test.get_temp_dir()
    FLAGS.include_unknowns = 1.0
    FLAGS.max_seq_length = 20
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      vocab_writer.write(("\n".join(self.VOCAB_TOKENS) + "\n").encode("utf-8"))
      vocab_writer.flush()
      FLAGS.vocab_file = vocab_writer.name
    self.test_file = os.path.join(self.test_tmpdir, "nq-unittest.jsonl.gz")

  def test_minimal_examples(self):
    num_examples = 20
    minimal_example = {
        "annotations": [],
        "long_answer_candidates": [],
        "question_text": "",
        "document_url": "",
        "document_title": "",
        "example_id": 1
    }
    self.write_examples([minimal_example] * num_examples)
    output_examples = nq_data_utils.get_nq_examples(self.test_file)
    self.assertEqual(num_examples, len(list(output_examples)))

  def test_example_metadata(self):
    example = {
        "annotations": [],
        "long_answer_candidates": [],
        "question_text": "test_q",
        "document_url": "test_url",
        "document_title": "test_title",
        "example_id": 10
    }
    self.write_examples([example])
    output_example = next(nq_data_utils.get_nq_examples(self.test_file))
    self.assertEqual(output_example["name"], "test_title")
    self.assertEqual(output_example["id"], "10")
    self.assertEqual(output_example["questions"][0]["input_text"], "test_q")
    self.assertEqual(output_example["answers"][0]["input_text"], "long")

  def test_multi_candidate_document(self):
    example = {
        "annotations": [{
            "long_answer": {
                "candidate_index": 1,
                "start_token": 0,
                "end_token": 3,
                "entity_map": {}
            },
            "short_answers": [{
                "start_token": 2,
                "end_token": 3,
                "entity_map": {}
            }],
            "yes_no_answer": "NONE"
        }],
        "long_answer_candidates": [{
            "start_token": 0,
            "end_token": 3,
            "top_level": True,
            "entity_map": {}
        }, {
            "start_token": 0,
            "end_token": 3,
            "top_level": True
        }],
        "document_tokens": [{
            "token": "<P>",
            "html_token": True
        }, {
            "token": "the",
            "html_token": False
        }, {
            "token": "document",
            "html_token": False
        }],
        "question_text": "the question",
        "document_url": "",
        "document_title": "",
        "example_id": 1
    }
    self.write_examples([example])

    # The document in this case should be a single string with all contexts.
    output_example = next(nq_data_utils.get_nq_examples(self.test_file))
    self.assertEqual(
        "[ContextId=-1] [NoLongAnswer] [ContextId=0] [Paragraph=1] the document "
        "[ContextId=1] [Paragraph=2] the document", output_example["contexts"])

    passages, spans, _ = self.make_tf_examples(output_example, is_training=True)
    self.assertEqual([
        "[CLS] [Q] the question [SEP] [ContextId=-1] [NoLongAnswer] "
        "[ContextId=0] [Paragraph=1] the docum [SEP] [SEP] [PAD] [PAD] "
        "[PAD] [PAD] [PAD] [PAD] [PAD]",
        "[CLS] [Q] the question [SEP]ent [ContextId=1] [Paragraph=2] the document [SEP] "
        "[SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
    ], passages)
    self.assertEqual(["[CLS]", "document"], spans)

    passages, _, tok_maps = self.make_tf_examples(
        output_example, is_training=False)
    self.assertEqual([
        "[CLS] [Q] the question [SEP] [ContextId=-1] [NoLongAnswer] "
        "[ContextId=0] [Paragraph=1] the docum [SEP] [SEP] [PAD] [PAD] "
        "[PAD] [PAD] [PAD] [PAD] [PAD]",
        "[CLS] [Q] the question [SEP]ent [ContextId=1] [Paragraph=2] the document [SEP] "
        "[SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
    ], passages)
    self.assertEqual(
        [[-1] * 9 + [1, 2] + [-1]*9, [-1] * 5 + [2, -1, -1, 1, 2, 2] + [-1]*9],
        tok_maps)


if __name__ == "__main__":
  tf.test.main()
