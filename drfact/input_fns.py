# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Classes for processing different datasets into a common format."""

import collections
import json
import random

from bert import tokenization
from language.labs.drkit import input_fns as input_utils
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from tensorflow.contrib import data as contrib_data


class Example(object):
  """A single training/test example for QA."""

  def __init__(
      self,
      qas_id,
      question_text,
      subject_entity,  # The concepts mentioned by the question.
      answer_entity=None,  # The concept(s) in the correct choice
      choice2concepts=None,  # for evaluation
      correct_choice=None,  # for evaluation
      exclude_set=None):  # concepts in the question and wrong choices
    self.qas_id = qas_id
    self.question_text = question_text
    self.subject_entity = subject_entity
    self.answer_entity = answer_entity
    self.choice2concepts = choice2concepts
    self.correct_choice = correct_choice
    self.exclude_set = exclude_set

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    if self.answer_mention:
      s += ", answer_mention: %d" % self.answer_mention[0]
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               qas_id,
               qry_tokens,
               qry_input_ids,
               qry_input_mask,
               qry_entity_id,
               answer_entity=None,
               exclude_set=None):
    self.qas_id = qas_id
    self.qry_tokens = qry_tokens
    self.qry_input_ids = qry_input_ids
    self.qry_input_mask = qry_input_mask
    self.qry_entity_id = qry_entity_id
    self.answer_entity = answer_entity
    self.exclude_set = exclude_set


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training, has_bridge):
    self.filename = filename
    self.is_training = is_training
    self.has_bridge = has_bridge
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    # The feature object is actually of Example class.
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    features = collections.OrderedDict()
    features["qas_ids"] = create_bytes_feature(feature.qas_id)
    features["qry_input_ids"] = create_int_feature(feature.qry_input_ids)
    features["qry_input_mask"] = create_int_feature(feature.qry_input_mask)
    features["qry_entity_id"] = create_int_feature(feature.qry_entity_id)

    if self.is_training:
      features["answer_entities"] = create_int_feature(feature.answer_entity)
      features["exclude_set"] = create_int_feature(feature.exclude_set)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def convert_examples_to_features(examples, tokenizer, max_query_length,
                                 entity2id, output_fn):
  """Loads a data file into a list of `InputBatch`s."""
  for (example_index, example) in tqdm(
      enumerate(examples), desc="Converting Examples to Features"):
    (qry_input_ids, qry_input_mask,
     qry_tokens) = input_utils.get_tokens_and_mask(example.question_text,
                                                   tokenizer, max_query_length)
    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s", example.qas_id)
      tf.logging.info(
          "qry_tokens: %s",
          " ".join([tokenization.printable_text(x) for x in qry_tokens]))
      tf.logging.info("qry_input_ids: %s",
                      " ".join([str(x) for x in qry_input_ids]))
      tf.logging.info("qry_input_mask: %s",
                      " ".join([str(x) for x in qry_input_mask]))
      subj_concepts_str = ", ".join(
          ["%s %d" % (e, entity2id.get(e, -1)) for e in example.subject_entity])
      tf.logging.info("qry_entity_id: %s", subj_concepts_str)
      ans_concepts_str = ", ".join(["%d" % (e,) for e in example.answer_entity])
      tf.logging.info("num of answer entity: %d", len(example.answer_entity))
      tf.logging.info("answer entity: %s", ans_concepts_str)
      exc_concepts_str = ", ".join(
          ["%s %d" % (e, entity2id.get(e, -1)) for e in example.exclude_set])
      tf.logging.info("exclude_set entity: %s", exc_concepts_str)

    feature = InputFeatures(
        qas_id=example.qas_id.encode("utf-8"),
        qry_tokens=qry_tokens,
        qry_input_ids=qry_input_ids,
        qry_input_mask=qry_input_mask,
        qry_entity_id=[entity2id.get(ee, 0) for ee in example.subject_entity],
        answer_entity=example.answer_entity,
        exclude_set=[entity2id.get(ee, 0) for ee in example.exclude_set])

    # Run callback
    output_fn(feature)


def input_fn_builder(input_file, is_training, drop_remainder,
                     names_to_features):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, names_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class OpenCSRDataset(object):
  """Reads the open commonsense reasoning dataset and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    del subject_mention_probability

    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file, entity2id)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=False)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.VarLenFeature(tf.int64),
    }
    if is_training:
      names_to_features["answer_entities"] = tf.VarLenFeature(tf.int64)
      names_to_features["exclude_set"] = tf.VarLenFeature(tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=False,
        names_to_features=names_to_features)

  def read_examples(self, queries_file, entity2id):
    """Read a json file into a list of Example."""
    self.max_qry_answers = 0
    num_qrys_without_answer, num_qrys_without_all_answers = 0, 0
    num_qrys_without_entity, num_qrys_without_all_entities = 0, 0
    tf.logging.info("Reading examples from %s", queries_file)
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for line in tqdm(reader, desc="Reading from %s" % reader.name):
        item = json.loads(line.strip())

        qas_id = item["_id"]
        question_text = item["question"]
        choice2concepts = item["choice2concepts"]
        answer_txt = item["answer"]
        assert answer_txt in choice2concepts
        answer_entities = []

        for answer_concept in choice2concepts[answer_txt]:
          if answer_concept in entity2id:
            # Note: add an arg for decide if only use the longest concept.
            answer_entities.append(entity2id[answer_concept])

        if not answer_entities:
          num_qrys_without_answer += 1
          if self.is_training:
            continue
        if len(answer_entities) != len(item["supporting_facts"]):
          num_qrys_without_all_answers += 1

        subject_entities = []
        for entity in item["entities"]:
          if entity["kb_id"].lower() in entity2id:
            subject_entities.append(entity["kb_id"].lower())
        if not subject_entities:
          num_qrys_without_entity += 1
          if self.is_training:
            continue
        if len(subject_entities) != len(item["entities"]):
          num_qrys_without_all_entities += 1
        if len(answer_entities) > self.max_qry_answers:
          self.max_qry_answers = len(answer_entities)
          tf.logging.warn("%s has %d linked entities", qas_id,
                          len(subject_entities))
        # Define the exclude_entities as the question entities,
        # and the concepts mentioned by wrong choices.
        exclude_entities = subject_entities[:]
        for choice, concepts in choice2concepts.items():
          if choice == answer_txt:
            continue
          for non_answer_concept in concepts:
            if non_answer_concept in entity2id:
              exclude_entities.append(non_answer_concept.lower())

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=subject_entities,
            answer_entity=answer_entities,
            correct_choice=answer_txt,
            choice2concepts=choice2concepts,
            exclude_set=exclude_entities)
        examples.append(example)

    tf.logging.info("Number of valid questions = %d", len(examples))
    tf.logging.info("Questions without any answer = %d",
                    num_qrys_without_answer)
    tf.logging.info("Questions without all answers = %d",
                    num_qrys_without_all_answers)
    tf.logging.info("Questions without any entity = %d",
                    num_qrys_without_entity)
    tf.logging.info("Questions without all entities = %d",
                    num_qrys_without_all_entities)
    tf.logging.info("Maximum answers per question = %d", self.max_qry_answers)

    return examples
