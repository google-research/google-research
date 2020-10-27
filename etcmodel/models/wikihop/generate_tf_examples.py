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

"""Binary for generating TF examples using WikiHop json data file."""

import json
import os
from typing import Iterator

import apache_beam as beam
import tensorflow.compat.v1 as tf

from etcmodel.models import tokenization
from etcmodel.models.wikihop import data_utils

tf.compat.v1.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "input_train_json_filepath", None,
    "The fully qualified path to the train input json file containing the data "
    "for the task. If None, `input_tf_records_path` has to be specified. "
    "For predict mode, this should not be None.")

flags.DEFINE_string(
    "input_dev_json_filepath", None,
    "The fully qualified path to the dev input json file containing the data "
    "for the task. If None, `input_tf_records_path` has to be specified. "
    "For predict mode, this should not be None.")

flags.DEFINE_enum("tokenizer_type", "ALBERT", ["BERT", "ALBERT"],
                  "The tokenizer to be used.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "spm_model_path", None,
    "Path to a SentencePiece model file to use instead of a BERT vocabulary "
    "file. If given, we use the tokenization code from ALBERT instead of BERT. "
    "Should be set whe using ALBERT tokenizer.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ETC model) to start "
    "fine-tuning.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("long_seq_len", 4096,
                     "The total input sequence length to pad to for training.")

flags.DEFINE_integer("global_seq_len", 430,
                     "The raw maximum global input sequence length.")

flags.DEFINE_integer(
    "max_num_sentences", 200,
    "The max total number of sentences to be used across all the docs before. "
    "truncation.  Must be <= 214 (384 - 27 - 79 - 63), where 27 represents the "
    "maximum number of query WordPieces, 79 the number of candidates, 63 the "
    "number of docs per example for the WikiHop dataset. We mirror the query "
    "the global input, have a token per doc.")

flags.DEFINE_bool(
    "shuffle_docs_within_example", False,
    "True, if the docs within a single WikiHopExample should "
    "be shuffled.")

flags.DEFINE_string("output_dir_path", "", "The output dir for tf examples.")

flags.DEFINE_integer("num_train_shards", 100,
                     "Number of shards to output train TF Examples into.")

flags.DEFINE_integer("num_dev_shards", 1,
                     "Number of shards to output dev TF Examples into.")


class WikiHopExampleToTfExamplesFn(beam.DoFn):
  """DoFn for converting WikiHopExample to tf.Example."""

  def setup(self):
    super().setup()
    if FLAGS.tokenizer_type == "BERT":
      tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                    FLAGS.init_checkpoint)
      if not FLAGS.vocab_file:
        raise ValueError("vocab_file should be specified when using "
                         "BERT tokenizer.")
      self._tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    elif FLAGS.tokenizer_type == "ALBERT":
      if not FLAGS.spm_model_path:
        raise ValueError("spm_model_path should be specified when using "
                         "ALBERT tokenizer.")
      self._tokenizer = tokenization.FullTokenizer(
          vocab_file=None,
          do_lower_case=None,
          spm_model_file=FLAGS.spm_model_path)
    else:
      raise ValueError("Unexpected tokenizer_type found: {}".format(
          FLAGS.tokenizer_type))

  def process(self, example) -> Iterator[tf.train.Example]:
    converter = data_utils.WikiHopTFExampleConverter(
        tokenizer=self._tokenizer,
        global_seq_len=FLAGS.global_seq_len,
        long_seq_len=FLAGS.long_seq_len,
        max_num_sentences=FLAGS.max_num_sentences)
    yield converter.convert_single_example(example=example)


def pipeline(root):
  """Beam pipeline to run."""
  train_tf_examples = (
      root
      | "ReadTrainInputFile" >> beam.io.textio.ReadFromText(
          FLAGS.input_train_json_filepath)
      | "TrainParseExample" >> beam.FlatMap(
          data_utils.WikiHopExample.parse_examples,
          shuffle_docs_within_example=FLAGS.shuffle_docs_within_example)
      | "TrainReshuffle" >> beam.transforms.Reshuffle()
      | "TrainConvertToTfExample" >> beam.ParDo(WikiHopExampleToTfExamplesFn()))

  dev_tf_examples = (
      root
      | "ReadDevInputFile" >> beam.io.textio.ReadFromText(
          FLAGS.input_dev_json_filepath)
      | "DevParseExample" >> beam.FlatMap(
          data_utils.WikiHopExample.parse_examples)
      | "DevReshuffle" >> beam.transforms.Reshuffle()
      | "DevConvertToTfExample" >> beam.ParDo(WikiHopExampleToTfExamplesFn()))

  _ = (
      train_tf_examples
      | "CountTrainTfExamples" >> beam.combiners.Count.Globally()
      | "TrainJsonCounts" >> beam.Map(lambda x: json.dumps(x, indent=2))
      | "TrainWriteToText" >> beam.io.WriteToText(
          os.path.join(FLAGS.output_dir_path, "train_stats"),
          shard_name_template="",  # To force unsharded output.
      ))

  _ = (
      dev_tf_examples
      | "CountDevTfExamples" >> beam.combiners.Count.Globally()
      | "DevJsonCounts" >> beam.Map(lambda x: json.dumps(x, indent=2))
      | "DevWriteToText" >> beam.io.WriteToText(
          os.path.join(FLAGS.output_dir_path, "dev_stats"),
          shard_name_template="",  # To force unsharded output.
      ))

  _ = (
      train_tf_examples
      | "WriteTrainTFRecords" >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_dir_path, "train/tf_examples"),
          coder=beam.coders.ProtoCoder(tf.train.Example),
          file_name_suffix=".record",
          num_shards=FLAGS.num_train_shards))
  _ = (
      dev_tf_examples
      | "WriteDevTFRecords" >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_dir_path, "dev/tf_examples"),
          coder=beam.coders.ProtoCoder(tf.train.Example),
          file_name_suffix=".record",
          num_shards=FLAGS.num_dev_shards))


def main(unused_argv):
  # run the pipeline:
  p = beam.Pipeline()
  pipeline(p)
  result = p.run()
  result.wait_until_finish()


if __name__ == "__main__":
  tf.app.run()
