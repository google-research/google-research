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

"""A pipeline to preprocess NQ data in ETC example format."""

import glob
import gzip
import json
import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from etcmodel.models.nq import preproc_nq_lib

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None,
                    "Path to input NQ examples.")

flags.DEFINE_string("output_dir", None,
                    "Path for writing preprocessed NQ examples.")

flags.DEFINE_integer("stride", 2048,
                     "Token length stride for splitting documents.")

flags.DEFINE_integer("seq_len", 4096, "Total window size in word pieces.")

flags.DEFINE_integer("global_seq_len", 230, "Total number of global tokens.")

flags.DEFINE_integer("question_len", 32,
                     "Maximum question length in word pieces.")

flags.DEFINE_multi_integer("global_token_types", [0, 1, 2],
                           "Global token types for sentence, CLS, and question "
                           "word piece tokens respectively.")

flags.DEFINE_enum("tokenizer_type", "BERT", ["BERT", "ALBERT"],
                  "Specifies which tokenizers to use.")

flags.DEFINE_string(
    "vocab_file", None,
    "Path to a wordpiece vocabulary to be used with the tokenizer. "
    "This is ignored when using the ALBERT tokenizer if 'spm_model_path' is "
    "specified.")

flags.DEFINE_boolean("do_lower_case", True,
                     "Whether to lower case text. This is ignored when using "
                     "the ALBERT tokenizer if 'spm_model_path' is specified.")

flags.DEFINE_string(
    "spm_model_path", None,
    "Path to a SentencePiece model file to use with the ALBERT tokenizer. "
    "For example: 'my_folder/vocab_gpt.model'.")

flags.DEFINE_integer("num_shards", 256,
                     "Number of output shards to generate.")

flags.DEFINE_boolean("predict_la_when_no_sa", False,
                     "Whether to predict long answer where there is no short "
                     "answer.")

flags.DEFINE_float("include_unknown_rate", 1.0,
                   "Proportion of windows that do not contain an answer to "
                   "keep on the final dataset.")

flags.DEFINE_float("include_unknown_rate_for_unanswerable", None,
                   "Proportion of windows that do not contain an answer to "
                   "keep on the final dataset for instances that actually do "
                   "not have an answer at all. If not specified, this will be "
                   "set to 'include_unknown_rate'*4.")

flags.DEFINE_boolean("fixed_blocks", False,
                     "If 'True', this just sets the global input to a uniform "
                     "sequence of sentence tokens, each of them mapping to a "
                     "set of long input tokens of fixed length.")

flags.DEFINE_integer("fixed_block_size", 27,
                     "How many word piece tokens per global token when "
                     "'fixed_blocks' is True.")


def pipeline():
  """Method to pass into flume runner."""
  for split in ["train", "dev"]:
    input_path = os.path.join(FLAGS.input_dir,
                              "%s/nq-%s-??.jsonl.gz" % (split, split))
    output_path = os.path.join(FLAGS.output_dir, "nq-%s.tfrecords" % split)
    input_files = glob.glob(input_path)
    writers = []
    for i in range(FLAGS.num_shards):
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.io.TFRecordCompressionType.GZIP)
      file_name = (output_path + "-" + str(i).zfill(5) + "-of-" +
                   str(FLAGS.num_shards).zfill(5))
      writer = tf.python_io.TFRecordWriter(file_name, options=options)
      writers.append(writer)

    preprocessor = preproc_nq_lib.NQPreprocessor(
        stride=FLAGS.stride,
        seq_len=FLAGS.seq_len,
        global_seq_len=FLAGS.global_seq_len,
        question_len=FLAGS.question_len,
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case,
        predict_la_when_no_sa=FLAGS.predict_la_when_no_sa,
        include_unknown_rate=FLAGS.include_unknown_rate,
        include_unknown_rate_for_unanswerable=(
            FLAGS.include_unknown_rate_for_unanswerable),
        include_html_tokens=True,
        global_token_types=FLAGS.global_token_types,
        spm_model_path=FLAGS.spm_model_path,
        tokenizer_type=FLAGS.tokenizer_type,
        is_train=split == "train",
        fixed_blocks=FLAGS.fixed_blocks,
        fixed_block_size=FLAGS.fixed_block_size)

    input_lines = 0
    num_instances = 0
    for input_file in input_files:
      print("processing " + input_file + "...")
      with gzip.open(input_file, "rt") as f:
        for line in f:
          input_lines += 1
          example = json.loads(line)
          preproc_example = preprocessor.to_tf_example(example)
          for instance in preprocessor.split_example(preproc_example):
            serialized = instance.SerializeToString()
            writers[num_instances % len(writers)].write(serialized)
            num_instances += 1
            if num_instances % 100 == 0:
              print("    " + str(num_instances) + " instances from " +
                    str(input_lines) + " input examples.")
      print("    " + str(num_instances) + " generated so far for " + split)

    # close all the output shard writers:
    for writer in writers:
      writer.close()


def main(unused_args):
  # Run the pipeline.
  pipeline()


if __name__ == "__main__":
  app.run(main)
