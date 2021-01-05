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

"""A beam pipeline to preprocess NQ data in ETC example format."""

import os
from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options

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

flags.DEFINE_integer(
    "direct_num_workers", 1,
    "Number of workers to use for the Beam DirectRunner. "
    "Increasing this should speed up example generation, "
    "but DirectRunner appears to run out of memory quickly "
    "when using more workers.")


def pipeline(root):
  """Method to pass into flume runner."""
  for split in ["train", "dev"]:
    input_path = os.path.join(FLAGS.input_dir,
                              "%s/nq-%s-??.jsonl.gz" % (split, split))
    output_path = os.path.join(FLAGS.output_dir, "nq-%s.tfrecords" % split)
    serialized_examples = (
        root
        | "Read-" + split >> beam.io.textio.ReadFromText(
            input_path, validate=False)
        | "Shuffle-" + split >> beam.transforms.Reshuffle()
        | "Prep-" + split >> beam.ParDo(
            preproc_nq_lib.NQPreprocessFn(
                FLAGS.stride, FLAGS.seq_len, FLAGS.global_seq_len,
                FLAGS.question_len, FLAGS.vocab_file, FLAGS.do_lower_case,
                FLAGS.global_token_types, FLAGS.spm_model_path,
                FLAGS.tokenizer_type, split == "train",
                FLAGS.predict_la_when_no_sa, FLAGS.include_unknown_rate,
                FLAGS.include_unknown_rate_for_unanswerable, FLAGS.fixed_blocks,
                FLAGS.fixed_block_size))
        | "Split-" + split >> beam.ParDo(
            preproc_nq_lib.NQSplitFn(
                FLAGS.stride, FLAGS.seq_len, FLAGS.global_seq_len,
                FLAGS.question_len, FLAGS.vocab_file, FLAGS.do_lower_case,
                FLAGS.global_token_types, FLAGS.spm_model_path,
                FLAGS.tokenizer_type, split == "train",
                FLAGS.predict_la_when_no_sa, FLAGS.include_unknown_rate,
                FLAGS.include_unknown_rate_for_unanswerable, FLAGS.fixed_blocks,
                FLAGS.fixed_block_size))
        | "Shuffle2-" + split >> beam.transforms.Reshuffle())
    _ = (
        serialized_examples
        | "WriteTFExample-" + split >> beam.io.WriteToTFRecord(
            output_path,
            num_shards=FLAGS.num_shards,
            compression_type=beam.io.filesystem.CompressionTypes.GZIP))

    # Write count to file.
    _ = (
        serialized_examples
        | "Count-" + split >> beam.combiners.Count.Globally()
        | "WriteCount-" + split >> beam.io.WriteToText(
            os.path.join(FLAGS.output_dir, "%s_count.txt" % split),
            shard_name_template="",  # To force unsharded output.
        ))


def main(unused_args):
  # Run the pipeline.
  options = pipeline_options.PipelineOptions(
      runner="DirectRunner",
      direct_running_mode="multi_processing",
      direct_num_workers=FLAGS.direct_num_workers)
  p = beam.Pipeline(options=options)
  pipeline(p)
  p.run().wait_until_finish()


if __name__ == "__main__":
  app.run(main)
