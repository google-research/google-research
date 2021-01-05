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

"""Binary for generating tf examples using HotpotQA json data file."""
import json

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options
import tensorflow.compat.v1 as tf

from etcmodel.models.hotpotqa import generate_tf_examples_lib as lib

FLAGS = flags.FLAGS

flags.DEFINE_string("input_json_filename", "",
                    "The json filename for the HotpotQA data.")
flags.DEFINE_string("output_tf_examples_file_path_prefix", "",
                    "The output tf examples file path prefix.")
flags.DEFINE_integer("num_shards", 256, "Number of output shards to generate.")
flags.DEFINE_string("output_tf_examples_stat_filename", "",
                    "The output tf examples statistics filename.")

flags.DEFINE_string(
    "spm_model_file", "",
    ("The SentencePiece tokenizer model file that the ETC model was trained on."
     "If not None, the `vocab_file` is ignored."))
flags.DEFINE_string(
    "vocab_file", "",
    "The WordPiece tokenizer vocabulary file that the ETC model was trained on."
)

flags.DEFINE_integer("global_seq_length", 256,
                     "The sequence length of global tokens for the ETC model.")
flags.DEFINE_integer("long_seq_length", 2048,
                     "The sequence length of long tokens for the ETC model.")

flags.DEFINE_bool(
    "is_training", True,
    ("Whether in training mode. In training mode, labels are also stored in the"
     "tf.Examples."))

flags.DEFINE_enum("answer_encoding_method", "span", ["span", "bio"],
                  "The answer encoding method.")

flags.DEFINE_integer(
    "direct_num_workers", 0,
    "Number of workers to use for the Beam DirectRunner. "
    "Increasing this should speed up example generation, "
    "but DirectRunner appears to run out of memory quickly "
    "when using more workers. 0 is automatically using all available workers.")

flags.DEFINE_bool(
    "debug", True,
    ("Whether in debug mode. In debug mode, extra information is also stored in"
     "the tf.Examples."))


def pipeline(root):
  """Beam pipeline to run."""
  tf_examples = (
      root
      | "ReadFromJSON" >> beam.io.textio.ReadFromText(FLAGS.input_json_filename)
      | "ParseHotpotQAExamples" >> beam.FlatMap(lib.HotpotQAExample.from_json)
      | "Reshuffle" >> beam.transforms.Reshuffle()
      | "ConvertToTfExample" >> beam.ParDo(
          lib.HotpotQAExampleToTfExamplesFn(
              FLAGS.global_seq_length, FLAGS.long_seq_length, FLAGS.is_training,
              FLAGS.answer_encoding_method, FLAGS.spm_model_file,
              FLAGS.vocab_file, FLAGS.debug)))

  _ = (
      tf_examples
      | "CountTfExamples" >> beam.combiners.Count.Globally()
      | beam.Map(lambda x: json.dumps(x, indent=2))
      | beam.io.WriteToText(
          FLAGS.output_tf_examples_stat_filename,
          shard_name_template="",  # To force unsharded output.
      ))
  _ = (
      tf_examples
      | "WriteTFExample" >> beam.io.WriteToTFRecord(
          FLAGS.output_tf_examples_file_path_prefix,
          coder=beam.coders.ProtoCoder(tf.train.Example),
          num_shards=FLAGS.num_shards))


def main(unused_args):
  # run the pipeline:
  options = pipeline_options.PipelineOptions(
      runner="DirectRunner",
      direct_running_mode="multi_processing",
      direct_num_workers=FLAGS.direct_num_workers)
  p = beam.Pipeline(options=options)
  pipeline(p)
  p.run().wait_until_finish()


if __name__ == "__main__":
  app.run(main)
