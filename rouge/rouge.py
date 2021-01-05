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

r"""Main routine to calculate ROUGE scores across text files.

Designed to replicate scores computed by the ROUGE perl implementation as
closely as possible.

Output is a text file in CSV format.

Sample usage:

rouge ---rouge_types=rouge1,rouge2,rougeL \
    --target_filepattern=*.targets \
    --prediction_fliepattern=*.decodes \
    --output_filename=scores.csv \
    --use_stemmer

Which is equivalent to calling the perl ROUGE script as:

ROUGE-1.5.5.pl -m -e ./data -n 2 -a /tmp/rouge/settings.xml

Where settings.xml provides target and decode text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from rouge import io
from rouge import rouge_scorer
from rouge import scoring

flags.DEFINE_string("target_filepattern", None,
                    "Files containing target text.")
flags.DEFINE_string("prediction_filepattern", None,
                    "Files containing prediction text.")
flags.DEFINE_string("output_filename", None,
                    "File in which to write calculated ROUGE scores as a CSV.")
flags.DEFINE_string("delimiter", "\n",
                    "Record delimiter  in files.")
flags.DEFINE_list("rouge_types", ["rouge1", "rouge2", "rougeL"],
                  "List of ROUGE types to calculate.")
flags.DEFINE_boolean("use_stemmer", False,
                     "Whether to use Porter stemmer to remove common suffixes.")
flags.DEFINE_boolean("aggregate", True,
                     "Write aggregates if this is set to True")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  scorer = rouge_scorer.RougeScorer(FLAGS.rouge_types, FLAGS.use_stemmer)
  aggregator = scoring.BootstrapAggregator() if FLAGS.aggregate else None
  io.compute_scores_and_write_to_csv(
      FLAGS.target_filepattern,
      FLAGS.prediction_filepattern,
      FLAGS.output_filename,
      scorer,
      aggregator,
      delimiter=FLAGS.delimiter)


if __name__ == "__main__":
  flags.mark_flag_as_required("target_filepattern")
  flags.mark_flag_as_required("prediction_filepattern")
  flags.mark_flag_as_required("output_filename")
  app.run(main)
