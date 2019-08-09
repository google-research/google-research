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

r"""Converts an NQ dataset into smaller shards.

Notes:
Takes a dataset shard number and splits it to smaller shards of 1000
Usage: python nq_sharder.py --split train --input_data_dir nq/train/
                            --output_data_dir sharded_nq/train --task_id 10

       This command would split nq-train-10.jsonl.gz to further smaller files
       of 1000 examples of the form:
         nq-train-1000.jsonl.gz... nq-train-1010.jsonl.gz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tensorflow as tf

from fat.fat_bert_nq import nq_data_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "split", "train",
    "Train and dev split to read from. Accepted values: ['train', 'dev', 'test']"
)

flags.DEFINE_string("input_data_dir", "", "input_data_dir")

flags.DEFINE_string("output_data_dir", "", "output_data_dir")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("split_size", 1000, "Number of examples in every split")


def get_output_fp(shard_counter):
  output_file = nq_data_utils.get_sharded_filename(FLAGS.output_data_dir,
                                                   FLAGS.split, FLAGS.task_id,
                                                   shard_counter, "jsonl.gz")
  op = gzip.GzipFile(fileobj=tf.gfile.Open(output_file, "w"))
  return op


def main(_):
  shard_counter = 0
  input_file = nq_data_utils.get_nq_filename(FLAGS.input_data_dir, FLAGS.split,
                                             FLAGS.task_id, "jsonl.gz")
  op = get_output_fp(shard_counter)
  counter = 0
  for line in nq_data_utils.get_nq_examples(input_file):
    op.write((line.decode("utf-8")).encode("utf-8"))
    counter += 1
    if counter % FLAGS.split_size == 0:
      op.close()
      shard_counter += 1
      op = get_output_fp(shard_counter)


if __name__ == "__main__":
  tf.app.run()
