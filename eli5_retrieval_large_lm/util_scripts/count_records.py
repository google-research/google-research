# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Count the number of entries in a tfrecord

Example of use:
  python count_records.py
  --path=/usr/local/google/home/julesgm/ram_drive/blocks.tfr \
  --suspected_total=13353718
  --logger_levels=__main__:DEBUG,retrieval_while_decoding.utils:DEBUG
"""
import logging
import os

from absl import app
from absl import flags
from absl import logging as absl_logging
import tensorflow as tf
import tqdm
import utils


LOGGER = logging.getLogger(__name__)
FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, "Path to the TFRecord file")
flags.DEFINE_integer("suspected_total", None,
                     "Total you think might be correct. Gives a progress bar.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  absl_logging.use_python_logging()
  utils.log_module_args(LOGGER, argv[0])

  blocks_dataset = tf.data.TFRecordDataset(FLAGS.path,
                                           buffer_size=512 * 1024 * 1024,
                                           num_parallel_reads=os.cpu_count())

  with utils.log_duration(LOGGER, "main", "count"):
    count = sum(1 for _ in tqdm.tqdm(blocks_dataset,
                                     total=FLAGS.suspected_total))
    print(f"The number of entries is `{count}`")


if __name__ == "__main__":
  app.run(main)
