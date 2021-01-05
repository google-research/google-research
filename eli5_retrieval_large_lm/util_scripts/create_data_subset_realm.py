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

r"""Makes a subset of the REALM data to quicken the development of main.py.

Example of use:
python create_data_subset_realm.py \
--source_text_path=/usr/local/google/home/julesgm/ram_drive/blocks.tfr \
--source_embeddings_prefix=/usr/local/google/home/julesgm/ram_drive\
/cc_news_pretrained/embedder/encoded/encoded.ckpt \
--subset_text_path=/usr/local/google/home/julesgm/subset/subset_text.tfr \
--subset_embeddings_ds_path=/usr/local/google/home/julesgm/subset/encoded.ckpt \
--source_total=13353718 \
--subset_total=5000 \
--logger_levels=__main__:DEBUG,retrieval_while_decoding.utils:DEBUG
"""

import logging
import operator
import os
from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
import tqdm
import utils


LOGGER = logging.getLogger(__name__)
FLAGS = flags.FLAGS

flags.DEFINE_string("source_text_path", None,
                    "Path to the TFRecord file with text.")
flags.DEFINE_string("source_embeddings_prefix", None,
                    "Path to the TFRecord file with embeddings_ds.")
flags.DEFINE_string("subset_text_path", None,
                    "Path to the TFRecord file with text.")
flags.DEFINE_string("subset_embeddings_ds_path", None,
                    "Path to the TFRecord file with embeddings_ds.")
flags.DEFINE_integer("source_total", None,
                     "Number of points in the original records")
flags.DEFINE_integer("subset_total", None,
                     "Number of points desired for in the subset records.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  absl_logging.use_python_logging()
  utils.log_module_args(LOGGER, argv[0])

  # Some checks for the flags
  utils.check_exists(FLAGS.source_text_path)
  utils.check_exists(os.path.dirname(FLAGS.subset_text_path))
  utils.check_exists(os.path.dirname(FLAGS.subset_embeddings_ds_path))
  utils.check_operator(operator.lt, FLAGS.subset_total, FLAGS.source_total)

  utils.check_glob_prefix(FLAGS.source_embeddings_prefix)

  # Select a random subset
  with utils.log_duration(LOGGER, "main", "preparing indices"):
    indices = np.random.choice(FLAGS.source_total, FLAGS.subset_total,
                               replace=False)
    indices.sort()

  # Process the textual data
  # Much (5 min vs 2 h) faster than iterating through the records and writing
  # only those we want. An hypothesis for this is that
  # get_single_element would allow to get elements without parsing all of the
  # elements along the way, like simply iterating through the records would.
  # Or did they get constant time indexing in TFRecords?
  # Inspired by the ORQA codebase:
  # https://github.com/google-research/language/blob/master/language/orqa/models/orqa_model.py#L147
  with utils.log_duration(LOGGER, "main", "preparing data"):
    text_ds = tf.data.TFRecordDataset(FLAGS.source_text_path,
                                      buffer_size=512 * 1024 * 1024,
                                      num_parallel_reads=os.cpu_count())
    text_ds = text_ds.batch(FLAGS.source_total)
    text_ds = tf.data.experimental.get_single_element(text_ds)
    subset = tf.gather(text_ds, tf.constant(indices))

  with utils.log_duration(LOGGER, "main", "writing text data"):
    with tf.io.TFRecordWriter(FLAGS.subset_text_path) as text_writer:
      for text in tqdm.tqdm(subset, total=FLAGS.subset_total):
        text = text.numpy()
        # REALM's data uses no packaging of the data into features, etc.
        text_writer.write(text)

  with utils.log_duration(LOGGER, "main", "All of the embedding task"):
    # Process the embeddings data
    with tf.device("/cpu:0"):
      with utils.log_duration(LOGGER, "main", "Loading the checkpoint"):
        embs = tf.train.load_checkpoint(FLAGS.source_embeddings_prefix
                                        ).get_tensor("block_emb")
        utils.check_equal(embs.shape[0], FLAGS.source_total)

      with utils.log_duration(LOGGER, "main", "taking a subset of the indices"):
        subset = embs[indices]

      tf_db = tf.Variable(subset, shape=subset.shape)
      ckpt = tf.train.Checkpoint(block_emb=tf_db)

      with utils.log_duration(LOGGER, "main", "Saving the checkpoint"):
        ckpt.save(FLAGS.subset_embeddings_ds_path)

    LOGGER.debug("Done")


if __name__ == "__main__":
  app.run(main)
