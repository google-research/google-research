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

r"""Tests the recall of ScaNN with certain settings over a certain dataset.
"""

import logging
import operator
import os
import time
from typing import Union

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import retrievers
import scann_utils
import tensorflow as tf
import tqdm
import utils


LOGGER = logging.getLogger(__name__)
SCRIPT_DIRECTORY = os.path.dirname(__file__)


def exact_search(num_neighbors, vectors, db
                 ):
  product = tf.linalg.matmul(vectors, db, transpose_b=True)
  _, top_k = tf.math.top_k(product, k=num_neighbors, sorted=False)
  return top_k

################################################################################
# Flag Definitions
################################################################################
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", None, ["any", "all"],
                  "Recall computation mode. `any` gives the fraction of the "
                  "time that the correct point is in the `num_neighbor` points."
                  " `all` computes the fraction of the `num_neighbor` "
                  "predicted points that are present in "
                  "a set with the target point and `num_neighbor` - 1 other "
                  "points.")
flags.DEFINE_string("scann_config_path",
                    os.path.join(SCRIPT_DIRECTORY, "configs", "scann_configs",
                                 "default_config.json"),
                    "Configuration file for the ScaNN MIPS library.")
flags.DEFINE_integer("batch_size", 2, "Size of the batches for the retrievals.")
flags.DEFINE_integer("num_neighbors", 10, "Number of neighbors to retrieve.")
flags.DEFINE_integer("test_how_many", 100, "How many top_k's to test.")
flags.DEFINE_string("embeddings_ckpt_path", None,
                    "Path to the checkpoint containing the embeddings.")
flags.DEFINE_string("output_dir", None,
                    "Where to save the results.")
flags.DEFINE_integer("print_every_n_batches", 1,
                     "Log information every n number of batches.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  absl_logging.use_python_logging()
  utils.log_module_args(LOGGER, argv[0])

  utils.check_exists(FLAGS.scann_config_path)
  utils.check_glob_prefix(FLAGS.embeddings_ckpt_path)
  utils.check_exists(FLAGS.output_dir)
  if not tf.io.gfile.isdir(FLAGS.output_dir):
    raise RuntimeError("Output dir needs to be a directory.")

  ##############################################################################
  # Setup: Build the ScaNN (Scam) searcher
  ##############################################################################
  with utils.log_duration(LOGGER, "main", "load_scann_searcher"):
    checkpoint_path = os.path.join(FLAGS.embeddings_ckpt_path)
    # The conversion to a ScannConfig object enforces that all the fields we
    # expect are present in the json file.
    scann_config = retrievers.ScannConfig(**utils.from_json_file(
        FLAGS.scann_config_path))
    block_emb, scann_searcher = scann_utils.load_scann_searcher(
        var_name="block_emb", checkpoint_path=checkpoint_path,
        **vars(scann_config))
  utils.check_operator(operator.ge, block_emb.shape[0], FLAGS.test_how_many)

  ##############################################################################
  # Recall Computation
  ##############################################################################
  LOGGER.debug(block_emb.shape)
  utils.check_operator(operator.ge, block_emb.shape[0], FLAGS.test_how_many)
  with utils.log_duration(LOGGER, "main", "all retrievals & comparisons"):
    LOGGER.debug("block_emb.shape: %s", str(block_emb.shape))
    LOGGER.debug("FLAGS.test_how_many: %d", FLAGS.test_how_many)
    all_indices = np.random.choice(block_emb.shape[0], FLAGS.test_how_many,
                                   replace=False)
    count_total = 0
    count_good = 0
    for i, idx_start in tqdm.tqdm(enumerate(range(0, len(all_indices),
                                                  FLAGS.batch_size))):
      indices = all_indices[idx_start:idx_start + FLAGS.batch_size]
      vectors = tf.gather(block_emb, indices)

      if FLAGS.mode == "all":
        with utils.log_duration(LOGGER, "main", "exact_search"):
          labels = exact_search(FLAGS.num_neighbors, vectors, block_emb)
      elif FLAGS.mode == "any":
        labels = tf.cast(tf.expand_dims(indices, - 1), tf.int32)
      else:
        raise RuntimeError(FLAGS.mode)

      with utils.log_duration(LOGGER, "main", "scann_search"):
        predictions, _ = scann_searcher.search_batched(vectors)
      good = tf.sets.intersection(labels, predictions)
      count_good += len(good.values)
      count_total += tf.math.reduce_prod(labels.shape)
      ratio = count_good / count_total
      if i % FLAGS.print_every_n_batches == 0 and i != 0:
        LOGGER.debug("Recall so far: %f %%", 100 * ratio)

  final_recall = count_good / count_total
  LOGGER.debug("Final recall for mode `%(mode)s` with `%(num_neighbors)d` "
               "neighbors: %(recall)f %%",
               dict(mode=FLAGS.mode, num_neighbors=FLAGS.num_neighbors,
                    recall=100 * final_recall))
  LOGGER.debug("%d true positives over %d points.", count_good, count_total)

  ##############################################################################
  # Build the output object and save it.
  ##############################################################################
  output = {}
  output["flags"] = {flag.name: flag.value for flag
                     in FLAGS.flags_by_module_dict()[argv[0]]}
  output["recall"] = float(final_recall)
  # Redundant but easier to read
  output["count_goods"] = int(count_good)
  output["count_total"] = int(count_total)
  output_path = os.path.join(FLAGS.output_dir, "test_recall_" +
                             time.strftime("results_%Y%m%d-%H%M%S.json"))
  utils.to_json_file(output_path, output)

if __name__ == "__main__":
  app.run(main)
