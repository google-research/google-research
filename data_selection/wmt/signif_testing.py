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

"""Significance testing for data selection results.

Use bootstraps to compute significance values for variance in data
selection losses / BLEUs with different configurations.
"""

import collections
import concurrent.futures
import csv
import itertools
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from data_selection.wmt import bleu


FLAGS = flags.FLAGS

flags.DEFINE_string("decode_dir", default=None, help="Directory of decodes.")


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree"s leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

  def pre_pmap(xs):
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)

  def post_pmap(xs):
    return jax.tree_map(lambda x: x[0], xs)

  return post_pmap(host_psum(pre_pmap(in_tree)))


def get_decodes():
  """read decodes from FS."""
  logging.info("Reading Decodes")
  decodes = []
  loss_name = "/decodes.csv"
  with tf.io.gfile.GFile(FLAGS.decode_dir + loss_name, "r") as h:
    reader = csv.reader(h)
    for row in reader:
      decodes.append(row)
  logging.info("Done with decodes")
  return decodes


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Starting")

  decodes = get_decodes()

  def worker(indexes, decodes):
    chosen = [
        x for i, x in enumerate(decodes) if i in indexes
    ]
    references = [x[1] for x in chosen]
    predictions = [x[2] for x in chosen]
    bleu_matches = bleu.bleu_partial(references, predictions)
    all_bleu_matches = per_host_sum_pmap(bleu_matches)
    bleu_score = bleu.complete_bleu(*all_bleu_matches)
    logging.info(bleu_score)
    return bleu_score

  logging.info("Worker defined")

  indexes = []
  for _ in range(1000):
    index = np.random.choice(len(decodes), 10000)
    indexes.append(index)
  with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
    bleu_scores = list(executor.map(worker, indexes,
                                    itertools.repeat(decodes)))
  samples_name = "/bleu_samples.csv"
  with tf.io.gfile.GFile(FLAGS.decode_dir + samples_name, "w") as h:
    writer = csv.writer(h)
    for score in bleu_scores:
      writer.writerow([score])


if __name__ == "__main__":
  app.run(main)
