# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Reads a TypeMixtureTopicDistribution and samples from it."""

from collections.abc import Sequence
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np
import tensorflow as tf

from topics_api_data_release import type_mixture_distribution


TypeMixtureTopicDistribution = (
    type_mixture_distribution.TypeMixtureTopicDistribution
)

_DISTRIBUTION_PATH = flags.DEFINE_string(
    "distribution_path",
    None,
    "Path to a recordio file containing a list of TypeDistribution protos.",
)

_TOPIC_TAXONOMY_PATH = flags.DEFINE_string(
    "topic_taxonomy_path",
    None,
    "Path to the topic taxonomy json file.",
    required=True,
)

_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", None, "The number of topic sets to sample.", required=True
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    "output_prefix",
    None,
    "The output will be saved as a sharded tfrecord file with file names of the"
    " form [output_prefix]_#####-of-#####.tfrecord, where the number of shards"
    " is specified by the num_shards command line flag.",
    required=True,
)

_NUM_SHARDS = flags.DEFINE_integer(
    "num_shards",
    100,
    "The number of shards to use for the output file.",
)

_SEED = flags.DEFINE_integer("seed", 0, "The random seed to use.")


def load_topic_ids(path):
  """Returns a list of topic ids in the order they appear in the taxonomy."""
  with open(path, "r") as fh:
    records = json.load(fh)
  return np.array([record["topic_id"] for record in records])


def write_output(tfexamples):
  """Writes a collection of tf.train.Examples to a sharded TFRecord file.

  Args:
    tfexamples: The tf.train.Examples to write.
  """
  num_shards = _NUM_SHARDS.value
  paths = [
      f"{_OUTPUT_PREFIX.value}_{i:05d}-of-{num_shards:05d}.tfrecord"
      for i in range(num_shards)
  ]
  writers = [tf.io.TFRecordWriter(path) for path in paths]
  for i, ex in enumerate(tfexamples):
    writers[i % len(writers)].write(ex.SerializeToString())
  for writer in writers:
    writer.close()


def convert_sample_to_tf_example(
    user_id, sample, topic_ids
):
  """Converts a single topic set sample to a tf.train.Example.

  The tf.train.Example will have the following features:
    - `user_id`: A bytes feature that contains the provided user_id.
    - `epoch_{t}_topics` for `t in range(num_weeks)`: An int feature that
      contains the topics in the user's profile for epoch `t`.
    - `epoch_{t}_weights` for `t in range(num_epochs)`: A float feature that
      contains the topic weights for epoch `t`. These weights will always be
      set to 1.0.

  Args:
    user_id: The user id to set for this sample.
    sample: A jax array of shape [num_weeks, num_slots] representing a single
      users sampled topic indices for each week and slot.
    topic_ids: A list of topic ids such that the topic with index i corresponds
      to the topic with id topic_ids[i].

  Returns:
    A tf.train.Example encoding the sample.
  """
  features = {
      "user_id": tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[bytes(f"user {user_id}", "utf-8")]
          )
      ),
  }
  num_weeks, num_slots = sample.shape
  for w in range(num_weeks):
    features[f"epoch_{w}_topics"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=topic_ids[sample[w, :]])
    )
    features[f"epoch_{w}_weights"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[1.0] * num_slots)
    )

  return tf.train.Example(features=tf.train.Features(feature=features))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  topic_ids = load_topic_ids(_TOPIC_TAXONOMY_PATH.value)

  # Read the TypeDistribution json.
  dist_load_start = time.time()
  with open(_DISTRIBUTION_PATH.value, "r") as fh:
    dist = TypeMixtureTopicDistribution.from_json(fh.read(), topic_ids)
  logging.info(
      "Loaded distribution with %d types in %.2f seconds.",
      dist.theta.shape[0],
      time.time() - dist_load_start,
  )

  sampling_start = time.time()
  rng_key = jax.random.PRNGKey(_SEED.value)
  sampled_topic_indices = dist.sample_topic_indices(rng_key, _NUM_SAMPLES.value)
  jax.block_until_ready(sampled_topic_indices)
  logging.info(
      "Sampled %d synthetic users in %.2f seconds.",
      _NUM_SAMPLES.value,
      time.time() - sampling_start,
  )
  # Convert samples to a numpy array for faster slicing compared to jax.Array.
  sampled_topic_indices = np.array(sampled_topic_indices)

  convert_to_tfexample_start = time.time()
  tfexamples = []
  for i in range(_NUM_SAMPLES.value):
    tfexamples.append(
        convert_sample_to_tf_example(
            user_id=i, sample=sampled_topic_indices[i, Ellipsis], topic_ids=topic_ids
        )
    )
  logging.info(
      "Converted %d synthetic users to tf.train.Examples in %.2f seconds.",
      _NUM_SAMPLES.value,
      time.time() - convert_to_tfexample_start,
  )

  write_start = time.time()

  os.makedirs(os.path.dirname(_OUTPUT_PREFIX.value), exist_ok=True)
  write_output(tfexamples)
  logging.info(
      "Wrote %d synthetic users to %s in %.2f seconds.",
      _NUM_SAMPLES.value,
      _OUTPUT_PREFIX.value,
      time.time() - write_start,
  )


if __name__ == "__main__":
  app.run(main)
