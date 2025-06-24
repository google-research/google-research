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

"""Fits a TypeMixtureTopicDistribution to a collection of target statistics."""

from collections.abc import Sequence
import json

from absl import app
from absl import flags
import jax
import optax

from topics_api_data_release import distribution_optimizer
from topics_api_data_release import topics_query_builder
from topics_api_data_release import type_mixture_distribution

_WITHIN_WEEK_STATS_PATH = flags.DEFINE_string(
    "within_week_stats_path",
    None,
    "Path to the csv file containing within-week statistics.",
    required=True,
)
_ACROSS_WEEK_STATS_PATH = flags.DEFINE_string(
    "across_week_stats_path",
    None,
    "Path to the json file containing across-week statistics.",
    required=True,
)
_TOPIC_TAXONOMY_PATH = flags.DEFINE_string(
    "topic_taxonomy_path",
    None,
    "Path to the topic taxonomy json file.",
    required=True,
)
_NUM_WEEKS = flags.DEFINE_integer(
    "num_weeks", None, "The number of weeks to fit.", required=True
)
_NUM_TYPES = flags.DEFINE_integer(
    "num_types", None, "The number of types to fit.", required=True
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", None, "The path to write the output to. The output is a json"
    " file containing the parameters of the fitted distribution.", required=True
)
_SEED = flags.DEFINE_integer("seed", 0, "The random seed to use.")
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 8192, "The number of queries used per training step."
)
_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs",
    2000,
    "The number of epochs (passes through the queries) to optimize over.",
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 1.0, "The ADAM learning rate parameter."
)
_INIT_STD_DEV = flags.DEFINE_float(
    "init_std_dev",
    0.1,
    "The standard deviation of the Gaussian distribution used to initialize the"
    " synthetic topics distribution parameters.",
)


def load_topic_ids(path):
  """Returns a list of topic ids in the order they appear in the taxonomy."""
  with open(path, "r") as fh:
    records = json.load(fh)
  return [record["topic_id"] for record in records]


def load_across_statistics(path):
  """Loads statistics from a json file and return topic ids with frequencies."""
  with open(path, "r") as fh:
    records = json.load(fh)
  stats = {}
  for record in records:
    topic_1 = record["week_1_topic_id"]
    topic_2 = record["week_2_topic_id"]
    frequency = record["frequency"]
    stats[(topic_1, topic_2)] = frequency
  return stats


def load_within_statistics(path):
  """Loads statistics from a json file and return topic ids with frequencies."""
  with open(path, "r") as fh:
    records = json.load(fh)
  stats = {}
  for record in records:
    topic_1 = record["topic_1_id"]
    topic_2 = record["topic_2_id"]
    frequency = record["frequency"]
    stats[(topic_1, topic_2)] = frequency
  return stats


def extract_single_topic_stats(
    within_week_stats,
):
  """Returns the single-topic stats extracted from within-week stats."""
  num_top_topics = 5
  stats = {}
  for (topic_1, topic_2), target in within_week_stats.items():
    for t in [topic_1, topic_2]:
      stats[t] = stats.get(t, 0.0) + target / (num_top_topics - 1)
  return stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  topic_ids = load_topic_ids(_TOPIC_TAXONOMY_PATH.value)
  across_week_stats = load_across_statistics(_ACROSS_WEEK_STATS_PATH.value)
  within_week_stats = load_within_statistics(_WITHIN_WEEK_STATS_PATH.value)
  single_topic_stats = extract_single_topic_stats(within_week_stats)

  problem = topics_query_builder.create_topics_optimization_problem(
      num_weeks=_NUM_WEEKS.value,
      topics=topic_ids,
      single_topic_stats=single_topic_stats,
      within_week_stats=within_week_stats,
      across_week_stats=across_week_stats,
  )

  rng_key = jax.random.PRNGKey(_SEED.value)

  init_key, rng_key = jax.random.split(rng_key)
  init_dist = type_mixture_distribution.TypeMixtureTopicDistribution.initialize_randomly(
      rng_key=init_key,
      num_types=_NUM_TYPES.value,
      num_weeks=_NUM_WEEKS.value,
      num_slots=5,
      num_topics=len(topic_ids),
  )

  final_dist = distribution_optimizer.fit_distribution(
      rng_key=rng_key,
      initial_distribution=init_dist,
      queries=problem.queries,
      targets=problem.targets,
      weights=problem.weights,
      batch_size=_BATCH_SIZE.value,
      num_epochs=_NUM_EPOCHS.value,
      loss_fn=lambda g, t: (g - t) ** 2,
      optimizer=optax.adam(_LEARNING_RATE.value),
  )

  final_dist_json = final_dist.format_as_json(topic_ids)
  with open(_OUTPUT_PATH.value, "w") as fh:
    fh.write(final_dist_json)

if __name__ == "__main__":
  app.run(main)
