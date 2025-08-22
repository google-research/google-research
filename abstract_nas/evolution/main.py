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

"""Entry point for an evolutionary NAS experiment."""

from collections.abc import Sequence
import csv
import functools
import os
import time
from typing import List

from absl import app
from absl import flags
from absl import logging
import ml_collections as mlc
from ml_collections.config_flags import config_flags

from abstract_nas.evolution import utils as evol_utils
from abstract_nas.train import config as train_config
from abstract_nas.train import train
from abstract_nas.zoo import utils as zoo_utils

_MAX_NUM_TRIALS = flags.DEFINE_integer("max_num_trials", 50,
                                       "The total number of trials to run.")
_RESULTS_DIR = flags.DEFINE_string(
    "results_dir", "", "The directory in which to write the results.")
_WRITE_CHECKPOINTS = flags.DEFINE_bool(
    "checkpoints", False, "Whether to checkpoint the evolved models.")
_STUDY_NAME = flags.DEFINE_string(
    "study_name", "abstract_nas_demo",
    "The name of this study, used for writing results.")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config")


def save_results(population):
  """Saves results as csv."""
  results_dir = _RESULTS_DIR.value
  study_name = _STUDY_NAME.value
  if not results_dir:
    logging.warn("No results_dir defined, so skipping saving results.")
    return

  results_filename = f"{results_dir}/{study_name}.csv"
  if os.path.exists(results_filename):
    logging.warn("Results file %s already exists, so skipping saving results.",
                 results_filename)
    return

  logging.info("Writing results to %s.", results_filename)
  data = [(generation, i.creation_time_sec, i.completion_time_sec, i.accuracy,
           i.im_sec_core_train, i.flops, i.num_params)
          for generation, i in enumerate(population)]
  with open(results_filename, "wt") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow([
        "generation", "creation_time_sec", "completion_time_sec", "accuracy",
        "im_sec_core_train", "gflops", "params_m"
    ])
    writer.writerows(data)


def run_experiment(config):
  """Runs an evolutionary NAS experiment."""

  model = config.train.model_name.lower()
  num_classes = config.train.dataset.num_classes
  spatial_size = config.train.dataset.input_shape[0]
  model_fn = zoo_utils.get_model_fn(model, num_classes=num_classes,
                                    spatial_res=spatial_size)

  population_manager = evol_utils.PopulationManager(config)
  mutator = evol_utils.ModelMutator(config)
  make_config = functools.partial(train_config.Config, config_dict=config)

  write_checkpoints = _WRITE_CHECKPOINTS.value
  results_dir = _RESULTS_DIR.value
  if write_checkpoints and results_dir:
    make_path = lambda x: f"{results_dir}/checkpoint/generation{x}"
  else:
    make_path = lambda _: None

  population = []
  base_individuals = list(range(config.evolution.num_seed))
  for generation_id in range(_MAX_NUM_TRIALS.value):
    start_time = int(time.time())

    if generation_id < config.evolution.num_seed:
      new_graph, new_constants, new_blocks = model_fn()
    else:
      parent_id = population_manager.select_individuals(
          [p[1] for p in population],
          num_suggestions_hint=1,
          base_individuals=base_individuals)[0]
      parent_graph, parent_constants, parent_blocks = population[parent_id][0]
      new_graph, new_constants, new_blocks, _ = mutator.mutate(
          parent_graph, parent_constants, parent_blocks, model_fn,
          generation_id)

    if new_graph:
      trial_config = make_config(
          graph=(new_graph, new_constants),
          output_dir=make_path(generation_id))
      results, _, _ = train.train_and_eval(trial_config)

      end_time = int(time.time())

      individual = evol_utils.Individual(
          completion_time_sec=end_time,
          creation_time_sec=start_time,
          accuracy=results.acc * 100,
          im_sec_core_train=results.im_sec_core_train,
          flops=results.flops / 10**9,
          num_params=results.num_params / 10**6)
    else:
      end_time = int(time.time())
      individual = evol_utils.Individual(
          completion_time_sec=end_time,
          creation_time_sec=start_time,
          accuracy=-1.,
          im_sec_core_train=float("inf"),
          flops=float("inf"),
          num_params=float("inf"))
    logging.info("Generation %d: %s", generation_id, individual)

    model_and_individual = ((new_graph, new_constants, new_blocks), individual)
    population.append(model_and_individual)

  save_results([p[1] for p in population])


def main(argv):
  del argv
  run_experiment(FLAGS.config)


if __name__ == "__main__":
  app.run(main)
