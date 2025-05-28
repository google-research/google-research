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

"""Parse scoring results to compute exposure for canary data, and parse task performance.


Library for loading the results of the T5X evaluation jobs and outputs the
exposure based on the perplexities. Similarly, load and parses the metrics
computed.
"""

import collections
import glob
import json
import multiprocessing as mp
import os
import pathlib
from typing import Sequence, Mapping, Any, Tuple, Optional

from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests.secret_sharer import exposures

from learn_to_forget.transformers import constants


feature_description = {
    'inputs': tf.io.FixedLenFeature([], dtype=tf.string),
    'targets': tf.io.FixedLenFeature([], dtype=tf.string),
    'num_repetitions': tf.io.FixedLenFeature((), dtype=tf.int64),
}


def get_canaries(freq: Optional[int]) -> Mapping[str, int]:
  """Gets the target canaries and their frequencies from the TFExample files."""
  canary_freq = collections.defaultdict()
  for path in glob.glob(
      os.path.join(constants.CANARY_TFDS_PATH, 'train_*tfrecord*')):
    train_dataset = tf.data.TFRecordDataset(path)
    for record in train_dataset:
      parsed_record = tf.io.parse_single_example(record, feature_description)
      parsed_freq = parsed_record['num_repetitions'].numpy()
      if parsed_freq != freq:
        continue
      targets = parsed_record['targets'].numpy().decode('utf-8')
      canary_freq[targets] = parsed_freq

  return canary_freq


def parse_perplexities(
    output_dir: str, canary_freq: Mapping[str, int]
) -> Tuple[Mapping[int, Sequence[float]], Sequence[float]]:
  """Reads the inference scoring results from a given directory."""

  perplexities_test_set = []
  perplexities_train = collections.defaultdict(list)

  for path in glob.glob(os.path.join(output_dir, '*-score.jsonl*')):
    with open(path, 'r') as f:
      for line in f:
        infer_result = json.loads(line)
        target = infer_result['inputs']['targets_pretokenized']
        if target in canary_freq:
          perplexities_train[canary_freq[target]].append(
              -float(infer_result['score']))
        else:
          perplexities_test_set.append(-float(infer_result['score']))

  if not perplexities_test_set or not perplexities_train:
    raise ValueError('Empty perplexities!')
  return perplexities_train, perplexities_test_set


def compute_exposures(
    batch_size: str,
    train_step: str,
    experiment_dir: str,
    canary_freq: Mapping[str, int],
    parse_individual_exposures: bool = False,
) -> Mapping[str, Any]:
  """Computes the exposures from a set of scoring results at a given path.

  Args:
    batch_size: The batch size of the experiment (unlearning/training).
    train_step: The checkpoint or train step of the model.
    experiment_dir: The path to the scoring results.
    canary_freq: The mapping between the canaries seen during training and their
      frequency.
    parse_individual_exposures: The option to include the individual exposures
      or not.

  Returns:
    The batch_size, train_step, average exposures via extrapolation, average
    exposures via interpolation. If `parse_individual_exposures` is set to True,
    then the invididual exposures are returned.
  """
  result = {'batch_size': int(batch_size), 'train_step': int(train_step)}
  perplexities = parse_perplexities(experiment_dir, canary_freq)
  exposures_extr = exposures.compute_exposure_extrapolation(
      perplexities=perplexities[0],
      perplexities_reference=perplexities[1])

  # Compute the average exposure
  result['exposure_extrapolation'] = {
      int(freq): np.mean(exposures_extr[freq]) for freq in perplexities[0]
  }
  if parse_individual_exposures:
    result['list_exposures_extrapolation'] = {
        int(freq): exposures_extr[freq].tolist() for freq in perplexities[0]
    }

  exposures_inter = exposures.compute_exposure_interpolation(
      perplexities=perplexities[0],
      perplexities_reference=perplexities[1])

  result['exposure_interpolation'] = {
      int(freq): np.mean(exposures_inter[freq]) for freq in perplexities[0]
  }

  if parse_individual_exposures:
    result['list_exposures_interpolation'] = {
        int(freq): exposures_inter[freq].tolist() for freq in perplexities[0]
    }

  return result


def parse_metrics(
    batch_size: str, train_step: str,
    output_dir: str) -> Mapping[str, Any]:
  """Parse the bleu and sequence accuracy metrics for tasks."""
  metrics = {}

  for path in glob.glob(
      os.path.join(output_dir, 'inference_eval/*-metrics.jsonl')):
    with open(path, 'r') as f:
      for line in f:
        result = json.loads(line)
        if int(train_step) != int(result['step']):
          raise ValueError('Incorrect train step')
        metrics['batch_size'] = int(batch_size)
        metrics['train_step'] = int(train_step)
        metrics['bleu'] = result['bleu']
        metrics['sequence_accuracy'] = result['sequence_accuracy']
        break

  return metrics


def get_experiment_config(
    experiments_path: str,
    training_experiment_xid: str) -> Sequence[Tuple[str, str, Any]]:
  """Extracts the config mapping batch size and train step to an experiment path.
  """
  jobs = []
  for experiment_dir in glob.glob(experiments_path):
    if os.path.exists(os.path.join(experiment_dir, 'config.gin')):
      # Parse the checkpoint path and hence train step from the config
      logging.debug('Search in %s', experiment_dir)
      with open(os.path.join(experiment_dir, 'config.gin'), 'r') as f:
        for line in f:
          if training_experiment_xid in line:
            # Get the parent directory of the checkpoint path
            parent_path = pathlib.Path(line.strip(' \'\n')).parents[0]
            train_step = pathlib.Path(line.strip(' \'\n')).name.split('_')[1]
            logging.info(
                'This job was evaluating %s. Opening config.gin to get batch_size'
                , parent_path.as_posix())
            config_path = parent_path / 'config.gin'
            if not config_path.exists():
              logging.debug('%s does not exist. Ignoring %s',
                            config_path.as_posix(), parent_path.as_posix())
              continue
            # Parse the batch size from the config
            with config_path.open() as configf:
              for config_line in configf:
                if 'BATCH_SIZE = ' in config_line:
                  batch_size = config_line.strip(' \n').split('=')[1].strip()

                  if (batch_size, train_step) in jobs:
                    print('Found duplicate results for {}, {}!'.format(
                        batch_size, train_step))
                  else:
                    jobs.append((batch_size, train_step, experiment_dir))
                  break
            break

  return jobs


def parse_task_results(experiments_path: str, training_experiment_xid: str,
                       output_res_path: str) -> None:
  """Parses and writes to json the performance results from multiple checkpoints."""
  jobs = get_experiment_config(experiments_path, training_experiment_xid)

  logging.debug(jobs)
  logging.info('Total jobs: %d', len(jobs))

  task_results = []

  def accumulate(result):
    task_results.append(result)

  pool = mp.Pool(processes=6)
  for job in jobs:
    pool.apply_async(
        parse_metrics,
        args=(
            job[0],
            job[1],
            job[2],
        ),
        callback=accumulate,
        error_callback=print)
  pool.close()
  pool.join()

  with open(
      os.path.join(output_res_path, 'performance_results.jsonl'), 'w') as f:
    for entry in task_results:
      json.dump(entry, f)
      f.write('\n')


def parse_exposure_results(experiments_path: str, training_experiment_xid: str,
                           output_res_path: str, freq: int) -> None:
  """Parses and writes to csv the exposure results of multiple experiments."""
  jobs = get_experiment_config(experiments_path, training_experiment_xid)

  logging.debug(jobs)
  logging.info('Total jobs: %d', len(jobs))

  exposure_results = []
  canary_freq = get_canaries(freq)

  def accumulate(result):
    exposure_results.append(result)

  pool = mp.Pool(processes=6)
  for job in jobs:
    pool.apply_async(
        compute_exposures,
        args=(
            job[0],
            job[1],
            job[2],
            canary_freq,
        ),
        callback=accumulate,
        error_callback=print)
  pool.close()
  pool.join()

  with open(
      os.path.join(output_res_path, 'exposure_results.jsonl'), 'w') as f:
    for entry in exposure_results:
      json.dump(entry, f)
      f.write('\n')
