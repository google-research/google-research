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

"""Utilities for processing the experimental artifacts for evaluation."""

import glob
import json
import os
import re

from vrdu import benchmark_utils


def decode_split_name(split_name):
  """Decodes the split file name to get multiple meta data of the experiment."""
  pattern = r'([a-zA-Z]+)-([0-9a-zA-Z_-]+)-train_(\d+)-test_(\d+)-valid_(\d+)-SD_(\d+)'
  match = re.match(pattern, split_name)
  if match is not None:
    return {
        'dataset': match.group(1),
        'task': match.group(2),
        'train_size': int(match.group(3)),
        'test_size': int(match.group(4)),
        'valid_size': int(match.group(5)),
        'seed': int(match.group(6))}
  else:
    raise ValueError('The split file name %s cannot be decoded.' % split_name)


def load_experiments(base_dirpath, extraction_dirpath):
  """Loads splits and model extraction results.

  Args:
    base_dirpath: Path that contains both `main/` which stores the dataset
      documents and `few_shot-splits/` which stores the split JSON files of
      the tasks to be evaluated.
    extraction_dirpath: Path that contains the extraction results which all
      start with the corresponding split file name and end with
      `-test_predictions.json`.

  Returns:
    Object of class `DataUtils` which loads the dataset and List of Dictionary,
    each of which describes the paths of extraction and split file of one task.

  Raises:
    FileNotFoundError: raised when no files is found in either `base_dirpath` or
    `extraction_dirpath`.
  """
  ground_truth = benchmark_utils.DataUtils(os.path.join(base_dirpath, 'main/'))

  extraction_files = sorted(
      glob.glob(os.path.join(extraction_dirpath, '*-test_predictions.json')))
  if not extraction_files:
    raise FileNotFoundError(
        'Extraction file path %s does not exist or contain any extractions.' %
        extraction_dirpath)

  experiments = []
  for extraction_file in extraction_files:
    split_name = os.path.split(extraction_file)[1].replace(
        '-test_predictions.json', '')
    split_file = os.path.join(base_dirpath,
                              'few_shot-splits/{}.json'.format(split_name))
    if not os.path.exists(split_file):
      raise FileNotFoundError('Split file %s is not found.' % split_file)
    split_info = decode_split_name(split_name)
    experiments.append({
        'split_file': split_file,
        'extraction_file': extraction_file
    } | split_info)
  return ground_truth, experiments


def evaluate_experiments(benchmark, experiments):
  """Evaluates the extraction results by computing F1 scores."""
  evals = []
  for experiment in experiments:
    extractions = json.load(open(experiment['extraction_file']))
    benchmark.update_splits(experiment['split_file'])
    eval_res = benchmark_utils.evaluate(extractions, benchmark)
    eval_res = {f'metric-{k}': v for k, v in eval_res.items() if 'f1' in k}
    evals.append(experiment | eval_res)
  return evals
