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

"""Functions for concurrently running self-consistency experiments on datasets."""

from concurrent import futures
import dataclasses
import json
import os
import pickle

import pandas as pd
import tensorflow.compat.v2 as tf
import tqdm

from cisc.src import confidence_extraction
from cisc.src import self_consistency
from cisc.src.datasets import bbh
from cisc.src.datasets import dataset as dataset_lib
from cisc.src.datasets import gsm8k
from cisc.src.datasets import math as math_dataset
from cisc.src.datasets import mmlu_pro
from cisc.src.runners import runner as runner_lib

SelfConsistencyResult = self_consistency.SelfConsistencyResult
ConfidenceOptions = list[int] | list[str]


gfile = tf.io.gfile


@dataclasses.dataclass()
class ExperimentConfiguration:
  """Holds all the experiment meta-parameters."""

  num_traces: int
  num_rows: int
  temperature: float
  max_num_tokens: int

  # The confidence extraction configuration.
  confidence_config: confidence_extraction.AggregatedConfidenceExtractionConfig

  tag: str = ""  # Free text tag to identify the experiment.

  def __post_init__(self):
    """Validations."""
    assert self.max_num_tokens > 0

  def to_path_str(self):
    return (
        str(self)
        .removeprefix("ExperimentConfiguration(")
        .replace(", ", "_")
        .removesuffix(")")
    )


@dataclasses.dataclass()
class ExperimentResult:
  """The result of running the experiment on a single dataset."""

  # Config used for running the experiment.
  experiment_configuration: ExperimentConfiguration
  # The name of the dataset - e.g., MMLU.
  dataset_name: str
  # The results of the self-consistency eval.
  self_consistency_results: list[SelfConsistencyResult]
  # Populated after `populate_results_df` is called. This is convenient for
  # not storing the results on disk twice.
  results_df: pd.DataFrame | None = None

  def get_results_df(self):
    """Populates the results dataframe from the self-consistency results."""
    if self.results_df is not None:
      return self.results_df
    self.results_df = self_consistency.results_to_dataframe(
        self.self_consistency_results
    )
    for expected_col in [
        "question_id",
        "verbal_confidence",
        "answer",
        "is_correct",
    ]:
      if expected_col not in self.results_df.columns:
        raise ValueError(
            f"Expected column {expected_col} not found in the results"
            " dataframe."
        )
    return self.results_df


def write_to_disk_as_pickle(
    exp_results,
    dir_name,
    file_name="experiment_output.pkl",
    also_write_config=True,
):
  """Writes the experiment results and configuration to CNS file."""
  gfile.makedirs(dir_name)
  file_name = os.path.join(dir_name, file_name)
  print("Writing to cns as pickle: ", file_name)
  with gfile.GFile(file_name, "wb") as f:  # this is not working ...
    pickle.dump(exp_results, f)

  config = exp_results.experiment_configuration
  if config and also_write_config:
    file_name = os.path.join(dir_name, "experiment_conf.pkl")
    with gfile.GFile(file_name, "wb") as f:
      json.dump(dataclasses.asdict(config), f)


def enrich_traces_with_confidence_inplace(
    sc_result,
    runner,
    config,
):
  """Enriches the traces with confidence."""
  # Consider adding parallelism here. Saying that, we should have a lot of
  # parallelism already at this stage, and I don't want to overload the
  # server (we already had crashes before - see cl/708778110).
  for trace in sc_result.traces:
    trace.confidence = confidence_extraction.generate_confidence(
        trace.prompt,
        trace.response,
        trace.answer_span,
        runner,
        config,
        trace.confidence,
    )
  return sc_result


def run_question_answering_on_dataset(
    runner,
    dataset,
    config,
    pbar,
    max_workers,
):
  """Runs the `runner` on the `dataset` concurrently.

  Args:
    runner: the runner which runs the model
    dataset: the dataset on which to run the eval
    config: experiment meta-prameters
    pbar: progress bar to update while running
    max_workers: max number of concurrent workers to use

  Returns:
    self_consistency_results: the results of the self-consistency eval
  """
  self_consistency_futures: list[futures.Future[SelfConsistencyResult]] = []
  with futures.ThreadPoolExecutor(
      min(max_workers, len(dataset.data))
  ) as executor:
    for _, row in dataset.data.iterrows():
      formatted_question = dataset.format_question(row.question)
      self_consistency_futures.append(
          executor.submit(
              self_consistency.run_self_consistency,
              runner=runner,
              question_id=row.question_id,
              prompt=formatted_question,
              temp=config.temperature,
              num_traces=config.num_traces,
              num_tokens=config.max_num_tokens,
              dataset=dataset,
          )
      )

    # Wait and update progress bar.
    for _ in futures.as_completed(self_consistency_futures):
      pbar.update(config.num_traces)

  # Parse futures.
  self_consistency_results = []
  for i, f in enumerate(self_consistency_futures):
    if f.exception():
      raise f.exception()
    original_row = dataset.data.iloc[i]
    result = f.result()
    result.golden_label = original_row.golden_label
    result.original_row = original_row
    self_consistency_results.append(result)
  return self_consistency_results


def load_dataset(
    dataset_name,
):
  if dataset_name == "GSM8K":
    return gsm8k.get_dataset()
  elif dataset_name == "MMLU":
    return mmlu_pro.get_dataset()
  elif dataset_name == "MATH":
    return math_dataset.get_dataset()
  elif dataset_name == "BBH":
    return bbh.get_dataset()
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")


def enrich_dataset_with_confidence(
    input_results,
    runner,
    config,
    pbar,
    max_workers,
):
  """Enriches the traces with confidence."""
  self_consistency_futures: list[futures.Future[SelfConsistencyResult]] = []
  with futures.ThreadPoolExecutor(
      min(max_workers, len(input_results))
  ) as executor:
    for result in input_results:
      self_consistency_futures.append(
          executor.submit(
              enrich_traces_with_confidence_inplace,
              sc_result=result,
              runner=runner,
              config=config,
          )
      )

    # Wait and update progress bar.
    for _ in futures.as_completed(self_consistency_futures):
      pbar.update(1)

  # Parse futures.
  self_consistency_results = []
  for f in self_consistency_futures:
    if f.exception():
      raise f.exception()
    self_consistency_results.append(f.result())
  return self_consistency_results


def run_confidence_extraction_on_experiment_results(
    runner,
    experiment_results,
    config,
    max_workers,
    output_base_dir,
):
  """Runs the confidence extraction on the experiment results."""
  for experiment_result in experiment_results:
    experiment_result.experiment_configuration.confidence_config = config
    sc_results = experiment_result.self_consistency_results
    with tqdm.tqdm(total=len(sc_results)) as pbar:
      experiment_result.self_consistency_results = (
          enrich_dataset_with_confidence(
              sc_results, runner, config, pbar, max_workers
          )
      )
    write_to_disk_as_pickle(
        experiment_result,
        dir_name=os.path.join(output_base_dir, experiment_result.dataset_name),
    )
  return experiment_results


def run_question_answering_on_datasets(
    runner,
    dataset_names,
    config,
    max_workers,
    output_base_dir,
    seed=1337,
):
  """Runs the `runner` on the `dataset` concurrently.

  Args:
    runner: the runner which runs the model
    dataset_names: the names of the dataset_names to run on. Subset of ["GSM8K",
      "MMLU", "MATH"].
    config: the experiment meta-prameters
    max_workers: max number of concurrentworkers to use
    output_base_dir: the base directory to write the results to.
    seed: the seed to use when sampling the dataset (i.e., when we don't want to
      run on the entire dataset). Sampling is important (instead of taking
      head), as dataset like MMLU might be sorted by category.

  Returns:
    a list of dataframes, each containing the results of a single dataset.
  """
  if not dataset_names:
    raise ValueError("No dataset names provided.")
  all_datasets_outputs = []
  for dataset_name in dataset_names:
    # Run on dataset.
    dataset = load_dataset(dataset_name)
    num_rows = min(config.num_rows, len(dataset.data))
    dataset.data = dataset.data.sample(num_rows, random_state=seed)
    print(f"Running on dataset: {dataset_name}")
    with tqdm.tqdm(total=num_rows * config.num_traces) as pbar:
      results = run_question_answering_on_dataset(
          runner, dataset, config, pbar, max_workers
      )
    output = ExperimentResult(
        experiment_configuration=config,
        dataset_name=dataset_name,
        self_consistency_results=results,
    )
    all_datasets_outputs.append(output)

    write_to_disk_as_pickle(
        output, dir_name=os.path.join(output_base_dir, dataset_name)
    )

  return all_datasets_outputs


def load_dataset_from_disk(
    dataset_dir, file_name="experiment_output.pkl"
):
  """Loads the experiment results from the given directory."""
  with gfile.Open(os.path.join(dataset_dir, file_name), "rb") as f:
    exp_result = pickle.load(f)
  return exp_result


def get_dataset_dirs(
    dir_name,
    version = None,
):
  """Returns the list of dataset directories in a given model directory.

    It is assumed the model directory structure is as follows:
      `dir_name`
        ├── version A
            ├── DATASET 1
            │   ├── [experiment_output_file]
            │   ├── [experiment_conf_file]
            └── DATASET 2
            └── ...
        ├── version B
        ...

  Args:
    dir_name: the top level directory.
    version: if given, takes this version. Othewise asserts that there is only
      one version and takes it.

  Returns:
    A list of fully qualified dataset directories - one for each dataset.
  """
  versions_dir = gfile.ListDir(dir_name)
  if version is None:
    if len(versions_dir) != 1:
      raise ValueError(
          f"Expected exactly one version in {dir_name}, got"
          f" {len(versions_dir)}. Available versions: {versions_dir}"
      )
    version = versions_dir[0]
  else:
    if version not in versions_dir:
      raise ValueError(
          f"Version {version} not found in {dir_name}. Available versions:"
          f" {versions_dir}"
      )
  dataset_dirs = []
  version_dir = os.path.join(dir_name, version)
  for dataset_name in gfile.ListDir(version_dir):
    dataset_full_path = os.path.join(version_dir, dataset_name)
    if not gfile.IsDirectory(dataset_full_path):
      continue
    dataset_dirs.append(dataset_full_path)
  return dataset_dirs


def load_all_experiment_results(
    dir_name,
    version = None,
    file_name="experiment_output.pkl",
):
  """Loads all the experiment results from the given directory.

  See `get_dataset_dirs` for the expected directory structure.

  Args:
    dir_name: the top level directory.
    version: if given, takes this version. Othewise asserts that there is only
      one version and takes it.
    file_name: the name of the file to load.

  Returns:
    A list of `ExperimentResult` - one for each dataset.
  """
  dataset_dirs = get_dataset_dirs(dir_name, version)
  with futures.ThreadPoolExecutor(max_workers=len(dataset_dirs)) as executor:
    return list(
        executor.map(
            lambda x: load_dataset_from_disk(x, file_name), dataset_dirs
        )
    )
