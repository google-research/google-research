# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Functions for data loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import pandas as pd


def _list_subdirs(base_dir):
  """Lists all subdirectories in base_dir, raising ValueError if none exist."""
  subdirs = []
  for dir_entry in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, dir_entry)):
      subdirs.append(dir_entry)

  if not subdirs:
    raise ValueError("No subdirectories found in {}".format(base_dir))

  return subdirs


def load_study(study_dir,
               num_trials=None,
               load_complete_trials=True,
               load_incomplete_trials=False,
               load_infeasible_trials=False):
  """Loads measurements for all trials in a study.

  A study is a metaparameter search for a given workload and batch size. A trial
  is a training run of a particular metaparameter configuration within the
  study.

  Args:
    study_dir: Directory containing 'study.json' and subdirectories
      corresponding to individual trials.
    num_trials: The number of trials to load. Default is to load all trials.
    load_complete_trials: Whether to load complete trials.
    load_incomplete_trials: Whether to load incomplete trials.
    load_infeasible_trials: Whether to load infeasible trials.

  Returns:
    study_metadata: A dict of study metadata.
    study_measurements: A Pandas DataFrame indexed by (trial_id, step).

  Raises:
    ValueError: If none of load_complete_trials, load_incomplete_trials, or
      load_infeasible_trials is True.
  """
  # Determine which trial statuses to load.
  status_whitelist = set()
  if load_complete_trials:
    status_whitelist.add("COMPLETE")
  if load_incomplete_trials:
    status_whitelist.add("INCOMPLETE")
  if load_infeasible_trials:
    status_whitelist.add("INFEASIBLE")
  if not status_whitelist:
    raise ValueError(
        "At least one of load_complete_trials, load_incomplete_trials, or "
        "load_infeasible_trials must be True.")

  trial_ids = []
  measurements_tables = []

  # Load the study metadata.
  with open(os.path.join(study_dir, "study.json")) as study_file:
    study_metadata = json.load(study_file)
  study_metadata["trials"] = collections.OrderedDict()

  # Find all trial directories.
  trial_dirs = _list_subdirs(study_dir)
  trial_dirs.sort(key=int)  # Trial directory names are integers.

  for trial_dir in trial_dirs:
    # Load trial metadata.
    trial_dir = os.path.join(study_dir, trial_dir)
    with open(os.path.join(trial_dir, "metadata.json")) as metadata_file:
      trial_metadata = json.load(metadata_file)

    # Ignore trials with the wrong status.
    status = trial_metadata["status"]
    if status_whitelist and status not in status_whitelist:
      continue

    # Add trial metadata to the study metadata.
    trial_id = trial_metadata["trial_id"]
    trial_ids.append(trial_id)
    study_metadata["trials"][trial_id] = trial_metadata

    # Read the measurements.
    measurements_file = os.path.join(trial_dir, "measurements.csv")
    measurements_tables.append(pd.read_csv(measurements_file, index_col="step"))

    if num_trials and len(trial_ids) >= num_trials:
      break  # Already loaded the required number of trials.

  # Validate the number of trials.
  if not trial_ids:
    raise ValueError("No trials with status {} found in {}".format(
        list(status_whitelist), study_dir))

  if num_trials and len(trial_ids) != num_trials:
    raise ValueError(
        "Requested {} trials with status {}, but found only {} trials in {}"
        .format(num_trials, list(status_whitelist), len(trial_ids), study_dir))

  study_measurements = pd.concat(
      measurements_tables, keys=trial_ids, names=["trial_id"])
  return study_metadata, study_measurements


def load_workload(workload_dir,
                  num_trials=None,
                  load_complete_trials=True,
                  load_incomplete_trials=False,
                  load_infeasible_trials=False):
  """Loads all studies within a given workload.

  A workload is a triplet of (dataset, model, optimizer). A study is a
  metaparameter search for a given workload and batch size. A study is
  comprised of trials. A trial is a training run of a particular metaparameter
  configuration.

  Args:
    workload_dir: Directory containing subdirectories corresponding to
      individual studies.
    num_trials: The number of trials to load per study. Default is to load all
      trials.
    load_complete_trials: Whether to load complete trials.
    load_incomplete_trials: Whether to load incomplete trials.
    load_infeasible_trials: Whether to load infeasible trials.

  Returns:
    workload_metadata: A dict containing the metadata for each study.
    workload_table: A Pandas DataFrame indexed by (batch_size, trial_id, step).
  """
  batch_sizes = []
  study_tables = []
  workload_metadata = collections.OrderedDict()

  study_dirs = _list_subdirs(workload_dir)
  study_dirs.sort(key=int)  # Study directory names are integers.

  for study_dir in study_dirs:
    study_metadata, study_measurements = load_study(
        os.path.join(workload_dir, study_dir),
        num_trials=num_trials,
        load_complete_trials=load_complete_trials,
        load_incomplete_trials=load_incomplete_trials,
        load_infeasible_trials=load_infeasible_trials)

    batch_size = int(study_metadata["batch_size"])
    batch_sizes.append(batch_size)
    study_tables.append(study_measurements)
    workload_metadata[batch_size] = study_metadata

  print("Loaded {} batch sizes for {} on {} with optimizer {}".format(
      len(batch_sizes), study_metadata["model"], study_metadata["dataset"],
      study_metadata["optimizer"]))

  workload_table = pd.concat(
      study_tables, keys=batch_sizes, names=["batch_size"])
  return workload_metadata, workload_table
