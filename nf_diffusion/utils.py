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

"""Router for loading different datasets according to the configuration.

The dataset files are stored in nf_diffusion/datasets
"""
import importlib
from typing import Tuple

import jax
import ml_collections
import tensorflow as tf


def merge_batch_stats(replicated_state):
  """Merge model batch stats."""
  if hasattr(replicated_state, "batch_stats") and jax.tree.leaves(
      replicated_state.batch_stats
  ):
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
    return replicated_state.replace(
        batch_stats=cross_replica_mean(replicated_state.batch_stats)
    )
  else:
    return replicated_state


_DEFAULT_PACKAGE_PATH = "nf_diffusion"


def load_trainer_module(
    trainer_name,
    base_package_name = _DEFAULT_PACKAGE_PATH,
):
  """Load the trainer module according to the configuration.

  Args:
    trainer_name: The name of the trainer, which refers to importing the Module
      under [base_package_name].
    base_package_name: The name of the based package.

  Returns:
    mod: Trainer module that includes required functions for the internal loop
      of [train_and_evaluate] function.
  """

  mod = importlib.import_module(f"{base_package_name}.trainers.{trainer_name}")
  return mod


def load_tester_module(
    tester_name,
    base_package_name = _DEFAULT_PACKAGE_PATH,
):
  """Load the trainer module according to the configuration.

  Args:
    tester_name: The name of the tester, which refers to importing the Module
      under [base_package_name].
    base_package_name: The name of the based package.

  Returns:
    mod: Trainer module that includes required functions for the internal loop
      of [train_and_evaluate] function.
  """

  mod = importlib.import_module(f"{base_package_name}.testers.{tester_name}")
  return mod


def create_datasets(
    config,
    data_rng,
    base_package_name = _DEFAULT_PACKAGE_PATH,
):
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations for reproducible training.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.
    base_package_name: The name of the base package.

  Returns:
    A tuple with the total number of training batches info, the training dataset
    and the evaluation dataset.
  """
  data_name = config.data.name.lower()
  mod = importlib.import_module(f"{base_package_name}.datasets.{data_name}")
  return mod.create_datasets(config, data_rng)
