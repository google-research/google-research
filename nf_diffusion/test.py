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

"""Continuous testing code."""

import json
import os

from absl import logging
from clu import checkpoint
from clu import metric_writers
from etils import epath
import jax
import ml_collections
import numpy as np

from nf_diffusion import utils


def test(config, workdir):
  """Runs a training and evaluation loop with multiple accelerators.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Load the trainer config and module
  host_id = jax.process_index()
  config_path = epath.Path(config.train_path) / "config.json"
  logging.info("Config path: %s", config_path)
  train_config = json.loads(config_path.read_text())
  if isinstance(train_config, str):
    train_config = json.loads(train_config)
  train_config = ml_collections.ConfigDict(initial_dictionary=train_config)
  trainer = utils.load_trainer_module(train_config.trainer.name)

  test_dir_name = config.get("test_dir_name", "test")
  workdir = epath.Path(workdir)
  if config.get("test_path", None) is None:
    testdir = workdir / test_dir_name
  else:
    testdir = epath.Path(config.test_path) / test_dir_name
  workdir.mkdir(parents=True, exist_ok=True)
  testdir.mkdir(parents=True, exist_ok=True)

  # Deterministic training.
  rng = jax.random.PRNGKey(config.seed)
  rng = jax.random.fold_in(rng, host_id)

  # Load training side information.
  # 1. Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  data_info, train_ds, val_ds = utils.create_datasets(train_config, data_rng)
  num_train_steps = data_info.num_train_steps
  # 2. Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, _, state = trainer.create_train_state(
      train_config, model_rng, data_info
  )
  # Now loading tester information
  # 1. Loading testing dataset
  if config.test_split == "train":
    test_ds = train_ds
  elif config.test_split == "val":
    test_ds = val_ds
  elif config.test_split == "test":
    # Load test data from test configuration
    _, _, test_ds = utils.create_datasets(config, data_rng)

  # 2. Get writer information (to put into tensorboard)
  logging.info("Set up metric writers.")
  writer = metric_writers.create_default_writer(
      testdir, just_logging=jax.process_index() > 0
  )
  writer.write_hparams(dict(config))
  # Load tester module, checkpoint steps
  tester = utils.load_tester_module(config.tester.name)
  checkpoint_steps = []

  logging.info("Initialize loop")
  while True:
    # Load newst checkpointing of the model and the input pipeline.
    checkpoint_dir = os.path.join(config.train_path, "checkpoints")
    ckpt = checkpoint.MultihostCheckpoint(
        checkpoint_dir,
        max_to_keep=train_config.get("max_checkpoints_to_keep", 2),
    )
    state = ckpt.restore_or_initialize(state)
    curr_checkpoint_step = int(state.step)
    logging.info("Resume step: %d", curr_checkpoint_step)
    logging.info("Max training step: %d", num_train_steps)
    logging.info("Existing checkpoints: %s", checkpoint_steps)
    if curr_checkpoint_step in checkpoint_steps:
      sleep_seconds = int(config.get("sleep_time_duration", 60))
      logging.info("Checkpoint exist, sleep for %d seconds", sleep_seconds)
      # NOTE: this will cause the machine to kill the job, we will recompute the
      #       metrics on the same old checkpoint in case new checkpoint hasn't
      #       comes in before the loop completed.
      # time.sleep(sleep_seconds)
      # continue
    checkpoint_steps.append(curr_checkpoint_step)

    eval_metrics = None
    with metric_writers.ensure_flushes(writer):
      rng, rng_e = jax.random.split(rng)
      eval_metrics, eval_info = tester.test(
          config, train_config, model, state, test_ds, rng_e
      )

      if eval_metrics is not None:
        eval_metrics_cpu = jax.tree_util.tree_map(
            np.array, eval_metrics.compute()
        )
        eval_metrics_cpu = {
            "test/{}".format(k): v for k, v in eval_metrics_cpu.items()
        }
        writer.write_scalars(int(state.step), eval_metrics_cpu)

      if hasattr(tester, "eval_visualize"):
        tester.eval_visualize(
            config, writer, int(state.step), model, state, eval_info, testdir
        )
    logging.info("Finishing evaluation step %d", int(state.step))
    if curr_checkpoint_step >= num_train_steps:
      logging.info(
          "Finishing evaluate all checkpoints %d/%d",
          curr_checkpoint_step,
          num_train_steps,
      )
      break

    if not config.get("continue_run", True):
      break
  logging.info("Finishing evaluation.")
