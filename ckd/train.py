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

# pylint: disable=missing-module-docstring
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=consider-using-from-import
# pylint: disable=missing-function-docstring

import argparse
import os
import random

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now
# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
import lavis.tasks as tasks
from lavis.tasks import *
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def parse_args():
  parser = argparse.ArgumentParser(description="Training")

  parser.add_argument(
      "--cfg-path", required=True, help="path to configuration file."
  )
  parser.add_argument(
      "--options",
      nargs="+",
      help=(
          "override some settings in the used config, the key-value pair "
          "in xxx=yyy format will be merged into config file (deprecate), "
          "change to --cfg-options instead."
      ),
  )

  args = parser.parse_args()
  # if 'LOCAL_RANK' not in os.environ:
  #     os.environ['LOCAL_RANK'] = str(args.local_rank)

  return args


def setup_seeds(config):
  seed = config.run_cfg.seed + get_rank()

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  cudnn.benchmark = False
  cudnn.deterministic = True


def get_runner_class(cfg):
  """Get runner class from config. Default to epoch-based runner."""
  runner_cls = registry.get_runner_class(
      cfg.run_cfg.get("runner", "runner_base")
  )

  return runner_cls


def main():
  # pylint: disable=line-too-long
  # allow auto-dl completes on main process without timeout when using NCCL backend.
  # os.environ["NCCL_BLOCKING_WAIT"] = "1"

  # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
  job_id = now()

  cfg = Config(parse_args())

  init_distributed_mode(cfg.run_cfg)

  setup_seeds(cfg)

  # set after init_distributed_mode() to only log on master.
  setup_logger()

  cfg.pretty_print()

  task = tasks.setup_task(cfg)
  datasets = task.build_datasets(cfg)
  model = task.build_model(cfg)

  runner = get_runner_class(cfg)(
      cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
  )
  runner.train()


if __name__ == "__main__":
  main()
