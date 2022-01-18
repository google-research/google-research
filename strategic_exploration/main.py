# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

import argparse
import logging
import numpy as np
import os
import random
import socket
import torch

from gtd.io import save_stdout
from gtd.utils import Config
from strategic_exploration.hrl.training_run import HRLTrainingRuns
from os.path import abspath, dirname, join

# CONFIGS ARE MERGED IN THE FOLLOWING ORDER:
# 1. configs in args.config_paths, from left to right
# 2. task config
# 3. config_strings

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--check_commit', default='strict')
arg_parser.add_argument('-d', '--description', default='None.')
arg_parser.add_argument('-n', '--name', default='unnamed')
arg_parser.add_argument('-r', '--seed', default=0, type=int)
arg_parser.add_argument('-x', '--reward_configs', action='append', default=[])
arg_parser.add_argument('config_paths', nargs='+')
args = arg_parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# create run
runs = HRLTrainingRuns(check_commit=(args.check_commit == 'strict'))

config_paths = args.config_paths
if len(config_paths) == 1 and config_paths[0].isdigit():
  configs = [Config.from_file(p) for p in args.reward_configs]
  run = runs.clone(int(config_paths[0]), configs, args.name)
else:
  # new run according to configs
  configs = [Config.from_file(p) for p in config_paths]

  # merge all configs together
  config = Config.merge(configs)  # later configs overwrite earlier configs
  run = runs.new(config, name=args.name)  # new run from config

  run.metadata['description'] = args.description
  run.metadata['name'] = args.name

run.metadata['host'] = socket.gethostname()

# start training
run.workspace.add_file('stdout', 'stdout.txt')
run.workspace.add_file('stderr', 'stderr.txt')

with save_stdout(run.workspace.root):
  try:
    run.train()
  finally:
    run.close()
