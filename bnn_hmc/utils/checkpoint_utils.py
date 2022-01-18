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

# Copyright 2020 The Google Research Authors.
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
"""Utility functions for checkpointing."""

import os
import pickle
import re
from haiku._src.data_structures import FlatMapping
from enum import Enum

_CHECKPOINT_FORMAT_STRING = "model_step_{}.pt"


class InitStatus(Enum):
  INIT_RANDOM = 0
  INIT_CKPT = 1
  LOADED_PREEMPTED = 2


def load_checkpoint(path):
  with open(path, "rb") as f:
    checkpoint_dict = pickle.load(f)
  return checkpoint_dict


def save_checkpoint(path, checkpoint_dict):
  with open(path, "wb") as f:
    pickle.dump(checkpoint_dict, f)


def _checkpoint_pattern():
  pattern_string = _CHECKPOINT_FORMAT_STRING.format("(?P<step>[0-9]+)")
  return re.compile(pattern_string)


def _match_checkpoint_pattern(name):
  pattern = _checkpoint_pattern()
  return pattern.match(name)


def name_is_checkpoint(name):
  return bool(_match_checkpoint_pattern(name))


def parse_checkpoint_name(name):
  match = _match_checkpoint_pattern(name)
  return int(match.group("step"))


def make_checkpoint_name(step):
  return _CHECKPOINT_FORMAT_STRING.format(step)


def initialize(dirname, init_checkpoint):
  checkpoints = filter(name_is_checkpoint, os.listdir(dirname))
  checkpoints = list(checkpoints)
  if checkpoints:
    checkpoint_iteration = map(parse_checkpoint_name, checkpoints)
    start_iteration = max(checkpoint_iteration)
    start_checkpoint_path = (
        os.path.join(dirname, make_checkpoint_name(start_iteration)))
    checkpoint_dict = load_checkpoint(start_checkpoint_path)
    checkpoint_dict["filename_iteration"] = start_iteration
    return checkpoint_dict, InitStatus.LOADED_PREEMPTED
  else:
    if init_checkpoint is not None:
      return load_checkpoint(init_checkpoint), InitStatus.INIT_CKPT
    else:
      return None, InitStatus.INIT_RANDOM


def make_hmc_checkpoint_dict(iteration, params, state, key, step_size, accepted,
                             num_ensembled, ensemble_predictions):
  checkpoint_dict = {
      "iteration": iteration,
      "params": params,
      "state": state,
      "key": key,
      "step_size": step_size,
      "accepted": accepted,
      "num_ensembled": num_ensembled,
      "ensemble_predictions": ensemble_predictions
  }
  return checkpoint_dict


def parse_hmc_checkpoint_dict(checkpoint_dict):
  if "iteration" not in checkpoint_dict.keys():
    checkpoint_dict["iteration"] = checkpoint_dict["filename_iteration"]
  if "state" not in checkpoint_dict.keys():
    checkpoint_dict["state"] = FlatMapping({})
  for key in ["accepted", "num_ensembled", "ensemble_predicted_probs"]:
    if key not in checkpoint_dict.keys():
      checkpoint_dict[key] = None
  field_names = [
      "iteration", "params", "state", "key", "step_size", "accepted",
      "num_ensembled", "ensemble_predictions"
  ]
  return [checkpoint_dict[name] for name in field_names]


def make_sgd_checkpoint_dict(iteration, params, net_state, opt_state, key):
  checkpoint_dict = {
      "iteration": iteration,
      "params": params,
      "net_state": net_state,
      "key": key,
      "opt_state": opt_state
  }
  return checkpoint_dict


def parse_sgd_checkpoint_dict(checkpoint_dict):
  field_names = ["iteration", "params", "net_state", "opt_state", "key"]
  return [checkpoint_dict[name] for name in field_names]


def make_sgmcmc_checkpoint_dict(iteration, params, net_state, opt_state, key,
                                num_ensembled, predictions,
                                ensemble_predictions):
  checkpoint_dict = {
      "iteration": iteration,
      "params": params,
      "net_state": net_state,
      "opt_state": opt_state,
      "key": key,
      "num_ensembled": num_ensembled,
      "predictions": predictions,
      "ensemble_predictions": ensemble_predictions
  }
  return checkpoint_dict


def parse_sgmcmc_checkpoint_dict(checkpoint_dict):
  field_names = [
      "iteration", "params", "net_state", "opt_state", "key", "num_ensembled",
      "predictions", "ensemble_predictions"
  ]
  return [checkpoint_dict[name] for name in field_names]
