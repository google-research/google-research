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

"""Helpers to define experiments for xmanager runs.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
  name: str
  update_mem: bool
  warmstart: bool
  warmstart_batch_fraction: float
  warmstart_update_model: bool
  reset_memory: bool
  reset_nbd_loader: bool


EXPERIMENTS = [
    ExperimentConfig(
        name="transductive",
        update_mem=True,
        warmstart=True,
        warmstart_batch_fraction=0.2,
        warmstart_update_model=True,
        reset_memory=False,
        reset_nbd_loader=False,
    ),
    ExperimentConfig(
        name="transfer_no_warmstart",
        update_mem=True,
        warmstart=False,
        warmstart_batch_fraction=0.0,
        warmstart_update_model=False,
        reset_memory=True,
        reset_nbd_loader=True,
    ),
    ExperimentConfig(
        name="transfer_warmstart",
        update_mem=True,
        warmstart=True,
        warmstart_batch_fraction=0.2,
        warmstart_update_model=False,
        reset_memory=True,
        reset_nbd_loader=True,
    )
]
