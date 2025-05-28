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

"""Scirpt contaning function to build dataset."""

import os
from absl import logging

from omnimatte3D.src.datasets import davis

DATASET_DICT = {
    "davis": davis.Davis,
}


def create_datasets(args):
  train_ds = DATASET_DICT[args.dataset.name]("train", args)
  eval_ds = DATASET_DICT[args.dataset.name]("test", args)
  return train_ds, eval_ds, None
