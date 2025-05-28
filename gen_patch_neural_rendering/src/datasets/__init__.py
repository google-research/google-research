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

"""Script containing function to build dataset."""

import os
from absl import logging

from gen_patch_neural_rendering.src.datasets import eval_ff_epipolar
from gen_patch_neural_rendering.src.datasets import eval_ibr_epipolar
from gen_patch_neural_rendering.src.datasets import ff_epipolar
from gen_patch_neural_rendering.src.utils import file_utils

dataset_dict = {
    "ff_epipolar": ff_epipolar.FFEpipolar,
}


def create_train_dataset(args, scene):
  """Function to create dataset."""

  train_ds = dataset_dict[args.dataset.name]("train", args, scene)

  return train_ds


def create_finetune_dataset(args):
  """Create the dataset for finetuning."""
  scene = args.dataset.eval_scene
  train_ds = eval_ibr_epipolar.EvalIBREpipolar("train", args, scene)
  eval_ds = eval_ibr_epipolar.EvalIBREpipolar("test", args, scene, train_ds)
  return train_ds, eval_ds


def create_eval_dataset(args):
  """Function to create eval dataset.

  Args:
    args: experiment config.

  Returns:
    train_ds: train  dataset.
    eval_ds_list: a dict of eval datasets.
  """

  if args.dataset.eval_dataset == "llff":
    eval_ds_list = {}
    if not args.dataset.eval_scene:
      scene_list = [
          "fern", "flower", "fortress", "horns", "leaves", "orchids", "room",
          "trex"
      ]
    else:
      scene_list = [args.dataset.eval_scene]

    if args.dev_run:
      scene_list = ["fern"]

    for scene in scene_list:
      logging.info("Loading eval scene {} ===============".format(scene))  # pylint: disable=logging-format-interpolation
      train_ds = eval_ibr_epipolar.EvalIBREpipolar("train", args, scene)
      eval_ds = eval_ibr_epipolar.EvalIBREpipolar("test", args, scene, train_ds)
      eval_ds_list[scene] = eval_ds

  elif args.dataset.eval_dataset == "shiny-6":
    eval_ds_list = {}
    if not args.dataset.eval_scene:
      scene_list = ["crest", "food", "giants", "pasta", "seasoning", "tools"]
    else:
      scene_list = [args.dataset.eval_scene]

    if args.dev_run:
      scene_list = ["crest"]

    for scene in scene_list:
      logging.info("Loading eval scene {} ===============".format(scene))  # pylint: disable=logging-format-interpolation
      train_ds = eval_ff_epipolar.EvalFFEpipolar("train", args, scene)
      eval_ds = eval_ff_epipolar.EvalFFEpipolar("test", args, scene, train_ds)
      eval_ds_list[scene] = eval_ds

  return train_ds, eval_ds_list
