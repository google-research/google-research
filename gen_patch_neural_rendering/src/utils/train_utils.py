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

"""Utilities for training."""

import os

from clu import metrics
import flax
import ml_collections
import optax

from gen_patch_neural_rendering.src.utils import file_utils


@flax.struct.dataclass
class Stats:
  loss: float
  psnr: float
  loss_c: float
  psnr_c: float
  weight_l2: float


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  total_loss: metrics.Average.from_output("total_loss")
  train_loss: metrics.Average.from_output("loss")
  train_loss_std: metrics.Std.from_output("loss")
  train_loss_c: metrics.Average.from_output("loss_c")
  train_loss_c_std: metrics.Std.from_output("loss_c")
  learining_rate: metrics.LastValue.from_output("learning_rate")
  train_psnr: metrics.Average.from_output("psnr")
  train_psnr_c: metrics.Average.from_output("psnr_c")
  weight_l2: metrics.Average.from_output("weight_l2")


def create_learning_rate_fn(config,):
  """Create learning rate schedule."""
  # Linear warmup
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=config.train.lr_init,
      transition_steps=config.train.warmup_steps)

  if config.train.scheduler == "linear":
    decay_fn = optax.linear_schedule(
        init_value=config.train.lr_init,
        end_value=0.,
        transition_steps=config.train.max_steps - config.train.warmup_steps)
  elif config.train.scheduler == "cosine":
    cosine_steps = max(config.train.max_steps - config.train.warmup_steps, 1)
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.train.lr_init, decay_steps=cosine_steps)
  elif config.train.scheduler == "step":
    step_steps = max(config.train.max_steps - config.train.warmup_steps, 1)  # pylint: disable=unused-variable

    def schedule(count):
      return config.train.lr_init * (0.5**(count // 50000))

    decay_fn = schedule

  else:
    raise NotImplementedError

  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, decay_fn], boundaries=[config.train.warmup_steps])
  return schedule_fn


def get_train_scene_list(config):
  """Function to get the list of scenes.

  Args:
    config: experiment config.

  Returns:
    scene_path_list: list of scenes.
  """

  if config.dataset.name == "ff_epipolar":
    corrupted_and_test_list = [
        "howardzhou_010_internal_drawing_vase", "howardzhou_059_narcissus",
        "howardzhou_087_yellow_chain_links",
        "howardzhou_089_whilte_bmw_x3_front", "howardzhou_085_sweet_onions",
        "qq18", "qq33", "data2_fernvlsb", "data2_hugetrike", "data2_trexsanta",
        "data3_orchid", "data5_leafscene", "data5_lotr", "data5_redflower"
    ]
    scene_path_list = file_utils.listdir(config.dataset.ff_base_dir)
    scene_path_list = list(set(scene_path_list) - set(corrupted_and_test_list))

  elif config.dataset.name == "dtu":
    with file_utils.open_file(
        os.path.join(config.dataset.dtu_base_dir, "configs", "lists",
                     "dtu_train_all.txt")) as f:
      scene_path_list = [
          line.rstrip().decode("utf-8") for line in f.readlines()
      ]
  elif config.dataset.name == "blender_rot":
    scene_path_list = ["lego"]

  return scene_path_list
