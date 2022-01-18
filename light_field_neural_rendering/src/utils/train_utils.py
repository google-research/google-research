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

"""Utilites for training."""

from clu import metrics
import flax
import flax.linen as nn
import ml_collections
import numpy as np
import optax


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

  decay_fn = optax.linear_schedule(
      init_value=config.train.lr_init,
      end_value=0.,
      transition_steps=config.train.max_steps - config.train.warmup_steps)
  #cosine_steps = max(config.train.max_steps - config.train.warmup_steps, 1)
  #decay_fn = optax.cosine_decay_schedule(
  #    init_value=config.train.lr_init,
  #    decay_steps=cosine_steps)

  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, decay_fn], boundaries=[config.train.warmup_steps])
  return schedule_fn
